import os
import time
import argparse
import contextlib
from tabulate import tabulate
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from jpwr.ctxmgr import get_power
import platform
from slugify import slugify

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Benchmark')
    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument('--arch', default='resnet50', choices=models.__dict__.keys(),
                        help='Model architecture (default: resnet50)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument( "--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--precision', type=str, choices=['fp32', 'tf32', 'fp16', 'bf16'], 
                        default='fp16',
                        help='Specify precision mode (fp32, tf32, fp16, bf16)')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable Distributed Data Parallel (DDP)')
    parser.add_argument("--compiler", type=str, default=None,
                        choices=["trace", "inductor", "aot_eager", "None"],
                        help="Optimization to enable for model")
    parser.add_argument("--compiler_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="Mode for torch.compile(backend=inductor)")
    parser.add_argument('--train_optim', action='store_true',
                        help='torch.compile(train_func)')    
    parser.add_argument('--channels_last', type=bool, default=True,
                    help='Use channels last memory format for tensors.')
    parser.add_argument('--save_checkpoint',action='store_true',
                        help='save model checkpoint at end of training' )   
    parser.add_argument('--loss_logging', action='store_true',
                        help='Log train and validation loss')    
                           
    return parser.parse_args()

@functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    else:
        # only one process
        return True

def get_local_rank():
    """Return the local rank of this process."""
    return int(os.getenv('LOCAL_RANK', 0))

def print0(*args, **kwargs):
    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)

def save0(*args, **kwargs):
    """Pass the given arguments to `torch.save`, but only on the root
    process.
    """
    if is_root_process():
        torch.save(*args, **kwargs)

def all_reduce_avg(tensor):
    """Return the average of the given tensor across all processes."""
    result = tensor.clone()
    if dist.is_initialized():
        dist.all_reduce(result, dist.ReduceOp.AVG)
    return result

def get_device(local_rank):
    """Return the device to benchmark on. """
    if torch.cuda.is_available() and not torch.version.hip:
        device = torch.device('cuda', local_rank)
        print0(f'Using {torch.cuda.get_device_name(0)} GPU')
        torch.cuda.set_device(device)
    elif torch.version.hip and torch.cuda.is_available():  # ROCm support
        device = torch.device('cuda', local_rank)
        print0(f'Using {torch.cuda.get_device_name(0)} GPU')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
        print0('No GPUs Detected, using CPU')

    return device

def get_dtype(precision: str):
    """ Setting the precision. """
    if precision == 'tf32':
        return torch.float32  # TF32 19 bits mantissa mapped only on NVIDIA 
    elif precision == 'fp16':
        return torch.float16  # FP16 is 16-bit floating point precision
    elif precision == 'bf16':
        return torch.bfloat16  # BF16 (Brain Float) is a 16-bit floating point precision
    else:
        return torch.float32  # FP32 default with 23 bits mantissa
    
def set_optimisations(args, model, device):

    if args.compiler == "trace":
        input = torch.randn((args.batch_size, 3, 256, 256)).to(device)
        model_opt = torch.jit.trace(model, input)
        return model_opt, model_opt.parameters()

    # torch.compile mode options for inductor:
    # default: optimizes for large models, low compile-time
    #          and no extra memory usage
    # reduce-overhead: optimizes to reduce the framework overhead
    #                and uses some extra memory. Helps speed up small models
    # max-autotune: optimizes to produce the fastest model,
    #               but takes a very long time to compile

    if args.compiler == "inductor" and not args.train_optim:
        model_opt = torch.compile(model, backend='inductor', mode=args.compiler_mode)
        return model_opt, model_opt.parameters()

    if args.compiler == "aot_eager" and not args.train_optim:
        model_opt = torch.compile(model, backend='aot_eager')
        return model_opt, model_opt.parameters()

    return model, model.parameters()
    
class data_prefetcher():
    """Based on prefetcher from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        if input is None:
            raise StopIteration
        return input, target

def create_datasets(args, device):
    """Return the train and validation datasets wrapped
    in a dataloader.
    """

    dataset = datasets.FakeData(
        size=10000,
        transform=transforms.ToTensor(),
    )

    valid_length = len(dataset) // 10
    train_length = len(dataset) - valid_length

    train_dataset, valid_dataset= torch.utils.data.random_split(
        dataset,
        [train_length, valid_length],
    )

    train_sampler = None
    valid_sampler = None
    if dist.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
 
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=args.num_workers > 0,
        collate_fn=None,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=valid_sampler,
        persistent_workers=args.num_workers > 0,
        collate_fn=None,
    )

    return train_loader, val_loader, dataset

class NoScale:
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass

@contextlib.contextmanager
def scaling(enable_mp, dtype):
    if enable_mp:
        with torch.autocast(device_type="cuda",dtype=dtype):
            yield
    else:
        yield

def initialize_model(args, device):
    model = models.__dict__[args.arch]().to(device)

    if args.precision == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
    elif args.precision in ['fp16', 'bf16']:
        torch.backends.cuda.matmul.allow_tf32 = False

    model, params = set_optimisations(args, model, device)
    args.global_batch_size = args.world_size * args.batch_size

    if args.channels_last:
        args.memory_format = torch.channels_last
    else:
        args.memory_format = torch.contiguous_format

    model = model.to(memory_format=args.memory_format)

    # Scale learning rate based on global batch size
    args.lr = args.lr
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params,lr=args.lr)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    enable_mixed_precision = True if args.precision in ['fp16', 'bf16'] and torch.cuda.is_available() else False
    if enable_mixed_precision:
        scaler = torch.cuda.amp.GradScaler(init_scale=1,
                                           growth_factor=2,
                                           backoff_factor=0.5,
                                           growth_interval=100,
                                           enabled=enable_mixed_precision)
    else:
        scaler = NoScale()
   
    return model, criterion, optimizer, scheduler, scaler, enable_mixed_precision

# Training loop
def train_one_epoch(args, model, criterion, optimizer, scaler, 
                    train_loader, epoch, enable_mp, device, dtype):
    if hasattr(model, 'train'):
        model.train()

    step = 0
    data_iterator = data_prefetcher(train_loader)
    data_iterator = iter(data_iterator)

    for i, (inputs, targets) in enumerate(data_iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with scaling(enable_mp, dtype):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss = loss.contiguous()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        loss_avg = all_reduce_avg(loss).item()
        step += 1
        if step % 10 == 0 and args.loss_logging:
                print0(f'[{epoch}/{args.epochs}; {step - 1}] train loss: {loss_avg:.5f}')
  
# Validation loop
def val_one_epoch(args, model, criterion, val_loader, epoch,
                  enable_mp, device, dtype):
    """Evaluate the model return the global
    loss over the entire validation  set.
    """

    if hasattr(model, 'eval'):
        model.eval() 
    
    data_iterator = data_prefetcher(val_loader)
    data_iterator = iter(data_iterator)

    with torch.no_grad(): 
        loss = 0.0
        for inputs, targets in data_iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            with scaling(enable_mp, dtype):
                outputs = model(inputs)
                loss += criterion(outputs, targets)
    loss /= len(val_loader)
    val_loss = all_reduce_avg(loss).item()
    if args.loss_logging:
        print0(f'[{epoch}/{args.epochs}] valid loss: {val_loss:.5f}')
    
def main():
    args = parse_arguments()

    method_list = []
    if os.getenv("ACCELERATOR") in ["MI250", "MI300X"]:
      from jpwr.gpu.rocm import power
      method_list.append(power())
      gpu_name = "AMD"
    else:
      # Energy measurement using nvidia-smi
      from jpwr.gpu.pynvml import power
      method_list.append(power())
      gpu_name = "NVIDIA"
    if os.getenv("ACCELERATOR") in ["GH200", "JEDI"]:
      from jpwr.sys.gh import power
      method_list.append(power())
    
    args.world_size = 1
    if args.distributed:
        dist.init_process_group(backend='nccl')
        # Different random seed for each process.
        torch.random.manual_seed(args.seed + dist.get_rank())
        args.world_size = dist.get_world_size()

    args.local_rank = get_local_rank()
    device = get_device(args.local_rank)
    print0('############## Arguments ##############')
    print0(''.join(f'{k}: {v}\n' for k, v in vars(args).items()))
    print0('#######################################')

    with get_power(method_list, 100) as training_scope:

        data_time = time.time()
        train_loader, val_loader, dataset = create_datasets(args, device)
        print0(f'Creating datasets took {np.round(time.time()- data_time,3)} s')

        model, criterion, optimizer, scheduler, scaler, enable_mp = initialize_model(args, device)
        if args.train_optim:
            if args.compiler == 'inductor':
                train_opt = torch.compile(train_one_epoch, backend='inductor', mode=args.compiler_mode)
            else:
                train_opt = torch.compile(train_one_epoch, backend='aot_eager')
        else:
            train_opt = train_one_epoch
        
        if dist.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                gradient_as_bucket_view=True,
            )
    
        images_per_epoch = len(dataset)
        print0(f'Starting training with {images_per_epoch} images...')
        results = []
        epoch_time = []
        images_per_sec = []

        print0(f'Starting warm-up epochs ...')
        warmup_time = time.time()
        # warm up epochs
        for epoch in range(5):
            if dist.is_initialized():
                train_loader.sampler.set_epoch(epoch)
            train_opt(args, model, 
                    criterion, 
                    optimizer, 
                    scaler, 
                    train_loader,
                    epoch, 
                    enable_mp,
                    device,
                    dtype=get_dtype(args.precision)
                )
            val_one_epoch(args, model,
                        criterion, 
                        val_loader,
                        epoch,
                        enable_mp,
                        device,
                        dtype=get_dtype(args.precision)
                )
            if dist.is_initialized():
                torch.cuda.synchronize()
        print0(f'End of warm-up epochs (took {np.round(time.time()-warmup_time,3)} s)...')

        train_time = time.time()
        for epoch in range(1, args.epochs+1 ):
            start_time = time.time()
            if dist.is_initialized():
                train_loader.sampler.set_epoch(epoch)

            train_opt(args, model, 
                    criterion, 
                    optimizer, 
                    scaler, 
                    train_loader,
                    epoch, 
                    enable_mp,
                    device,
                    dtype=get_dtype(args.precision)
                )
            scheduler.step()
            val_one_epoch(args, model,
                                    criterion, 
                                    val_loader,
                                    epoch,
                                    enable_mp,
                                    device,
                                    dtype=get_dtype(args.precision)
                                    )
            
            if dist.is_initialized():
                torch.cuda.synchronize()

            epoch_time.append(time.time() - start_time)
            images_per_sec.append(images_per_epoch / epoch_time[-1])
        
            if epoch % 10 == 0 or epoch == args.epochs:
                results.append([int(epoch), images_per_sec[-1], epoch_time[-1]])

    if args.save_checkpoint:
        save0({
            'epoch': epoch,
            'arch': args.arch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, "checkpoint.pt")
    
    print0(f'Training for {args.epochs} epoch(s) took {np.round(time.time()-train_time,3)} s')
    print0(tabulate(results, headers=["Epoch", "Images/s", "Time (s)"], tablefmt="grid", \
                    floatfmt=(".0f", ".0f", ".3f")))
    print0(f'Average Images/s: {int(np.mean(images_per_sec))}')

    ##################################
    #        ENERGY STATS HERE       #
    ##################################

    print0('-' * 64)  

    energy_df, additional_data = training_scope.energy()
    nodename  = platform.node()
    rankid    = int(os.getenv("SLURM_PROCID"))
    ranks     = int(os.getenv("SLURM_NTASKS"))
    accelerator = os.getenv("ACCELERATOR")
    power_file_base = f"{accelerator}_power.csv"
    power_file = power_file_base.replace("csv", f"{rankid}.csv")
    training_scope.df["nodename"] = nodename
    training_scope.df["rank"] = rankid
    if not os.path.exists(power_file):
        training_scope.df.to_csv(power_file)
    energy_df["nodename"] = nodename
    energy_df["rank"] = rankid
    energy_file = power_file.replace("csv", f"energy.csv")
    if not os.path.exists(energy_file):
        energy_df.to_csv(energy_file)
    print0("\n=== Energy Statistics (Rank 0 Only) ===")
    print0(f"Energy-per-GPU-list integrated(Wh): \n{energy_df.to_string()}")
    for k,v in additional_data.items():
        additional_path = power_file.replace("csv", f"{slugify(k)}.csv")
        print0(f"Writing {k} df to {additional_path}")
        v.T.to_csv(additional_path)
        print0(f"Energy-per-GPU-list from {k}(Wh): {v.to_string()}")

    print0('-' * 64)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()