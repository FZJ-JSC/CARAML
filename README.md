# CARAML 

**C**ompact **A**utomated **R**eproducible **A**ssessment of **M**achine **L**earning (**CARAML**)  is a benchmark to assess main stream Computer Vision and Natrual Language Processing work loads on novel accelerators. It is developed and tested on systems of Jülich Supercomputing Centre (JSC).

CARAML benchmark is automated and made compact with the help of [JUBE](https://apps.fz-juelich.de/jsc/jube/docu/index.html), a scripting based framekwork to easily create benchmark sets, run those sets on different computer systems and evaluate the results. Additionally, the benchmarks are supplemented with power/energy measuring feature using [jpwr](https://github.com/FZJ-JSC/jpwr).

With the usage of `JUBE` CARAML provides easy and reproducible way to benchmark different systems and model configurations with minimal effort.

## Tested Accelerators:

CARAML has been tested on the [JURECA-DC EVALUATION PLATFORM](https://apps.fz-juelich.de/jsc/hps/jureca/evaluation-platform-overview.html), [JURECA-DC](https://apps.fz-juelich.de/jsc/hps/jureca/configuration.html), [JEDI](https://apps.fz-juelich.de/jsc/hps/jedi/index.html#) and [WEST-AI Nodes](https://westai.de/services/hardware/). These include the  accelerators: 

- AMD MI200 node with 4 $\times$ MI250 GPUs (`tag: MI250`)
- Graphcore IPU-POD4 M2000 with 4 $\times$ GC200 IPUs (`tag: GC200`)
- NVIDIA Ampere node (SXM)with 4 $\times$ A100 GPUs (`tag: A100`)
- NVIDIA Hopper node (PCIe) with 4 $\times$ H100 GPUs (`tag: H100`)
- NVIDIA Hopper node (NVLink) with 4 $\times$ H100 GPUs (`tag: WAIH100`)
- NVIDIA Grace-Hopper chip with 1 $\times$ GH200 GPU (`tag: GH200`)
- NVIDIA Grace-Hopper Node with 4 $\times$ GH200 GPUs (`tag: JEDI`)

# Benchmark

CARAML currently offers two benchmarks written in `python`:
- Computer Vision: [ResNet50](./resnet50/) benchmark  implemented in TensorFlow curated from forked versions of 
    - [tensorflow/benchmarks](https://github.com/tensorflow/benchmarks) for NVIDIA and AMD 
    - [graphcore/examples](https://github.com/graphcore/examples) for Graphcore

- GPT Language Model: [LLM-training](./llm_training/) implemented in PyTorch curated from 
    - [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git) with commit: `f7727433293427bef04858f67b2889fe9b177d88` and [patch](./llm_training/aux/nvidia_energy_llm_fix.patch) applied  for NVIDIA
    - [Megatron-LM-ROCm](https://github.com/bigcode-project/Megatron-LM.git) with commit: `21045b59127cd2d5509f1ca27d81fae7b485bd22` and [patch](./llm_training/aux/amd_energy_llm_fix.patch) applied for AMD 
    - [graphcore/examples](https://github.com/graphcore/examples) forked version for Graphcore

# Requirements

To run the benchmark `JUBE` must be installed. Refer to [JUBE Installation Documentation](https://apps.fz-juelich.de/jsc/jube/docu/tutorial.html#installation). The containers are deployed using [Apptainer](https://apptainer.org/) images and SLURM on the accelerators.

## Dataset

For ResNet50, either download the `ImageNet LSVRC 2012` dataset from the [source](http://image-net.org/download) or [via kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) (Disk space required: 144 GB) or use tag `synthetic` with `JUBE` to use synthetic data for benchmark.

For LLM training, a subset (790 samples, 10 MB) of the small version of the [OSCAR](https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz) dataset that is pre-processed using [GPT-2 tokenizers](./llm_training/aux/tokenizers/) is provided in [llm_data](./llm_training/llm_data/).

# Implementation

## ResNet50

The `JUBE` file [resnet50_benchmark.xml](./resnet50/resnet50_benchmark.xml) sets up the enviroment by

- Pulling TensorFlow containers and `pip` installing additional packages using [get_tensorflow_container.sh](./resnet50/get_tensorflow_container.sh) file
- Cloning:
    - [tf_cnn_benchmarks](https://github.com/chelseajohn/tf_cnn_benchmarks) (forked version) for NVIDIA & AMD 
    - [examples](https://github.com/chelseajohn/examples) (forked version) for Graphcore

The performance is measured in terms of `images/sec`.

## LLM-Training

The `JUBE` file [llm_benchmark_nvidia_amd.yaml](./llm_training/llm_benchmark_nvidia_amd.yaml) and [llm_benchmark_ipu.yaml](./llm_training/llm_benchmark_ipu.yaml) sets up the environent by
- Pulling PyTorch containers and `pip` installing additional packages using [get_pytorch_container.sh](./llm_training/get_pytorch_container.sh) file
- Cloning:
    - [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git) with commit: `f7727433293427bef04858f67b2889fe9b177d88` and applying [patch](./llm_training/aux/nvidia_energy_llm_fix.patch) using [setup_llm.sh](./llm_training/setup_llm.sh) file for NVIDIA,
    - [Megatron-LM-ROCm](https://github.com/bigcode-project/Megatron-LM.git) with commit: `21045b59127cd2d5509f1ca27d81fae7b485bd22` and applying [patch](./llm_training/aux/amd_energy_llm_fix.patch) using [setup_llm_amd.sh](./llm_training/setup_amd_llm.sh) file for AMD 
    - [examples](https://github.com/chelseajohn/examples) (forked version) for Graphcore

The performance is measured  in terms of  `tokens/sec`.

# Execution 

Clone this repository and `cd` into it as 

```bash
git clone https://github.com/FZJ-JSC/CARAML.git
cd CARAML
```
## ResNet50
Set the required `system` and `model` parameters and the path to downloaded ImageNet data in [resnet50_benchmark.xml](./resnet50/resnet50_benchmark.xml)

- To pull the required container use `container` tag as:
    - NVIDIA A100 and H100 GPUs
    ```bash
    jube run  resnet50/resnet50_benchmark.xml --tag container H100
    ```
    - NVIDIA GH200 and JEDI GPUs
    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag container GH200
    ```
    - AMD
    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag container MI250
    ```
    - Graphcore
    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag container GC200
    ```
- To run the benchmark with defined configurations do
    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag A100
    ```
    OR with `synthetic data`

    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag A100 synthetic
    ```

    `A100` can be replaced with `H100`, `WAIH100`, `GH200`, `JEDI`, `MI250` and `GC200` for the respective systems.

- After the benchmark has been executed, use `jube continue` to postprocess results
    ```bash
   jube continue resnet50/resnet50_benchmark_run -i last
   ```

- After the postprocessing, to get the result do
   ```bash
    jube result resnet50/resnet50_benchmark_run -i last
   ```
- Example result 
    ```bash
    JobID,System,Version,Queue,Runtime(s),Model,Dataset,Nodes,Devices,Tasks/Node,Threads/Task,GlobalBatchSize,BatchSize/Device,Images/second,EnergyFile
    13077565,MI250,2024.01,dc-mi200,54.71,resnet50_v2,ImageNet,1,8,8,4,64,8,2107.00,CARAML/resnet50/resnet50_benchmark_run/000004/000002_combine_energy/work/combined_energy.csv
    
    JobID,System,Version,Queue,Runtime(s),Model,Dataset,Nodes,Devices,Tasks/Node,Threads/Task,GlobalBatchSize,BatchSize/Device,Images/second,EnergyFile
    13082568,GC200,2024.01,dc-ipu,1.0,resnet50_mlperf_pod4_bs20,ImageNet,1,4,1,12,32,8,3556.18,CARAML/resnet50/resnet50_benchmark_run/000000/000000_execute/work/GC200_power.0.energy.csv

    JobID,System,Version,Queue,Runtime(s),Model,Dataset,Nodes,Devices,Tasks/Node,Threads/Task,GlobalBatchSize,BatchSize/Device,Images/second,EnergyFile
    13080521,H100,2024.01,dc-h100,89.67,resnet50_v2,ImageNet,1,4,4,4,32,8,1994.69,CARAML/resnet50/resnet50_benchmark_run/000000/000001_combine_energy/work/combined_energy.csv
    ```
## LLM-Training
Set the required `system` and `model` parameters  in [llm_benchmark_nvidia_amd.yaml](./llm_training/llm_benchmark_nvidia_amd.yaml)
for NVIDIA and AMD devices and in [llm_benchmark_ipu.yaml](./llm_training/llm_benchmark_ipu.yaml) for Graphcore

- To pull the required container and build packages use `container` tag as:
    - NVIDIA A100 and H100 GPUs
    ```bash
    jube run llm_training/llm_benchmark_nvidia_amd.yaml --tag container H100
    ```

    - NVIDIA GH200 and JEDI GPUs
    ```bash
    jube run llm_training/llm_benchmark_nvidia_amd.yaml--tag container GH200
    ```
    - AMD MI250
    ```bash
    jube run llm_training/llm_benchmark_nvidia_amd.yaml--tag container MI250
    ``` 
   - Graphcore GC200
    ```bash
    jube run llm_training/llm_benchmark_ipu.yaml --tag container 
    ```

- To run the benchmark with defined configurations for `800M` GPT model with OSCAR data do
    ```bash
    jube run llm_training/llm_benchmark_nvidia_amd.yaml --tag 800M A100
    ```
    `A100` can be replaced with `H100`, `WAIH100`, `GH200`, `JEDI` and `MI250` for the respective systems and `800M` can be replaced with `175B` and `13B` for systems with more node resources like `JEDI`,`H100`, `A100` and `MI250`.

- To run the benchmark with defined configurations for `117M` GPT model on Graphcore with synthetic data  do
    ```bash
    jube run llm_training/llm_benchmark_ipu.yaml --tag 117M synthetic
    ```
    If tag `synthetic` is not given, the benchmark will use OSCAR data

- After the benchmark has been executed, use `jube continue` to postprocess results
    ```bash
    jube continue llm_training/llm_benchmark_nvidia_amd_run -i last
   ```
    OR

   ```bash
    jube continue llm_training/llm_benchmark_ipu_run -i last
   ```

- After the postprocessing, to get the result do
   ```bash
   jube result llm_training/llm_benchmark_nvidia_amd_run -i last
   ```
    OR

   ```bash
   jube result llm_training/llm_benchmark_ipu_run -i last
   ```
 

- Example result

```bash
JobID,System,Version,Queue,JobTime,Runtime(min),Model,ModelSize,Dataset,Nodes,Devices,GlobalBatchSize,PipelineParallel,TensorParallel,DataParallel,Iterations,Time/iteration(s),Tokens/second,Avg_TFLOPs/GPU,EnergyFile
13077019,MI250,2024.01,dc-mi200,00:15:00,10,GPT,800M,OSCAR,1,8,32,1,1,8,750,0.74,88620.76,69.35,CARAML/llm_training/llm_benchmark_nvidia_amd_run/000006/000002_combine_energy/work/combined_energy.csv

JobID,System,Version,Queue,JobTime,Model,ModelSize,Dataset,Nodes,Devices,DataParallel,IPU/replica,GlobalBatchSize,Time/iteration(s),Tokens/second,EnergyFile
13011841,GC200,2024.01,dc-ipu,00:40:00,GPT,117M,Synthetic,1,4,1,4,2048,11.17,183.37,CARAML/llm_training/llm_benchmark_ipu_run/000003/000002_combine_energy/work/combined_energy.csv

JobID,System,Version,Queue,JobTime,Runtime(min),Model,ModelSize,Dataset,Nodes,Devices,GlobalBatchSize,PipelineParallel,TensorParallel,DataParallel,Iterations,Time/iteration(s),Tokens/second,Avg_TFLOPs/GPU,EnergyFile
3914,JEDI,2024.01,all,00:34:00,30,GPT,800M,OSCAR,1,4,2048,1,1,4,25,26.52,158152.80,321.65,CARAML/llm_training/llm_benchmark_nvidia_amd_run/000025/000002_combine_energy/work/combined_energy.csv
```

# JSC Specific Fixes
In order to use PyTorch `torch run` API on JSC systems [fixed_torch_run.py](./llm_training/aux/fixed_torch_run.py) fix is required. The fix solves the issue defined [here](https://github.com/pytorch/pytorch/pull/81691).

Additionally the `hostname` is appended with an `i` for allowing communication over InfiniBand as described [here](https://apps.fz-juelich.de/jsc/hps/juwels/known-issues.html#ip-connectivity-on-compute-nodes).