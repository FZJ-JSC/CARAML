#!/usr/bin/env bash

set -euox pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `llm_training` directory of benchmark" >&2
    exit 1
fi

NVIDIA_X86_ACCELERATORS=(A100 H100 WAIH100)
NVIDIA_ARM_ACCELERATORS=(Jedi GH200)

DONE_FILE=$BENCH_DIR/llm_nvidia_build_done
PATCH_APPLIED=$BENCH_DIR/megatron_patch_applied

if [ -f $DONE_FILE ]; then
    echo "$DONE_FILE exists, exiting" >&2
    exit 0
fi

export CUDA_VISIBLE_DEVICES=0

export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"

cd $BENCH_DIR


echo "ACCELERATOR=$ACCELERATOR"

if ! [ -f "$BENCH_DIR"/nvidia_torch_wrap.sh ] && [[ " ${NVIDIA_X86_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; then
    echo "creating NVIDIA X86 wrapper"
    printf "%s\n"  "export PYTHONPATH=$BENCH_DIR/nvidia_torch_packages/local/lib/python3.10/dist-packages:\$PYTHONPATH" "\$*" > "$BENCH_DIR"/nvidia_torch_wrap.sh
    chmod u+rwx "$BENCH_DIR"/nvidia_torch_wrap.sh
elif ! [ -f "$BENCH_DIR"/gh_torch_wrap.sh ] && [[ " ${NVIDIA_ARM_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; then
    echo "creating NVIDIA ARM wrapper"
    printf "%s\n"  "export PYTHONPATH=$BENCH_DIR/gh_torch_packages/local/lib/python3.10/dist-packages:\$PYTHONPATH" "\$*" > "$BENCH_DIR"/gh_torch_wrap.sh
    chmod u+rwx "$BENCH_DIR"/gh_torch_wrap.sh
fi

# Clone Megatron-LM
if ! [ -d "Megatron-LM" ]; then
   git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
else
   echo "Megatron-LM directory exists at $BENCH_DIR/ !" >&2
fi

# Where the Megatron-LM code is stored
MEGATRON_LM_REPO="$BENCH_DIR"/Megatron-LM
[ "x$MEGATRON_LM_REPO" = x ] \
    && echo 'Please set `MEGATRON_LM_REPO` in `llm_variables.bash`' && return 1 >&2

cd "$MEGATRON_LM_REPO"
# fixing the commit 
git checkout f7727433293427bef04858f67b2889fe9b177d88 

# apply add_tflops_logging.patch
if ! [ -f "$PATCH_APPLIED" ]; then
    git apply "$BENCH_DIR"/aux/nvidia_energy_llm_fix.patch
    touch $PATCH_APPLIED
fi

# Modified PyTorch launcher for JSC systems 
if ! [ -f "fixed_torch_run.py" ]; then
  ln -sf "$BENCH_DIR"/aux/fixed_torch_run.py ./fixed_torch_run.py
fi
if ! [ -f "get_power_nvidia.py" ]; then
  ln -sf "$BENCH_DIR"/aux/get_power_nvidia.py ./get_power_nvidia.py
fi
cd ..
touch $DONE_FILE

echo "LLM training benchmark setup done!" >&2
