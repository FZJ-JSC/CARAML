#!/usr/bin/env bash

set -euox pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `llm_training` directory of benchmark" >&2
    exit 1
fi

DONE_FILE=$BENCH_DIR/llm_amd_build_done
PATCH_APPLIED=$BENCH_DIR/megatron_rocm_patch_applied

if [ -f $DONE_FILE ]; then
    echo "$DONE_FILE exists" >&2
    echo "To rebuild setup delete $DONE_FILE,exiting" >&2
    exit 0
fi

export ROCM_VISIBLE_DEVICES=0

export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"

cd $BENCH_DIR

if ! [ -f "$BENCH_DIR"/amd_torch_wrap.sh ]; then
    printf "%s\n"  "export PYTHONPATH=$BENCH_DIR/amd_torch_packages/lib/python3.9/site-packages:\$PYTHONPATH" "\$*" > "$BENCH_DIR"/amd_torch_wrap.sh
    chmod u+rwx "$BENCH_DIR"/amd_torch_wrap.sh
fi

# Clone Megatron-LM for ROCm
if ! [ -d "Megatron-LM-ROCm" ]; then
   git clone https://github.com/bigcode-project/Megatron-LM.git Megatron-LM-ROCm
else
   echo "Megatron-LM-ROCm directory exists at $BENCH_DIR/ !" >&2
fi

cd "$BENCH_DIR"/Megatron-LM-ROCm
# fixing the commit 
git checkout 21045b59127cd2d5509f1ca27d81fae7b485bd22

# apply rocm_patch
if ! [ -f "$PATCH_APPLIED" ]; then
    git apply "$BENCH_DIR"/aux/amd_energy_llm_fix.patch
    touch $PATCH_APPLIED
fi

# Modified PyTorch launcher for JSC systems 
if ! [ -f "fixed_torch_run.py" ]; then
  ln -sf "$BENCH_DIR"/aux/fixed_torch_run.py ./fixed_torch_run.py
fi

# Power script without jpwr
# if ! [ -f "get_power_rsmi.py" ]; then
#   ln -sf "$BENCH_DIR"/aux/get_power_rsmi.py ./get_power_rsmi.py
# fi

cd ..
touch $DONE_FILE

echo "LLM training benchmark setup for AMD done!" >&2
