#!/usr/bin/env bash

set -euox pipefail

if [ "x$ROOT_DIR" = "x" ]; then
    echo "ROOT_DIR is not set. Please set it to the root directory of benchmark" >&2
    exit 1
fi

DONE_FILE=$ROOT_DIR/nlp_build_done
PATCH_APPLIED=$ROOT_DIR/megatron_patch_applied

if [ -f $DONE_FILE ]; then
    echo "$DONE_FILE exists, exiting"
    exit 0
fi

export PIP_CACHE_DIR=$ROOT_DIR/pip_cache

source $ROOT_DIR/llm_variables.bash || exit 1

export CUDA_VISIBLE_DEVICES=0

export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"

cd $ROOT_DIR
# Clone Megatron-LM
if ! [ -d "Megatron-LM" ]; then
   git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
else
   echo "Megatron-LM directory exists at $ROOT_DIR/ !"
fi

# Where the Megatron-LM code is stored
MEGATRON_LM_REPO="$ROOT_DIR"/Megatron-LM
[ "x$MEGATRON_LM_REPO" = x ] \
    && echo 'Please set `MEGATRON_LM_REPO` in `llm_variables.bash.' && return 1

cd "$MEGATRON_LM_REPO"
# fixing the commit 
git checkout f7727433293427bef04858f67b2889fe9b177d88 

#### apply add_tflops_logging.patch
if ! [ -f "$PATCH_APPLIED" ]; then
    git apply "$ROOT_DIR"/aux/add_tflops_logging.patch
    touch $PATCH_APPLIED
fi

## Modified PyTorch launcher for JSC systems 
ln -sf "$ROOT_DIR"/aux/fixed_torch_run.py ./fixed_torch_run.py

cd ..
touch $DONE_FILE

echo "LLM training benchmark setup done!"
