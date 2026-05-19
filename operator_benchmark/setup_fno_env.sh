set -e pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `operator_benchmark` directory" >&2
    exit 1
fi

export ROOT_DIR=$BENCH_DIR/..
export CUDA_VISIBLE_DEVICES=0

echo "Using ACCELERATOR=$ACCELERATOR"
NVIDIA_X86_ACCELERATORS=(A100 H100 WAIH100 MILA)
NVIDIA_ARM_ACCELERATORS=(JUPITER GH200)

PYTORCH_CONTAINER_FILE_NVIDIA_X86=$ROOT_DIR/containers/ngc2602_pytorch211_cuda13_nccl2289_py312.sif
PYTORCH_CONTAINER_FILE_NVIDIA_ARM=$ROOT_DIR/containers/ngc2602_pytorch211_cuda13_nccl2289_py312_arm.sif
PYTORCH_CONTAINER_FILE_DONE=$BENCH_DIR/fno_container_done

PYTORCH_PACKAGES_NVIDIA_x86=$BENCH_DIR/nvidia_fno_packages_x86
PYTORCH_PACKAGES_NVIDIA_ARM=$BENCH_DIR/nvidia_fno_packages_arm
PYTORCH_PACKAGES_FILE_NVIDIA_x86=$BENCH_DIR/nvidia_fno_packages_x86_installed
PYTORCH_PACKAGES_FILE_NVIDIA_ARM=$BENCH_DIR/nvidia_fno_packages_arm_installed
DONE_FILE_x86=$BENCH_DIR/fno_setup_x86_done
DONE_FILE_ARM=$BENCH_DIR/fno_setup_arm_done

if ! [ -d "$ROOT_DIR/containers" ]; then
    mkdir -p "$ROOT_DIR/containers"
fi

if ! [ -d "$ROOT_DIR/containers/tmp_dir" ]; then
   mkdir -p "$ROOT_DIR/containers/tmp_dir"
fi

if [ -f $PYTORCH_CONTAINER_FILE_DONE ]; then
    echo "Required containers exists at $ROOT_DIR/containers/" >&2
    echo "To rebuild containers delete $PYTORCH_CONTAINER_FILE_DONE" >&2
else
    export APPTAINER_CACHEDIR=$(mktemp -d -p $ROOT_DIR/containers/tmp_dir)
    export APPTAINER_TMPDIR=$(mktemp -d -p $ROOT_DIR/containers/tmp_dir)
fi


##### Installing Containers #####
if [ "$ACCELERATOR" = "GH200" ]; then
    if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA_ARM ]; then
        echo "$PYTORCH_CONTAINER_FILE_NVIDIA_ARM" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-02.html
        apptainer pull $PYTORCH_CONTAINER_FILE_NVIDIA_ARM docker://nvcr.io/nvidia/pytorch:26.02-py3 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_NVIDIA_ARM"  >&2
    fi
else
    if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA_X86 ]; then
        echo "$PYTORCH_CONTAINER_FILE_NVIDIA_X86" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-02.html
        if [ "$ACCELERATOR" = "MILA" ]; then
            ml load singularity/3.7.1
            singularity pull $PYTORCH_CONTAINER_FILE_NVIDIA_X86 docker://nvcr.io/nvidia/pytorch:26.02-py3 >&2
        else
            apptainer pull $PYTORCH_CONTAINER_FILE_NVIDIA_X86 docker://nvcr.io/nvidia/pytorch:26.02-py3 >&2
        fi
        echo "Done pulling $PYTORCH_CONTAINER_FILE_NVIDIA_X86"  >&2
    fi
fi

if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA_X86 ] && [ -f $PYTORCH_CONTAINER_FILE_NVIDIA_ARM ]; then
    touch $PYTORCH_CONTAINER_FILE_DONE
    echo "Done pulling Operator Learning Pytorch Containers!" >&2
fi

rm -rf $APPTAINER_CACHEDIR
rm -rf $APPTAINER_TMPDIR

##### Installing Requirements #####
if [[ " ${NVIDIA_X86_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; then
    CONTAINER=$PYTORCH_CONTAINER_FILE_NVIDIA_X86
    PYTORCH_PACKAGES_NVIDIA=$PYTORCH_PACKAGES_NVIDIA_x86
    PYTORCH_PACKAGES_FILE_NVIDIA=$PYTORCH_PACKAGES_FILE_NVIDIA_x86
    NVIDIA_WRAP="$BENCH_DIR"/nvidia_x86_fno_wrap.sh
    DONE_FILE=$DONE_FILE_x86
else
    CONTAINER=$PYTORCH_CONTAINER_FILE_NVIDIA_ARM
    PYTORCH_PACKAGES_NVIDIA=$PYTORCH_PACKAGES_NVIDIA_ARM
    PYTORCH_PACKAGES_FILE_NVIDIA=$PYTORCH_PACKAGES_FILE_NVIDIA_ARM
    NVIDIA_WRAP="$BENCH_DIR"/nvidia_arm_fno_wrap.sh
    DONE_FILE=$DONE_FILE_ARM
fi

if [ -f $DONE_FILE ]; then
    echo "No additional packages required for $ACCELERATOR" >&2
else
    mkdir -p $PYTORCH_PACKAGES_NVIDIA
    export PIP_USER=0
    if [ "$ACCELERATOR" = "MILA" ]; then
        singularity exec --nv -B $SCRATCH $CONTAINER \
            python -m pip install \
            --prefix=$PYTORCH_PACKAGES_NVIDIA \
            --no-cache-dir \
            --no-deps \
            -r $ROOT_DIR/requirements/nvidia_fno_torch_requirements.txt \
            >&2
    else
        apptainer exec $CONTAINER \
            python -m pip install \
            --prefix=$PYTORCH_PACKAGES_NVIDIA \
            --no-cache-dir \
            --no-deps \
            -r $ROOT_DIR/requirements/nvidia_fno_torch_requirements.txt \
            >&2
    fi
fi

# clone operator_learning code
cd $BENCH_DIR
if ! [ -d "operator_learning" ]; then
    git clone https://github.com/chelseajohn/operator_learning.git operator_learning
    cd operator_learning
    if [ "$ACCELERATOR" = "MILA" ]; then
        singularity exec --nv -B $SCRATCH $CONTAINER \
            python -m pip install --prefix=$PYTORCH_PACKAGES_NVIDIA -e .
    else
        apptainer exec $CONTAINER \
            python -m pip install --prefix=$PYTORCH_PACKAGES_NVIDIA -e .
    fi
    cd ..
else
    echo "operator_learning directory exists at $BENCH_DIR/ !" >&2
fi

# clone fno data from hugging face
# TODO: add pic3d data into hf
if ! [ -d "fno_data" ]; then
    git clone https://huggingface.co/datasets/chelseajohn/FNOBenchmark fno_data
else
    echo "operator_learning directory exists at $BENCH_DIR/ !" >&2
fi

# clone pySDC 
cd $PYTORCH_PACKAGES_NVIDIA/local/lib/python*/dist-packages/
if ! [ -d "pySDC" ]; then
    git clone https://github.com/Parallel-in-Time/pySDC.git pySDC
    cd pySDC
    if [ "$ACCELERATOR" = "MILA" ]; then
        singularity exec --nv -B $SCRATCH $CONTAINER \
            python -m pip install --prefix=$PYTORCH_PACKAGES_NVIDIA -e .
    else
        apptainer exec $CONTAINER \
            python -m pip install --prefix=$PYTORCH_PACKAGES_NVIDIA -e .
    fi
else
    echo "operator_learning directory exists at $BENCH_DIR/ !" >&2
fi
cd $BENCH_DIR
touch $PYTORCH_PACKAGES_FILE_NVIDIA
echo "Done building additional packages for $ACCELERATOR in $PYTORCH_PACKAGES_NVIDIA" >&2
# Creating wrapper for external torch packages
if ! [ -f $NVIDIA_WRAP ]; then
    echo "creating NVIDIA Container wrapper"
    printf "%s\n"  "export PYTHONPATH=$PYTORCH_PACKAGES_NVIDIA/local/lib/python3.12/dist-packages:$BENCH_DIR/operator_learning:\$PYTHONPATH" "export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real/libcuda.so.1" "\$*" > $NVIDIA_WRAP
    chmod u+rwx $NVIDIA_WRAP
fi
touch $DONE_FILE

echo "FNO benchmarking setup for NVIDIA $ACCELERATOR done!" >&2
