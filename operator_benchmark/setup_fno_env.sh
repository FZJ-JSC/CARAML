set -e pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `operator_learning` directory of benchmark" >&2
    exit 1
fi

export ROOT_DIR=$BENCH_DIR/..
export CUDA_VISIBLE_DEVICES=0

echo "Using ACCELERATOR=$ACCELERATOR"
NVIDIA_X86_ACCELERATORS=(A100 H100 WAIH100)
NVIDIA_ARM_ACCELERATORS=(JUPITER GH200)

PYTORCH_CONTAINER_FILE_NVIDIA_X86=$ROOT_DIR/containers/ngc2508_pytorch28_cuda13_nccl2277_py312.sif
PYTORCH_CONTAINER_FILE_NVIDIA_ARM=$ROOT_DIR/containers/ngc2508_pytorch28_cuda13_nccl2277_py312_arm.sif
PYTORCH_CONTAINER_FILE_DONE=$BENCH_DIR/fno_container_done

PYTORCH_PACKAGES_NVIDIA=$BENCH_DIR/nvidia_fno_packages
PYTORCH_PACKAGES_FILE_NVIDIA=$BENCH_DIR/nvidia_fno_packages_installed
DONE_FILE=$BENCH_DIR/fno_setup_done

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
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html
        apptainer pull $PYTORCH_CONTAINER_FILE_NVIDIA_ARM docker://nvcr.io/nvidia/pytorch:25.08-py3 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_NVIDIA_ARM"  >&2
    fi
else
    if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA_X86 ]; then
        echo "$PYTORCH_CONTAINER_FILE_NVIDIA_X86" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html
        apptainer pull $PYTORCH_CONTAINER_FILE_NVIDIA_X86 docker://nvcr.io/nvidia/pytorch:25.08-py3 >&2
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
if ! [ -f $PYTORCH_PACKAGES_FILE_NVIDIA ] && \
   { [[ " ${NVIDIA_X86_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]] || [[ " ${NVIDIA_ARM_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; }; then

    mkdir -p $PYTORCH_PACKAGES_NVIDIA
    export PIP_USER=0

    if [[ " ${NVIDIA_X86_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; then
        CONTAINER=$PYTORCH_CONTAINER_FILE_NVIDIA_X86
    else
        CONTAINER=$PYTORCH_CONTAINER_FILE_NVIDIA_ARM
    fi

    apptainer exec $CONTAINER \
        python -m pip install \
        --prefix=$PYTORCH_PACKAGES_NVIDIA \
        --no-cache-dir \
        -r $ROOT_DIR/requirements/nvidia_fno_torch_requirements.txt \
        >&2
    
    
    # clone operator_learning code
    cd $BENCH_DIR
    if ! [ -d "operator_learning" ]; then
        git clone -b profiling https://github.com/chelseajohn/operator_learning.git operator_learning
        cd operator_learning
        apptainer exec $CONTAINER \
            python -m pip install --prefix=$PYTORCH_PACKAGES_NVIDIA -e .
        cd ..
    else
        echo "operator_learning directory exists at $BENCH_DIR/ !" >&2
    fi

    # clone pySDC 
    cd $PYTORCH_PACKAGES_NVIDIA/local/lib/python3.12/dist-packages/
    if ! [ -d "pySDC" ]; then
        git clone https://github.com/Parallel-in-Time/pySDC.git pySDC
        cd pySDC
        apptainer exec $CONTAINER \
            python -m pip install --prefix=$PYTORCH_PACKAGES_NVIDIA -e .
    else
        echo "operator_learning directory exists at $BENCH_DIR/ !" >&2
    fi
    cd $BENCH_DIR
    touch $PYTORCH_PACKAGES_FILE_NVIDIA
    echo "Done building additional packages for $ACCELERATOR in $PYTORCH_PACKAGES_NVIDIA" >&2
else
    echo "No additional packages required for $ACCELERATOR" >&2
fi


# Creating wrapper for external torch packages
if ! [ -f "$BENCH_DIR"/nvidia_fno_wrap.sh ]; then
    echo "creating NVIDIA Container wrapper"
    printf "%s\n"  "export PYTHONPATH=$PYTORCH_PACKAGES_NVIDIA/local/lib/python3.12/dist-packages:$BENCH_DIR/operator_learning:\$PYTHONPATH" "\$*" > "$BENCH_DIR"/nvidia_fno_wrap.sh
    chmod u+rwx "$BENCH_DIR"/nvidia_fno_wrap.sh
fi

touch $DONE_FILE

echo "FNO benchmarking setup for NVIDIA $ACCELERATOR done!" >&2
