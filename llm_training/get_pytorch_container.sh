set -euox pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `llm_training` directory of benchmark" >&2
    exit 1
fi

export ROOT_DIR=$BENCH_DIR/..
export CUDA_VISIBLE_DEVICES=0
export ROCM_VISIBLE_DEVICES=0


NVIDIA_X86_ACCELERATORS=(A100 H100 WAIH100)
NVIDIA_ARM_ACCELERATORS=(JEDI GH200)

PYTORCH_CONTAINER_FILE_NVIDIA_X86=$ROOT_DIR/containers/ngc2406_pytorch24_cuda125_nccl2215_py310.sif
PYTORCH_CONTAINER_FILE_NVIDIA_ARM=$ROOT_DIR/containers/ngc2402_pytorch23_cuda123_nccl219_py310_arm.sif
PYTORCH_CONTAINER_FILE_IPU=$ROOT_DIR/containers/ipu_pytorch20_poplar33_py38.sif
PYTORCH_CONTAINER_FILE_AMD=$ROOT_DIR/containers/amd_pytorch21_rocm612_rccl2186_py39.sif
PYTORCH_CONTAINER_FILE_DONE=$BENCH_DIR/pytorch_container_done

PYTORCH_PACKAGES_IPU=$BENCH_DIR/ipu_torch_packages
PYTORCH_PACKAGES_FILE_IPU=$BENCH_DIR/ipu_torch_packages_installed
PYTORCH_PACKAGES_AMD=$BENCH_DIR/amd_torch_packages
PYTORCH_PACKAGES_FILE_AMD=$BENCH_DIR/amd_torch_packages_installed
PYTORCH_PACKAGES_NVIDIA_X86=$BENCH_DIR/nvidia_x86_torch_packages
PYTORCH_PACKAGES_FILE_NVIDIA_X86=$BENCH_DIR/nvidia_x86_torch_packages_installed
PYTORCH_PACKAGES_NVIDIA_ARM=$BENCH_DIR/nvidia_arm_torch_packages
PYTORCH_PACKAGES_FILE_NVIDIA_ARM=$BENCH_DIR/nvidia_arm_torch_packages_installed

if ! [ -d "$ROOT_DIR/containers" ]; then
    mkdir -p "$ROOT_DIR/containers"
fi

if ! [ -d "$ROOT_DIR/containers/tmp_dir" ]; then
   mkdir "$ROOT_DIR/containers/tmp_dir"
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
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html
        apptainer pull $PYTORCH_CONTAINER_FILE_NVIDIA_ARM docker://nvcr.io/nvidia/pytorch:24.02-py3 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_NVIDIA_ARM"  >&2
    fi
elif [ "$ACCELERATOR" = "GC200" ]; then
    if [ -f $PYTORCH_CONTAINER_FILE_IPU ]; then
        echo "$PYTORCH_CONTAINER_FILE_IPU" exists >&2
    else
        # https://hub.docker.com/layers/graphcore/pytorch/3.3.0-ubuntu-20.04-20230703/images/sha256-7f65b5ff5bdc2dad3c112e45e380dc2549113d3eec181d4cf04df6a006cd42a4?context=explore
        apptainer pull $PYTORCH_CONTAINER_FILE_IPU docker://graphcore/pytorch:3.3.0-ubuntu-20.04-20230703 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_IPU"  >&2
    fi
elif [ "$ACCELERATOR" = "MI250" ]; then
    if [ -f $PYTORCH_CONTAINER_FILE_AMD ]; then
        echo "$PYTORCH_CONTAINER_FILE_AMD" exists >&2
    else
        # https://hub.docker.com/layers/rocm/pytorch/rocm6.1.2_ubuntu20.04_py3.9_pytorch_release-2.1.2/images/sha256-e3c1c3cde0886689b139daad7a62ad24af3f292855f683d7b28806ae9f1d2a7e?context=explore
        apptainer pull $PYTORCH_CONTAINER_FILE_AMD docker://rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_release-2.1.2 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_AMD"  >&2
    fi
else
    if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA_X86 ]; then
        echo "$PYTORCH_CONTAINER_FILE_NVIDIA_X86" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-06.html
        apptainer pull $PYTORCH_CONTAINER_FILE_NVIDIA_X86 docker://nvcr.io/nvidia/pytorch:24.06-py3 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_NVIDIA_X86"  >&2
    fi
fi

if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA_X86 ] && [ -f $PYTORCH_CONTAINER_FILE_NVIDIA_ARM ] && [ -f $PYTORCH_CONTAINER_FILE_IPU ] && [ -f $PYTORCH_CONTAINER_FILE_AMD ]; then
    touch $PYTORCH_CONTAINER_FILE_DONE
    echo "Done pulling LLM Pytorch Containers!" >&2
fi

rm -rf $APPTAINER_CACHEDIR
rm -rf $APPTAINER_TMPDIR

##### Installing Requirements #####
if ! [ -f $PYTORCH_PACKAGES_FILE_IPU ] && [ "$ACCELERATOR" = "GC200" ]; then
    mkdir -p $PYTORCH_PACKAGES_IPU
    export PIP_USER=0 
    apptainer exec --cleanenv $PYTORCH_CONTAINER_FILE_IPU \
                    $BENCH_DIR/ipu_torch_wrap.sh \
                    python -m pip install \
                    --prefix=$PYTORCH_PACKAGES_IPU \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/requirements/ipu_torch_requirements.txt \
                    >&2
    touch $PYTORCH_PACKAGES_FILE_IPU
    echo "Done building additional packages for $ACCELERATOR in $PYTORCH_PACKAGES_IPU " >&2
elif ! [ -f $PYTORCH_PACKAGES_FILE_AMD ] && [ "$ACCELERATOR" = "MI250" ]; then
    mkdir -p $PYTORCH_PACKAGES_AMD
    export PIP_USER=0 
    apptainer exec --cleanenv $PYTORCH_CONTAINER_FILE_AMD \
                    python -m pip install \
                    --prefix=$PYTORCH_PACKAGES_AMD \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/requirements/amd_torch_requirements.txt \
                    >&2
    touch $PYTORCH_PACKAGES_FILE_AMD
    echo "Done building additional packages for $ACCELERATOR in $PYTORCH_PACKAGES_AMD " >&2
elif ! [ -f $PYTORCH_PACKAGES_FILE_NVIDIA_X86 ] && [[ " ${NVIDIA_X86_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; then
    mkdir -p $PYTORCH_PACKAGES_NVIDIA_X86
    export PIP_USER=0 
    apptainer exec --cleanenv $PYTORCH_CONTAINER_FILE_NVIDIA_X86 \
                    python -m pip install \
                    --prefix=$PYTORCH_PACKAGES_NVIDIA_X86 \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/requirements/nvidia_x86_torch_requirements.txt \
                    >&2
    touch $PYTORCH_PACKAGES_FILE_NVIDIA_X86
    echo "Done building additional packages for $ACCELERATOR in $PYTORCH_PACKAGES_NVIDIA_X86 " >&2
elif ! [ -f $PYTORCH_PACKAGES_FILE_NVIDIA_ARM ] && [[ " ${NVIDIA_ARM_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; then
    mkdir -p $PYTORCH_PACKAGES_NVIDIA_ARM
    export PIP_USER=0 
    apptainer exec --cleanenv $PYTORCH_CONTAINER_FILE_NVIDIA_ARM \
                    python -m pip install \
                    --prefix=$PYTORCH_PACKAGES_NVIDIA_ARM \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/requirements/nvidia_arm_torch_requirements.txt \
                    >&2
    touch $PYTORCH_PACKAGES_FILE_NVIDIA_ARM
    echo "Done building additional packages for $ACCELERATOR in $PYTORCH_PACKAGES_NVIDIA_ARM " >&2
else
    echo "No additional packages required for $ACCELERATOR" >&2
fi

