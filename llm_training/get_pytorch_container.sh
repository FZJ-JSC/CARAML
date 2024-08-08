set -euox pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `llm_training` directory of benchmark" >&2
    exit 1
fi

export ROOT_DIR=$BENCH_DIR/..
export CUDA_VISIBLE_DEVICES=0
export ROCM_VISIBLE_DEVICES=0

PYTORCH_CONTAINER_FILE_NVIDIA=$ROOT_DIR/containers/ngc2406_pytorch24_cuda125_nccl2215_py310.sif
PYTORCH_CONTAINER_FILE_GH=$ROOT_DIR/containers/ngc2402_pytorch23_cuda123_nccl219_py310_arm.sif
PYTORCH_CONTAINER_FILE_GC200=$ROOT_DIR/containers/ipu_pytorch20_poplar33_py38.sif
PYTORCH_CONTAINER_FILE_MI250=$ROOT_DIR/containers/amd_pytorch21_rocm612_rccl2186_py39.sif
PYTORCH_CONTAINER_FILE_DONE=$BENCH_DIR/pytorch_container_done

PYTORCH_PACKAGES_GC200=$BENCH_DIR/ipu_torch_packages
PYTORCH_PACKAGES_FILE_GC200=$BENCH_DIR/ipu_torch_packages_installed
PYTORCH_PACKAGES_MI250=$BENCH_DIR/amd_torch_packages
PYTORCH_PACKAGES_FILE_MI250=$BENCH_DIR/amd_torch_packages_installed

if ! [ -d "$ROOT_DIR/containers" ]; then
    mkdir -p "$ROOT_DIR/containers"
fi

if ! [ -d "$ROOT_DIR/containers" ]; then
    mkdir "$ROOT_DIR/tmp_dir"
fi


if [ -f $PYTORCH_CONTAINER_FILE_DONE ]; then
    echo "$PYTORCH_CONTAINER_FILE_DONE exists, exiting" >&2
    exit 0
else
    export APPTAINER_CACHEDIR=$(mktemp -d -p $ROOT_DIR/tmp_dir)
    export APPTAINER_TMPDIR=$(mktemp -d -p $ROOT_DIR/tmp_dir)
fi

if [ "$ACCELERATOR" = "GH200" ]; then
    if [ -f $PYTORCH_CONTAINER_FILE_GH ]; then
        echo "$PYTORCH_CONTAINER_FILE_GH" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html
        apptainer pull $PYTORCH_CONTAINER_FILE_GH docker://nvcr.io/nvidia/pytorch:24.02-py3 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_GH"  >&2
    fi
elif [ "$ACCELERATOR" = "GC200" ]; then
    if [ -f $PYTORCH_CONTAINER_FILE_GC200 ]; then
        echo "$PYTORCH_CONTAINER_FILE_GC200" exists >&2
    else
        # https://hub.docker.com/layers/graphcore/pytorch/3.3.0-ubuntu-20.04-20230703/images/sha256-7f65b5ff5bdc2dad3c112e45e380dc2549113d3eec181d4cf04df6a006cd42a4?context=explore
        apptainer pull $PYTORCH_CONTAINER_FILE_GC200 docker://graphcore/pytorch:3.3.0-ubuntu-20.04-20230703 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_GC200"  >&2
    fi
elif [ "$ACCELERATOR" = "MI250" ]; then
    if [ -f $PYTORCH_CONTAINER_FILE_MI250 ]; then
        echo "$PYTORCH_CONTAINER_FILE_MI250" exists >&2
    else
        # https://hub.docker.com/layers/rocm/pytorch/rocm6.1.2_ubuntu20.04_py3.9_pytorch_release-2.1.2/images/sha256-e3c1c3cde0886689b139daad7a62ad24af3f292855f683d7b28806ae9f1d2a7e?context=explore
        apptainer pull $PYTORCH_CONTAINER_FILE_MI250 docker://rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_release-2.1.2 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_MI250"  >&2
    fi
else
    if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA ]; then
        echo "$PYTORCH_CONTAINER_FILE_NVIDIA" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-06.html
        apptainer pull $PYTORCH_CONTAINER_FILE_NVIDIA docker://nvcr.io/nvidia/pytorch:24.06-py3 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_NVIDIA"  >&2
    fi
fi

if ! [ -f $PYTORCH_PACKAGES_FILE_GC200 ] && [ "$ACCELERATOR" = "GC200" ]; then
    mkdir -p $PYTORCH_PACKAGES_GC200
    export PIP_USER=0 
    apptainer exec --cleanenv $PYTORCH_CONTAINER_FILE_GC200 \
                    $BENCH_DIR/ipu_torch_wrap.sh \
                    python -m pip install \
                    --prefix=$PYTORCH_PACKAGES_GC200 \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/ipu_gc200_torch_requirements.txt \
                    >&2
    touch $PYTORCH_PACKAGES_FILE_GC200
    echo "Done building additional packages for $ACCELERATOR in $PYTORCH_PACKAGES_GC200 " >&2
elif ! [ -f $PYTORCH_PACKAGES_FILE_MI250 ] && [ "$ACCELERATOR" = "MI250" ]; then
    mkdir -p $PYTORCH_PACKAGES_MI250
    export PIP_USER=0 
    apptainer exec --cleanenv $PYTORCH_CONTAINER_FILE_MI250 \
                    python -m pip install \
                    --prefix=$PYTORCH_PACKAGES_MI250 \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/amd_mi250_torch_requirements.txt \
                    >&2
    touch $PYTORCH_PACKAGES_FILE_MI250
    echo "Done building additional packages for $ACCELERATOR in $PYTORCH_PACKAGES_MI250 " >&2
else
    echo "No additional packages required for $ACCELERATOR" >&2
fi

if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA ] && [ -f $PYTORCH_CONTAINER_FILE_GH ] && [ -f $PYTORCH_CONTAINER_FILE_GC200 ]  && [ -f $PYTORCH_CONTAINER_FILE_MI250 ]; then
    touch $PYTORCH_CONTAINER_FILE_DONE
    echo "Done pulling LLM Pytorch Containers!" >&2
fi

rm -rf $APPTAINER_CACHEDIR
rm -rf $APPTAINER_TMPDIR
