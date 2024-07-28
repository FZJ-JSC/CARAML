set -euox pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `llm_training` directory of benchmark" >&2
    exit 1
fi

export ROOT_DIR=$BENCH_DIR/..

PYTORCH_CONTAINER_FILE_NVIDIA=$ROOT_DIR/containers/ngc_2406_pytorch24_py310_cuda125_nccl2215.sif
PYTORCH_CONTAINER_FILE_GH=$ROOT_DIR/containers/ngc_2402_pytorch23_py310_cuda123_nccl219_arm.sif
PYTORCH_CONTAINER_FILE_DONE=$BENCH_DIR/pytorch_container_done

if ! [ -d "$ROOT_DIR/containers" ]; then
    mkdir -p "$ROOT_DIR/containers/tmp_dir"
fi

if [ -f $PYTORCH_CONTAINER_FILE_DONE ]; then
    echo "$PYTORCH_CONTAINER_FILE_DONE exists, exiting" >&2
    exit 0
else
    export APPTAINER_CACHEDIR=$(mktemp -d -p $ROOT_DIR/containers/tmp_dir)
    export APPTAINER_TMPDIR=$(mktemp -d -p $ROOT_DIR/containers/tmp_dir)
fi

if [ "$ACCELERATOR" = "GH200" ]; then
    if [ -f $PYTORCH_CONTAINER_FILE_GH ]; then
        echo "$PYTORCH_CONTAINER_FILE_GH" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html
        apptainer pull $PYTORCH_CONTAINER_FILE_GH docker://nvcr.io/nvidia/pytorch:24.02-py3 >&2
        echo "Done pulling $PYTORCH_CONTAINER_FILE_GH"  >&2
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

if [ -f $PYTORCH_CONTAINER_FILE_NVIDIA ] && [ -f $PYTORCH_CONTAINER_FILE_GH ]; then
    touch $PYTORCH_CONTAINER_FILE_DONE
    echo "Done pulling NVIDIA LLM Pytorch Containers!" >&2
fi

rm -rf $APPTAINER_CACHEDIR
rm -rf $APPTAINER_TMPDIR
