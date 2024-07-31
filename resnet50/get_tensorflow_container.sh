set -euox pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `resnet50` directory of benchmark" >&2
    exit 1
fi

export ROOT_DIR=$BENCH_DIR/..

TENSORFLOW_CONTAINER_FILE_NVIDIA=$ROOT_DIR/containers/ngc2301_tf115_cuda1201_nccl2165_py38.sif
TENSORFLOW_CONTAINER_FILE_GH=$ROOT_DIR/containers/ngc2301_tf115_cuda1201_nccl2165_py38_arm.sif
TENSORFLOW_CONTAINER_FILE_MI250=$ROOT_DIR/containers/amd_tf27_rocm50_py39-dev.sif
TENSORFLOW_CONTAINER_FILE_GC200=$ROOT_DIR/containers/ipu_tf263_poplar310_py38.sif
TENSORFLOW_CONTAINER_FILE_DONE=$BENCH_DIR/tensorflow_container_done

TENSORFLOW_PACKAGES_MI250=$BENCH_DIR/amd_tensorflow_packages
TENSORFLOW_PACKAGES_GC200=$BENCH_DIR/ipu_tensorflow_packages
TENSORFLOW_PACKAGES_FILE_MI250=$BENCH_DIR/amd_tensorflow_packages_installed
TENSORFLOW_PACKAGES_FILE_GC200=$BENCH_DIR/ipu_tensorflow_packages_installed


if ! [ -d "$ROOT_DIR/containers" ]; then
    mkdir -p "$ROOT_DIR/containers/tmp_dir"
fi

if [ -f $TENSORFLOW_CONTAINER_FILE_DONE]; then
    echo "$TENSORFLOW_CONTAINER_FILE_DONE exists, exiting" >&2
    exit 0
else
    export APPTAINER_CACHEDIR=$(mktemp -d -p $ROOT_DIR/containers/tmp_dir)
    export APPTAINER_TMPDIR=$(mktemp -d -p $ROOT_DIR/containers/tmp_dir)
fi

if [ "$ACCELERATOR" = "GH200" ]; then
    if [ -f $TENSORFLOW_CONTAINER_FILE_GH ]; then
        echo "$TENSORFLOW_CONTAINER_FILE_GH" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-01.html
        apptainer pull $TENSORFLOW_CONTAINER_FILE_GH docker://nvcr.io/nvidia/tensorflow:23.01-tf1-py3 >&2
        echo "Done pulling $TENSORFLOW_CONTAINER_FILE_GH" >&2
    fi
elif [ "$ACCELERATOR" = "H100" ]; then
    if [ -f $TENSORFLOW_CONTAINER_FILE_NVIDIA ]; then
        echo "$TENSORFLOW_CONTAINER_FILE_NVIDIA" exists >&2
    else
       # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-01.html
        apptainer pull $TENSORFLOW_CONTAINER_FILE_NVIDIA docker://nvcr.io/nvidia/tensorflow:23.01-tf1-py3 >&2
        echo "Done pulling $TENSORFLOW_CONTAINER_FILE_NVIDIA" >&2
    fi
elif [ "$ACCELERATOR" = "MI250" ]; then
    if [ -f $TENSORFLOW_CONTAINER_FILE_MI250 ]; then
        echo "$TENSORFLOW_CONTAINER_FILE_MI250" exists >&2
    else
        # https://hub.docker.com/layers/rocm/tensorflow/rocm5.0-tf2.7-dev/images/sha256-664fbd3e38234f5b4419aa54b2b81664495ed0a9715465678f2bc14ea4b7ae16?context=explore
        apptainer pull $TENSORFLOW_CONTAINER_FILE_MI250 docker://rocm/tensorflow:rocm5.0-tf2.7-dev >&2
        echo "Done pulling $TENSORFLOW_CONTAINER_FILE_MI250" >&2
    fi
else
    if [ -f $TENSORFLOW_CONTAINER_FILE_GC200 ]; then
        echo "$TENSORFLOW_CONTAINER_FILE_GC200" exists >&2
    else
        # https://hub.docker.com/layers/john856/caraml/tensorflow_poplar310_tf263_mpi4py/images/sha256-57cb664cb1e3493657c576b07a0d274363e1097e62820d2f7e03db5e68fe1f0e?context=repo
        apptainer pull $TENSORFLOW_CONTAINER_FILE_GC200 docker://john856/caraml:tensorflow_poplar310_tf263_mpi4py >&2
        echo "Done pulling $TENSORFLOW_CONTAINER_FILE_GC200" >&2
    fi
fi

if ! [ -f $TENSORFLOW_PACKAGES_FILE_MI250 ] && [ "$ACCELERATOR" = "MI250" ]; then
    mkdir -p $TENSORFLOW_PACKAGES_MI250
    apptainer exec $TENSORFLOW_CONTAINER_FILE_MI250 \
                    python -m pip install \
                    --upgrade pip setuptools distlib \
                    >&2 
    export PIP_USER=0 
    apptainer exec $TENSORFLOW_CONTAINER_FILE_MI250 \
                    python -m pip install \
                    --prefix=$TENSORFLOW_PACKAGES_MI250 \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/amd_mi250_tensorflow_requirements.txt\
                     >&2
    touch $TENSORFLOW_PACKAGES_FILE_MI250
    echo "Done building additional packages for $ACCELERATOR in $TENSORFLOW_PACKAGES_MI250" >&2

elif ! [ -f $TENSORFLOW_PACKAGES_FILE_GC200 ] && [ "$ACCELERATOR" = "GC200" ]; then
    mkdir -p $TENSORFLOW_PACKAGES_GC200
    apptainer exec $TENSORFLOW_CONTAINER_FILE_GC200 \
                    python -m pip install \
                    --upgrade pip setuptools distlib \
                    >&2
    export PIP_USER=0 
    apptainer exec $TENSORFLOW_CONTAINER_FILE_GC200 \
                    python -m pip install \
                    --prefix=$TENSORFLOW_PACKAGES_GC200 \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/ipu_gc200_tensorflow_requirements.txt \
                     >&2
    touch $TENSORFLOW_PACKAGES_FILE_GC200
    echo "Done building additional packages for $ACCELERATOR in $TENSORFLOW_PACKAGES_GC200" >&2

else
    echo "No additional packages required for $ACCELERATOR" >&2
fi

if [ -f $TENSORFLOW_CONTAINER_FILE_NVIDIA ] && \
   [ -f $TENSORFLOW_CONTAINER_FILE_GH  ] && \
   [ -f $TENSORFLOW_CONTAINER_FILE_MI250 ] && \
   [ -f $TENSORFLOW_CONTAINER_FILE_GC200 ]; then
        touch $TENSORFLOW_CONTAINER_FILE_DONE
        echo "Done pulling TensorFlow Containers!" >&2
fi

rm -rf $APPTAINER_CACHEDIR
rm -rf $APPTAINER_TMPDIR