set -euox pipefail

if [ "x$BENCH_DIR" = "x" ]; then
    echo "BENCH_DIR is not set. Please set it to the `resnet50` directory of benchmark" >&2
    exit 1
fi

export ROOT_DIR=$BENCH_DIR/..

NVIDIA_X86_ACCELERATORS=(A100 H100 WAIH100)
NVIDIA_ARM_ACCELERATORS=(JEDI GH200)

TENSORFLOW_CONTAINER_FILE_NVIDIA_X86=$ROOT_DIR/containers/ngc2301_tf115_cuda1201_nccl2165_py38.sif
TENSORFLOW_CONTAINER_FILE_NVIDIA_ARM=$ROOT_DIR/containers/ngc2301_tf115_cuda1201_nccl2165_py38_arm.sif
TENSORFLOW_CONTAINER_FILE_AMD=$ROOT_DIR/containers/amd_tf27_rocm50_py39-dev.sif
TENSORFLOW_CONTAINER_FILE_IPU=$ROOT_DIR/containers/ipu_tf263_poplar310_py38.sif
TENSORFLOW_CONTAINER_FILE_DONE=$BENCH_DIR/tensorflow_container_done

TENSORFLOW_PACKAGES_AMD=$BENCH_DIR/amd_tensorflow_packages
TENSORFLOW_PACKAGES_IPU=$BENCH_DIR/ipu_tensorflow_packages
TENSORFLOW_PACKAGES_NVIDIA_X86=$BENCH_DIR/nvidia_x86_tensorflow_packages
TENSORFLOW_PACKAGES_NVIDIA_ARM=$BENCH_DIR/nvidia_arm_tensorflow_packages
TENSORFLOW_PACKAGES_FILE_AMD=$BENCH_DIR/amd_tensorflow_packages_installed
TENSORFLOW_PACKAGES_FILE_IPU=$BENCH_DIR/ipu_tensorflow_packages_installed
TENSORFLOW_PACKAGES_FILE_NVIDIA_X86=$BENCH_DIR/nvidia_x86_tensorflow_packages_installed
TENSORFLOW_PACKAGES_FILE_NVIDIA_ARM=$BENCH_DIR/nvidia_arm_tensorflow_packages_installed


if ! [ -d "$ROOT_DIR/containers" ]; then
    mkdir -p "$ROOT_DIR/containers"
fi

if ! [ -d "$ROOT_DIR/containers/tmp_dir" ]; then
   mkdir "$ROOT_DIR/containers/tmp_dir"
fi


if [ -f $TENSORFLOW_CONTAINER_FILE_DONE]; then
    echo "Required containers exists at $ROOT_DIR/containers/" >&2
    echo "To rebuild containers delete $TENSORFLOW_CONTAINER_FILE_DONE" >&2
else
    export APPTAINER_CACHEDIR=$(mktemp -d -p $ROOT_DIR/containers/tmp_dir)
    export APPTAINER_TMPDIR=$(mktemp -d -p $ROOT_DIR/containers/tmp_dir)
fi

##### Installing Containers #####
if [ "$ACCELERATOR" = "GH200" ]; then
    export CUDA_VISIBLE_DEVICES=0
    if [ -f $TENSORFLOW_CONTAINER_FILE_NVIDIA_ARM ]; then
        echo "$TENSORFLOW_CONTAINER_FILE_NVIDIA_ARM" exists >&2
    else
        # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-01.html
        apptainer pull $TENSORFLOW_CONTAINER_FILE_NVIDIA_ARM docker://nvcr.io/nvidia/tensorflow:23.01-tf1-py3 >&2
        echo "Done pulling $TENSORFLOW_CONTAINER_FILE_NVIDIA_ARM" >&2
    fi
elif [ "$ACCELERATOR" = "H100" ]; then
    export CUDA_VISIBLE_DEVICES=0
    if [ -f $TENSORFLOW_CONTAINER_FILE_NVIDIA_X86 ]; then
        echo "$TENSORFLOW_CONTAINER_FILE_NVIDIA_X86" exists >&2
    else
       # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-01.html
        apptainer pull $TENSORFLOW_CONTAINER_FILE_NVIDIA_X86 docker://nvcr.io/nvidia/tensorflow:23.01-tf1-py3 >&2
        echo "Done pulling $TENSORFLOW_CONTAINER_FILE_NVIDIA_X86" >&2
    fi
elif [ "$ACCELERATOR" = "MI250" ]; then
    export ROCM_VISIBLE_DEVICES=0
    if [ -f $TENSORFLOW_CONTAINER_FILE_AMD ]; then
        echo "$TENSORFLOW_CONTAINER_FILE_AMD" exists >&2
    else
        # https://hub.docker.com/layers/rocm/tensorflow/rocm5.0-tf2.7-dev/images/sha256-664fbd3e38234f5b4419aa54b2b81664495ed0a9715465678f2bc14ea4b7ae16?context=explore
        apptainer pull $TENSORFLOW_CONTAINER_FILE_AMD docker://rocm/tensorflow:rocm5.0-tf2.7-dev >&2
        echo "Done pulling $TENSORFLOW_CONTAINER_FILE_AMD" >&2
    fi
else
    if [ -f $TENSORFLOW_CONTAINER_FILE_IPU ]; then
        echo "$TENSORFLOW_CONTAINER_FILE_IPU" exists >&2
    else
        # https://hub.docker.com/layers/john856/caraml/tensorflow_poplar310_tf263_mpi4py/images/sha256-57cb664cb1e3493657c576b07a0d274363e1097e62820d2f7e03db5e68fe1f0e?context=repo
        apptainer pull $TENSORFLOW_CONTAINER_FILE_IPU docker://john856/caraml:tensorflow_poplar310_tf263_mpi4py >&2
        echo "Done pulling $TENSORFLOW_CONTAINER_FILE_IPU" >&2
    fi
fi

if [ -f $TENSORFLOW_CONTAINER_FILE_NVIDIA_X86 ] && [ -f $TENSORFLOW_CONTAINER_FILE_NVIDIA_ARM  ] && [ -f $TENSORFLOW_CONTAINER_FILE_AMD ] && [ -f $TENSORFLOW_CONTAINER_FILE_IPU ]; then
    touch $TENSORFLOW_CONTAINER_FILE_DONE
    echo "Done pulling TensorFlow Containers!" >&2
fi

rm -rf $APPTAINER_CACHEDIR
rm -rf $APPTAINER_TMPDIR

##### Installing Requirements #####
if ! [ -f $TENSORFLOW_PACKAGES_FILE_AMD ] && [ "$ACCELERATOR" = "MI250" ]; then
    mkdir -p $TENSORFLOW_PACKAGES_AMD
    apptainer exec $TENSORFLOW_CONTAINER_FILE_AMD \
                    python -m pip install \
                    --upgrade pip setuptools distlib \
                    >&2 
    export PIP_USER=0 
    apptainer exec $TENSORFLOW_CONTAINER_FILE_AMD \
                    python -m pip install \
                    --prefix=$TENSORFLOW_PACKAGES_AMD \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/requirements/amd_tensorflow_requirements.txt\
                     >&2
    touch $TENSORFLOW_PACKAGES_FILE_AMD
    echo "Done building additional packages for $ACCELERATOR in $TENSORFLOW_PACKAGES_AMD" >&2
elif ! [ -f $TENSORFLOW_PACKAGES_FILE_IPU ] && [ "$ACCELERATOR" = "GC200" ]; then
    mkdir -p $TENSORFLOW_PACKAGES_IPU
    apptainer exec $TENSORFLOW_CONTAINER_FILE_IPU \
                    python -m pip install \
                    --upgrade pip setuptools distlib \
                    >&2
    export PIP_USER=0 
    apptainer exec $TENSORFLOW_CONTAINER_FILE_IPU \
                    python -m pip install \
                    --prefix=$TENSORFLOW_PACKAGES_IPU \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/requirements/ipu_tensorflow_requirements.txt \
                     >&2
    touch $TENSORFLOW_PACKAGES_FILE_IPU
    echo "Done building additional packages for $ACCELERATOR in $TENSORFLOW_PACKAGES_IPU" >&2
elif ! [ -f $TENSORFLOW_PACKAGES_FILE_NVIDIA_X86 ] && [[ " ${NVIDIA_X86_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; then
    mkdir -p $TENSORFLOW_PACKAGES_NVIDIA_X86
    apptainer exec $TENSORFLOW_CONTAINER_FILE_NVIDIA_X86 \
                    python -m pip install \
                    --upgrade pip setuptools distlib \
                    >&2 
    export PIP_USER=0 
    apptainer exec $TENSORFLOW_CONTAINER_FILE_NVIDIA_X86 \
                    python -m pip install \
                    --prefix=$TENSORFLOW_PACKAGES_NVIDIA_X86 \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/requirements/nvidia_x86_tensorflow_requirements.txt\
                     >&2
    touch $TENSORFLOW_PACKAGES_FILE_NVIDIA_X86
    echo "Done building additional packages for $ACCELERATOR in $TENSORFLOW_PACKAGES_NVIDIA_X86" >&2
elif ! [ -f $TENSORFLOW_PACKAGES_FILE_NVIDIA_ARM ] && [[ " ${NVIDIA_ARM_ACCELERATORS[@]} " == *" $ACCELERATOR "* ]]; then
    mkdir -p $TENSORFLOW_PACKAGES_NVIDIA_ARM
    apptainer exec $TENSORFLOW_CONTAINER_FILE_NVIDIA_ARM \
                    python -m pip install \
                    --upgrade pip setuptools distlib \
                    >&2 
    export PIP_USER=0 
    apptainer exec $TENSORFLOW_CONTAINER_FILE_NVIDIA_ARM \
                    python -m pip install \
                    --prefix=$TENSORFLOW_PACKAGES_NVIDIA_ARM \
                    --ignore-installed --no-deps \
                    --no-cache-dir \
                    -r $BENCH_DIR/requirements/nvidia_arm_tensorflow_requirements.txt\
                     >&2
    touch $TENSORFLOW_PACKAGES_FILE_NVIDIA_ARM
    echo "Done building additional packages for $ACCELERATOR in $TENSORFLOW_PACKAGES_NVIDIA_ARM" >&2
else
    echo "No additional packages required for $ACCELERATOR" >&2
fi


