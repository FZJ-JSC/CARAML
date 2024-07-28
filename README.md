# CARAML 

**C**ompact **A**utomated **R**eproducible **A**ssessment of **M**achine **L**earning (CARAM)  is a benchmark to assess main stream Computer Vision and Natrual Language Processing worloads on novel accelerators. It was developed and tested on systems of Jülich Supercomputing Centre (JSC).

CARAML benchmarking is made compact and automated with the help of [JUBE](https://apps.fz-juelich.de/jsc/jube/docu/index.html), a scripting based framekwork to easily create benchmark sets, run those sets on different computer systems and evaluate the results. Additionally, the benchmarks are supplemented with power/energy measuring feauture.

With the usage of `JUBE` CARAML provides easy and reproducible way to benchmark different system and model configurations with minimal effort.

## Tested Accelerators:

CARAML has been tested on the [JURECA-SC EVALUATION PLATFORM](https://apps.fz-juelich.de/jsc/hps/jureca/evaluation-platform-overview.html), [JURECA-DC](https://apps.fz-juelich.de/jsc/hps/jureca/configuration.html), [JEDI](https://apps.fz-juelich.de/jsc/hps/jedi/index.html#) and [WEST AI Nodes](https://westai.de/services/hardware/) accelerators that include:

- AMD MI200 node with 4 $\times$ MI250 GPUs (`tag: MI250`)
- Graphcore IPU-POD4 M2000 with 4 $\times$ GC200 IPUs (`tag: GC200`)
- NVIDIA Ampere node (SXM)with 4 $\times$ A100 GPUs (`tag: A100`)
- NVIDIA Hopper node (PCIe) with 4 $\times$ H100 GPUs (`tag: H100`)
- NVIDIA Hopper node (NVLink) with 4 $\times$ H100 GPUs (`tag: WAIH100`)
- NVIDIA Grace-Hopper chip with 1 $\times$ GH200 GPU (`tag: GH200`)
- NVIDIA Grace-Hopper Node with 4 $\times$ GH200 GPUs (`tag: Jedi`)

# Benchmark

CARAML currently offers two benchmarks written in `python`:
- Computer Vision: [ResNet50](./resnet50/) benchmark  implemented in TensorFlow curated from forked versions of [tensorflow/benchmarks](https://github.com/tensorflow/benchmarks) and [graphcore/examples](hhttps://github.com/graphcore/examples)

- GPT Language Model: [LLM-training](./llm_training/) implemented in PyTorch curated from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git) with commit: `f7727433293427bef04858f67b2889fe9b177d88` and [patch](./llm_training/aux/add_tflops_logging.patch) applied

# Requirements

To run the benchmark `JUBE` must be installed. Refer to [JUBE Installation Documentation](https://apps.fz-juelich.de/jsc/jube/docu/tutorial.html#installation). The containers are deployed using [Apptainer](https://apptainer.org/) images and SLURM on the accelerators.

For ResNet50, download the `ImageNet LSVRC 2012` dataset from the [source](http://image-net.org/download) or [via kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)(Disk space required: 144GB)

In order to use PyTorch `torch run` API on JSC systems [fixed_torch_run.py](./llm_training/aux/fixed_torch_run.py) fix is required which is used to solve the issue defined [here](https://github.com/pytorch/pytorch/pull/81691).
Additionally the `hostname` is appended with an `i` for allowing communication over InfiniBand as described [here](https://apps.fz-juelich.de/jsc/hps/juwels/known-issues.html#ip-connectivity-on-compute-nodes).

# Implementation

## ResNet50

The `JUBE` file [resnet50_benchmark.xml](./resnet50/resnet50_benchmark.xml) sets up the environent by
    - Pulling TensorFlow containers and `pip` installing additional packages required for AMD and Graphcore using [get_tensorflow_container.sh](./resnet50/get_tensorflow_container.sh) file
    - Cloning [tf_cnn_benchmarks](https://github.com/chelseajohn/tf_cnn_benchmarks)(forked version) for NVIDIA & AMD 
    and [examples](https://github.com/chelseajohn/examples)(forked version) for Graphcore

The performance is measured using the throughput in terms of `images/sec`

## LLM-Training

In [llm_data](./llm_training/llm_data/), a subset (790 samples, 10 MB) of the small version of the [Oscar](https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz) dataset that is already pre-processed using [GPT-2 tokenizers](./llm_training/aux/tokenizers/) is provided

The `JUBE` file [nlp_benchmark_nvidia.yaml](./llm_training/nlp_benchmark_nvidia.yaml) sets up the environent by
    - Pulling PyTorch containers using [get_pytorch_container.sh](./llm_training/get_pytorch_container.sh) file
    - Cloning [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git) with commit: `f7727433293427bef04858f67b2889fe9b177d88` and applying [patch](./llm_training/aux/add_tflops_logging.patch) using [setup_llm.sh](./llm_training/setup_llm.sh) file

The performance is measured using the throughput in terms of `TFLOPs/GPU` and `tokens/s`

# How to run 

Clone this repository and `cd` into it as 

```bash
git clone https://github.com/FZJ-JSC/CARAML.git
cd CARAML
```
## ResNet50
Set the required `system` and `model` parameters and the path to downloaded ImageNet data in [resnet50_benchmark.xml](./resnet50/resnet50_benchmark.xml)

- To pull the required container use `container` tag as:
    - NVIDIA A100 and H100 
    ```bash
    jube run  --tag container H100
    ```
    - NVIDIA GH200 
    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag container GH200
    ```
    - AMD
    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag container MI250
    ```
    - Graphcore
    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag container GC200
    ```
- To run the benchmark with defined configurations do
    ```bash
    jube run resnet50/resnet50_benchmark.xml --tag A100
    ```
    `A100` can be replaced with `H100`, `WAIH100`, `GH200`, `Jedi`, `MI250` and `GC200` for the respective systems.

- After the benchmark has been executed, to get the result do
   ```bash
   jube result resnet50/resnet50_benchmark_run -i last
   ```
- Example result 
    ```bash
    Job ID,System,Version,Queue,Runtime,Model,Nodes,Devices,Tasks/Node,Threads/Task,GlobalBatchSize,BatchSize/Device,Images/sec
    13004679,WAIH100,2024.01,dc-wai,27.36,resnet50_v2,1,1,1,18,16,16,1542.78
    ```
## LLM-Training
Set the required `system` and `model` parameters  in [nlp_benchmark_nvidia.yaml](./llm_training/nlp_benchmark_nvidia.yaml)

- To pull the required container use `container` tag as:
    - NVIDIA A100 and H100 
    ```bash
    jube run llm_training/nlp_benchmark_nvidia.yaml --tag container H100
    ```
    - NVIDIA GH200
    ```bash
    jube run llm_training/nlp_benchmark_nvidia.yaml--tag container GH200
    ```
  
- To run the benchmark with defined configurations for `800M` GPT model do
    ```bash
    jube run llm_training/nlp_benchmark_nvidia.yaml --tag 800M A100
    ```
    `A100` can be replaced with `H100`, `WAIH100`, `GH200` and `Jedi` for the respective systems and `800M` can be replaced with `175B` and `13B` for systems with more than 1 node resources like `Jedi` and `H100` and `A100`.

- After the benchmark has been executed, to get the result do
   ```bash
   jube result llm_training/llm_benchmark_nvidia_run -i last
   ```
- Example result
```bash
JobID,System,Version,Queue,JobTime,ModelSize,Nodes,Devices,BatchSize,PipelineParallel,TensorParallel,Iterations,Time/iteration(s),Tokens/sec,Avg_TFLOPs/GPU
3349,Jedi,2024.01,all,00:30:00,800M,1,4,2048,1,1,5,26.50,158258.30,321.83
```