# SPipe

Hybrid GPU and CPU Pipeline for Training LLMs under Memory Pressure

## Introduction

SPipe is a LLM training framework that enables efficient utilization of multiple GPUs and CPUs for training under limited compute and memory resources. SPipe presents two key techniques for pipeline parallelism (PP) to efficiently leverage CPU offloading: (1) Decoupled Pass Assignment and (2) Asynchronous CPU Optimizer. 

(1) stores model parameters in CPU memory rather than GPU memory, and assigns forward and backward passes to two separate GPUs that access the shared CPU-resident parameters. This design effectively reduces inter-GPU pipeline bubbles that arise in conventional settings where a single GPU performs both passes. (2) performs the GPU backward passes and CPU optimizer steps for the already-completed parameters in parallel. This parallelism alleviates pipeline bubbles that occur when there is no overlap between GPU and CPU processing. 

Additionally, SPipe provides a set of supporting techniques: Fine-grained Stage Partitioning to eliminate inter-GPU pipeline bubbles, Asynchronous Checkpoint Communication to optimize GPU communication, and Bypassing and Rollback mechanisms to ensure the efficiency and correctness of the asynchronous optimizer.

## Repository Organization

```
spipe/
├── csrc/                   # SPipe C++ package
│  ├── spipe_helper/        #    - SHMEM+RDMA backend
│  ├── spipe_cpu_adam/      #    - CPU optimizer
│  ├── common/              #    - Utility functions
│  └── external/            #    - External libraries
│
├── megatron/               # SPipe Python package
│  └── spipe/               #    - Pipeline schedule
│
├── external/               # External libraries
│
├── examples/               # SPipe usage examples
├── data/                   # Data vocab, merge file
├── scripts/                # Scripts to reproduce AE
└── results/                # Execution log files
```

## Build

### Dependencies
- Python 3.8
- UCX
- ompi
- CUDA 12.4
- cuDNN 8.5.0
- nv_peer_mem

Setting these up is detailed in [PREREQUISITES](./Prerequisites.md).

### Clone project
```bash
git clone --recurse-submodules https://github.com/mcrl/spipe.git
```

### Environment variables
Set environment variables in bash profile:
```bash
export SPIPE_ROOT=/path/to/spipe
export CUDA_ROOT=/path/to/cuda
export UCX_ROOT=/path/to/ucx/installation
export MPI_ROOT=/path/to/ompi/installation
export SPIPE_CONDA=<conda_env>

export PATH="$CUDA_ROOT/bin:$MPI_ROOT/bin:$UCX_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$MPI_ROOT/lib:$UCX_ROOT/lib:$LD_LIBRARY_PATH"
```

### Essential installation
```bash
# conda
conda create -n $SPIPE_CONDA python=3.8
conda activate $SPIPE_CONDA

# PyTorch 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Apex 741bdf5
git clone https://github.com/NVIDIA/apex && cd apex && git checkout 741bdf5
CUDA_HOME=$CUDA_ROOT TORCH_CUDA_ARCH_LIST="<cuda;arch;list>" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install -r requirements.txt

# mpi4py
MPICC=/path/to/mpicc python -m pip install mpi4py --no-cache

# submodule DeepSpeed (checkout 5f631ab & cherry-pick a4cd550)
cd $SPIPE_ROOT/csrc/external/DeepSpeed && TORCH_CUDA_ARCH_LIST="<cuda;arch;list>" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j16" --no-cache -v --disable-pip-version-check

# misc
pip install cmake ninja regex pillow pybind11 pyyaml typing-extensions six psutil nvtx py-cpuinfo einops transformers

# spipe_helper
cd $SPIPE_ROOT/csrc/spipe_helper && CUDA_BUILD_DIR=$CUDA_ROOT MPI_BUILD_DIR=$MPI_ROOT pip install .
```

## Examples

TBD

## Artifact Evaluation

See [Artifact Evaluation](./AE.md) for the details to reproduce the results in the [paper]().

## References
If you find SPipe relevant to your research, please consider citing:
```
@inproceedings{spipe-pact25,
    title = {{SPipe}: Hybrid {GPU} and {CPU} Pipeline for Training {LLMs} under Memory Pressure},
    author = {Ryu, Junyeol and Jeong, Yujin and Park, Daeyoung and Kim, Jinpyo and Kim, Heehoon and Lee, Jaejin},
    booktitle = {Proceedings of the 34th ACM/IEEE/IFIP International Conference on Parallel Architectures and Compilation Techniques},
    year = {2025},
}
```

## Contact

Junyeol Ryu (jyeol.ryu@gmail.com)