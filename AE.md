# Artifact Evaluation

This document provides instructions for obtaining the artifact, performing necessary preprocessing, and executing experiments using the provided scripts.

```python
git clone --recurse-submodules https://github.com/mcrl/spipe.git
```

## Prerequisites

- CUDA 12.4
- cuDNN v.8.5.0

## Download Artifact Package

The artifact package includes the spipe source code along with compatible versions of UCX, OpenMPI, and APEX.

```shell
$ wget https://github.com/mcrl/spipe/releases/download/spipe-aec/spipe-aec.tar.gz
$ tar xf spipe-aec.tar.gz
```

## Installation

Execute the following initialization scripts **once**:

```bash
source spipe-aec/spipe/scripts/setup_env.sh
source spipe-aec/spipe/scripts/setup_mpi.sh
source spipe-aec/spipe/scripts/setup_conda.sh
source spipe-aec/spipe/scripts/setup_data.sh
```

- The `setup_env.sh` sets necessary environment variables such as `SPIPE_ROOT` used in the later steps. This script should be executed before any other scripts for each shell open.
- The `setup_mpi.sh` installs UCX and cuda-aware MPI.
- The `setup_conda.sh` creates a conda environment and installs PyTorch and spipe dependencies.
- The `setup_dataset.sh` downloads and preprocesses the dataset for deep learning experiments.

## Evaluation

```bash
bash eval_speedup.sh
bash eval_batch_scaling.sh
bash eval_optimizations.sh
```

- The `eval_speedup.sh` evaluates speedup between DeepSpeed, Mobius, Megatron, and SPipe.
- The `eval_batch_scaling.sh` evaluates scaling of micro-batch size and mini-batch size for SPipe.
- The `eval_optimizations.sh` evaluates impact of adding system optimizations to SPipe.

## Expectation

TBD