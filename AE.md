# Artifact Evaluation

This document provides instructions for obtaining the artifact, performing necessary preprocessing, and executing experiments using the provided scripts.

```python
git clone --branch PACT-AE --recurse-submodules https://github.com/mcrl/spipe.git
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
$ source spipe-aec/spipe/script/setup_env.sh
$ source spipe-aec/spipe/script/setup_mpi.sh
$ source spipe-aec/spipe/script/setup_conda.sh
$ source spipe-aec/spipe/script/setup_data.sh
```

- The `setup_env.sh` sets necessary environment variables such as `SPIPE_ROOT` used in the later steps. This script should be executed before any other scripts for each shell open.
- The `setup_mpi.sh` installs UCX and cuda-aware MPI.
- The `setup_conda.sh` creates a conda environment and installs PyTorch and spipe dependencies.
- The `setup_dataset.sh` downloads and preprocesses the dataset for deep learning experiments.

## Experiment Workflow

```
source spipe-aec/spipe/script/setup_env.sh
sbatch -p spipe -N 1 $SPIPE_ROOT/examples/run.sh -j spipe -n llama2 -w 16 -z 2048 -s 10 -f 2 -b 6 -t 3 -l 1 -m 1 -g 16 -v 1 -u 2 -o stage
```
