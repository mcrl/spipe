# Artifact Evaluation

This document provides instructions for obtaining the artifact, performing necessary preprocessing, and executing experiments using the provided scripts.

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

Before run the scripts, execute the `setup_env.sh` to set the environment variables.

```bash
bash $SPIPE_ROOT/scripts/eval_speedup.sh
bash $SPIPE_ROOT/scripts/eval_batch_scaling.sh
bash $SPIPE_ROOT/scripts/eval_optimizations.sh
```

- The `eval_speedup.sh` evaluates speedup between DeepSpeed, Mobius, Megatron, and SPipe. 
- The `eval_batch_scaling.sh` evaluates scaling of micro-batch size and mini-batch size for SPipe.
- The `eval_optimizations.sh` evaluates impact of adding system optimizations to SPipe.

Our experiment defaults to the V100 cluster.
For the RTX3090 cluster experiment in Figure 10 of the paper, please execute it with the settings below.
(It is recommended to separate the conda environment of the V100 cluster and the RTX3090 cluster.)

```bash
conda activate spipe-pact-3090
PARTITION=spipe-3090 bash $SPIPE_ROOT/scripts/eval_speedup.sh
```

## Expectation
The scripts we provide are based on the Slurm environment. When execute the training scripts, log files named `slurm-<jobId>.out` are generated for each job.

By executing the script below in the directory where these output files are located, you can extract each job’s configuration and elapsed time based on the log outputs.

```bash
bash $SPIPE_ROOT/scripts/result_extract.sh
```

After running the above script, an `actual.csv` file will be created under the `/results` directory. We provide an `expected.csv` file that contains the experimental results from our paper.

Using the script below, you can compare the two result files and calculate the error between them.

```bash
bash $SPIPE_ROOT/scripts/result_compare.sh
```
