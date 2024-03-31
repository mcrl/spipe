# Spiral helper

## Setup
```
# UCX
export PATH=/path/to/ucx/build/bin:${PATH}
export LD_LIBRARY_PATH=/path/to/ucx/build/lib:${LD_LIBRARY_PATH}

# MPI
export PATH=/path/to/mpi/build/bin:${PATH}
export LD_LIBRARY_PATH=/path/to/mpi/build/lib:${LD_LIBRARY_PATH}

```

## Installation
```
MPI_BUILD_DIR=/path/to/mpi/build CUDA_BUILD_DIR=/path/to/cuda/build pip install .
```