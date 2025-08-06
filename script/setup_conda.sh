
#!/bin/bash

# Create conda environment
conda create -n spipe-pact python=3.8 -y
conda activate spipe-pact

# Install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install apex
cd $SPIPE_AEC_ROOT/apex
## git clone https://github.com/NVIDIA/apex && cd apex && git checkout 741bdf5
CUDA_HOME=$CUDA_ROOT TORCH_CUDA_ARCH_LIST="7.0;8.6" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
pip install -r requirements.txt

# Install mpi4py
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py --no-cache

# misc
pip install cmake ninja regex pillow pybind11 pyyaml typing-extensions six psutil nvtx py-cpuinfo einops transformers

# Install DeepSpeed
cd $SPIPE_ROOT/csrc/external/DeepSpeed
TORCH_CUDA_ARCH_LIST="7.0;8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j16" --no-cache -v --disable-pip-version-check

# Install spiral_helper
cd $SPIPE_ROOT/csrc/spipe_helper
CUDA_BUILD_DIR=$CUDA_ROOT MPI_BUILD_DIR=$MPI_ROOT pip install .
