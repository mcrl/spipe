### UCX (1.14.1)

```bash
git clone https://github.com/openucx/ucx && cd ucx && git checkout ddd634 
./autogen.sh
mkdir build && cd build
../contrib/configure-release --prefix=${UCX_ROOT} --with-cuda=${CUDA_ROOT} --enable-mt && make -j`nproc` install
# Update PATH and LD_LIBRARY_PATH
```

### Open MPI (4.1.0)
```bash
git clone --recursive https://github.com/open-mpi/ompi.git && cd ompi && git checkout 424151
./autogen.pl
mkdir build && cd build
../configure --prefix=${MPI_ROOT} --with-ucx=${UCX_ROOT} --with-cuda=${CUDA_ROOT} && make -j`nproc` install
# Update PATH and LD_LIBRARY_PATH
```

### nv_peer_mem

Run following command if you want to use GPUDirectRDMA and have its capable GPUs (e.g., V100). If not (e.g., RTX 3090), this isn't necessary. 
```bash
insmod /lib/modules/5.4.0-100-generic/updates/dkms/nvidia-peermem.ko
```