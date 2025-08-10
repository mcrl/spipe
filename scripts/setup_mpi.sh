#!/bin/bash

# Install ucx (ddd634)
cd $SPIPE_AEC_ROOT/ucx
# git clone https://github.com/openucx/ucx && cd ucx && git checkout ddd634
./autogen.sh
mkdir -p build && cd build
../contrib/configure-release --prefix=${UCX_ROOT} --with-cuda=${CUDA_ROOT} --enable-mt && make -j`nproc` install

# Install ompi (424151)
cd $SPIPE_AEC_ROOT/ompi
# git clone --recursive https://github.com/open-mpi/ompi.git && cd ompi && git checkout 424151
./autogen.pl
mkdir -p build && cd build
../configure --prefix=${MPI_ROOT} --with-ucx=${UCX_ROOT} --with-cuda=${CUDA_ROOT} && make -j`nproc` install

# Fix: manually link libopen-pal for our OpenMPI versions
ln -s $MPI_ROOT/lib/libopen-pal.so.0.0.0 $MPI_ROOT/lib/libopen-pal.so.40
