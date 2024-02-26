import pathlib
import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from glob import glob

srcpath = pathlib.Path(__file__).parent.absolute()

setup(
    name='spiral_helper',
    ext_modules=[
        CUDAExtension(
            name='spiral_helper',
            sources = sorted(glob('spiral_helper/*.cpp')),
            include_dirs=[
                srcpath / 'external/spdlog/include',
                os.path.join(os.environ['MPI_BUILD_DIR'], 'include'),
            ],
            library_dirs=[
                os.path.join(os.environ['MPI_BUILD_DIR'], 'lib'),
            ],
            libraries=['mpi', 'mpi_cxx', 'rt', 'pthread', 'cuda', 'cudart'], # linker. -lmpi, -lmpi_cxx
            extra_compile_args=['-g', '-fvisibility=hidden']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })