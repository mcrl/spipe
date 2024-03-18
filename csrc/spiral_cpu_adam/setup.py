import pathlib
import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from glob import glob

srcpath = pathlib.Path(__file__).parent.absolute()

# TODO (SpiralPipe) generalize paths
setup(
    name='spiral_cpu_adam',
    ext_modules=[
        CUDAExtension(
            name='spiral_cpu_adam',
            sources = sorted(glob('*.cpp')),
            include_dirs=[
                '/home/n1/junyeol/asplos2025/Megatron-LM-mcrl/csrc/external/spdlog/include',
                '/home/n1/junyeol/asplos2025/Megatron-LM-mcrl/csrc/external/DeepSpeed/csrc/includes',
                '/home/n1/junyeol/asplos2025/Megatron-LM-mcrl/csrc/common',
                os.path.join(os.environ['CUDA_BUILD_DIR'], 'include'),
            ],
            library_dirs=[
                os.path.join(os.environ['CUDA_BUILD_DIR'], 'lib64'),
            ],
            libraries=['rt', 'pthread', 'cuda', 'cudart'],
            extra_compile_args=['-g', '-fvisibility=hidden']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })