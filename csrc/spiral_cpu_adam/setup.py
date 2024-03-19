import pathlib
import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from glob import glob

csrc = os.path.abspath(__file__ + '/../../')

setup(
    name='spiral_cpu_adam',
    ext_modules=[
        CUDAExtension(
            name='spiral_cpu_adam',
            sources = [
                os.path.join(csrc, 'spiral_cpu_adam/cpu_adam.cpp'),
                os.path.join(csrc, 'external/DeepSpeed/csrc/adam/cpu_adam_impl.cpp')
            ],
            include_dirs=[
                os.path.join(csrc, 'external/spdlog/include'),
                os.path.join(csrc, 'external/DeepSpeed/csrc/includes'),
                os.path.join(csrc, 'common'),
                os.path.join(os.environ['CUDA_BUILD_DIR'], 'include'),
            ],
            library_dirs=[
                os.path.join(os.environ['CUDA_BUILD_DIR'], 'lib64'),
            ],
            libraries=['rt', 'pthread', 'cuda', 'cudart', 'cublas', 'curand'],
            extra_compile_args=['-g', '-fvisibility=hidden', '-fopenmp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })