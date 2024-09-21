import os
from deepspeed.ops.op_builder import CPUAdamBuilder


class SpiralCPUAdamBuilder(CPUAdamBuilder):
    BUILD_VAR = "DS_BUILD_SPIRAL_CPU_ADAM"
    NAME = "spiral_cpu_adam"

    csrc = os.path.join(os.environ["MEGATRON_PATH"], "csrc")

    def __init__(self):
        super(CPUAdamBuilder, self).__init__(name=self.NAME)

    def absolute_name(self):
        return f"deepspeed.ops.adam.{self.NAME}_op"

    def sources(self):
        if self.build_for_cpu:
            return [
                os.path.join(self.csrc, "spiral_cpu_adam/cpu_adam.cpp"),
            ]

        return [
            os.path.join(self.csrc, "spiral_cpu_adam/cpu_adam.cpp"),
            os.path.join(
                self.csrc, "external/DeepSpeed/csrc/common/custom_cuda_kernel.cu"
            ),
        ]

    def libraries_args(self):
        args = super().libraries_args()
        if self.build_for_cpu:
            return args

        if not self.is_rocm_pytorch():
            args += ["curand"]

        return args

    def include_paths(self):
        import torch

        include_dirs=[
            os.path.join(self.csrc, 'external/spdlog/include'),
            os.path.join(self.csrc, "spiral_cpu_adam"),
            os.path.join(self.csrc, 'common'),
            os.path.join(self.csrc, 'external/DeepSpeed/csrc/includes'),
        ]

        if self.build_for_cpu:
            CUDA_INCLUDE = []
        elif not self.is_rocm_pytorch():
            CUDA_INCLUDE = [
                os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
            ]
        else:
            CUDA_INCLUDE = []
        return include_dirs + CUDA_INCLUDE
