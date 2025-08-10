#include "util.hpp"
#include <cuda.h>
#include <stdio.h>

#define _DEBUG_GDR false

// Check GPU Direct RDMA support. Return 1 if supported, 0 otherwise.
int check_gdr_support(int device_id)
{
  int gdr_support = 0;

  CUdevice dev;
  CHECK_CUDA_DRIVER(cuDeviceGet(&dev, device_id));

#if CUDA_VERSION >= 11030
  int drv_version;
  CHECK_CUDA_DRIVER(cuDriverGetVersion(&drv_version));

  // Starting from CUDA 11.3, CUDA provides an ability to check GPUDirect RDMA
  // support.
  if (drv_version >= 11030) {
    CHECK_CUDA_DRIVER(cuDeviceGetAttribute(
        &gdr_support, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev));
    if (_DEBUG_GDR)
      printf("GPUDirect RDMA support: %d\n", gdr_support);
    return gdr_support;
  }
#endif
  // TODO (SPipe) Check check_gdr_support() in
  // https://github.com/NVIDIA/gdrcopy/blob/master/tests/common.cpp
  if (_DEBUG_GDR)
    printf("GPUDirect RDMA support check for CUDA version < 11.3 is currently "
           "not supported.\n");
  if (_DEBUG_GDR)
    printf("GPUDirect RDMA support: 0\n");
  return gdr_support;
}