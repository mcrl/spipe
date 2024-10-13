#pragma once

#include <cassert>
#include <vector>
#include <algorithm>
#include <sched.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(status_));                                    \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

#define CHECK_CUDA_DRIVER(call)                                                \
  do {                                                                         \
    CUresult status_ = call;                                                   \
    const char* err_str = nullptr;                                             \
    if (status_ != CUDA_SUCCESS) {                                             \
      cuGetErrorString(status_, &err_str);                                     \
      fprintf(stderr, "CUDA driver error (%s:%d): %s\n", __FILE__, __LINE__,   \
              err_str);                                                        \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

#define CHECK_NCCL(call)                                                       \
  do {                                                                         \
    ncclResult_t status_ = call;                                               \
    if (status_ != ncclSuccess && status_ != ncclInProgress) {                 \
      fprintf(stderr, "NCCL error (%s:%d): %s\n", __FILE__, __LINE__,          \
              ncclGetErrorString(status_));                                    \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

#define CHECK_MPI(call)                                                        \
  do {                                                                         \
    int code = call;                                                           \
    if (code != MPI_SUCCESS) {                                                 \
      char estr[MPI_MAX_ERROR_STRING];                                         \
      int elen;                                                                \
      MPI_Error_string(code, estr, &elen);                                     \
      fprintf(stderr, "MPI error (%s:%d): %s\n", __FILE__, __LINE__, estr);    \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

#define CHECK_ERRNO(call)                                                      \
  do {                                                                         \
    int code = call;                                                           \
    if (code != 0) {                                                           \
      fprintf(stderr, "ERRNO error (%s:%d): %s(%d)\n", __FILE__, __LINE__,     \
              strerror(errno), errno);                                         \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

void set_affinity(const std::vector<int>& excluded_cpus = {}) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);

  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (std::find(excluded_cpus.begin(), excluded_cpus.end(), i) == excluded_cpus.end()) {
      CPU_SET(i, &cpu_set);
    }
  }

  assert(sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set) == 0);
}

std::vector<int> get_affinity() {
  std::vector<int> cpu_affinity;

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  assert(sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set) == 0);

  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &cpu_set)) {
      cpu_affinity.push_back(i);
    }
  }

  return cpu_affinity;
}
