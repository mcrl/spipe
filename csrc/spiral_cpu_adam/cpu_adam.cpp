#include "cpu_adam.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <torch/csrc/cuda/Event.h>
#include <torch/extension.h>
#include <type_traits>
#include <unordered_map>

#if defined(__ENABLE_CUDA__)
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"
#include <cuda_runtime_api.h>
#endif

#include "thread_pool.hpp"
#include "util.hpp"
#include <cuda_runtime.h>
#include <future>
#include <spdlog/spdlog.h>
#include <thread>
#include <unistd.h>

#define _DEBUG_CPU_ADAM true
#define _USE_THREAD_POOL true

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

#if _USE_THREAD_POOL
static ThreadPool pool(64); // TODO (SpiralPipe) make num threads configurable
static vector<future<int>> futures; // Storage for futures returned by threads
#else
static vector<std::thread> threads;
#endif

int spiral_create_adam_optimizer(int optimizer_id,
                                 float alpha,
                                 float betta1,
                                 float betta2,
                                 float eps,
                                 float weight_decay,
                                 bool adamw_mode,
                                 bool should_log)
{

  spdlog::info("Creating spiral adam optimizer");

  auto opt = std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps,
                                              weight_decay, adamw_mode);

  s_optimizers[optimizer_id] = opt;

  if (should_log) {
    std::string avx_type = "";
#if defined(__AVX512__)
    avx_type = "AVX512";
#else
#if defined(__AVX256__)
    avx_type = "AVX2";
#else
    avx_type = "scalar";
#endif
#endif

    printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
           optimizer_id, avx_type.c_str());
    printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
           alpha, betta1, betta2, weight_decay, (int)adamw_mode);
  }

  return 0;
}

int spiral_destroy_adam_optimizer(int optimizer_id)
{
  s_optimizers.erase(optimizer_id);

  return 0;
}

int _spiral_adam_step(int optimizer_id,
                      size_t step,
                      float lr,
                      float beta1,
                      float beta2,
                      float epsilon,
                      float weight_decay,
                      bool bias_correction,
                      torch::Tensor& params,
                      torch::Tensor& grads,
                      torch::Tensor& exp_avg,
                      torch::Tensor& exp_avg_sq,
                      long ev_long)
{
  if (ev_long == 0) {
    throw std::runtime_error("Event is not recorded");
  } else if (ev_long == -1) {
    if (_DEBUG_CPU_ADAM)
      spdlog::info("Event is not provided. Skip synchronization");
  } else {
    if (_DEBUG_CPU_ADAM)
      spdlog::info("Spiral CPU Adam: event={} wait", ev_long);
    cudaEvent_t ev = (cudaEvent_t)ev_long;
    CHECK_CUDA(cudaEventSynchronize(ev));
    if (_DEBUG_CPU_ADAM)
      spdlog::info("Spiral CPU Adam: event={} wait done!", ev_long);
  }

  if (_DEBUG_CPU_ADAM)
    spdlog::info("Spiral CPU Adam: Start step");

  auto params_c = params.contiguous();
  auto grads_c = grads.contiguous();
  auto exp_avg_c = exp_avg.contiguous();
  auto exp_avg_sq_c = exp_avg_sq.contiguous();

  // assert(params.options().dtype() == grads.options().dtype());

  float* params_ptr = (float*)params_c.data_ptr();
  float* grads_ptr = (float*)grads_c.data_ptr();
  float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
  float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

  std::shared_ptr<Adam_Optimizer> opt =
      std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
  opt->IncrementStep(step, beta1, beta2);
  opt->update_state(lr, epsilon, weight_decay, bias_correction);

  opt->Step_8(params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr,
              params_c.numel(), nullptr,
              (params.options().dtype() == at::kHalf));

  if (_DEBUG_CPU_ADAM)
    spdlog::info("Spiral CPU Adam: End step");

#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
  opt->SynchronizeStreams();

  if (_DEBUG_CPU_ADAM)
    spdlog::info("Spiral CPU Adam: Sync stream");
#endif

  return 0;
}

int _spiral_adam_step_plus_copy(int optimizer_id,
                                size_t step,
                                float lr,
                                float beta1,
                                float beta2,
                                float epsilon,
                                float weight_decay,
                                bool bias_correction,
                                torch::Tensor& params,
                                torch::Tensor& grads,
                                torch::Tensor& exp_avg,
                                torch::Tensor& exp_avg_sq,
                                torch::Tensor& device_params,
                                long ev_long)
{
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
  if (ev_long == 0) {
    throw std::runtime_error("Event is not recorded");
  } else if (ev_long == -1) {
    if (_DEBUG_CPU_ADAM)
      spdlog::info("Event is not provided. Skip synchronization");
  } else {
    if (_DEBUG_CPU_ADAM)
      spdlog::info("Spiral CPU Adam: event={} wait", ev_long);
    cudaEvent_t ev = (cudaEvent_t)ev_long;
    CHECK_CUDA(cudaEventSynchronize(ev));
    if (_DEBUG_CPU_ADAM)
      spdlog::info("Spiral CPU Adam: event={} wait done!", ev_long);
  }

  if (_DEBUG_CPU_ADAM)
    spdlog::info("Spiral CPU Adam: Start step");

  auto params_c = params.contiguous();
  auto device_params_c = device_params.contiguous();
  auto exp_avg_c = exp_avg.contiguous();
  auto exp_avg_sq_c = exp_avg_sq.contiguous();
  auto grads_c = grads.contiguous();

  float* params_ptr = (float*)params_c.data_ptr();
  float* grads_ptr = (float*)grads_c.data_ptr();
  ds_half_precision_t* device_params_ptr =
      (ds_half_precision_t*)device_params_c.data_ptr();
  float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
  float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

  std::shared_ptr<Adam_Optimizer> opt =
      std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
  opt->IncrementStep(step, beta1, beta2);
  opt->update_state(lr, epsilon, weight_decay, bias_correction);
  opt->Step_8(params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr,
              params_c.numel(), device_params_ptr,
              (params.options().dtype() == at::kHalf));

  if (_DEBUG_CPU_ADAM)
    spdlog::info("Spiral CPU Adam: End step");

  opt->SynchronizeStreams();
  if (_DEBUG_CPU_ADAM)
    spdlog::info("Spiral CPU Adam: Sync stream");
#else
  assert(false);
#endif
  return 0;
}

int spiral_adam_step(int optimizer_id,
                     size_t step,
                     float lr,
                     float beta1,
                     float beta2,
                     float epsilon,
                     float weight_decay,
                     bool bias_correction,
                     torch::Tensor& params,
                     torch::Tensor& grads,
                     torch::Tensor& exp_avg,
                     torch::Tensor& exp_avg_sq,
                     long ev_long)
{
#if _USE_THREAD_POOL
  futures.emplace_back(pool.submit(_spiral_adam_step, optimizer_id, step, lr,
                                   beta1, beta2, epsilon, weight_decay,
                                   bias_correction, params, grads, exp_avg,
                                   exp_avg_sq, ev_long));
#else
  threads.emplace_back(_spiral_adam_step, optimizer_id, step, lr, beta1, beta2,
                       epsilon, weight_decay, bias_correction, std::ref(params),
                       std::ref(grads), std::ref(exp_avg), std::ref(exp_avg_sq),
                       ev_long);
#endif
  return 0;
}

int spiral_adam_step_plus_copy(int optimizer_id,
                               size_t step,
                               float lr,
                               float beta1,
                               float beta2,
                               float epsilon,
                               float weight_decay,
                               bool bias_correction,
                               torch::Tensor& params,
                               torch::Tensor& grads,
                               torch::Tensor& exp_avg,
                               torch::Tensor& exp_avg_sq,
                               torch::Tensor& device_params,
                               long ev_long)
{
#if _USE_THREAD_POOL
  futures.emplace_back(
      pool.submit(_spiral_adam_step_plus_copy, optimizer_id, step, lr, beta1,
                  beta2, epsilon, weight_decay, bias_correction, params, grads,
                  exp_avg, exp_avg_sq, device_params, ev_long));
#else
  threads.emplace_back(_spiral_adam_step_plus_copy, optimizer_id, step, lr,
                       beta1, beta2, epsilon, weight_decay, bias_correction,
                       std::ref(params), std::ref(grads), std::ref(exp_avg),
                       std::ref(exp_avg_sq), std::ref(device_params), ev_long);
#endif
  return 0;
}

void spiral_adam_synchronize()
{
  if (_DEBUG_CPU_ADAM)
    spdlog::info("Spiral CPU Adam: Synchronize");

#if _USE_THREAD_POOL
  for (auto& f : futures) {
    if (f.get() != 0) {
      // Non-zero future value indicates an error
      throw std::runtime_error("Error produced during Adam step is detected");
    }
  }
  futures.clear();
#else
  for (auto& t : threads) {
    t.join();
  }
  threads.clear();
#endif

  if (_DEBUG_CPU_ADAM)
    spdlog::info("Spiral CPU Adam: Synchronize done!");
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("adam_update", &spiral_adam_step, "SpiralPipe CPU Adam update (C++)");
  m.def("adam_update_copy", &spiral_adam_step_plus_copy,
        "SpiralPipe CPU Adam update and param copy (C++)");
  m.def("create_adam", &spiral_create_adam_optimizer,
        "SpiralPipe CPU Adam (C++)");
  m.def("destroy_adam", &spiral_destroy_adam_optimizer,
        "SpiralPipe CPU Adam destroy (C++)");
  m.def("adam_sync", &spiral_adam_synchronize,
        "SpiralPipe CPU Adam join threads (C++)");
}
