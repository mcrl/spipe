#include "cpu_adam.hpp"
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
#include <mutex>
#include <thread>
#include <unistd.h>

#define _DEBUG_OPTIMIZER false // for debugging Tensor values

struct ThreadSafeOptimizer {
  std::unordered_map<int, std::shared_ptr<void>>
      group_s_opts; // Optimizer shared by a parameter group
  ThreadPool pool;
  std::vector<std::future<int>>
      futures;           // Storage for futures returned by threads
  const int nparams;     // Number of parameters to update
  int nparams_submitted; // Number of parameters submitted to pool
  const bool should_log;
  std::mutex m;

  ThreadSafeOptimizer(
      std::unordered_map<int, std::shared_ptr<void>> group_s_opts,
      const int nparams,
      const int pool_size,
      const bool should_log)
    : group_s_opts(group_s_opts), pool(pool_size), nparams(nparams),
      nparams_submitted(0), should_log(should_log)
  {}
};

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

int spiral_create_adam_optimizer(int optimizer_id,
                                 int ngroups,
                                 int nparams,
                                 int pool_size,
                                 float alpha,
                                 float betta1,
                                 float betta2,
                                 float eps,
                                 float weight_decay,
                                 bool adamw_mode,
                                 bool should_log)
{
  /*
   * Each backward stage has a ThreadSafeOptimizer and a ThreadPool.
   * - ThreadSafeOptimizer has a SpiralCPUAdamOptimizer for each parameter group
   * (mainly a weight group and a bias group).
   * - ThreadPool is shared between all ThreadSafeOptimizers.
   */
  std::unordered_map<int, std::shared_ptr<void>> group_s_opts;
  for (int i = 0; i < ngroups; i++) {
    group_s_opts[i] = std::make_shared<SpiralAdamOptimizer>(
        alpha, betta1, betta2, eps, weight_decay, adamw_mode);
  }

  if (pool_size == 0 || pool_size > nparams)
    pool_size = nparams;
  else if (pool_size < 0)
    throw std::runtime_error("Invalid thread pool size");

  s_optimizers[optimizer_id] = std::make_shared<ThreadSafeOptimizer>(
      group_s_opts, nparams, pool_size, should_log);

  if (should_log) {
    printf("[%ld] ThreadSafeOptimizer #%d is created with %d threads for %d "
           "groups %d "
           "params.\n",
           (long)getpid(), optimizer_id, pool_size, ngroups, nparams);
  }

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
    printf("Adam Optimizer is created with %s arithmetic capability.\n"
           "Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
           avx_type.c_str(), alpha, betta1, betta2, weight_decay,
           (int)adamw_mode);
  }

  return 0;
}

int spiral_destroy_adam_optimizer(int optimizer_id)
{
  auto ts_opt =
      std::static_pointer_cast<ThreadSafeOptimizer>(s_optimizers[optimizer_id]);
  bool should_log = ts_opt->should_log;

  s_optimizers.erase(optimizer_id);

  if (should_log) {
    printf("[%ld] ThreadSafeOptimizer #%d is destroyed.\n", (long)getpid(),
           optimizer_id);
  }

  return 0;
}

int _spiral_adam_step(int optimizer_id,
                      int group_id,
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
  auto ts_opt =
      std::static_pointer_cast<ThreadSafeOptimizer>(s_optimizers[optimizer_id]);

  if (ev_long == 0) {
    throw std::runtime_error("Event is not recorded");
  } else if (ev_long == -1) {
    if (ts_opt->should_log)
      printf("(pid:%ld,tid:%ld) Skip null event\n", (long)getpid(),
             (long)gettid());
  } else {
    if (ts_opt->should_log)
      printf("(pid:%ld,tid:%ld) Wait event:%ld\n", (long)getpid(),
             (long)gettid(), ev_long);

    cudaEvent_t ev = (cudaEvent_t)ev_long;
    CHECK_CUDA(cudaEventSynchronize(ev));
  }

  auto params_c = params.contiguous();
  auto grads_c = grads.contiguous();
  auto exp_avg_c = exp_avg.contiguous();
  auto exp_avg_sq_c = exp_avg_sq.contiguous();

  // assert(params.options().dtype() == grads.options().dtype());

  float* params_ptr = (float*)params_c.data_ptr();
  float* grads_ptr = (float*)grads_c.data_ptr();
  float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
  float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

  if (_DEBUG_OPTIMIZER) {
    printf("Updating param=(%p,%f,%zu) with grad=(%p,%f,%zu), "
           "momentum=(%p,%f,%zu), variance=(%p,%f,%zu)\n",
           params_ptr, at::mean(params).item().toFloat(), params_c.numel(),
           grads_ptr, at::mean(grads).item().toFloat(), grads_c.numel(),
           exp_avg_ptr, at::mean(exp_avg).item().toFloat(), exp_avg_c.numel(),
           exp_avg_sq_ptr, at::mean(exp_avg_sq).item().toFloat(),
           exp_avg_sq_c.numel());
  }

  auto group_s_opt = std::static_pointer_cast<SpiralAdamOptimizer>(
      ts_opt->group_s_opts[group_id]);

  // Modifying the states of group_s_opt is safe, since param_group shares the
  // same state value refer to megatron/spiral/cpu_adam.py step()
  group_s_opt->IncrementStep(step, beta1, beta2);
  group_s_opt->update_state(lr, epsilon, weight_decay, bias_correction);

  group_s_opt->Step_8(params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr,
                      params_c.numel(), nullptr,
                      (params.options().dtype() == at::kHalf));

  if (_DEBUG_OPTIMIZER) {
    printf("Updated param=(%p,%f,%zu) with grad=(%p,%f,%zu), "
           "momentum=(%p,%f,%zu), variance=(%p,%f,%zu)\n",
           params_ptr, at::mean(params).item().toFloat(), params_c.numel(),
           grads_ptr, at::mean(grads).item().toFloat(), grads_c.numel(),
           exp_avg_ptr, at::mean(exp_avg).item().toFloat(), exp_avg_c.numel(),
           exp_avg_sq_ptr, at::mean(exp_avg_sq).item().toFloat(),
           exp_avg_sq_c.numel());
  }

#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
  group_s_opt->SynchronizeStreams();
#endif
  return 0;
}

int _spiral_adam_step_plus_copy(int optimizer_id,
                                int group_id,
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
  auto ts_opt =
      std::static_pointer_cast<ThreadSafeOptimizer>(s_optimizers[optimizer_id]);

  if (ev_long == 0) {
    throw std::runtime_error("Event is not recorded");
  } else if (ev_long == -1) {
    if (ts_opt->should_log)
      printf("(pid:%ld,tid:%ld) Skip null event\n", (long)getpid(),
             (long)gettid());
  } else {
    if (ts_opt->should_log)
      printf("(pid:%ld,tid:%ld) Wait event:%ld\n", (long)getpid(),
             (long)gettid(), ev_long);

    cudaEvent_t ev = (cudaEvent_t)ev_long;
    CHECK_CUDA(cudaEventSynchronize(ev));
  }

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

  if (_DEBUG_OPTIMIZER) {
    printf("Updating param=(%p,%f,%zu) device_parm=(%p,%f,%zu) with "
           "grad=(%p,%f,%zu), "
           "momentum=(%p,%f,%zu), variance=(%p,%f,%zu)\n",
           params_ptr, at::mean(params).item().toFloat(), params_c.numel(),
           device_params_ptr, at::mean(device_params).item().toFloat(),
           device_params_c.numel(), grads_ptr, at::mean(grads).item().toFloat(),
           grads_c.numel(), exp_avg_ptr, at::mean(exp_avg).item().toFloat(),
           exp_avg_c.numel(), exp_avg_sq_ptr,
           at::mean(exp_avg_sq).item().toFloat(), exp_avg_sq_c.numel());
  }

  auto group_s_opt = std::static_pointer_cast<SpiralAdamOptimizer>(
      ts_opt->group_s_opts[group_id]);

  // Modifying the states of group_s_opt is safe, since param_group shares the
  // same state value refer to megatron/spiral/cpu_adam.py step()
  group_s_opt->IncrementStep(step, beta1, beta2);
  group_s_opt->update_state(lr, epsilon, weight_decay, bias_correction);

  group_s_opt->Step_8(params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr,
                      params_c.numel(), device_params_ptr,
                      (params.options().dtype() == at::kHalf));

  if (_DEBUG_OPTIMIZER) {
    printf("Updated param=(%p,%f,%zu) device_parm=(%p,%f,%zu) with "
           "grad=(%p,%f,%zu), "
           "momentum=(%p,%f,%zu), variance=(%p,%f,%zu)\n",
           params_ptr, at::mean(params).item().toFloat(), params_c.numel(),
           device_params_ptr, at::mean(device_params).item().toFloat(),
           device_params_c.numel(), grads_ptr, at::mean(grads).item().toFloat(),
           grads_c.numel(), exp_avg_ptr, at::mean(exp_avg).item().toFloat(),
           exp_avg_c.numel(), exp_avg_sq_ptr,
           at::mean(exp_avg_sq).item().toFloat(), exp_avg_sq_c.numel());
  }

  group_s_opt->SynchronizeStreams();
#else
  assert(false);
#endif
  return 0;
}

int spiral_adam_step(int optimizer_id,
                     int group_id,
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
  auto ts_opt =
      std::static_pointer_cast<ThreadSafeOptimizer>(s_optimizers[optimizer_id]);

  std::lock_guard<std::mutex> lck(ts_opt->m);

  if (ts_opt->should_log) {
    printf("(pid:%ld) ThreadSafeOptimizer #%d param #%d step called with "
           "param.ptr=%p, grad.ptr=%p\n",
           (long)getpid(), optimizer_id, ts_opt->nparams_submitted,
           params.data_ptr(), grads.data_ptr());
  }

  ts_opt->futures.emplace_back(
      ts_opt->pool.submit(_spiral_adam_step, optimizer_id, group_id, step, lr,
                          beta1, beta2, epsilon, weight_decay, bias_correction,
                          params, grads, exp_avg, exp_avg_sq, ev_long));
  ts_opt->nparams_submitted++;

  return 0;
}

int spiral_adam_step_plus_copy(int optimizer_id,
                               int group_id,
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
  auto ts_opt =
      std::static_pointer_cast<ThreadSafeOptimizer>(s_optimizers[optimizer_id]);

  std::lock_guard<std::mutex> lck(ts_opt->m);

  if (ts_opt->should_log) {
    printf("(pid:%ld) ThreadSafeOptimizer #%d param #%d step called with "
           "param.ptr=%p, grad.ptr=%p\n",
           (long)getpid(), optimizer_id, ts_opt->nparams_submitted,
           params.data_ptr(), grads.data_ptr());
  }

  ts_opt->futures.emplace_back(ts_opt->pool.submit(
      _spiral_adam_step_plus_copy, optimizer_id, group_id, step, lr, beta1,
      beta2, epsilon, weight_decay, bias_correction, params, grads, exp_avg,
      exp_avg_sq, device_params, ev_long));
  ts_opt->nparams_submitted++;

  return 0;
}

void spiral_adam_synchronize(int optimizer_id)
{
  auto ts_opt =
      std::static_pointer_cast<ThreadSafeOptimizer>(s_optimizers[optimizer_id]);

  while (true) {
    std::unique_lock<std::mutex> lck(ts_opt->m);
    if (ts_opt->should_log) {
      printf("(pid:%ld) ThreadSafeOptimizer #%d sync param update "
             "(submitted:%d/total:%d)\n",
             (long)getpid(), optimizer_id, ts_opt->nparams_submitted,
             ts_opt->nparams);
    }
    if (ts_opt->nparams_submitted == ts_opt->nparams) {
      assert(ts_opt->futures.size() ==
             ts_opt->nparams); // assert all params submitted
      break;
    }
  }

  size_t fcnt = 0;
  for (auto& f : ts_opt->futures) {
    if (f.get() != 0) {
      // Non-zero future value indicates an error
      throw std::runtime_error("Error produced during Adam step is detected");
    }
    fcnt++;
  }
  assert(fcnt == ts_opt->nparams); // assert all submitted finished

  {
    std::lock_guard<std::mutex> lck(ts_opt->m);
    ts_opt->futures.clear();
    ts_opt->nparams_submitted = 0;
    ts_opt->pool.flush(ts_opt->nparams);
  }

  if (ts_opt->should_log) {
    printf("(pid:%ld) ThreadSafeOptimizer #%d post-sync param update\n",
           (long)getpid(), optimizer_id);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("adam_update", &spiral_adam_step, "SpiralPipe CPU Adam update (C++) ");
  m.def("adam_update_copy", &spiral_adam_step_plus_copy,
        "SpiralPipe CPU Adam update and param copy (C++)");
  m.def("create_adam", &spiral_create_adam_optimizer,
        "SpiralPipe CPU Adam (C++)");
  m.def("destroy_adam", &spiral_destroy_adam_optimizer,
        "SpiralPipe CPU Adam destroy (C++)");
  m.def("adam_sync", &spiral_adam_synchronize,
        "SpiralPipe CPU Adam join threads (C++)");
}
