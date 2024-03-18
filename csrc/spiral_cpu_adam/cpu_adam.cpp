#include <torch/extension.h>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cpu_adam.h"

#if defined(__ENABLE_CUDA__)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"
#endif

#include <spdlog/spdlog.h>
#include "thread_pool.hpp"

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

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

    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

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
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

int spiral_destroy_adam_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

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
                 torch::Tensor& exp_avg_sq)
{
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
                           torch::Tensor& device_params)
{
    return 0;
}

void end_optty_step() {

};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &spiral_adam_step, "SpiralPipe CPU Adam update (C++)");
    m.def("adam_update_copy",
          &spiral_adam_step_plus_copy,
          "SpiralPipe CPU Adam update and param copy (C++)");
    m.def("create_adam", &spiral_create_adam_optimizer, "SpiralPipe CPU Adam (C++)");
    m.def("destroy_adam", &spiral_destroy_adam_optimizer, "SpiralPipe CPU Adam destroy (C++)");
    m.def("end_optty_step", &end_optty_step, "SpiralPipe CPU Adam end_optty_step (C++)");
}
