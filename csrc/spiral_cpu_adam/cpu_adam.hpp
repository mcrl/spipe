// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX // Windows idiosyncrasy
                 // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include "simd.hpp"
#include <cassert>
#include <stdio.h>
#include <torch/extension.h>
#include <cmath>
typedef unsigned short ds_half_precision_t;

#define STEP(SPAN)                                                             \
  void Step_##SPAN(float* _params, float* grads, float* _exp_avg,              \
                   float* _exp_avg_sq, size_t _param_size);

#define ROLLBACK(SPAN)                                                         \
  void Rollback_##SPAN(float* _params, float* grads, float* _exp_avg,          \
                       float* _exp_avg_sq, size_t _param_size);

class SpiralAdamOptimizer {
public:
  SpiralAdamOptimizer(float alpha = 1e-3,
                        float betta1 = 0.9,
                        float betta2 = 0.999,
                        float eps = 1e-8,
                        float weight_decay = 0,
                        bool adamw_mode = true)
    : _alpha(alpha), _betta1(betta1), _betta2(betta2), _eps(eps),
      _weight_decay(weight_decay), _betta1_t(1.0), _betta2_t(1.0), _step(0),
      _adamw_mode(adamw_mode)
  {
  }
  ~SpiralAdamOptimizer() {}

#if defined(__AVX512__) or defined(__AVX256__)
  template <int span>
  void Step_AVX(size_t* rounded_size,
                float* _params,
                float* grads,
                float* _exp_avg,
                float* _exp_avg_sq,
                size_t param_size);
  
  template <int span>
  void Rollback_AVX(size_t* rounded_size,
                    float* _params,
                    float* grads,
                    float* _exp_avg,
                    float* _exp_avg_sq,
                    size_t _param_size);
#endif

  STEP(1)
  STEP(4)
  STEP(8)
  ROLLBACK(1)
  ROLLBACK(4)
  ROLLBACK(8)
  inline void IncrementStep(size_t step, float beta1, float beta2)
  {
    if (beta1 != _betta1 || beta2 != _betta2) {
      _step = step;
      _betta1 = beta1;
      _betta2 = beta2;
      _betta1_t = std::pow(_betta1, step);
      _betta2_t = std::pow(_betta2, step);
    } else {
      _step++;
      if (_step != step) {
        _betta1_t = std::pow(_betta1, step);
        _betta2_t = std::pow(_betta2, step);
        _step = step;
      } else {
        _betta1_t *= _betta1;
        _betta2_t *= _betta2;
      }
    }
  }
  inline void update_state(float lr,
                           float epsilon,
                           float weight_decay,
                           bool bias_correction)
  {
    _alpha = lr;
    _eps = epsilon;
    _weight_decay = weight_decay;

    _bias_correction1 = 1.0f;
    _bias_correction2 = 1.0f;
    if (bias_correction == 1) {
      _bias_correction1 = 1 - _betta1_t;
      _bias_correction2 = 1 / sqrt(1 - _betta2_t);
    }
  }

private:
  float _alpha;
  float _betta1;
  float _betta2;
  float _eps;
  float _weight_decay;

  float _betta1_t;
  float _betta2_t;
  size_t _step;

  float _bias_correction1;
  float _bias_correction2;

  bool _adamw_mode;
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
void SpiralAdamOptimizer::Step_AVX(size_t* rounded_size,
                                   float* _params,
                                   float* grads,
                                   float* _exp_avg,
                                   float* _exp_avg_sq,
                                   size_t _param_size)
{
  size_t new_rounded_size = 0;
  int rshft = 0;

  AVX_Data betta1_4;
  betta1_4.data = SIMD_SET(_betta1);
  AVX_Data betta2_4;
  betta2_4.data = SIMD_SET(_betta2);

  float betta1_minus1 = 1 - _betta1;
  float betta2_minus1 = 1 - _betta2;
  AVX_Data betta1_minus1_4;
  betta1_minus1_4.data = SIMD_SET(betta1_minus1);
  AVX_Data betta2_minus1_4;
  betta2_minus1_4.data = SIMD_SET(betta2_minus1);

  AVX_Data bias2_sqrt;
  bias2_sqrt.data = SIMD_SET(_bias_correction2);

  AVX_Data eps_4;
  eps_4.data = SIMD_SET(_eps);

  float step_size = -1 * _alpha / _bias_correction1;
  AVX_Data step_size_4;
  step_size_4.data = SIMD_SET(step_size);

  float w_decay = -1 * _alpha * _weight_decay;
  AVX_Data weight_decay4;
  if (_weight_decay > 0)
    weight_decay4.data =
        (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
  new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
  for (size_t t = 0; t < new_rounded_size; t += TILE) {
    size_t copy_size = TILE;
    if ((t + TILE) > new_rounded_size)
      copy_size = new_rounded_size - t;
    size_t offset = copy_size + t;
#pragma omp parallel for
    for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
      AVX_Data grad_4[span];
      simd_load<span>(grad_4, grads + (i >> rshft), false);

      AVX_Data momentum_4[span];
      simd_load<span>(momentum_4, _exp_avg + i, false);

      AVX_Data variance_4[span];
      simd_load<span>(variance_4, _exp_avg_sq + i, false);

      AVX_Data param_4[span];
      simd_load<span>(param_4, _params + (i >> rshft), false);

      if (_weight_decay > 0 && !_adamw_mode) {
        simd_fma<span>(grad_4, param_4, weight_decay4, grad_4);
      }

      simd_mul<span>(momentum_4, momentum_4, betta1_4);
      simd_fma<span>(momentum_4, grad_4, betta1_minus1_4, momentum_4);
      simd_mul<span>(variance_4, variance_4, betta2_4);
      simd_mul<span>(grad_4, grad_4, grad_4);
      simd_fma<span>(variance_4, grad_4, betta2_minus1_4, variance_4);
      simd_sqrt<span>(grad_4, variance_4);
      simd_fma<span>(grad_4, grad_4, bias2_sqrt, eps_4);
      simd_div<span>(grad_4, momentum_4, grad_4);

      if (_weight_decay > 0 && _adamw_mode) {
        simd_fma<span>(param_4, param_4, weight_decay4, param_4);
      }

      simd_fma<span>(param_4, grad_4, step_size_4, param_4);

      simd_store<span>(_params + (i >> rshft), param_4, false);
      simd_store<span>(_exp_avg + i, momentum_4, false);
      simd_store<span>(_exp_avg_sq + i, variance_4, false);
    }
  }
  *rounded_size = new_rounded_size;
}
#endif

void SpiralAdamOptimizer::Step_1(float* _params,
                                   float* grads,
                                   float* _exp_avg,
                                   float* _exp_avg_sq,
                                   size_t _param_size)
{
  size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
  Step_AVX<1>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
  if (_param_size > rounded_size) {
    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;

    float step_size = -1 * _alpha / _bias_correction1;
    float w_decay = -1 * _alpha * _weight_decay;

    for (size_t t = rounded_size; t < _param_size; t += TILE) {
      size_t copy_size = TILE;
      if ((t + TILE) > _param_size)
        copy_size = _param_size - t;
      size_t offset = copy_size + t;
#pragma omp parallel for
      for (size_t k = t; k < offset; k++) {
        float grad = grads[k];
        float param = _params[k];
        float momentum = _exp_avg[k];
        float variance = _exp_avg_sq[k];
        if (_weight_decay > 0 && !_adamw_mode) {
          grad = param * _weight_decay + grad;
        }
        momentum = momentum * _betta1;
        momentum = grad * betta1_minus1 + momentum;

        variance = variance * _betta2;
        grad = grad * grad;
        variance = grad * betta2_minus1 + variance;

        grad = sqrt(variance);
        grad = grad * _bias_correction2 + _eps;
        grad = momentum / grad;
        if (_weight_decay > 0 && _adamw_mode) {
          param += w_decay * param;
        }
        param = grad * step_size + param;

        _params[k] = param;
        _exp_avg[k] = momentum;
        _exp_avg_sq[k] = variance;
      }
    }
  }
}

void SpiralAdamOptimizer::Step_4(float* _params,
                                   float* grads,
                                   float* _exp_avg,
                                   float* _exp_avg_sq,
                                   size_t _param_size)
{
  size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
  Step_AVX<4>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
  if (_param_size > rounded_size)
    Step_1((_params + rounded_size), (grads + rounded_size),
           (_exp_avg + rounded_size), (_exp_avg_sq + rounded_size),
           (_param_size - rounded_size));
}

void SpiralAdamOptimizer::Step_8(float* _params,
                                   float* grads,
                                   float* _exp_avg,
                                   float* _exp_avg_sq,
                                   size_t _param_size)
{
  size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
  Step_AVX<8>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
  if (_param_size > rounded_size)
    Step_4((_params + rounded_size), (grads + rounded_size),
           (_exp_avg + rounded_size), (_exp_avg_sq + rounded_size),
           (_param_size - rounded_size));
}

#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
void SpiralAdamOptimizer::Rollback_AVX(size_t* rounded_size,
                                       float* _params,
                                       float* grads,
                                       float* _exp_avg,
                                       float* _exp_avg_sq,
                                       size_t _param_size)
{
  size_t new_rounded_size = 0;
  int rshft = 0;

  AVX_Data betta1_4;
  betta1_4.data = SIMD_SET(_betta1);
  AVX_Data betta2_4;
  betta2_4.data = SIMD_SET(_betta2);

  float betta1_minus1 = 1 - _betta1;
  float betta2_minus1 = 1 - _betta2;
  AVX_Data minus_betta1_minus1_4;
  minus_betta1_minus1_4.data = SIMD_SET(betta1_minus1 * -1);
  AVX_Data minus_betta2_minus1_4;
  minus_betta2_minus1_4.data = SIMD_SET(betta2_minus1 * -1);

  AVX_Data bias2_sqrt;
  bias2_sqrt.data = SIMD_SET(_bias_correction2);

  AVX_Data eps_4;
  eps_4.data = SIMD_SET(_eps);

  float step_size = -1 * _alpha / _bias_correction1;
  AVX_Data minus_step_size_4;
  minus_step_size_4.data = SIMD_SET(step_size * -1);

  float w_decay_plus1 = -1 * _alpha * _weight_decay + 1;
  AVX_Data weight_decay4;
  if (_weight_decay > 0)
    weight_decay4.data =
        (_adamw_mode ? SIMD_SET(w_decay_plus1) : SIMD_SET(_weight_decay));
  new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
  for (size_t t = 0; t < new_rounded_size; t += TILE) {
    size_t copy_size = TILE;
    if ((t + TILE) > new_rounded_size)
      copy_size = new_rounded_size - t;
    size_t offset = copy_size + t;

#pragma omp parallel for
    for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
      AVX_Data grad_4[span];
      simd_load<span>(grad_4, grads + (i >> rshft), false);

      AVX_Data momentum_4[span];
      simd_load<span>(momentum_4, _exp_avg + i, false);

      AVX_Data variance_4[span];
      simd_load<span>(variance_4, _exp_avg_sq + i, false);

      AVX_Data param_4[span];
      simd_load<span>(param_4, _params + (i >> rshft), false);

      AVX_Data buf_4[span];
      simd_sqrt<span>(buf_4, variance_4);
      simd_fma<span>(buf_4, buf_4, bias2_sqrt, eps_4);
      simd_div<span>(buf_4, momentum_4, buf_4);

      simd_fma<span>(param_4, buf_4, minus_step_size_4, param_4);

      if (_weight_decay > 0 && _adamw_mode) {
        simd_div<span>(param_4, param_4, weight_decay4);
      }

      if (_weight_decay > 0 && !_adamw_mode) {
        simd_fma<span>(grad_4, param_4, weight_decay4, grad_4);
      }

      simd_fma<span>(momentum_4, grad_4, minus_betta1_minus1_4, momentum_4);
      simd_div<span>(momentum_4, momentum_4, betta1_4);

      simd_mul<span>(grad_4, grad_4, grad_4);
      simd_fma<span>(variance_4, grad_4, minus_betta2_minus1_4, variance_4);
      simd_div<span>(variance_4, variance_4, betta2_4);

      // If the fp32 precision error is negative, the sqrt operation will result in nan
      simd_negative_to_zero<span>(variance_4);

      simd_store<span>(_params + (i >> rshft), param_4, false);
      simd_store<span>(_exp_avg + i, momentum_4, false);
      simd_store<span>(_exp_avg_sq + i, variance_4, false);
    }
  }
  *rounded_size = new_rounded_size;
}
#endif

void SpiralAdamOptimizer::Rollback_1(float* _params,
                                     float* grads,
                                     float* _exp_avg,
                                     float* _exp_avg_sq,
                                     size_t _param_size)
{
  size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
  Rollback_AVX<1>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif

  if(_param_size > rounded_size) {
    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;

    float step_size = -1 * _alpha / _bias_correction1;
    float w_decay = -1 * _alpha * _weight_decay;

    for (size_t t = rounded_size; t < _param_size; t += TILE) {
      size_t copy_size = TILE;
      if ((t + TILE) > _param_size)
        copy_size = _param_size - t;
      size_t offset = copy_size + t;
#pragma omp parallel for
      for (size_t k = t; k < offset; k++) {
        float grad = grads[k];
        float param = _params[k];
        float momentum = _exp_avg[k];
        float variance = _exp_avg_sq[k];

        float buf = momentum / (sqrt(variance) * _bias_correction2 + _eps);
        param = param - buf * step_size;
        if (_weight_decay > 0 && _adamw_mode) {
          param = param / (1 + w_decay);
        }

        if (_weight_decay > 0 && !_adamw_mode) {
          grad = param * _weight_decay + grad;
        }

        momentum = momentum - grad * betta1_minus1;
        momentum = momentum / _betta1;

        grad = grad * grad;
        variance = variance - grad * betta2_minus1;
        variance = variance / _betta2;

        // If the fp32 precision error is negative, the sqrt operation will result in nan
        if (variance < 0) {
          variance = 0;
        }

        _params[k] = param;
        _exp_avg[k] = momentum;
        _exp_avg_sq[k] = variance;
      }
    }
  }
}

void SpiralAdamOptimizer::Rollback_4(float* _params,
                                     float* grads,
                                     float* _exp_avg,
                                     float* _exp_avg_sq,
                                     size_t _param_size)
{
  size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
  Rollback_AVX<4>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
  if (_param_size > rounded_size)
    Rollback_1((_params + rounded_size), (grads + rounded_size),
               (_exp_avg + rounded_size), (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size));
}

void SpiralAdamOptimizer::Rollback_8(float* _params,
                                     float* grads,
                                     float* _exp_avg,
                                     float* _exp_avg_sq,
                                     size_t _param_size)
{
  size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
  Rollback_AVX<8>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
  if (_param_size > rounded_size)
    Rollback_4((_params + rounded_size), (grads + rounded_size),
               (_exp_avg + rounded_size), (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size));
}
