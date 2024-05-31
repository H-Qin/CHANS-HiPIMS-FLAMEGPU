// High-Performance Integrated hydrodynamic Modelling System ***hybrid***
// @author: Jiaheng Zhao (Hemlab)
// @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
// @contact: j.zhao@lboro.ac.uk
// @software: hipims_hybrid
// @time: 07.01.2021
// This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
// Feel free to use and extend if you are a ***member of hemlab***.
// #include <torch/extension.h>
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

namespace
{
}

template <typename scalar_t>
__global__ void frictionCalculation_kernel(
    int N, int32_t *__restrict__ wetMask, scalar_t *__restrict__ h_update,
    scalar_t *__restrict__ qx_update, scalar_t *__restrict__ qy_update,
    scalar_t *__restrict__ z_update, uint8_t *__restrict__ landuse,
    scalar_t *__restrict__ h, scalar_t *__restrict__ wl,
    scalar_t *__restrict__ qx, scalar_t *__restrict__ qy,
    scalar_t *__restrict__ z, scalar_t *__restrict__ manning,
    scalar_t *__restrict__ dt)
{

  scalar_t h_small = 1.0e-6;
  scalar_t g = 9.81;
  scalar_t q_norm, C_f;
  // scalar_t C_f_temp;

  // get the index of cell
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N)
  {
    int32_t i = wetMask[j];
    // first, we will update the water depth
    h[i] += h_update[i];
    if (h[i] < h_small)
    {
      qx[i] = 0.0;
      qy[i] = 0.0;
    }
    else
    {

      C_f =
          g * manning[landuse[i]] * manning[landuse[i]] * pow(h[i], -1.0 / 3.0);

      // increased manning coefficient
      // C_f_temp = g * min(manning[landuse[i]] * manning[landuse[i]] * 4.0, 0.36) * pow(h[i], -1.0 / 3.0);

      qx[i] += qx_update[i];
      qy[i] += qy_update[i];
      q_norm = sqrt(qx[i] * qx[i] + qy[i] * qy[i]);
      if (manning[landuse[i]] > 0.0)
      {
        // if (q_norm > 1.0e-10){
        if (abs(qx[i]) > 1.0e-10)
        {
          // if (C_f > 1.0e-15)

          // qx[i] = (1.0 -
          //          sqrt(1.0 + (4.0 * dt[0] * C_f) / (h[i] * h[i]) * q_norm)) /
          //         (-2.0 * dt[0] * C_f / (h[i] * h[i]) * (q_norm / qx[i]));

          auto temp = (1.0 -
                       sqrt(1.0 + (4.0 * dt[0] * C_f) / (h[i] * h[i]) * q_norm)) /
                      (-2.0 * dt[0] * C_f / (h[i] * h[i]) * (q_norm / qx[i]));

          // ==================================================//
          // add a limit for the supercritical flow
          // if fr>5.0, increase the manning value
          if (abs(temp) > 5.0 * h[i] * sqrt(h[i] * g))
          // if (abs(temp) > 10.0 * h[i])
          {
            qx[i] = 2.0 * temp - qx[i];
            // make sure the direction is the same with temp
            if (qx[i] * temp <= 0.0)
            {
              qx[i] = 0.0;
            }
            // qx[i] = (1.0 -
            //          sqrt(1.0 + (4.0 * dt[0] * C_f_temp) / (h[i] * h[i]) * q_norm)) /
            //         (-2.0 * dt[0] * C_f_temp / (h[i] * h[i]) * (q_norm / qx[i]));
          }
          else
          {
            qx[i] = temp;
          }
          // ==================================================//
        }
        if (abs(qy[i]) > 1.0e-10)
        {
          // if (C_f > 1.0e-15)

          // qy[i] = (1.0 -
          //          sqrt(1.0 + (4.0 * dt[0] * C_f) / (h[i] * h[i]) * q_norm)) /
          //         (-2.0 * dt[0] * C_f / (h[i] * h[i]) * (q_norm / qy[i]));
          auto temp = (1.0 -
                       sqrt(1.0 + (4.0 * dt[0] * C_f) / (h[i] * h[i]) * q_norm)) /
                      (-2.0 * dt[0] * C_f / (h[i] * h[i]) * (q_norm / qy[i]));

          // ==================================================//
          // add a limit for the supercritical flow
          // if fr>5.0, increase the manning value

          if (abs(temp) > 5.0 * h[i] * sqrt(h[i] * g))
          // if (abs(temp) > 10.0 * h[i])
          {
            qy[i] = 2.0 * temp - qy[i];
            // make sure the direction is the same with temp
            if (qy[i] * temp <= 0.0)
            {
              qy[i] = 0.0;
            }

            // qy[i] = (1.0 -
            //          sqrt(1.0 + (4.0 * dt[0] * C_f_temp) / (h[i] * h[i]) * q_norm)) /
            //         (-2.0 * dt[0] * C_f_temp / (h[i] * h[i]) * (q_norm / qy[i]));
          }
          else
          {
            qy[i] = temp;
          }
          // ==================================================//
        }
        // }
      }
    }
    // h[i] += h_update[i];
    z[i] += z_update[i];
    wl[i] = z[i] + h[i];

    h_update[i] = 0.0;
    qx_update[i] = 0.0;
    qy_update[i] = 0.0;
    z_update[i] = 0.0;
  }
}

void frictionCalculation_cuda(at::Tensor wetMask, at::Tensor h_update,
                              at::Tensor qx_update, at::Tensor qy_update,
                              at::Tensor z_update, at::Tensor landuse,
                              at::Tensor h, at::Tensor wl, at::Tensor qx,
                              at::Tensor qy, at::Tensor z, at::Tensor manning,
                              at::Tensor dt)
{
  const int N = wetMask.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "frictioncuda_Calculation", ([&] {
        frictionCalculation_kernel<
            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
            N, wetMask.data<int32_t>(), h_update.data<scalar_t>(),
            qx_update.data<scalar_t>(), qy_update.data<scalar_t>(),
            z_update.data<scalar_t>(), landuse.data<uint8_t>(),
            h.data<scalar_t>(), wl.data<scalar_t>(), qx.data<scalar_t>(),
            qy.data<scalar_t>(), z.data<scalar_t>(), manning.data<scalar_t>(),
            dt.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
