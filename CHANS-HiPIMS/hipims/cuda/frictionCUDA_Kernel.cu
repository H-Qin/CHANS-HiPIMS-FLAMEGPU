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

template <typename scalar_t>
__global__ void frictionCalculation_kernel(
    int N, int32_t *__restrict__ wetMask, scalar_t *__restrict__ qx_update,
    scalar_t *__restrict__ qy_update, uint8_t *__restrict__ landuse,
    scalar_t *__restrict__ h, scalar_t *__restrict__ qx,
    scalar_t *__restrict__ qy, scalar_t *__restrict__ manning,
    scalar_t *__restrict__ dt) {

  scalar_t h_small = 1.0e-6;
  scalar_t g = 9.81;

  // get the index of cell

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    int32_t i = wetMask[j];

    scalar_t q_sum = sqrtf(qx[i] * qx[i] + qy[i] * qy[i]);

    scalar_t chezy_para =
        manning[landuse[i]] * manning[landuse[i]] * g * powf(h[i], -2.33333333);

    scalar_t SM_00 =
        q_sum < h_small
            ? 1.0
            : 1.0 + dt[0] * (chezy_para * (q_sum + qx[i] * qx[i] / q_sum));
    scalar_t SM_11 =
        q_sum < h_small
            ? 1.0
            : 1.0 + dt[0] * (chezy_para * (q_sum + qy[i] * qy[i] / q_sum));

    scalar_t friction_x = chezy_para * qx[i] * q_sum;
    scalar_t friction_y = chezy_para * qy[i] * q_sum;

    qx_update[i] -= (SM_00 == 1.0 ? qx[i] : friction_x * dt[0] / SM_00);
    qx_update[i] -= (SM_11 == 1.0 ? qy[i] : friction_y * dt[0] / SM_11);
  }
}

void frictionCalculation_cuda(at::Tensor wetMask, at::Tensor qx_update,
                              at::Tensor qy_update, at::Tensor landuse,
                              at::Tensor h, at::Tensor qx, at::Tensor qy,
                              at::Tensor manning, at::Tensor dt) {
  const int N = wetMask.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "frictioncuda_Calculation", ([&] {
        frictionCalculation_kernel<
            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
            N, wetMask.data<int32_t>(), qx_update.data<scalar_t>(),
            qy_update.data<scalar_t>(), landuse.data<uint8_t>(),
            h.data<scalar_t>(), qx.data<scalar_t>(), qy.data<scalar_t>(),
            manning.data<scalar_t>(), dt.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
