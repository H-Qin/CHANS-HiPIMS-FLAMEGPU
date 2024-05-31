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
#include <math.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void infiltrationCalculation_kernel(
    int N, int32_t *__restrict__ wetMask, scalar_t *__restrict__ h_update,
    uint8_t *__restrict__ landuse, scalar_t *__restrict__ h,
    scalar_t *__restrict__ hydraulic_conductivity,
    scalar_t *__restrict__ capillary_head,
    scalar_t *__restrict__ water_content_diff,
    scalar_t *__restrict__ cumulative_depth, scalar_t *__restrict__ dt) {
  // get the index of cell
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    int32_t i = wetMask[j];
    uint8_t land_type = landuse[i];
    scalar_t _h = h[i];
    scalar_t K_s = hydraulic_conductivity[land_type];
    scalar_t phi_s = capillary_head[land_type];
    scalar_t delta_theta = water_content_diff[land_type];
    scalar_t F_0 = cumulative_depth[i];
    scalar_t total_head = phi_s + _h;
    scalar_t F_1 = 0.5 * (F_0 + dt[0] * K_s +
                          sqrt((F_0 + dt[0] * K_s) * (F_0 + dt[0] * K_s) +
                               4.0 * dt[0] * K_s * total_head * delta_theta));
    scalar_t delta_F = min(_h, F_1 - F_0);
    cumulative_depth[i] += delta_F;
    h_update[i] -= delta_F;
  }
}

void infiltrationCalculation_cuda(at::Tensor wetMask, at::Tensor h_update,
                                  at::Tensor landuse, at::Tensor h,
                                  at::Tensor hydraulic_conductivity,
                                  at::Tensor capillary_head,
                                  at::Tensor water_content_diff,
                                  at::Tensor cumulative_depth, at::Tensor dt) {
  const int N = wetMask.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "infiltrationcuda_Calculation", ([&] {
        infiltrationCalculation_kernel<
            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
            N, wetMask.data<int32_t>(), h_update.data<scalar_t>(),
            landuse.data<uint8_t>(), h.data<scalar_t>(),
            hydraulic_conductivity.data<scalar_t>(),
            capillary_head.data<scalar_t>(),
            water_content_diff.data<scalar_t>(),
            cumulative_depth.data<scalar_t>(), dt.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
