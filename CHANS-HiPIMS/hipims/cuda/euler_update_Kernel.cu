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
__global__ void euler_update_kernel(
    int N, int32_t *__restrict__ updateMask, scalar_t *__restrict__ h_update,
    scalar_t *__restrict__ qx_update, scalar_t *__restrict__ qy_update,
    scalar_t *__restrict__ z_update, scalar_t *__restrict__ h,
    scalar_t *__restrict__ wl, scalar_t *__restrict__ z,
    scalar_t *__restrict__ qx, scalar_t *__restrict__ qy) {
  // get the index of cell

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    int32_t i = updateMask[j];
    h[i] += h_update[i];
    qx[i] += qx_update[i];
    qy[i] += qy_update[i];
    z[i] += z_update[i];
    wl[i] = h[i] + z[i];
  }
}

void euler_update_cuda(at::Tensor updateMask, at::Tensor h_update,
                       at::Tensor qx_update, at::Tensor qy_update,
                       at::Tensor z_update, at::Tensor h, at::Tensor wl,
                       at::Tensor z, at::Tensor qx, at::Tensor qy) {
  const int N = updateMask.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "euler_update", ([&] {
        euler_update_kernel<
            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
            N, updateMask.data<int32_t>(), h_update.data<scalar_t>(),
            qx_update.data<scalar_t>(), qy_update.data<scalar_t>(),
            z_update.data<scalar_t>(), h.data<scalar_t>(), wl.data<scalar_t>(),
            z.data<scalar_t>(), qx.data<scalar_t>(), qy.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
