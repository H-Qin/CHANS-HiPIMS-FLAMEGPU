// High-Performance Integrated hydrodynamic Modelling System ***hybrid***
// @author: Jiaheng Zhao (Hemlab)
// @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
// @contact: j.zhao@lboro.ac.uk
// @software: hipims_hybrid
// @time: 07.01.2021
// This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
// Feel free to use and extend if you are a ***member of hemlab***.
#include "gpu.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void fluxMask_kernel(const int N, bool *__restrict__ fluxMask,
                                scalar_t *__restrict__ h,
                                const int32_t *__restrict__ index,
                                scalar_t *__restrict__ t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_t h_small = 1.0e-6;
  int32_t n_index;
  if (i < N) {
    if (t[0] == 0.0) {
      fluxMask[i] = true;
    } else {
      fluxMask[i] = false;
      fluxMask[i] = fluxMask[i] || (h[i] > h_small);
      for (int nei = 1; nei < 5; nei++) {
        n_index = nei * N + i;
        n_index = index[n_index];
        if (n_index > 0) {
          fluxMask[i] = fluxMask[i] || (h[n_index] > h_small);
        }
      }
    }
  }
}

void fluxMask_cuda(at::Tensor fluxMask, at::Tensor h, at::Tensor index,
                   at::Tensor t) {
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int N = h.size(0);

  int thread_0 = 512;
  int block_0 = (N + 512 - 1) / 512;

  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "fluxMask_cuda", ([&] {
        fluxMask_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
            N, fluxMask.data<bool>(), h.data<scalar_t>(), index.data<int32_t>(),
            t.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
