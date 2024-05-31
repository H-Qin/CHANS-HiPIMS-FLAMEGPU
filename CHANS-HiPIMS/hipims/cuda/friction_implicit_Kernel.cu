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

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t norm(scalar_t *U) {
  return sqrtf(U[0] * U[0] + U[1] * U[1]);
}

template <typename scalar_t>
__device__ __forceinline__ void inverse(scalar_t *phi, scalar_t *in) {
  scalar_t det = phi[0] * phi[3] - phi[1] * phi[2];
  in[0] = phi[3] / det;
  in[1] = -phi[1] / det;
  in[2] = -phi[2] / det;
  in[3] = phi[0] / det;
}

template <typename scalar_t>
__device__ __forceinline__ void ManningNewton(scalar_t *S_b, scalar_t C_f,
                                              scalar_t dt, scalar_t *U,
                                              scalar_t *U_k1) {
  scalar_t S_f[2];
  S_f[0] = -C_f * norm(U) * U[0];
  S_f[1] = -C_f * norm(U) * U[1];

  U_k1[0] = U[0];
  U_k1[1] = U[1];

  if (sqrtf((S_b[0] + S_f[0]) * (S_b[0] + S_f[0]) +
            (S_b[1] + S_f[1]) * (S_b[1] + S_f[1])) <=
      1e-10) { // steady state, return directly
    return;
  }
  scalar_t epsilon = 0.001; // termination criteria
  scalar_t U_k[2] = {U[0], U[1]};
  scalar_t inv[4];
  scalar_t S[2];
  scalar_t rhs[2];
  scalar_t ijac[4];

  scalar_t J_xx, J_yy, J_xy;
  unsigned int k = 0;
  while (true) {

    if (norm(U_k) <= 1e-10) { // Jacobian matrix is 0
      inv[0] = 1.0;
      inv[1] = 0.0;
      inv[2] = 0.0;
      inv[3] = 1.0;
    } else {
      J_xx = -C_f * (2.0 * U_k[0] * U_k[0] + U_k[1] * U_k[1]) / norm(U_k);
      J_yy = -C_f * (U_k[0] * U_k[0] + 2.0 * U_k[1] * U_k[1]) / norm(U_k);
      J_xy = -C_f * (U_k[0] * U_k[1]) / norm(U_k);
      // ijac[0] = {1.0 - dt * J_xx, -dt * J_xy, -dt * J_xy, 1.0 - dt * J_yy};
      ijac[0] = 1.0 - dt * J_xx;
      ijac[1] = -dt * J_xy;
      ijac[2] = -dt * J_xy;
      ijac[3] = 1.0 - dt * J_yy;

      inverse(ijac, inv);
    }

    S[0] = S_b[0] - C_f * norm(U_k) * U_k[0];
    S[1] = S_b[1] - C_f * norm(U_k) * U_k[1];

    rhs[0] = dt * S[0] - U_k[0] + U[0];
    rhs[1] = dt * S[1] - U_k[1] + U[1];
    U_k1[0] = U_k[0] + inv[0] * rhs[0] + inv[1] * rhs[1];
    U_k1[0] = U_k[1] + inv[2] * rhs[0] + inv[3] * rhs[1];

    if (sqrtf((U_k1[0] - U_k[0]) * (U_k1[0] - U_k[0]) +
              (U_k1[1] - U_k[1]) * (U_k1[1] - U_k[1])) <= epsilon * norm(U_k) ||
        k > 10) {
      break;
    }
    U_k[0] = U_k1[0];
    U_k[1] = U_k1[1];
    k++;
  }
  return;
}
}

template <typename scalar_t>
__global__ void frictionCalculation_kernel(
    int N, int32_t *__restrict__ wetMask, scalar_t *__restrict__ qx_update,
    scalar_t *__restrict__ qy_update, uint8_t *__restrict__ landuse,
    scalar_t *__restrict__ h, scalar_t *__restrict__ qx,
    scalar_t *__restrict__ qy, scalar_t *__restrict__ manning,
    scalar_t *__restrict__ dt) {

  scalar_t h_small = 1.0e-6;
  scalar_t g = 9.81;
  scalar_t U_[2];
  scalar_t U_1[2];
  scalar_t acc_[2];
  scalar_t C_f;

  // get the index of cell

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    int32_t i = wetMask[j];
    C_f = g * manning[landuse[i]] * manning[landuse[i]] * pow(h[i], -4.0 / 3.0);
    U_[0] = qx[i] / h[i];
    U_[1] = qy[i] / h[i];
    acc_[0] = qx_update[i] / h[i];
    acc_[1] = qy_update[i] / h[i];

    ManningNewton(acc_, C_f, dt[0], U_, U_1);
    qx[i] = U_1[0] * h[i];
    qy[i] = U_1[1] * h[i];
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
