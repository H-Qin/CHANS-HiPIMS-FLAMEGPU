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
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <tuple>

// template <typename scalar_t>
// __global__ void
// timeControl_kernel(scalar_t *__restrict__ h, scalar_t *__restrict__ qx,
//                    scalar_t *__restrict__ qy, scalar_t *__restrict__ dx,
//                    scalar_t *__restrict__ CFL, scalar_t *__restrict__ t,
//                    scalar_t *__restrict__ dt) {

//   extern __shared__ float sha_partialMax2[];
//   const int tid = threadIdx.x;
//   const int gTid = blockIdx.x * (blockDim.x * 2) + tid;
//   // const int gTid = blockIdx.x * blockDim.x + tid;
//   const float g = 9.81;
//   const float h_small = 1.0e-6;

//   // 复制全局数据到共享内存

//   // Complex in0, in1;
//   // in0 = in[gTid];
//   // sha_partialMax2[tid] = in0.x*in0.x + in0.y*in0.y;
//   float h0, h1, qx0, qx1, qy0, qy1;
//   h0 = h[gTid];
//   qx0 = qx[gTid];
//   qy0 = qy[gTid];
//   sha_partialMax2[tid] =
//       h0 > h_small ? sqrtf(g * h0) + sqrtf(powf(qx0, 2.0) + powf(qy0, 2.0)) /
//       h0
//                    : 0.0;
//   h1 = h[gTid + blockDim.x];
//   qx1 = qx[gTid + blockDim.x];
//   qy1 = qy[gTid + blockDim.x];
//   sha_partialMax2[tid + blockDim.x] =
//       h1 > h_small ? sqrtf(g * h1) + sqrtf(powf(qx1, 2.0) + powf(qy1, 2.0)) /
//       h1
//                    : 0.0;

//   // block内前半部分与后半部分对应比较大小
//   if (blockDim.x > 512) {
//     __syncthreads();
//     if (sha_partialMax2[tid] < sha_partialMax2[tid + 1024])
//       sha_partialMax2[tid] = sha_partialMax2[tid + 1024];
//   }
//   if (blockDim.x > 256 && tid < 512) {
//     __syncthreads();
//     if (sha_partialMax2[tid] < sha_partialMax2[tid + 512])
//       sha_partialMax2[tid] = sha_partialMax2[tid + 512];
//   }
//   if (tid < 256) {
//     __syncthreads();
//     if (sha_partialMax2[tid] < sha_partialMax2[tid + 256])
//       sha_partialMax2[tid] = sha_partialMax2[tid + 256];
//   }
//   if (tid < 128) {
//     __syncthreads();
//     if (sha_partialMax2[tid] < sha_partialMax2[tid + 128])
//       sha_partialMax2[tid] = sha_partialMax2[tid + 128];
//   }
//   if (tid < 64) {
//     __syncthreads();
//     if (sha_partialMax2[tid] < sha_partialMax2[tid + 64])
//       sha_partialMax2[tid] = sha_partialMax2[tid + 64];
//   }

//   if (tid < 32) //步长stride小于等于warp
//   {
//     __syncthreads();
//     volatile float *vol_sh_max =
//         sha_partialMax2; // volatile 确保每次对共享内存的修改都即时修改
//     float register val_t;
//     for (int stride = 32; stride > 0; stride >>= 1) {
//       val_t = vol_sh_max[tid + stride];
//       if (vol_sh_max[tid] < val_t)
//         vol_sh_max[tid] = val_t;
//     }
//   }

//   if (tid == 0) {
//     dt[0] = CFL[0] * dx[0] / sha_partialMax2[0];
//     // t[0] = dt[0] + t[0];
//   }
// }

template <typename scalar_t>
__global__ void timeControl_kernel(const int n, scalar_t *__restrict__ g_idata,
                                   scalar_t *__restrict__ g_odata) {
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  scalar_t *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7 * blockDim.x < n) {
    scalar_t a1 = g_idata[idx];
    scalar_t a2 = g_idata[idx + blockDim.x];
    scalar_t a3 = g_idata[idx + 2 * blockDim.x];
    scalar_t a4 = g_idata[idx + 3 * blockDim.x];
    scalar_t b1 = g_idata[idx + 4 * blockDim.x];
    scalar_t b2 = g_idata[idx + 5 * blockDim.x];
    scalar_t b3 = g_idata[idx + 6 * blockDim.x];
    scalar_t b4 = g_idata[idx + 7 * blockDim.x];
    g_idata[idx] =
        max(max(max(max(max(max(max(a1, a2), a3), a4), b1), b2), b3), b4);
  }

  __syncthreads();

  // in-place reduction and complete unroll
  if (blockDim.x >= 1024 && tid < 512)
    idata[tid] = max(idata[tid], idata[tid + 512]);

  __syncthreads();

  if (blockDim.x >= 512 && tid < 256)
    idata[tid] = max(idata[tid], idata[tid + 256]);

  __syncthreads();

  if (blockDim.x >= 256 && tid < 128)
    idata[tid] = max(idata[tid], idata[tid + 128]);

  __syncthreads();

  if (blockDim.x >= 128 && tid < 64)
    idata[tid] = max(idata[tid], idata[tid + 64]);

  __syncthreads();

  // unrolling warp
  if (tid < 32) {
    volatile scalar_t *vmem = idata;
    vmem[tid] = max(vmem[tid], vmem[tid + 32]);
    vmem[tid] = max(vmem[tid], vmem[tid + 16]);
    vmem[tid] = max(vmem[tid], vmem[tid + 8]);
    vmem[tid] = max(vmem[tid], vmem[tid + 4]);
    vmem[tid] = max(vmem[tid], vmem[tid + 2]);
    vmem[tid] = max(vmem[tid], vmem[tid + 1]);
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

template <typename scalar_t>
__global__ void
timeControl_accer(const int N, int32_t *__restrict__ wetMask,
                  scalar_t *__restrict__ h_max, scalar_t *__restrict__ h,
                  scalar_t *__restrict__ qx, scalar_t *__restrict__ qy,
                  scalar_t *__restrict__ accer, scalar_t *__restrict__ dx,
                  scalar_t *__restrict__ CFL, scalar_t *__restrict__ t,
                  scalar_t *__restrict__ dt) {
  const scalar_t g = 9.81;
  const scalar_t h_small = 1.0e-6;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    // accer[i] = h[i] > h_small
    //                ? sqrtf(g * h[i]) +
    //                      sqrtf(powf(qx[i], 2.0) + powf(qy[i], 2.0)) / h[i]
    //                : 0.0;
    int32_t j = wetMask[i];
    h_max[j] = max(h[j], h_max[j]);
    // scalar_t accer_temp = sqrtf(g * h[j]) + max(abs(qx[j]), abs(qy[j])) /
    // h[j];
    // accer[i] = CFL[0] * dx[0] / accer_temp;
    accer[i] = h[j] > h_small
                   ? CFL[0] * dx[0] /
                         (sqrtf(g * h[j]) + max(abs(qx[j]), abs(qy[j])) / h[j])
                   : 60.0;

    // if (h[j] > h_small) {
    //   h_max[j] = max(h[j], h_max[j]);
    //   scalar_t accer_temp = sqrtf(g * h[j]) + max(abs(qx[j]), abs(qy[j])) /
    //   h[j];
    //   accer[i] = CFL[0] * dx[0] / accer_temp;
    // }
  }
}
// template <typename scalar_t>
// __global__ void
// update_maxDepth_kernel(const int N, int32_t *__restrict__ wetMask,
//                        scalar_t *__restrict__ h, scalar_t *__restrict__
//                        h_max) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < N) {
//     h_max[wetMask[i]] = max(h[wetMask[i]], h_max[wetMask[i]]);
//   }
// }

void timeControl_cuda(at::Tensor wetMask, at::Tensor accelerator,
                      at::Tensor h_max, at::Tensor h, at::Tensor qx,
                      at::Tensor qy, at::Tensor dx, at::Tensor CFL,
                      at::Tensor t, at::Tensor dt) {
  const int N = wetMask.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // printf("test max func: ", at::argmax(h_max));
  // int blocksize = 1024;

  // dim3 block(blocksize, 1);
  // dim3 grid((N + block.x - 1) / block.x, 1);

  // auto accer = torch::zeros_like(h);
  // auto accer_o = torch::zeros({grid.x / 8});

  // AT_DISPATCH_FLOATING_TYPES(
  //     h.type(), "timecuda_Control_0", ([&] {
  //       update_maxDepth_kernel<
  //           scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
  //           N, wetMask.data<int32_t>(), h.data<scalar_t>(),
  //           h_max.data<scalar_t>());
  //     }));
  // =============================================================
  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "timecuda_Control_1", ([&] {
        timeControl_accer<
            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
            N, wetMask.data<int32_t>(), h_max.data<scalar_t>(),
            h.data<scalar_t>(), qx.data<scalar_t>(), qy.data<scalar_t>(),
            accelerator.data<scalar_t>(), dx.data<scalar_t>(),
            CFL.data<scalar_t>(), t.data<scalar_t>(), dt.data<scalar_t>());
      }));
  // =============================================================
  // auto accer = torch::zeros_like(h);
  // AT_DISPATCH_FLOATING_TYPES(
  //     h.type(), "timecuda_Control_1", ([&] {
  //       timeControl_accer<
  //           scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
  //           N, h.data<scalar_t>(), qx.data<scalar_t>(), qy.data<scalar_t>(),
  //           accer.data<scalar_t>(), dx.data<scalar_t>(),
  //           CFL.data<scalar_t>(),
  //           t.data<scalar_t>(), dt.data<scalar_t>());
  //     }));

  // AT_DISPATCH_FLOATING_TYPES(
  //     h.type(), "timecuda_Control_2", ([&] {
  //       timeControl_kernel<
  //           scalar_t><<<GET_BLOCKS(N) / 8, CUDA_NUM_THREADS, 0, stream>>>(
  //           N, accer.data<scalar_t>(), accelerator.data<scalar_t>());
  //     }));
  // =============================================================

  // auto maxValue = 0.0;
  // for (int g = 0; g < grid.x; g++) {
  //   maxValue = max(accer_o[g].item<float>(), maxValue);
  // }

  // if (max > 1.0e-7) {
  //   dt = CFL * dx / maxValue;
  // }

  // auto max_and_index = accer_o.max(0, true);
  // auto max_value = std::get<0>(max_and_index).item<float>();
  // if (max_value > 1.0e-8) {
  //   dt = CFL * dx / max_value;
  // }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
