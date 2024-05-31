// #include <torch/extension.h>
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

namespace {}

template <typename scalar_t>
__global__ void sedi_Calculation_kernel(
    int N, int32_t *__restrict__ wetMask, scalar_t *__restrict__ h,
    scalar_t *__restrict__ C, scalar_t *__restrict__ h_update,
    scalar_t *__restrict__ hC_update, scalar_t *__restrict__ z_update,
    scalar_t *__restrict__ sedi_para, uint8_t *__restrict__ landuse) {

  //   scalar_t h_small = 1.0e-6;

  // get the index of cell
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    int32_t i = wetMask[j];
    int LandTypeNumber = (int)sedi_para[0];
    scalar_t p = sedi_para[LandTypeNumber * 4 + landuse[i] + 1];
    scalar_t h_cell = h[i] + h_update[i];
    scalar_t hc_cell = h[i] * C[i] - z_update[i] / (1.0 - p) + hC_update[i];
    C[i] = h_cell > 1.0e-6 ? hc_cell / h_cell : 0.0;
    C[i] = min(1.0 - p, max(C[i], 0.0));
  }
}

void sedi_c_euler_update_cuda(at::Tensor wetMask, at::Tensor h, at::Tensor C,
                              at::Tensor h_update, at::Tensor hC_update,
                              at::Tensor z_update, at::Tensor sedi_para,
                              at::Tensor landuse) {
  const int N = wetMask.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int thread_0 = 256;
  int block_0 = (N + thread_0 - 1) / thread_0;

  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "sedi_Calculation", ([&] {
        sedi_Calculation_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
            N, wetMask.data<int32_t>(), h.data<scalar_t>(), C.data<scalar_t>(),
            h_update.data<scalar_t>(), hC_update.data<scalar_t>(),
            z_update.data<scalar_t>(), sedi_para.data<scalar_t>(),
            landuse.data<uint8_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
