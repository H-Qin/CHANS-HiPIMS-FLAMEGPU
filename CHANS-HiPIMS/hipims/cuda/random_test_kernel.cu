// this is used for testing curand
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>

template <typename scalar_t>
__device__ scalar_t randomx(curandState states, scalar_t dx){
    scalar_t x;
    x = curand_uniform(&states) * dx;
    return x;
}

template <typename scalar_t>
__global__ void Random_kernel(const int N, 
                                scalar_t *h){
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N){
        scalar_t dx = 3.0;
        curandState states;

        int seed = j; // different seed per thread
        scalar_t cid1, cid2;
        curand_init(seed, j, 0, &states);  // 	Initialize CURAND

        for (int i=0; i < 2; i++){
            curand(&states);
            cid1 = randomx(states, dx);
            curand(&states);
            cid2 = randomx(states, dx);
            h[j*2] = cid1;
            h[j*2 + 1] = cid2;
        }
    }
}

void Random_cuda(at::Tensor h) {
    at::cuda::CUDAGuard device_guard(h.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = h.numel()/2;
    int thread_0 = 512;
    int block_0 = (N + 512 - 1) / 512;
    
    AT_DISPATCH_FLOATING_TYPES(
        h.type(), "Random", ([&] {
          Random_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
              N, h.data<scalar_t>());
        }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}