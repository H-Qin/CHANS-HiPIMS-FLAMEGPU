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
    int N, int32_t *__restrict__ wetMask, const int32_t *__restrict__ index,
    scalar_t *__restrict__ h, scalar_t *__restrict__ C,
    scalar_t *__restrict__ qx, scalar_t *__restrict__ qy,
    scalar_t *__restrict__ z, scalar_t *__restrict__ z_non_move,
    scalar_t *__restrict__ h_update, scalar_t *__restrict__ qx_update,
    scalar_t *__restrict__ qy_update, scalar_t *__restrict__ z_update,
    uint8_t *__restrict__ landuse, scalar_t *__restrict__ manning,
    scalar_t *__restrict__ sedi_para, scalar_t *__restrict__ dt,
    scalar_t *__restrict__ dx) {

  scalar_t h_small = 1.0e-6;
  scalar_t g = 9.81;

  // # sedi_para = {
  // # "N": 25,
  // #0     "rho_w": 1.0,
  // #1     "rho_s": 2.65,
  // #2     "d": 1.0e-3,
  // #3     "epsilon": 1.0,
  // #4     "p": 0.35,
  // #5     "repose_angle": math.pi / 6.0,
  // #6     "ThetaC_calibration": 1.,
  // # }

  // get the index of cell
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    int32_t i = wetMask[j];

    int LandTypeNumber = (int)sedi_para[0];

    // scalar_t erosionable = sedi_para[landuse[i] + 1];
    scalar_t rho_w = sedi_para[LandTypeNumber * 0 + landuse[i] + 1];
    scalar_t rho_s = sedi_para[LandTypeNumber * 1 + landuse[i] + 1];
    scalar_t d = sedi_para[LandTypeNumber * 2 + landuse[i] + 1];
    scalar_t epsilon = sedi_para[LandTypeNumber * 3 + landuse[i] + 1];
    scalar_t p = sedi_para[LandTypeNumber * 4 + landuse[i] + 1];
    scalar_t nu = 1.2e-6;
    scalar_t repose_angle = sedi_para[LandTypeNumber * 5 + landuse[i] + 1];
    scalar_t thetaC_cali = sedi_para[LandTypeNumber * 6 + landuse[i] + 1];
    scalar_t d_star = d * pow((rho_s / rho_w - 1.0) * g / (nu * nu), 1.0 / 3.0);

    // first, we will calculate the gradient of bottom
    // normal = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    int D8[8];
    scalar_t theta = 0.0;
    scalar_t qb = 0.0;
    scalar_t qb_star = 0.0;
    scalar_t theta_c =
        0.3 / (1.0 + 1.2 * d_star) + 0.055 * (1 - exp(-0.02 * d_star));

    theta_c *= thetaC_cali;

    D8[4] = index[N + i] == -1 ? i : index[N + i];
    D8[1] = index[2 * N + i] == -1 ? i : index[2 * N + i];
    D8[3] = index[3 * N + i] == -1 ? i : index[3 * N + i];
    D8[6] = index[4 * N + i] == -1 ? i : index[4 * N + i];

    // get the index of upper and lower of D8[4]
    D8[2] = index[2 * N + D8[4]] == -1 ? i : index[2 * N + D8[4]];
    D8[7] = index[4 * N + D8[4]] == -1 ? i : index[4 * N + D8[4]];
    // get the index of upper and lower of D8[3]
    D8[0] = index[2 * N + D8[3]] == -1 ? i : index[2 * N + D8[3]];
    D8[5] = index[4 * N + D8[3]] == -1 ? i : index[4 * N + D8[3]];

    scalar_t z_gradient_x = ((z[D8[2]] + 2.0 * z[D8[4]] + z[D8[7]]) -
                             (z[D8[0]] + 2.0 * z[D8[3]] + z[D8[5]])) /
                            (8.0 * dx[0]);
    scalar_t z_gradient_y = ((z[D8[0]] + 2.0 * z[D8[1]] + z[D8[2]]) -
                             (z[D8[5]] + 2.0 * z[D8[6]] + z[D8[7]])) /
                            (8.0 * dx[0]);

    scalar_t angle = (qx[i] * z_gradient_x + qy[i] * z_gradient_y) > h_small
                         ? -atan(sqrt(z_gradient_x * z_gradient_x +
                                      z_gradient_y * z_gradient_y))
                         : atan(sqrt(z_gradient_x * z_gradient_x +
                                     z_gradient_y * z_gradient_y));

    // theta_c *= sin(repose_angle - angle) / sin(repose_angle);
    theta_c *= sin(min(min(max(repose_angle - angle, 0.0), 2.0*repose_angle), 3.1415926/2.0)) / sin(repose_angle);

    theta = manning[landuse[i]] * manning[landuse[i]] * pow(h[i], -7.0 / 3.0) *
            (qx[i] * qx[i] + qy[i] * qy[i]) / ((rho_s / rho_w - 1) * d);
    qb = sqrt(qx[i] * qx[i] + qy[i] * qy[i]) * C[i];
    qb_star = epsilon * 8.0 * sqrt((rho_s / rho_w - 1) * d * g) * d *
              pow(max(0.0, theta - theta_c), 1.5);

    scalar_t delta_z = (qb - qb_star) * dt[0] / (1.0 - p);

    // set the maximum erosion
    delta_z = max(z_non_move[i] - z[i], delta_z);

    z_update[i] += delta_z;
    h_update[i] -= delta_z;

    // =====================================================
    // bottom elevation move to fluid effect
    // =====================================================
    scalar_t rho_m = rho_s * C[i] + rho_w * (1.0 - C[i]);

    qx_update[i] +=
        (rho_s - rho_w) / rho_m * delta_z * qx[i] / h[i] * (1.0 - p - C[i]);
    qy_update[i] +=
        (rho_s - rho_w) / rho_m * delta_z * qy[i] / h[i] * (1.0 - p - C[i]);

    // =====================================================
    // concentration effect
    // =====================================================

    // scalar_t c_gradient_x = ((C[D8[2]] + 2.0 * C[D8[4]] + C[D8[7]]) -
    //                          (C[D8[0]] + 2.0 * C[D8[3]] + C[D8[5]])) /
    //                         (8.0 * dx[0]);
    // scalar_t c_gradient_y = ((C[D8[0]] + 2.0 * C[D8[1]] + C[D8[2]]) -
    //                          (C[D8[5]] + 2.0 * C[D8[6]] + C[D8[7]])) /
    //                         (8.0 * dx[0]);

    // scalar_t rho_m = rho_s * C[i] + rho_w * (1.0 - C[i]);

    // qx_update[i] -=
    //     (rho_s - rho_w) / (2.0 * rho_m) * g * h[i] * h[i] * c_gradient_x;
    // qy_update[i] -=
    //     (rho_s - rho_w) / (2.0 * rho_m) * g * h[i] * h[i] * c_gradient_y;
    // =====================================================

    /* if (h[i] > h_small) {
      theta = manning[landuse[i]] * manning[landuse[i]] *
              pow(h[i], -7.0 / 3.0) * (qx[i] * qx[i] + qy[i] * qy[i]) /
              ((rho_s / rho_w - 1) * d);
      q_b = sqrt(qx[i] * qx[i] + qy[i] * qy[i]) * C[i];
      q_b_star = epsilon * 8.0 * sqrt((rho_s / rho_w - 1) * d * g) * d *
                 pow(max(0.0, theta - theta_c), 1.5);
    } */
  }
}

void sedi_mass_momentum_update_cuda(at::Tensor wetMask, at::Tensor index,
                                    at::Tensor h, at::Tensor C, at::Tensor qx,
                                    at::Tensor qy, at::Tensor z,
                                    at::Tensor z_non_move, at::Tensor h_update,
                                    at::Tensor qx_update, at::Tensor qy_update,
                                    at::Tensor z_update, at::Tensor landuse,
                                    at::Tensor manning, at::Tensor sedi_para,
                                    at::Tensor dt, at::Tensor dx) {
  const int N = wetMask.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int thread_0 = 256;
  int block_0 = (N + thread_0 - 1) / thread_0;

  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "sedi_Calculation", ([&] {
        sedi_Calculation_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
            N, wetMask.data<int32_t>(), index.data<int32_t>(),
            h.data<scalar_t>(), C.data<scalar_t>(), qx.data<scalar_t>(),
            qy.data<scalar_t>(), z.data<scalar_t>(),
            z_non_move.data<scalar_t>(), h_update.data<scalar_t>(),
            qx_update.data<scalar_t>(), qy_update.data<scalar_t>(),
            z_update.data<scalar_t>(), landuse.data<uint8_t>(),
            manning.data<scalar_t>(), sedi_para.data<scalar_t>(),
            dt.data<scalar_t>(), dx.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
