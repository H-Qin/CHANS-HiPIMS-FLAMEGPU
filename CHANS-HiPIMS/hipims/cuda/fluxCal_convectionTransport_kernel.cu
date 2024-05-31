#include "gpu.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
// #include <c10/cuda/CUDACachingAllocator.h>

// enum FIELD_TYPE { H = 0, WL = 1, QX = 2, QY = 3, Z = 4 };
// enum BOUNDARY_TYPE = {WALL_NON_SLIP = 3, WALL_SLIP = 4, OPEN = 5, HQ_GIVEN =
// 6};
// field
// 0: h, 1: wl, 2: qx, 3: qy, 4: z
// #define H 0
// #define WL 1
// #define QX 2
// #define QY 3
// #define Z 4

// // boundary type
// #define WALL_NON_SLIP 3
// #define WALL_SLIP 4
// #define OPEN 5
// #define HQ_GIVEN 6

namespace
{

  // update:
  // the field_c, field_l, field_r are chose to represent the central, left and
  // right field values

  template <typename scalar_t>
  __device__ __forceinline__ void
  hydroStaticReconstruction(scalar_t *field_c_h_z, scalar_t *field_r_h_z,
                            scalar_t *field_l, scalar_t *field_r,
                            const scalar_t h_small)
  {

    // ==================================================
    // jh's modified surface, slope will reconstructed to the lower part
    // ==================================================
    // scalar_t delta_b_interface = field_l[4] - field_r[4];

    // field_l[4] =
    //     min(field_r_h_z[1], field_c_h_z[1]) + max(delta_b_interface, 0.0);
    // field_r[4] =
    //     min(field_r_h_z[1], field_c_h_z[1]) + max(-delta_b_interface, 0.0);

    // field_l[1] =
    //     max(field_l[4] + field_l[0], min(field_r_h_z[1] + field_r_h_z[0],
    //                                      field_c_h_z[1] + field_c_h_z[0]));
    // field_r[1] =
    //     max(field_r[4] + field_r[0], min(field_r_h_z[1] + field_r_h_z[0],
    //                                      field_c_h_z[1] + field_c_h_z[0]));

    // field_l[4] = field_l[1] - field_l[0];
    // field_r[4] = field_r[1] - field_r[0];

    // scalar_t z_bm = max(field_l[4], field_r[4]);

    // field_l[0] = min(max(field_l[1] - z_bm, 0.0), field_c_h_z[0]);
    // field_r[0] = min(max(field_r[1] - z_bm, 0.0), field_r_h_z[0]);

    // field_l[4] = field_l[1] - field_l[0];
    //====================================================================
    // jh's center method
    //====================================================================
    // field_l[1] = field_l[0] + field_l[4];
    // field_r[1] = field_r[0] + field_r[4];

    // // if ((field_l[1] - field_r[1]) *
    // //         (field_c_h_z[0] + field_c_h_z[1] - field_r_h_z[0] -
    // field_r_h_z[1]) <
    // //     0.0) {
    // //   field_l[4] = min(field_l[1], field_r[1]) - field_l[0];
    // //   field_r[4] = min(field_l[1], field_r[1]) - field_r[0];
    // // }

    // scalar_t z_bm = min(max(field_l[4], field_r[4]), field_l[1]);

    // field_l[0] = max(field_l[1] - z_bm, 0.0);
    // field_r[0] = max(field_r[1] - z_bm, 0.0) - max(field_r[4] - z_bm, 0.0);
    // field_l[4] = z_bm;

    // jiaheng paper
    field_l[1] = field_l[0] + field_l[4];
    field_l[1] = min(field_l[1], max(field_c_h_z[0] + field_c_h_z[1],
                                     field_r_h_z[0] + field_r_h_z[1]));
    field_l[1] = max(field_l[1], min(field_c_h_z[0] + field_c_h_z[1],
                                     field_r_h_z[0] + field_r_h_z[1]));
    field_r[1] = field_r[0] + field_r[4];
    field_r[1] = min(field_r[1], max(field_c_h_z[0] + field_c_h_z[1],
                                     field_r_h_z[0] + field_r_h_z[1]));
    field_r[1] = max(field_r[1], min(field_c_h_z[0] + field_c_h_z[1],
                                     field_r_h_z[0] + field_r_h_z[1]));

    // if (field_c_h_z[0] + field_c_h_z[1] > field_r_h_z[0] + field_r_h_z[1]) {
    //   field_l[1] = max(field_l[1], field_r[1]);
    // } else {
    //   field_r[1] = max(field_l[1], field_r[1]);
    // }

    if (((field_c_h_z[0] + field_c_h_z[1]) - (field_r_h_z[0] + field_r_h_z[1])) *
            (field_l[1] - field_r[1]) <
        0.0)
    {
      scalar_t temp = (field_l[1] + field_r[1]) / 2.0;
      field_l[1] = temp;
      field_r[1] = temp;
    }

    field_l[4] = field_l[1] - field_l[0];
    field_r[4] = field_r[1] - field_r[0];
    auto z_bm = max(field_l[4], field_r[4]);

    field_l[0] = max(field_l[1] - z_bm, 0.0);
    field_r[0] = max(field_r[1] - z_bm, 0.0);

    // addiational treatment
    // if (field_l[0] <= h_small || field_r[0] <= h_small) {
    //   z_bm = max(field_c_h_z[1], field_r_h_z[1]);
    //   field_l[1] = field_c_h_z[0] + field_c_h_z[1];
    //   field_r[1] = field_r_h_z[0] + field_r_h_z[1];
    //   field_l[0] = max(field_l[1] - z_bm, 0.0);
    //   field_r[0] = max(field_r[1] - z_bm, 0.0);
    // }

    // jiaheng
    field_l[4] = field_l[1] - field_l[0];
    field_r[4] = field_r[1] - field_r[0];
    // Adusse
    // field_l[4] = z_bm;
    // field_r[4] = z_bm;

    // ====================================================================
    // field_l[4] = z_bm;
    // field_r[4] = z_bm;

    //   scalar_t delta_b_interface = field_r[4] - field_l[4];

    //   field_l[4] =
    //       max(field_r_h_z[1], field_c_h_z[1]) - max(delta_b_interface, 0.0);
    //   field_r[4] =
    //       max(field_r_h_z[1], field_c_h_z[1]) - max(-delta_b_interface, 0.0);

    //   field_l[1] =
    //       min(field_l[4] + field_l[0], max(field_r_h_z[1] + field_r_h_z[0],
    //                                        field_c_h_z[1] + field_c_h_z[0]));
    //   field_r[1] =
    //       min(field_r[4] + field_r[0], max(field_r_h_z[1] + field_r_h_z[0],
    //                                        field_c_h_z[1] + field_c_h_z[0]));

    //   field_l[4] = field_l[1] - field_l[0];
    //   field_r[4] = field_r[1] - field_r[0];

    //   scalar_t z_bm = max(field_l[4], field_r[4]);

    //   field_l[0] = max(field_l[1] - z_bm, 0.0);
    //   field_r[0] = max(field_r[1] - z_bm, 0.0);
    //   field_l[4] = field_l[1] - field_l[0];
    //   field_r[4] = field_r[1] - field_r[0];
  }

  template <typename scalar_t>
  __device__ __forceinline__ void
  hllcFluxSolver(scalar_t *field_c_h_z, scalar_t *field_r_h_z, scalar_t *field_l,
                 scalar_t *field_r, const scalar_t *normal,
                 const scalar_t *tangential, scalar_t &flux_h, scalar_t &flux_ch,
                 scalar_t &flux_qx, scalar_t &flux_qy, const scalar_t h_small,
                 const scalar_t g, const scalar_t dx, const scalar_t dt)
  {

    auto ux_l = field_l[0] < h_small ? 0.0 : field_l[2] / field_l[0];
    auto uy_l = field_l[0] < h_small ? 0.0 : field_l[3] / field_l[0];

    auto ux_r = field_r[0] < h_small ? 0.0 : field_r[2] / field_r[0];
    auto uy_r = field_r[0] < h_small ? 0.0 : field_r[3] / field_r[0];

    hydroStaticReconstruction(field_c_h_z, field_r_h_z, field_l, field_r,
                              h_small);

    if (field_l[0] < h_small && field_r[0] < h_small)
    {
      return;
    }
    else
    {
      auto u_L = ux_l * normal[0] + uy_l * normal[1];
      auto u_R = ux_r * normal[0] + uy_r * normal[1];

      auto v_L = -ux_l * normal[1] + uy_l * normal[0];
      auto v_R = -ux_r * normal[1] + uy_r * normal[0];

      auto qx_L = u_L * field_l[0]; // acturally the normal discharge
      auto qy_L = v_L * field_l[0]; // acturally the tan discharge
      auto qx_R = u_R * field_r[0];
      auto qy_R = v_R * field_r[0];

      scalar_t a_L = sqrt(g * field_l[0]);
      scalar_t a_R = sqrt(g * field_r[0]);

      scalar_t h_star = pow((a_L + a_R) / 2.0 + (u_L - u_R) / 4.0, 2.0) / g;
      scalar_t u_star = (u_L + u_R) / 2.0 + a_L - a_R;
      scalar_t a_star = sqrt(g * h_star);

      scalar_t s_L, s_R;

      s_L = field_l[0] <= h_small ? u_R - 2.0 * a_R
                                  : min(u_L - a_L, u_star - a_star);
      s_R = field_r[0] <= h_small ? u_L + 2.0 * a_L
                                  : max(u_R + a_R, u_star + a_star);

      scalar_t s_M =
          (s_L * field_r[0] * (u_R - s_R) - s_R * field_l[0] * (u_L - s_L)) /
          (field_r[0] * (u_R - s_R) - field_l[0] * (u_L - s_L));

      scalar_t h_flux_L = qx_L;
      scalar_t qx_flux_L = u_L * qx_L + 0.5 * g * field_l[0] * field_l[0];
      scalar_t qy_flux_L = u_L * qy_L;

      scalar_t h_flux_R = qx_R;
      scalar_t qx_flux_R = u_R * qx_R + 0.5 * g * field_r[0] * field_r[0];
      scalar_t qy_flux_R = u_R * qy_R;

      scalar_t h_flux_M = (s_R * h_flux_L - s_L * h_flux_R +
                           s_L * s_R * (field_r[0] - field_l[0])) /
                          (s_R - s_L);
      scalar_t qx_flux_M =
          (s_R * qx_flux_L - s_L * qx_flux_R + s_L * s_R * (qx_R - qx_L)) /
          (s_R - s_L);

      scalar_t h_flux, hc_flux, qx_flux, qy_flux;
      if (0.0 <= s_L)
      {
        h_flux = h_flux_L;
        hc_flux = h_flux * field_l[5];
        qx_flux = qx_flux_L;
        qy_flux = qy_flux_L;
      }
      else if (s_L < 0.0 && 0.0 <= s_M)
      {
        h_flux = h_flux_M;
        hc_flux = h_flux * field_l[5];
        qx_flux = qx_flux_M;
        qy_flux = h_flux_M * v_L;
      }
      else if (s_M < 0.0 && 0.0 <= s_R)
      {
        h_flux = h_flux_M;
        hc_flux = h_flux * field_r[5];
        qx_flux = qx_flux_M;
        qy_flux = h_flux_M * v_R;
      }
      else
      {
        h_flux = h_flux_R;
        hc_flux = h_flux * field_r[5];
        qx_flux = qx_flux_R;
        qy_flux = qy_flux_R;
      }
      flux_ch -= (hc_flux / dx) * dt;
      flux_h -= (h_flux / dx) * dt;
      flux_qx -= ((qx_flux * normal[0] + qy_flux * tangential[0]) / dx) * dt;
      flux_qy -= ((qx_flux * normal[1] + qy_flux * tangential[1]) / dx) * dt;
    }
  }

  template <typename scalar_t>
  __device__ __forceinline__ void boundary_WallNonSlip(scalar_t *field,
                                                       scalar_t *field_b)
  {
    field_b[0] = field[0];
    field_b[1] = field[1];
    field_b[4] = field[4];
    field_b[2] = -1.0 * field[2];
    field_b[3] = -1.0 * field[3];
    // field_b[2] = 0.0;
    // field_b[3] = 0.0;
    field_b[5] = field[5];
  }

  template <typename scalar_t>
  __device__ __forceinline__ void
  boundary_WallSlip(scalar_t *field, scalar_t *field_b, const scalar_t *normal)
  {
    field_b[0] = field[0];
    field_b[1] = field[1];
    field_b[4] = field[4];
    // field_b[2] = (normal[1] * normal[1] - normal[0] * normal[0]) * field[2];
    // field_b[3] = (normal[0] * normal[0] - normal[1] * normal[1]) * field[3];
    field_b[2] = (normal[0] * normal[0] - normal[1] * normal[1]) * field[2];
    field_b[3] = (normal[1] * normal[1] - normal[0] * normal[0]) * field[3];
    field_b[5] = field[5];
  }

  template <typename scalar_t>
  __device__ __forceinline__ void boundary_Open(scalar_t *field,
                                                scalar_t *field_b)
  {
    field_b[0] = field[0];
    field_b[1] = field[1];
    field_b[4] = field[4];
    field_b[2] = field[2];
    field_b[3] = field[3];
    field_b[5] = field[5];
  }

  template <typename scalar_t>
  __device__ __forceinline__ void
  boundary_Critical_Open(scalar_t *field, scalar_t *field_b, scalar_t manning,
                         scalar_t slope)
  {
    field_b[0] = field[0];
    field_b[4] = field[4];
    field_b[2] = field[2];
    field_b[3] = field[3];
    scalar_t q_norm = sqrt(field[2] * field[2] + field[3] * field[3]);
    scalar_t h_critical = pow(manning * q_norm / sqrt(slope), 0.6);
    field_b[0] = h_critical;
    field_b[1] = field_b[0] + field_b[4];
    field_b[5] = field[5];
  }

  template <typename scalar_t>
  __device__ __forceinline__ void minmod_tri(scalar_t x, scalar_t y, scalar_t z,
                                             scalar_t &r)
  {
    scalar_t theta = 1.0;
    if (x >= 0.0 && y >= 0.0 && z >= 0.0)
    {
      r = min(min(theta * x, 0.5 * y), theta * z);
    }
    else if (x <= 0.0 && y <= 0.0 && z <= 0.0)
    {
      r = max(max(theta * x, 0.5 * y), theta * z);
    }
    else
    {
      r = 0.0;
    }
  }

  template <typename scalar_t>
  __device__ __forceinline__ void minmod(scalar_t x, scalar_t y, scalar_t &r)
  {
    if (x * y <= 0.0)
    {
      r = 0.0;
    }
    else
    {
      r = fabs(x) < fabs(y) ? x : y;
    }
  }

  template <typename scalar_t>
  __device__ __forceinline__ void multiVanLeer(scalar_t minus, scalar_t plus,
                                               scalar_t &r)
  {
    if (minus * plus <= 0.0)
    {
      r = 0.0;
    }
    else
    {
      r = minus / plus;
      r = (2.0 * r) / (1 + r) * plus;
    }
  }

  template <typename scalar_t>
  __device__ __forceinline__ void
  boundary_HQ_given(scalar_t *field, scalar_t *field_b, const scalar_t *normal,
                    const scalar_t given_depth, const scalar_t given_qx,
                    const scalar_t given_qy)
  {
    field_b[0] = given_depth;
    field_b[4] = field[4];
    field_b[1] = field_b[4] + field_b[0];
    field_b[2] = given_qx;
    field_b[3] = given_qy;
    field_b[5] = field[5];
  }

  template <typename scalar_t>
  __device__ __forceinline__ void
  boundary_Q_given(scalar_t *field, scalar_t *field_b, const scalar_t *normal,
                   const scalar_t given_qx, const scalar_t given_qy)
  {

    field_b[0] = field[0];
    field_b[4] = field[4];
    field_b[1] = field_b[4] + field_b[0];
    field_b[2] = given_qx;
    field_b[3] = given_qy;
    field_b[5] = field[5];
  }

  template <typename scalar_t>
  __device__ __forceinline__ void
  boundary_conditions(const int boundaryType, scalar_t *field, scalar_t *field_b,
                      const scalar_t *normal, const scalar_t given_depth,
                      const scalar_t given_qx, const scalar_t given_qy)
  {
    switch (boundaryType)
    {
    case 3:
    {
      boundary_WallNonSlip(field, field_b);
      break;
    }
    case 4:
    {
      boundary_WallSlip(field, field_b, normal);
      break;
    }
    case 5:
    {
      boundary_Open(field, field_b);
      break;
    }
    case 6:
    {
      boundary_HQ_given(field, field_b, normal, given_depth, given_qx, given_qy);
      break;
    }
    case 7:
    {
      scalar_t a = 0.15;
      scalar_t b = 0.02;
      boundary_Critical_Open(field, field_b, a, b);
      break;
    }
    case 8:
    {
      boundary_Q_given(field, field_b, normal, given_qx, given_qy);
      break;
    }
    }
  }

  template <typename scalar_t>
  __global__ void fluxCalculation_kernel(
      const int M, const int N, const int depth_w, const int depth_h,
      const int discharge_w, const int discharge_h, int32_t *__restrict__ wetMask,
      scalar_t *__restrict__ h, scalar_t *__restrict__ C,
      scalar_t *__restrict__ wl, scalar_t *__restrict__ z,
      scalar_t *__restrict__ qx, scalar_t *__restrict__ qy,
      const int32_t *__restrict__ index, const scalar_t *__restrict__ normal,
      scalar_t *__restrict__ given_depth, scalar_t *__restrict__ given_discharge,
      scalar_t *__restrict__ dx, scalar_t *__restrict__ t,
      scalar_t *__restrict__ dt, scalar_t *__restrict__ h_flux,
      scalar_t *__restrict__ hc_flux, scalar_t *__restrict__ qx_flux,
      scalar_t *__restrict__ qy_flux)
  {

    scalar_t h_small = 1.0e-6;
    scalar_t g = 9.81;
    int timelevel_depth = 0;
    int timelevel_discharge = 0;

    // printf("The width of discharge is %d", depth_w);

    while ((t[0] > given_depth[timelevel_depth * depth_w]) &&
           (timelevel_depth < depth_h - 1))
    {
      timelevel_depth++;
    }
    while ((t[0] > given_discharge[timelevel_discharge * discharge_w]) &&
           (timelevel_discharge < discharge_h - 1))
    {
      timelevel_discharge++;
    }

    // if (t[0] < 0.01) {
    //   printf("The dimension of discharge is %d and %d", discharge_w, discharge_h);
    //   exit();
    // }

    // printf("h: %f", timelevel_depth);
    // printf("q: %f", timelevel_discharge);

    scalar_t field_c[6];
    scalar_t field_c_h_z[2];
    scalar_t field_r_h_z[2];
    scalar_t field_r[6];
    scalar_t _normal[2];
    scalar_t _tangential[2];
    scalar_t field_l_z, field_rr_z;
    scalar_t r_b = 0.0;

    int32_t c_type, c_id, r_id, l_id, rr_id;
    int32_t boundaryType, boundaryID;
    int n;
    scalar_t h_boundary, qx_boundary, qy_boundary;

    // get the index of cell

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w < M)
    {
      int32_t i = wetMask[w];
      c_type = index[i];
      c_id = i;

      for (int j = 0; j < 4; j++)
      {
        // _normal[2] = {normal[j * 2], normal[j * 2 + 1]};
        // _tangential[2] = {-_normal[1], _normal[0]};

        field_c[0] = h[c_id];
        field_c[1] = wl[c_id];
        field_c[2] = qx[c_id];
        field_c[3] = qy[c_id];
        field_c[4] = z[c_id];
        field_c[5] = C[c_id];
        field_c_h_z[0] = h[c_id];
        field_c_h_z[1] = z[c_id];

        _normal[0] = normal[j * 2];
        _normal[1] = normal[j * 2 + 1];
        _tangential[0] = -_normal[1];
        _tangential[1] = _normal[0];

        r_id = j + 1;
        // l_id = (r_id + 2) % 4;
        l_id = (r_id + 2) > 4 ? r_id - 2 : r_id + 2;

        l_id = index[l_id * N + c_id];
        r_id = index[r_id * N + c_id];
        if (l_id == -1)
        {
          field_l_z = z[c_id];
        }
        else
        {
          field_l_z = z[l_id];
        }

        if (r_id == -1)
        {
          field_rr_z = z[c_id];
          boundaryType = c_type;
          boundaryID = c_type;
          n = 0;
          while (boundaryType >= 10)
          {
            boundaryType = (boundaryType - (boundaryType % 10)) / 10;
            n++;
          }
          boundaryID = boundaryID - boundaryType * powf(10, n);
          // boundaryID = boundaryID - boundaryType * exp10f(n);

          h_boundary = given_depth[timelevel_depth * depth_w + boundaryID + 1];
          qx_boundary = given_discharge[timelevel_discharge * discharge_w + 1 +
                                        2 * boundaryID];
          qy_boundary = given_discharge[timelevel_discharge * discharge_w + 2 +
                                        2 * boundaryID];
          boundary_conditions(boundaryType, field_c, field_r, _normal, h_boundary,
                              qx_boundary, qy_boundary);

          field_r_h_z[0] = field_r[0];
          field_r_h_z[1] = field_r[4];

          // minmod reconstruction of bed elevation
          // not for boundary
        }
        else
        {
          // try to get the rr_id first

          rr_id = j + 1;
          rr_id = index[rr_id * N + r_id];
          if (rr_id == -1)
          {
            field_rr_z = z[r_id];
          }
          else
          {
            field_rr_z = z[rr_id];
          }

          field_r[0] = h[r_id];
          field_r[1] = wl[r_id];
          field_r[2] = qx[r_id];
          field_r[3] = qy[r_id];
          field_r[4] = z[r_id];
          field_r[5] = C[r_id];

          field_r_h_z[0] = field_r[0];
          field_r_h_z[1] = field_r[4];

          //   ==========================================
          // minmod reconstrucrion
          //   ==========================================

          // minmod_tri(field_c[4] - field_l_z, field_r[4] - field_l_z,
          //            field_r[4] - field_c[4], r_b);
          minmod(field_c[4] - field_l_z, field_r[4] - field_c[4], r_b);
          // if (abs(r_b) > 2.0 * field_c[0]) {
          //   r_b = r_b / abs(r_b) * 2.0 * field_c[0];
          // }
          if (field_r[0] > h_small)
          {
            field_c[4] += 0.5 * r_b;
          }
          // field_c[4] += 0.5 * r_b;

          minmod(field_r[4] - field_rr_z, field_c_h_z[1] - field_r[4], r_b);

          if (field_c[0] > h_small)
          {
            field_r[4] += 0.5 * r_b;
          }
          // field_r[4] += 0.5 * r_b;
        }

        if (field_c[0] < h_small && field_r[0] < h_small)
        {
          continue;
        }
        else
        {
          hllcFluxSolver(field_c_h_z, field_r_h_z, field_c, field_r, _normal,
                         _tangential, h_flux[i], hc_flux[i], qx_flux[i],
                         qy_flux[i], h_small, g, dx[0], dt[0]);

          qx_flux[i] -= ((0.5 * g * (field_c[0] + field_c_h_z[0]) *
                          (field_c[4] - field_c_h_z[1]) * _normal[0]) /
                         dx[0]) *
                        dt[0];
          qy_flux[i] -= ((0.5 * g * (field_c[0] + field_c_h_z[0]) *
                          (field_c[4] - field_c_h_z[1]) * _normal[1]) /
                         dx[0]) *
                        dt[0];
        }
      }
    }
    // __syncthreads();
  }
  // ===============================================
  // we will update the field in the euler update part
  // ===============================================
  template <typename scalar_t>
  __global__ void
  Update_kernel(const int N, scalar_t *__restrict__ h_flux,
                scalar_t *__restrict__ qx_flux, scalar_t *__restrict__ qy_flux,
                scalar_t *__restrict__ h, scalar_t *__restrict__ wl,
                scalar_t *__restrict__ z, scalar_t *__restrict__ qx,
                scalar_t *__restrict__ qy, scalar_t *__restrict__ dt)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
      h[i] += (h_flux[i]);
      qx[i] += (qx_flux[i]);
      qy[i] += (qy_flux[i]);
      wl[i] = h[i] + z[i];
    }
  }
}

void fluxCalculation_cuda(at::Tensor wetMask, at::Tensor h_flux,
                          at::Tensor hc_flux, at::Tensor qx_flux,
                          at::Tensor qy_flux, at::Tensor h, at::Tensor wl,
                          at::Tensor z, at::Tensor C, at::Tensor qx,
                          at::Tensor qy, at::Tensor index, at::Tensor normal,
                          at::Tensor given_depth, at::Tensor given_discharge,
                          at::Tensor dx, at::Tensor t, at::Tensor dt)
{
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int M = wetMask.size(0);
  if (M == 0)
  {
    return;
  }
  const int N = h.size(0);
  const int depth_h = given_depth.size(0);
  const int depth_w = given_depth.size(1);
  const int discharge_h = given_discharge.size(0);
  const int discharge_w = given_discharge.size(1);

  int thread_0 = 512;
  int block_0 = (M + 512 - 1) / 512;
  // int thread_1 = 512;
  // int block_1 = (N + 512 - 1) / 512;

  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "fluxCalculation_cuda", ([&]
                                         { fluxCalculation_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
                                               M, N, depth_w, depth_h, discharge_w, discharge_h,
                                               wetMask.data<int32_t>(), h.data<scalar_t>(), C.data<scalar_t>(),
                                               wl.data<scalar_t>(), z.data<scalar_t>(), qx.data<scalar_t>(),
                                               qy.data<scalar_t>(), index.data<int32_t>(), normal.data<scalar_t>(),
                                               given_depth.data<scalar_t>(), given_discharge.data<scalar_t>(),
                                               dx.data<scalar_t>(), t.data<scalar_t>(), dt.data<scalar_t>(),
                                               h_flux.data<scalar_t>(), hc_flux.data<scalar_t>(),
                                               qx_flux.data<scalar_t>(), qy_flux.data<scalar_t>()); }));

  // AT_DISPATCH_FLOATING_TYPES(
  //     h.type(), "fluxCalculation_cuda_1", ([&] {
  //       Update_kernel<scalar_t><<<block_1, thread_1, 0, stream>>>(
  //           N, h_flux.data<scalar_t>(), qx_flux.data<scalar_t>(),
  //           qy_flux.data<scalar_t>(), h.data<scalar_t>(),
  //           wl.data<scalar_t>(),
  //           z.data<scalar_t>(), qx.data<scalar_t>(), qy.data<scalar_t>(),
  //           dt.data<scalar_t>());
  //     }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
