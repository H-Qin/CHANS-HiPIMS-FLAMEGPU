// High-Performance Integrated hydrodynamic Modelling System ***hybrid***
// @author: Jiaheng Zhao (Hemlab)
// @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
// @contact: j.zhao@lboro.ac.uk
// @software: hipims_hybrid
// @time: 07.01.2021
// This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
// Feel free to use and extend if you are a ***member of hemlab***.
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void frictionCalculation_cuda(at::Tensor wetMask, at::Tensor h_update,
                              at::Tensor qx_update, at::Tensor qy_update,
                              at::Tensor z_update, at::Tensor landuse,
                              at::Tensor h, at::Tensor wl, at::Tensor qx,
                              at::Tensor qy, at::Tensor z, at::Tensor manning,
                              at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void friction_implicit_andUpdate_jh(at::Tensor wetMask, at::Tensor h_update,
                                    at::Tensor qx_update, at::Tensor qy_update,
                                    at::Tensor z_update, at::Tensor landuse,
                                    at::Tensor h, at::Tensor wl, at::Tensor qx,
                                    at::Tensor qy, at::Tensor z,
                                    at::Tensor manning, at::Tensor dt) {
  CHECK_INPUT(wetMask);
  CHECK_INPUT(landuse);
  CHECK_INPUT(h);
  CHECK_INPUT(qx);
  CHECK_INPUT(z);
  CHECK_INPUT(wl);
  CHECK_INPUT(qy);
  CHECK_INPUT(h_update);
  CHECK_INPUT(z_update);
  CHECK_INPUT(qx_update);
  CHECK_INPUT(qy_update);
  CHECK_INPUT(dt);
  CHECK_INPUT(manning);

  frictionCalculation_cuda(wetMask, h_update, qx_update, qy_update, z_update,
                           landuse, h, wl, qx, qy, z, manning, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("addFriction_eulerUpdate", &friction_implicit_andUpdate_jh,
        "Friction Updating, CUDA version");
}