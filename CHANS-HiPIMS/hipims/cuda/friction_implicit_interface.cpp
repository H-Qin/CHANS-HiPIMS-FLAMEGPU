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
void frictionCalculation_cuda(at::Tensor wetMask, at::Tensor qx_update,
                              at::Tensor qy_update, at::Tensor landuse,
                              at::Tensor h, at::Tensor qx, at::Tensor qy,
                              at::Tensor manning, at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void frictionCalculation_implicit(at::Tensor wetMask, at::Tensor qx_update,
                         at::Tensor qy_update, at::Tensor landuse, at::Tensor h,
                         at::Tensor qx, at::Tensor qy, at::Tensor manning,
                         at::Tensor dt) {
  CHECK_INPUT(wetMask);
  CHECK_INPUT(landuse);
  CHECK_INPUT(h);
  CHECK_INPUT(qx);
  CHECK_INPUT(qy);
  CHECK_INPUT(qx_update);
  CHECK_INPUT(qy_update);
  CHECK_INPUT(dt);
  CHECK_INPUT(manning);

  frictionCalculation_cuda(wetMask, qx_update, qy_update, landuse, h, qx, qy,
                           manning, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("addFriction", &frictionCalculation_implicit, "Friction Updating, CUDA version");
}