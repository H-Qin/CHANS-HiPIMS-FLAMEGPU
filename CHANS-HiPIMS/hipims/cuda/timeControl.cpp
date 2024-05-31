// High-Performance Integrated hydrodynamic Modelling System ***hybrid***
// @author: Jiaheng Zhao (Hemlab)
// @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
// @contact: j.zhao@lboro.ac.uk
// @software: hipims_hybrid
// @time: 07.01.2021
// This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
// Feel free to use and extend if you are a ***member of hemlab***.
#include <torch/extension.h>
// #include <vector>

// CUDA forward declarations
void timeControl_cuda(at::Tensor wetMask, at::Tensor accelerator,
                      at::Tensor h_max, at::Tensor h, at::Tensor qx,
                      at::Tensor qy, at::Tensor dx, at::Tensor CFL,
                      at::Tensor t, at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void timeControl(at::Tensor wetMask, at::Tensor accelerator, at::Tensor h_max,
                 at::Tensor h, at::Tensor qx, at::Tensor qy, at::Tensor dx,
                 at::Tensor CFL, at::Tensor t, at::Tensor dt) {
  CHECK_INPUT(wetMask);
  CHECK_INPUT(h);
  CHECK_INPUT(h_max);
  CHECK_INPUT(qx);
  CHECK_INPUT(qy);
  CHECK_INPUT(dx);
  CHECK_INPUT(CFL);
  CHECK_INPUT(dt);
  CHECK_INPUT(t);

  timeControl_cuda(wetMask, accelerator, h_max, h, qx, qy, dx, CFL, t, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("updateTimestep", &timeControl, "Time Updating, CUDA version");
}