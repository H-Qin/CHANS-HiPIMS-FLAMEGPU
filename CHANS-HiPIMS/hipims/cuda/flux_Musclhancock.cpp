// High-Performance Integrated hydrodynamic Modelling System ***hybrid***
// @author: Jiaheng Zhao (Hemlab)
// @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
// @contact: j.zhao@lboro.ac.uk
// @software: hipims_hybrid
// @time: 07.01.2021
// This is a beta version inhouse code of Hemlab used for high-performance flooding simulation.
// Feel free to use and extend if you are a ***member of hemlab***.
#include <torch/extension.h>
// CUDA forward declarations
void fluxCalculation_cuda(at::Tensor wetMask, at::Tensor h_flux,
                          at::Tensor qx_flux, at::Tensor qy_flux, at::Tensor h,
                          at::Tensor wl, at::Tensor z, at::Tensor qx,
                          at::Tensor qy, at::Tensor index, at::Tensor normal,
                          at::Tensor given_depth, at::Tensor given_discharge,
                          at::Tensor dx, at::Tensor t, at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void fluxCalculation_MusclHancock(at::Tensor wetMask, at::Tensor h_flux,
                              at::Tensor qx_flux, at::Tensor qy_flux,
                              at::Tensor h, at::Tensor wl, at::Tensor z,
                              at::Tensor qx, at::Tensor qy, at::Tensor index,
                              at::Tensor normal, at::Tensor given_depth,
                              at::Tensor given_q, at::Tensor dx, at::Tensor t,
                              at::Tensor dt) {
  CHECK_INPUT(h);
  CHECK_INPUT(wetMask);
  CHECK_INPUT(wl);
  CHECK_INPUT(z);
  CHECK_INPUT(qx);
  CHECK_INPUT(qy);
  CHECK_INPUT(index);
  CHECK_INPUT(dx);
  CHECK_INPUT(normal);
  CHECK_INPUT(given_q);
  CHECK_INPUT(given_depth);
  CHECK_INPUT(dt);
  CHECK_INPUT(t);

  fluxCalculation_cuda(wetMask, h_flux, qx_flux, qy_flux, h, wl, z, qx, qy,
                       index, normal, given_depth, given_q, dx, t, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("addFlux", &fluxCalculation_MusclHancock, "Flux Reconstructed Bed Calculation (CUDA)");
}
