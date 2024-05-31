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
void fluxMask_cuda(at::Tensor fluxMask, at::Tensor h, at::Tensor index, at::Tensor t);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void fluxMask(at::Tensor fluxMask, at::Tensor h, at::Tensor index, at::Tensor t) {
  CHECK_INPUT(fluxMask);
  CHECK_INPUT(h);
  CHECK_INPUT(index);
  CHECK_INPUT(t);

  fluxMask_cuda(fluxMask, h, index, t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &fluxMask, "fluxMask Updating, CUDA version");
}