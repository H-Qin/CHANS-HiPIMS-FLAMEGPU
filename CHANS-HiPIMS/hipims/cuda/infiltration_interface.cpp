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
void infiltrationCalculation_cuda(at::Tensor wetMask, at::Tensor h_update,
                                  at::Tensor landuse, at::Tensor h,
                                  at::Tensor hydraulic_conductivity,
                                  at::Tensor capillary_head,
                                  at::Tensor water_content_diff,
                                  at::Tensor cumulative_depth, at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void infiltrationCalculation(at::Tensor wetMask, at::Tensor h_update,
                             at::Tensor landuse, at::Tensor h,
                             at::Tensor hydraulic_conductivity,
                             at::Tensor capillary_head,
                             at::Tensor water_content_diff,
                             at::Tensor cumulative_depth, at::Tensor dt) {
  CHECK_INPUT(wetMask);
  CHECK_INPUT(landuse);
  CHECK_INPUT(h);
  CHECK_INPUT(h_update);
  CHECK_INPUT(dt);
  CHECK_INPUT(hydraulic_conductivity);
  CHECK_INPUT(capillary_head);
  CHECK_INPUT(water_content_diff);
  CHECK_INPUT(cumulative_depth);

  infiltrationCalculation_cuda(wetMask, h_update, landuse, h,
                               hydraulic_conductivity, capillary_head,
                               water_content_diff, cumulative_depth, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("addinfiltration", &infiltrationCalculation,
        "Infiltration Updating, CUDA version");
}