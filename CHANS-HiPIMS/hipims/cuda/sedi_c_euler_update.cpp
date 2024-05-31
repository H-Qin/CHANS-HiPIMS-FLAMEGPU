#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void sedi_c_euler_update_cuda(at::Tensor wetMask, at::Tensor h, at::Tensor C,
                              at::Tensor h_update, at::Tensor hC_update,
                              at::Tensor z_update, at::Tensor sedi_para,
                              at::Tensor landuse);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void sedi_c_euler_update(at::Tensor wetMask, at::Tensor h, at::Tensor C,
                         at::Tensor h_update, at::Tensor hC_update,
                         at::Tensor z_update, at::Tensor sedi_para,
                         at::Tensor landuse) {
  CHECK_INPUT(wetMask);
  CHECK_INPUT(h);
  CHECK_INPUT(h_update);
  CHECK_INPUT(hC_update);
  CHECK_INPUT(z_update);
  CHECK_INPUT(C);
  CHECK_INPUT(sedi_para);

  sedi_c_euler_update_cuda(wetMask, h, C, h_update, hC_update, z_update,
                           sedi_para, landuse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &sedi_c_euler_update,
        "sediment concentration update, CUDA version");
}