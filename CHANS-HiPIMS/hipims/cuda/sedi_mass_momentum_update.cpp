#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void sedi_mass_momentum_update_cuda(at::Tensor wetMask, at::Tensor index,
                                    at::Tensor h, at::Tensor C, at::Tensor qx,
                                    at::Tensor qy, at::Tensor z,
                                    at::Tensor z_non_move, at::Tensor h_update,
                                    at::Tensor qx_update, at::Tensor qy_update,
                                    at::Tensor z_update, at::Tensor landuse,
                                    at::Tensor manning, at::Tensor sedi_para,
                                    at::Tensor dt, at::Tensor dx);

// C++ interface
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void sedi_mass_momentum_update(at::Tensor wetMask, at::Tensor index,
                               at::Tensor h, at::Tensor C, at::Tensor qx,
                               at::Tensor qy, at::Tensor z, at::Tensor z_non_move,
                               at::Tensor h_update, at::Tensor qx_update,
                               at::Tensor qy_update, at::Tensor z_update,
                               at::Tensor landuse, at::Tensor manning,
                               at::Tensor sedi_para, at::Tensor dt,
                               at::Tensor dx)
{
  CHECK_INPUT(wetMask);
  CHECK_INPUT(index);
  CHECK_INPUT(landuse);
  CHECK_INPUT(h);
  CHECK_INPUT(qx);
  CHECK_INPUT(z);
  CHECK_INPUT(z_non_move);
  CHECK_INPUT(C);
  CHECK_INPUT(qy);
  CHECK_INPUT(h_update);
  CHECK_INPUT(qx_update);
  CHECK_INPUT(qy_update);
  CHECK_INPUT(z_update);
  CHECK_INPUT(sedi_para);
  CHECK_INPUT(dt);
  CHECK_INPUT(dx);
  CHECK_INPUT(manning);

  sedi_mass_momentum_update_cuda(wetMask, index, h, C, qx, qy, z, z_non_move,
                                 h_update, qx_update, qy_update, z_update,
                                 landuse, manning, sedi_para, dt, dx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("add_source", &sedi_mass_momentum_update,
        "sediment transport source calculation, CUDA version");
}