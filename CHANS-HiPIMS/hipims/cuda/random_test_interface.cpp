#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void Random_cuda(at::Tensor h);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void Random(at::Tensor h) {
  CHECK_INPUT(h);
 
  Random_cuda(h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &Random,
        "Particles Initializing, CUDA version");
}