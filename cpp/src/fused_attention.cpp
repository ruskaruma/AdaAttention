#include <torch/extension.h>

at::Tensor fused_attention_forward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float dropout_prob,
    bool deterministic);

at::Tensor fused_attention_forward(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_prob,
    bool deterministic) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  return fused_attention_forward_cuda(
      q, k, v, static_cast<float>(dropout_prob), deterministic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_attention_forward", &fused_attention_forward);
}

