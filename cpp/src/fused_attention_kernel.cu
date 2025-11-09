#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

bool needs_float32(at::ScalarType dtype) {
  return dtype == at::kHalf || dtype == at::kBFloat16;
}

}  // namespace

at::Tensor fused_attention_forward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float dropout_prob,
    bool deterministic) {
  TORCH_CHECK(q.dim() == 4, "q must have shape [B, H, S, D]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
  TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
  TORCH_CHECK(
      q.dtype() == k.dtype() && q.dtype() == v.dtype(),
      "q/k/v dtypes must match");
  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();
  auto v_contig = v.contiguous();
  const bool cast_inputs = needs_float32(q_contig.scalar_type());
  auto compute_dtype = cast_inputs ? at::kFloat : q_contig.scalar_type();
  auto q_working = cast_inputs ? q_contig.to(compute_dtype) : q_contig;
  auto k_working = cast_inputs ? k_contig.to(compute_dtype) : k_contig;
  auto v_working = cast_inputs ? v_contig.to(compute_dtype) : v_contig;

  auto scores = at::matmul(q_working, k_working.transpose(-2, -1));
  scores = at::softmax(scores, /*dim=*/-1);
  if (dropout_prob > 0.f && !deterministic) {
    scores = at::dropout(scores, dropout_prob, /*train=*/true);
  }
  auto out = at::matmul(scores, v_working);

  if (cast_inputs) {
    out = out.to(q_contig.dtype());
  }
  return out.contiguous();
}

