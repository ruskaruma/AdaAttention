import torch

from ._ext import fused_attention_forward, load_extension
from .fallback import reference_attention


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_prob: float = 0.0,
    deterministic: bool = False,
) -> torch.Tensor:
    use_fused = (
        torch.cuda.is_available()
        and q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and not deterministic
    )
    if not use_fused:
        return reference_attention(q, k, v, dropout_prob=dropout_prob, deterministic=deterministic)
    load_extension()
    return fused_attention_forward(
        q, k, v, dropout_prob=dropout_prob, deterministic=deterministic
    )

