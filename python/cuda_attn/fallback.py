from typing import Tuple

import torch


def _cast_for_compute(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    cast_needed = q.dtype in (torch.float16, torch.bfloat16)
    if cast_needed:
        return (
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            True,
        )
    return q, k, v, False


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_prob: float = 0.0,
    deterministic: bool = True,
) -> torch.Tensor:
    q_work, k_work, v_work, casted = _cast_for_compute(q, k, v)
    scores = torch.matmul(q_work, k_work.transpose(-2, -1))
    scores = torch.nn.functional.softmax(scores, dim=-1)
    if dropout_prob > 0.0 and not deterministic:
        scores = torch.nn.functional.dropout(scores, p=dropout_prob, training=True)
    out = torch.matmul(scores, v_work)
    if casted:
        out = out.to(q.dtype)
    return out

