import pytest
import torch

from cuda_attn.fallback import reference_attention
from cuda_attn.ops import attention


def make_inputs(device: torch.device, dtype: torch.dtype):
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16, device=device, dtype=dtype)
    k = torch.randn(2, 4, 8, 16, device=device, dtype=dtype)
    v = torch.randn(2, 4, 8, 16, device=device, dtype=dtype)
    return q, k, v

@pytest.mark.parametrize(
    "device,dtype",
    [
        (torch.device("cpu"), torch.float32),
        *(
            [
                (torch.device("cuda"), torch.float32),
                (torch.device("cuda"), torch.float16),
                (torch.device("cuda"), torch.bfloat16),
            ]
            if torch.cuda.is_available()
            else []
        ),
    ],
)
def test_attention_matches_reference(device, dtype):
    q, k, v = make_inputs(device, dtype)
    out_reference = reference_attention(q, k, v, dropout_prob=0.0, deterministic=True)
    out_attn = attention(q, k, v, dropout_prob=0.0, deterministic=True)
    assert torch.allclose(out_attn, out_reference, atol=1e-4, rtol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_path_matches_reference():
    q, k, v = make_inputs(torch.device("cuda"), torch.float16)
    out_reference = reference_attention(q, k, v, dropout_prob=0.0, deterministic=True)
    out_attn = attention(q, k, v, dropout_prob=0.0, deterministic=False)
    assert torch.allclose(out_attn, out_reference, atol=1e-3, rtol=1e-3)

