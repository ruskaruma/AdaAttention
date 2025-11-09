from pathlib import Path
from typing import Optional

import torch
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    str(ROOT / "cpp" / "src" / "fused_attention.cpp"),
    str(ROOT / "cpp" / "src" / "fused_attention_kernel.cu"),
]

_EXTENSION: Optional[object] = None

def load_extension(rebuild: bool = False):
    global _EXTENSION
    if _EXTENSION is not None and not rebuild:
        return _EXTENSION
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-arch=sm_89"]
    build_dir = ROOT / "build" / "extension"
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp_extension.USE_NINJA = False
    _EXTENSION = cpp_extension.load(
        name="cuda_attn_ext",
        sources=SOURCES,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        build_directory=str(build_dir),
        with_cuda=True,
    )
    return _EXTENSION

def fused_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_prob: float = 0.0,
    deterministic: bool = False,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fused attention")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise RuntimeError("Inputs must reside on CUDA")
    extension = load_extension()
    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    return extension.fused_attention_forward(
        qc, kc, vc, float(dropout_prob), bool(deterministic)
    )

