from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
README = ROOT / "README.md"


def read_readme() -> str:
    if README.exists():
        return README.read_text(encoding="utf-8")
    return ""


setup(
    name="cuda-attn",
    version="0.1.0",
    description="PyTorch C++/CUDA fused attention kernels (production-grade scaffold)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="ruskaruma",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=[
        CUDAExtension(
            name="cuda_attn_ext",
            sources=[
                "cpp/src/fused_attention.cpp",
                "cpp/src/fused_attention_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-arch=sm_89"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

