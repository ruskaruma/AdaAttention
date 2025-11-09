## AdaAttention
Fused attention kernels for PyTorch targeting Ada Lovelace GPUs (SM 8.9) with a deterministic fallback.

- Build with `python setup.py build_ext --inplace` or `python setup.py install`.
- Run tests via `uv run pytest -q`.
- Trigger extension rebuild using `uv run python -m tools.build_extension`.
- Benchmark forward throughput using `uv run python examples/bench.py`.
- Detailed architecture and optimization roadmap live in `ARCHITECTURE.md`.

