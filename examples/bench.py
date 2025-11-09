import time
import torch
from cuda_attn.ops import attention

def bench_one(batch: int, heads: int, seq: int, dim: int, runs: int = 10, dtype=torch.float16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    for _ in range(3):
        attention(q, k, v, dropout_prob=0.1, deterministic=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = time.time()
        attention(q, k, v, dropout_prob=0.1, deterministic=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)
    times.sort()
    median = times[len(times) // 2]
    tokens = batch * heads * seq
    print(
        f"B={batch} H={heads} S={seq} D={dim} dtype={dtype} "
        f"median_ms={median * 1000:.2f} tokens_per_s={tokens / median:.1f}"
    )
if __name__ == "__main__":
    configs = [(1, 8, 512, 64), (1, 12, 1024, 64), (1, 16, 2048, 64)]
    for cfg in configs:
        try:
            bench_one(*cfg)
        except RuntimeError as err:
            print(f"config={cfg} failed: {err}")

