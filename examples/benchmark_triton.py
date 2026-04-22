"""
Benchmark script comparing native PyTorch custom op vs Triton fused kernel.
"""

import torch
import time
from eml_pytorch import eml
from eml_pytorch.triton_kernel import eml_triton, TRITON_AVAILABLE

def benchmark():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    if not TRITON_AVAILABLE:
        print("Triton is not available. Exiting.")
        return

    # 测试不同规模的张量
    sizes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    for size in sizes:
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.rand(size, device='cuda', dtype=torch.float32) + 0.1

        # 预热
        for _ in range(10):
            _ = eml(x, y)
            _ = eml_triton(x, y)
        torch.cuda.synchronize()

        # 测试原生自定义算子
        start = time.perf_counter()
        for _ in range(100):
            _ = eml(x, y)
        torch.cuda.synchronize()
        time_native = time.perf_counter() - start

        # 测试 Triton 融合内核
        start = time.perf_counter()
        for _ in range(100):
            _ = eml_triton(x, y)
        torch.cuda.synchronize()
        time_triton = time.perf_counter() - start

        speedup = time_native / time_triton
        print(f"Size {size[0]}x{size[1]}:")
        print(f"  Native op : {time_native:.4f}s")
        print(f"  Triton op : {time_triton:.4f}s")
        print(f"  Speedup   : {speedup:.2f}x\n")

if __name__ == "__main__":
    benchmark()