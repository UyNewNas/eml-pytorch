"""
torch.compile 兼容性测试与性能基准脚本

测试 EML 算子在四种模式下的行为正确性与性能对比：
  1. Eager mode —— 直接调用 eml(x, y)
  2. torch.compile 模式 —— 对 eml 施加 torch.compile 自动优化
  3. Triton 融合内核 —— 调用 eml_triton(x, y)
  4. torch.compile + Triton —— 对 eml_triton 施加 torch.compile

验收标准：
  - 每种模式输出与 eager 模式的相对误差 < 1e-5
  - 输出加速比表格
  - Triton 不可用时自动降级，仅测试 eager 和 compile

注意：
  - torch.compile 默认使用 inductor 后端，该后端依赖 Triton 生成 GPU 内核。
    当 Triton 不可用时，脚本会自动回退到 aot_eager 后端（仅做图捕获与
    自动微分分解，不做代码生成优化），并在输出中标注实际使用的后端。
"""

import torch
import time
from eml_pytorch import eml
from eml_pytorch.triton_kernel import eml_triton, TRITON_AVAILABLE


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个张量之间的相对误差 (基于 Frobenius 范数)"""
    return (torch.norm(a - b) / torch.norm(b)).item()


def benchmark_mode(func, x, y, warmup=10, iters=100):
    """
    对给定函数进行预热和计时。

    Args:
        func: 接受 (x, y) 并返回张量的可调用对象
        x, y: 输入张量
        warmup: 预热迭代次数
        iters: 计时迭代次数

    Returns:
        (平均耗时秒数, 输出张量)
    """
    # 预热阶段：让 GPU 达到稳定频率，避免首次调用开销
    for _ in range(warmup):
        out = func(x, y)
    torch.cuda.synchronize()

    # 正式计时
    start = time.perf_counter()
    for _ in range(iters):
        out = func(x, y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters, out


def compile_with_fallback(fn, x, y):
    """
    尝试使用 inductor 后端编译并执行函数。
    若因缺少 Triton 而失败，自动回退到 aot_eager 后端。

    torch.compile 采用惰性编译策略：torch.compile(fn) 本身不会触发编译，
    仅在首次调用编译后的函数时才真正编译。因此需要在 try 中执行首次调用。

    Args:
        fn: 待编译的可调用对象
        x, y: 用于触发首次编译的示例输入

    Returns:
        (编译后的可调用对象, 实际使用的后端名称, 首次调用输出)
        若所有后端均失败，返回 (None, None, None)
    """
    backends = ["inductor", "aot_eager"]
    for backend in backends:
        try:
            compiled = torch.compile(fn, backend=backend)
            out = compiled(x, y)
            torch.cuda.synchronize()
            return compiled, backend, out
        except Exception as e:
            err_msg = str(e).lower()
            if "triton" in err_msg:
                print(f"       [Fallback] {backend} backend failed (Triton missing), "
                      f"trying next backend ...")
                torch._dynamo.reset()
                continue
            print(f"       [Warning] {backend} backend failed: {e}")
            torch._dynamo.reset()
            continue
    return None, None, None


def main():
    # ===== 环境检查 =====
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a GPU. Exiting.")
        return

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA device     : {torch.cuda.get_device_name(0)}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    print()

    # ===== 测试参数 =====
    size = (4096, 4096)
    dtype = torch.float32
    warmup = 10
    iters = 100

    # 生成输入张量：x 为标准正态分布，y 为 [0.1, 1.1) 均匀分布（避免 log(0)）
    x = torch.randn(size, device="cuda", dtype=dtype)
    y = torch.rand(size, device="cuda", dtype=dtype) + 0.1

    # ===== 模式 1: Eager mode =====
    # 直接调用 eml(x, y)，即 torch.ops.eml_pytorch.eml 的 Python 封装
    print("[1/4] Benchmarking Eager mode ...")
    time_eager, out_eager = benchmark_mode(eml, x, y, warmup, iters)
    print(f"       Eager avg time: {time_eager * 1000:.4f} ms")

    # ===== 模式 2: torch.compile 模式 =====
    # 将 eml 包装在函数中再编译，确保 torch.compile 能正确追踪算子调用
    # 默认使用 inductor 后端；若 Triton 不可用则自动回退到 aot_eager
    print("[2/4] Benchmarking torch.compile mode ...")

    def eml_compiled_fn(x, y):
        return eml(x, y)

    compiled_fn, compile_backend, out_first = compile_with_fallback(eml_compiled_fn, x, y)

    if compiled_fn is None:
        print("       [Skip] torch.compile is not available (all backends failed).")
        results = [("Eager", time_eager, out_eager, None)]
    else:
        # 使用首次编译调用产生的输出，避免额外预热
        # 后续 benchmark_mode 的 warmup 会进一步稳定计时
        time_compile, out_compile = benchmark_mode(compiled_fn, x, y, warmup, iters)
        compile_label = f"torch.compile({compile_backend})"
        print(f"       {compile_label} avg time: {time_compile * 1000:.4f} ms")

        # ===== 数值正确性验证: Eager vs torch.compile =====
        err_compile = relative_error(out_eager, out_compile)
        status_compile = "PASS" if err_compile < 1e-5 else "FAIL"
        print(f"       Correctness (Eager vs compile): rel_err={err_compile:.2e} [{status_compile}]")

        results = [
            ("Eager", time_eager, out_eager, None),
            (compile_label, time_compile, out_compile, err_compile),
        ]

    # ===== 模式 3 & 4: Triton 相关测试（仅在 Triton 可用时运行）=====
    if TRITON_AVAILABLE:
        # 模式 3: Triton 融合内核
        print("[3/4] Benchmarking Triton fused kernel ...")
        time_triton, out_triton = benchmark_mode(eml_triton, x, y, warmup, iters)
        print(f"       Triton avg time: {time_triton * 1000:.4f} ms")

        err_triton = relative_error(out_eager, out_triton)
        status_triton = "PASS" if err_triton < 1e-5 else "FAIL"
        print(f"       Correctness (Eager vs Triton): rel_err={err_triton:.2e} [{status_triton}]")

        results.append(("Triton", time_triton, out_triton, err_triton))

        # 模式 4: torch.compile + Triton
        # 对 eml_triton 施加 torch.compile，观察是否产生额外优化
        print("[4/4] Benchmarking torch.compile(Triton) ...")

        def eml_triton_compiled_fn(x, y):
            return eml_triton(x, y)

        compiled_triton_fn, ct_backend, _ = compile_with_fallback(eml_triton_compiled_fn, x, y)

        if compiled_triton_fn is None:
            print("       [Skip] torch.compile on Triton kernel failed.")
        else:
            time_compile_triton, out_compile_triton = benchmark_mode(
                compiled_triton_fn, x, y, warmup, iters
            )
            ct_label = f"torch.compile({ct_backend})+Triton"
            print(f"       {ct_label} avg time: {time_compile_triton * 1000:.4f} ms")

            err_compile_triton = relative_error(out_eager, out_compile_triton)
            status_ct = "PASS" if err_compile_triton < 1e-5 else "FAIL"
            print(
                f"       Correctness (Eager vs compile+Triton): "
                f"rel_err={err_compile_triton:.2e} [{status_ct}]"
            )

            results.append((ct_label, time_compile_triton, out_compile_triton, err_compile_triton))
    else:
        print("[3/4] Skipped — Triton is not available in this environment.")
        print("[4/4] Skipped — Triton is not available in this environment.")
        print()
        print("Hint: Install Triton with `pip install triton` to enable Triton benchmarks.")

    # ===== 输出汇总表格 =====
    print()
    print("=" * 78)
    print(f"  Benchmark Summary  |  Size: {size[0]}x{size[1]}  |  dtype: {dtype}")
    print("=" * 78)
    header = f"{'Mode':<30s} | {'Time (ms)':>10s} | {'Speedup vs Eager':>17s} | {'Correctness':>11s}"
    print(header)
    print("-" * 78)

    for name, t, _out, err in results:
        time_ms = t * 1000
        speedup = time_eager / t
        if err is not None:
            correctness = "PASS" if err < 1e-5 else f"FAIL({err:.2e})"
        else:
            correctness = "baseline"
        print(f"{name:<30s} | {time_ms:>10.4f} | {speedup:>16.2f}x | {correctness:>11s}")

    print("=" * 78)


if __name__ == "__main__":
    main()
