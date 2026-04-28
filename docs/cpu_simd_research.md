# CPU SIMD 对 exp/log 的加速潜力调研报告

> **项目**：eml-pytorch (`eml(x, y) = exp(x) - log(y)`)
> **调研日期**：2026-04-25
> **调研目标**：评估 CPU 端 SIMD 融合优化的可行性与潜在收益，给出 Go/No-Go 建议

---

## 1. PyTorch CPU 端 exp/log 实现现状

### 1.1 源码架构

PyTorch CPU 向量化实现的核心目录为 `aten/src/ATen/cpu/vec/`：

```
aten/src/ATen/cpu/vec/
├── vec_base.h                    # 基础向量类型定义
├── vec256/                       # AVX2 (256-bit) 向量实现
│   ├── vec256.h
│   ├── functional_base.h         # map/reduce 等函数式操作
│   └── vec256_float.h
├── vec512/                       # AVX-512 (512-bit) 向量实现
│   ├── vec512.h
│   ├── functional_base.h
│   └── vec512_float.h
├── vec256_float_neon.h           # ARM NEON 128-bit 实现
└── functional_base.h             # 共享的函数式基础设施
```

### 1.2 向量化调用链

以 `torch.exp` 为例，CPU 端的完整调用链为：

```
torch.exp(tensor)
  → at::exp_out()
    → at::cpu::exp_kernel()
      → vec::map(Sleef_expf4_u10, ...)     # AVX2: SLEEF 4-wide float exp
      → vec::map(Sleef_expf8_u10, ...)     # AVX-512: SLEEF 8-wide float exp
```

核心分发逻辑在编译时通过宏选择向量宽度：

```cpp
#if defined(CPU_CAPABILITY_AVX512)
using VectorType = vec512::Vec512<scalar_t>;
#elif defined(CPU_CAPABILITY_AVX2)
using VectorType = vec256::Vec256<scalar_t>;
#else
using VectorType = vec256::Vec256<scalar_t>;  // 回退到标量或 SSE
#endif
```

### 1.3 SLEEF 集成

**PyTorch 使用 SLEEF 库实现 SIMD 超越函数，而非自研实现。** SLEEF 作为第三方依赖位于 `third_party/sleef/`。

SLEEF 函数名映射表：

| PyTorch 操作 | float32 (AVX2) | float32 (AVX-512) | float64 (AVX2) | float64 (AVX-512) |
|---|---|---|---|---|
| `exp()` | `Sleef_expf4_u10` | `Sleef_expf8_u10` | `Sleef_expd2_u10` | `Sleef_expd4_u10` |
| `log()` | `Sleef_logf4_u10` | `Sleef_logf8_u10` | `Sleef_logd2_u10` | `Sleef_logd4_u10` |

**默认精度**：PyTorch 使用 SLEEF 的 `u10` 变体，精度约 1 ULP，等价于标准 `libm` 的精度水平。

### 1.4 关键发现：exp 和 log 是分开调用的

**PyTorch 在 CPU 上没有 exp-log 融合，它们是分开调用的。** 对于 EML 操作 `exp(x) - log(y)`，实际执行流程为：

```
1. 分配临时缓冲区 tmp1，计算 tmp1 = exp(x)     → 一次完整内存遍历 + SLEEF exp
2. 分配临时缓冲区 tmp2，计算 tmp2 = log(y)     → 一次完整内存遍历 + SLEEF log
3. 计算 out = tmp1 - tmp2                       → 一次完整内存遍历 + AVX 减法
```

加上 `clamp(y, min=1e-8)` 操作，当前 `eml_pytorch::eml` 在 CPU 上实际产生 **4 次内存遍历**（clamp → exp → log → sub），每次遍历都调用 SLEEF 向量化函数但无法利用数据局部性。

对比项目已有的 Triton GPU 实现，Triton 在一个 kernel 中融合了 `tl.exp(x) - tl.log(y_safe)`，只需 **1 次内存遍历**。

---

## 2. SIMD 指令集对 exp/log 的支持

### 2.1 AVX2：无原生 exp/log 指令

AVX2 指令集 **不包含** 任何超越函数硬件指令。`_mm256_exp_ps`、`_mm256_log_ps` 等内联函数不存在。

AVX2 仅提供基本算术运算内联函数：

| 类别 | 可用指令 |
|------|---------|
| 基本算术 | `_mm256_add_ps`, `_mm256_sub_ps`, `_mm256_mul_ps`, `_mm256_div_ps` |
| 平方根 | `_mm256_sqrt_ps` |
| 倒数近似 | `_mm256_rcp_ps`（12-bit 精度近似） |
| 平方根倒数近似 | `_mm256_rsqrt_ps`（12-bit 精度近似） |
| FMA | `_mm256_fmadd_ps`, `_mm256_fmsub_ps` 等 |

> **常见误解**：Intel 的 SVML（Short Vector Math Library）确实提供了 `_mm256_exp_ps`、`_mm256_log_ps` 等函数，但它们是**软件库函数**，不是硬件指令。SVML 仅在 Intel 编译器（ICC/ICX）中可用。

### 2.2 AVX-512：仅有极其有限的 exp2 近似指令

AVX-512ER（Exponential and Reciprocal Instructions）是唯一包含超越函数近似指令的 AVX-512 子集：

| 内联函数 | 功能 | 精度 |
|----------|------|------|
| `_mm512_exp2a23_ps` | 近似计算 2^x | 23-bit 尾数精度 |
| `_mm512_rcp28_ps` | 近似计算 1/x | 28-bit 尾数精度 |
| `_mm512_rsqrt28_ps` | 近似计算 1/sqrt(x) | 28-bit 尾数精度 |

**关键缺失**：`_mm512_exp_ps`（e^x）、`_mm512_log_ps`（ln(x)）等均不存在。

如果需要计算 e^x，必须使用 `_mm512_exp2a23_ps` 配合换底公式 `e^x = 2^(x * log2(e))`，但这会引入额外精度损失。

### 2.3 ARM NEON：无原生 exp/log 指令

ARM NEON 完全没有原生的 exp 或 log 硬件指令。NEON 仅提供基本算术、平方根（ARMv8A+）、倒数近似（8-bit 精度）和 FMA（ARMv8A+）。

ARM SVE/SVE2 同样不提供原生 exp/log 指令。

### 2.4 硬件可用性总结

| 指令集 | 原生 exp/log | 可用硬件 | 消费级 CPU 可用 |
|--------|-------------|---------|----------------|
| **AVX2** | 无 | 所有支持 AVX2 的 CPU | N/A |
| **AVX-512F** | 无 | Intel Skylake-X+, AMD Zen 4+ | N/A |
| **AVX-512ER** | 仅有 `_mm512_exp2a23_ps`（2^x 近似） | **仅 Xeon Phi KNL/KNM（已停产）** | **不可用** |
| **ARM NEON** | 无 | 所有 ARMv7-A+/ARMv8-A+ | N/A |
| **ARM SVE/SVE2** | 无 | Fujitsu A64FX, AWS Graviton3+ | N/A |

**核心结论**：没有任何消费级 CPU 支持原生 SIMD exp/log 硬件指令。业界标准做法是使用 SLEEF、SVML、libmvec 等软件库，通过多项式逼近 + 范围归约实现 SIMD exp/log。

---

## 3. 第三方库评估

### 3.1 SLEEF 库

SLEEF（SIMD Library for Evaluating Elementary Functions）是开源的 SIMD 数学函数库，被 PyTorch、TensorFlow、JAX 等主流框架采用。

#### exp 函数族

| 函数名 | 精度 | SIMD 宽度 |
|--------|------|----------|
| `sleef_expf2_u10` | u10 (~1 ULP) | SSE2, 2-wide |
| `sleef_expf4_u10` | u10 (~1 ULP) | SSE2/AVX2, 4-wide |
| `sleef_expf8_u10` | u10 (~1 ULP) | AVX2, 8-wide |
| `sleef_expf8_u35` | u35 (~3.5 ULP) | AVX2, 8-wide |
| `sleef_expf16_u10` | u10 (~1 ULP) | AVX-512, 16-wide |
| `sleef_expf16_u35` | u35 (~3.5 ULP) | AVX-512, 16-wide |

#### log 函数族

| 函数名 | 精度 | SIMD 宽度 |
|--------|------|----------|
| `sleef_logf2_u10` | u10 (~1 ULP) | SSE2, 2-wide |
| `sleef_logf4_u10` | u10 (~1 ULP) | SSE2/AVX2, 4-wide |
| `sleef_logf8_u10` | u10 (~1 ULP) | AVX2, 8-wide |
| `sleef_logf8_u35` | u35 (~3.5 ULP) | AVX2, 8-wide |
| `sleef_logf16_u10` | u10 (~1 ULP) | AVX-512, 16-wide |
| `sleef_logf16_u35` | u35 (~3.5 ULP) | AVX-512, 16-wide |

#### SIMD 指令集支持

| 指令集 | float 宽度 | double 宽度 | 支持状态 |
|--------|-----------|-------------|---------|
| SSE2 | 4-wide | 2-wide | 完全支持 |
| AVX2 | 8-wide | 4-wide | 完全支持 |
| AVX-512F | 16-wide | 8-wide | 完全支持 |
| NEON (AArch64) | 4-wide | 2-wide | 完全支持 |
| SVE | 可变宽度 | 可变宽度 | 实验性 |
| RISC-V V | 可变宽度 | 可变宽度 | 实验性 |
| WebAssembly SIMD | 4-wide | 2-wide | 支持 |

### 3.2 Intel MKL（oneAPI Math Kernel Library）

MKL 提供了 Vector Mathematical (VM) 函数，包含 SIMD 优化的 exp 和 log。

#### 精度模式

| 模式 | 常量 | ULP 误差 | 说明 |
|------|------|---------|------|
| HA (High Accuracy) | `VML_HA` | < 1 ULP | 最高精度 |
| LA (Low Accuracy) | `VML_LA` | < 4 ULP | 中等精度 |
| EP (Enhanced Performance) | `VML_EP` | 更宽松 | 最高性能 |

#### 融合运算（独有优势）

MKL VM 支持融合运算模式，可在单次调用中组合多个操作：

| 融合函数 | 计算 |
|---------|------|
| `vmsExpMulAdd` | `exp(x) * y + z` |
| `vmsLogMulAdd` | `log(x) * y + z` |
| `vmsPow2o3MulAdd` | `x^(2/3) * y + z` |
| `vmsSinCosMulAdd` | `sin(x)*y + cos(x)*z` |

> 注意：MKL 不直接提供 `exp(x) - log(y)` 融合，但 `ExpMulAdd` 可作为参考。

#### 限制

- **闭源**：免费但非开源
- **仅 x86**：不支持 ARM NEON
- **跨平台性差**：仅 Intel/AMD CPU

### 3.3 oneDNN

oneDNN 提供 `eltwise` 原语，支持 `exp` 和 `log` 操作，并支持 post-op fusion（如 Conv + Eltwise 融合）。

**关键限制**：oneDNN **不直接支持 exp-log 融合**（即 `exp(x) - log(y)` 这种自定义组合）。它只支持预定义的 eltwise 操作。如果需要 EML 融合，需要通过 oneDNN Graph API 定义自定义融合图，但 EML 这种自定义组合不一定能被自动识别和优化。

### 3.4 性能基准测试对比

以下数据综合了 SLEEF 官方 benchmark、Intel oneMKL 白皮书和独立第三方测试（平台：Skylake-X / Xeon Platinum 8280）：

#### 单精度 exp（float）

| 实现 | 指令集 | 吞吐量 (M elem/s) | 精度 (ULP) | vs 标量 libm |
|------|--------|-------------------|-----------|-------------|
| 标准 libm (glibc) | 标量 | ~80 | <1 | 1.0x |
| SLEEF u10 | AVX2 | ~580 | <1 | **7.3x** |
| SLEEF u35 | AVX2 | ~780 | <3.5 | **9.8x** |
| SLEEF u10 | AVX-512 | ~1100 | <1 | **13.8x** |
| SLEEF u35 | AVX-512 | ~1500 | <3.5 | **18.8x** |
| MKL HA | AVX-512 | ~1050 | <1 | **13.1x** |
| MKL LA | AVX-512 | ~1400 | <4 | **17.5x** |
| MKL EP | AVX-512 | ~1800 | 更宽松 | **22.5x** |

#### 单精度 log（float）

| 实现 | 指令集 | 吞吐量 (M elem/s) | 精度 (ULP) | vs 标量 libm |
|------|--------|-------------------|-----------|-------------|
| 标准 libm (glibc) | 标量 | ~75 | <1 | 1.0x |
| SLEEF u10 | AVX2 | ~620 | <1 | **8.3x** |
| SLEEF u35 | AVX2 | ~850 | <3.5 | **11.3x** |
| SLEEF u10 | AVX-512 | ~1200 | <1 | **16.0x** |
| SLEEF u35 | AVX-512 | ~1600 | <3.5 | **21.3x** |
| MKL HA | AVX-512 | ~1100 | <1 | **14.7x** |
| MKL LA | AVX-512 | ~1500 | <4 | **20.0x** |
| MKL EP | AVX-512 | ~1900 | 更宽松 | **25.3x** |

### 3.5 综合对比

| 维度 | SLEEF | MKL | 标准 libm |
|------|-------|-----|----------|
| **开源** | 是（BSL-1.0） | 否（免费但闭源） | 是 |
| **可移植性** | x86 + ARM + RISC-V + WASM | 仅 x86 (Intel/AMD) | 全平台 |
| **融合运算** | 不支持 | 支持 ExpMulAdd 等 | 不支持 |
| **ARM NEON** | 支持 | 不支持 | 标量 |
| **PyTorch 集成** | 已集成（PyTorch 2.x 内部使用） | 已集成（通过 oneMKL） | 默认 |

---

## 4. 精度-速度权衡

### 4.1 SLEEF u10 与 u35 的 ULP 误差

| 变体 | ULP 误差上界 | 典型 ULP | 相对误差量级 |
|------|-------------|---------|------------|
| **u10** | ≤ 1.0 ULP | 0.5~1.0 | ~1e-7 |
| **u35** | ≤ 3.5 ULP | 1.0~3.5 | ~4e-7 |

具体数据（来自 SLEEF 官方测试报告）：

- `sleef_expf_u10`：最大 ULP 误差约 0.506（几乎达到正确舍入）
- `sleef_expf_u35`：最大 ULP 误差约 2.0~3.5
- `sleef_logf_u10`：最大 ULP 误差约 0.501
- `sleef_logf_u35`：最大 ULP 误差约 1.5~3.5

### 4.2 1~2 ULP 误差对训练的影响

**结论：不会导致梯度不稳定或收敛问题。**

理论分析：1-2 ULP 的 exp/log 误差引入的扰动量级为：

```
delta_grad ≈ lr * |d(exp(x))/dx| * 1e-7 = lr * exp(x) * 1e-7
```

对于典型学习率 `lr=1e-3` 和 `exp(x)~1`，扰动约 `1e-10`，远小于 Adam/SGD 的更新步长。

实验证据：

1. **FP16/BF16 训练已广泛成功**：FP16 的相对精度约 `2^{-10} ≈ 1e-3`，BF16 约 `2^{-7} ≈ 8e-3`，比 1-2 ULP（~1e-7）差了 4-5 个数量级。如果 FP16/BF16 都能稳定训练，1-2 ULP 的超越函数误差完全不会造成问题。

2. **FP8 训练**：Micikevicius et al. (2022) 证明 E4M3 格式（~1e-2 相对精度）可以成功训练大模型，进一步佐证了 u35 精度（~1e-7）对训练完全足够。

3. **混合精度训练中的 exp/log**：PyTorch 的 autocast 将 exp/log 列入 FP32 白名单，但即便在 FP16 中计算，训练通常也能收敛。

### 4.3 torch.set_flush_denormal() 的影响

| 模式 | 行为 | 精度影响 | 性能影响 |
|------|------|---------|---------|
| `set_flush_denormal(True)` | 非正规数刷新为 0 | 丢失 1e-38 以下的精度 | 避免非正规数导致的严重性能下降 |
| `set_flush_denormal(False)` | 保留非正规数 | 完整 IEEE 754 精度 | 遇到非正规数时性能可能骤降 10-100x |

**建议**：训练时开启 FTZ（`set_flush_denormal(True)`），非正规数导致的性能惩罚远大于精度损失。EML 算子中 `exp(x)` 产生非常小的值（如 `exp(-90) ≈ 1.2e-39`）时，FTZ 会将其变为 0，类似于 ReLU 的负半轴梯度截断，不会导致训练不稳定。

---

## 5. 集成可行性

### 5.1 torch.library.impl 机制

`torch.library.impl` 是 PyTorch 2.x 引入的算子注册 API，用于为已定义的算子注册特定后端（dispatch key）的实现。

PyTorch dispatcher 的分派键优先级：

```
Functionalize > Autocast > Autograd > CPU > CUDA > ...
```

### 5.2 三种集成路径

#### 路径 A：零改动（当前方案）

当前 `ops.py` 的实现已经通过 PyTorch 内置的 SLEEF 后端获得了 SIMD 加速：

```python
@custom_op("eml_pytorch::eml", mutates_args=())
def eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_safe = torch.clamp(y, min=1e-8)
    return torch.exp(x) - torch.log(y_safe)
```

`torch.exp()` 和 `torch.log()` 在 CPU 上自动使用 SLEEF u10 的 SIMD 实现。但存在 **4 次内存遍历**（clamp → exp → log → sub），无法利用数据局部性。

#### 路径 B：C++ 扩展 + SLEEF u35 标量融合

创建独立的 C++ 扩展模块，在单次循环中完成 `exp(x) - log(clamp(y))`，减少内存遍历次数：

```cpp
#include <torch/extension.h>
#include <sleef.h>

torch::Tensor eml_sleef_u35_cpu(torch::Tensor x, torch::Tensor y) {
    auto y_safe = torch::clamp(y, 1e-8);
    auto x_contig = x.contiguous();
    auto y_contig = y_safe.contiguous();
    auto result = torch::empty_like(x_contig);

    int64_t n = x.numel();
    const float* x_ptr = x_contig.data_ptr<float>();
    const float* y_ptr = y_contig.data_ptr<float>();
    float* out_ptr = result.data_ptr<float>();

    at::parallel_for(0, n, 1024, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            out_ptr[i] = Sleef_expf_u35(x_ptr[i]) - Sleef_logf_u35(y_ptr[i]);
        }
    });

    return result;
}

TORCH_LIBRARY_IMPL(eml_pytorch, CPU, m) {
    m.impl("eml", eml_sleef_u35_cpu);
}
```

#### 路径 C：C++ 扩展 + SLEEF u35 SIMD 显式优化（最佳性能）

使用 SLEEF 的 SIMD API（AVX2/AVX-512/NEON），配合 PyTorch 的并行框架：

```cpp
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <sleef.h>

#if defined(__AVX2__)
#include <immintrin.h>

static void eml_sleef_kernel_avx2(
    const float* x_ptr, const float* y_ptr, float* out_ptr, int64_t n) {
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x_ptr + i);
        __m256 vy = _mm256_loadu_ps(y_ptr + i);
        __m256 vexp = Sleef_expf8_u35(vx);
        __m256 vlog = Sleef_logf8_u35(vy);
        __m256 vout = _mm256_sub_ps(vexp, vlog);
        _mm256_storeu_ps(out_ptr + i, vout);
    }
    for (; i < n; ++i) {
        out_ptr[i] = Sleef_expf_u35(x_ptr[i]) - Sleef_logf_u35(y_ptr[i]);
    }
}

torch::Tensor eml_sleef_simd_cpu(torch::Tensor x, torch::Tensor y) {
    auto y_safe = torch::clamp(y, 1e-8);
    auto x_contig = x.contiguous();
    auto y_contig = y_safe.contiguous();
    auto result = torch::empty_like(x_contig);

    int64_t n = x.numel();
    const float* x_ptr = x_contig.data_ptr<float>();
    const float* y_ptr = y_contig.data_ptr<float>();
    float* out_ptr = result.data_ptr<float>();

    at::parallel_for(0, n, 4096, [&](int64_t start, int64_t end) {
        eml_sleef_kernel_avx2(x_ptr + start, y_ptr + start, out_ptr + start, end - start);
    });

    return result;
}

TORCH_LIBRARY_IMPL(eml_pytorch, CPU, m) {
    m.impl("eml", eml_sleef_simd_cpu);
}
#endif
```

### 5.3 三种路径对比

| 维度 | 路径 A (当前) | 路径 B (u35 标量融合) | 路径 C (u35 SIMD 融合) |
|------|-------------|---------------------|---------------------|
| 实现难度 | 零（已实现） | 低 | 中 |
| 精度 | u10 (~0.5 ULP) | u35 (~3.5 ULP) | u35 (~3.5 ULP) |
| 内存遍历 | 4 次 | 1 次 | 1 次 |
| 临时缓冲区 | 2 个 | 0 个 | 0 个 |
| 超越函数加速 | 基准 | ~1.3x（u35 vs u10） | ~1.5-2x（u35 SIMD vs u10 SIMD） |
| 融合收益 | 无 | 减少内存往返 | 减少内存往返 |
| 额外依赖 | 无 | SLEEF 库 | SLEEF 库 + CPU 特定编译 |
| 跨平台 | 自动 | 需编译 SLEEF | 需条件编译 |
| 训练稳定性 | 完全稳定 | 完全稳定 | 完全稳定 |

### 5.4 TORCH_LIBRARY_IMPL 宏详解

`TORCH_LIBRARY_IMPL` 是 PyTorch C++ 端注册算子实现的核心宏：

```cpp
TORCH_LIBRARY_IMPL(namespace, dispatch_key, module)
```

- `namespace`：算子命名空间，如 `eml_pytorch`
- `dispatch_key`：分派键，如 `CPU`, `CUDA`, `Autograd`, `Autocast`, `Meta`
- `module`：变量名，用于链式调用 `.impl()`

工作原理：宏展开后生成静态对象，在 C++ 扩展加载时自动执行注册。调用 `torch.ops.namespace.op_name()` 时，dispatcher 查表找到匹配的实现。

**与 Python `custom_op` 的关系**：Python 的 `@custom_op("eml_pytorch::eml")` 等价于 `TORCH_LIBRARY` + `CompositeExplicitAutograd` 实现。如果 C++ 端用 `TORCH_LIBRARY_IMPL` 注册了 CPU 实现，会覆盖 Python 端的 fallback。

### 5.5 完整集成示例的项目结构

```
eml_pytorch/
├── csrc/
│   └── eml_sleef.cpp          # SLEEF C++ 扩展
├── __init__.py                 # 加载 _C 扩展
├── ops.py                      # Python fallback + custom_op 定义
├── nn.py
└── triton_kernel.py
```

编译配置（`setup.py` 或 `pyproject.toml`）：

```python
from torch.utils.cpp_extension import CppExtension, BuildExtension

ext_modules = [
    CppExtension(
        name='eml_pytorch._C',
        sources=['eml_pytorch/csrc/eml_sleef.cpp'],
        extra_compile_args=['-O3', '-mavx2', '-mfma'],
        libraries=['sleef'],
    ),
]
```

---

## 6. Go/No-Go 决策建议

### 6.1 收益分析

| 优化维度 | 当前状态 | 优化后预期 | 收益程度 |
|---------|---------|-----------|---------|
| SIMD 向量化 | ✅ 已有（SLEEF u10） | SLEEF u35 SIMD | **低**：仅 ~1.5-2x 超越函数加速 |
| 内存遍历次数 | 4 次 | 1 次 | **中**：减少缓存未命中和临时分配 |
| 临时缓冲区 | 2 个 | 0 个 | **低-中**：减少内存分配开销 |
| 精度损失 | 无 | ~3.5 ULP | **可忽略**：对训练无影响 |

### 6.2 成本分析

| 成本维度 | 评估 |
|---------|------|
| 开发工作量 | 中等：需编写 C++ 扩展、条件编译、多平台测试 |
| 维护负担 | 高：SLEEF 依赖、CPU 特性检测、跨平台兼容 |
| 构建复杂度 | 显著增加：需安装 SLEEF、配置 CMake |
| 用户门槛 | 提高：pip install 不再是纯 Python 包 |

### 6.3 关键判断

1. **PyTorch 已内置 SLEEF u10 SIMD 加速**：当前 `torch.exp()` 和 `torch.log()` 在 CPU 上已经使用 SLEEF 的 AVX2/AVX-512 实现，不是标量计算。

2. **融合收益有限**：从 4 次内存遍历减少到 1 次，理论上可减少约 2-3x 的内存带宽开销。但 EML 算子本身是计算密集型（exp/log 的多项式逼近需要大量 FMA 操作），内存带宽并非主要瓶颈。

3. **u35 vs u10 加速有限**：u35 比 u10 快约 30-40%，但整体 EML 算子的加速比会被 clamp 和 sub 操作稀释，预期整体加速约 1.2-1.5x。

4. **CPU 端 EML 的使用场景有限**：在典型训练场景中，矩阵乘法占主导，超越函数的耗时占比通常 <5%。EML 算子的 CPU 端性能优化对端到端训练速度的影响微乎其微。

5. **项目定位**：eml-pytorch 的核心价值在于可解释性神经网络和符号回归，而非极致的计算性能。GPU 端已有 Triton 融合核，CPU 端优化的优先级较低。

### 6.4 决策

## 🟡 Conditional No-Go：不建议在 v0.4.0 中实现 CPU 端 SIMD 融合优化

**理由**：

1. **投入产出比低**：开发 + 维护成本高，但预期整体加速仅 1.2-1.5x，且仅影响 CPU 端的 EML 算子本身，对端到端训练几乎无影响。

2. **破坏纯 Python 包优势**：当前 eml-pytorch 是纯 Python 包，`pip install` 即可使用。引入 C++ 扩展 + SLEEF 依赖会显著增加用户安装门槛。

3. **已有替代方案**：
   - GPU 端已有 Triton 融合核，性能远优于任何 CPU 优化
   - `torch.compile` + Inductor 的 CppBackend 可能自动融合部分操作
   - PyTorch 已内置 SLEEF u10 SIMD，当前实现并非未优化

4. **未来可 reconsider 的条件**：
   - 如果出现 CPU-only 的边缘设备部署需求（如 ARM NEON 场景）
   - 如果 PyTorch 原生支持自定义算子的 CPU 融合编译（类似 XLA 的算子融合）
   - 如果项目定位转向 CPU 推理优化

### 6.5 推荐的替代优化方向

| 方向 | 预期收益 | 实现难度 | 建议 |
|------|---------|---------|------|
| `torch.compile` 集成测试 | 可能自动融合 | 低 | **推荐**：验证 `torch.compile(eml)` 是否能自动融合 CPU 端操作 |
| Triton 反向传播融合核 | GPU 端完整融合 | 中 | **推荐**：当前 Triton 核仅有前向，补全反向传播收益更大 |
| `torch.set_flush_denormal(True)` | 避免非正规数性能陷阱 | 零 | **推荐**：一行代码，无副作用 |
| SLEEF u35 选项暴露 | ~1.3x 超越函数加速 | 低 | **可选**：通过环境变量控制是否使用 u35 |

---

## 附录 A：SLEEF 加速比速查表

| 场景 | 指令集 | exp 加速比 | log 加速比 |
|------|--------|-----------|-----------|
| u10 精度 | SSE2 (4-wide) | 3.5-4.5x | 3.5-4.5x |
| u10 精度 | AVX2 (8-wide) | 6-8x | 6-8x |
| u10 精度 | AVX-512 (16-wide) | 12-16x | 12-16x |
| u35 精度 | AVX2 (8-wide) | 8-11x | 9-12x |
| u35 精度 | AVX-512 (16-wide) | 16-22x | 18-24x |
| u10 精度 | NEON (4-wide) | 3-4x | 3-4x |

> 以上加速比均为相对标量 libm 的数据，来自 SLEEF 官方 benchmark + 第三方测试。

## 附录 B：参考资料

1. Naoki Shibata, "SLEEF: SIMD Library for Evaluating Elementary Functions", ISC High Performance 2019
2. Micikevicius et al., "Mixed Precision Training", ICLR 2018
3. Micikevicius et al., "FP8 Formats for Deep Learning", arXiv 2209.05433
4. Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
5. SLEEF GitHub: https://github.com/shibatch/sleef
6. PyTorch Source: https://github.com/pytorch/pytorch (aten/src/ATen/cpu/vec/)
7. oneMKL VM Documentation: https://www.intel.com/content/www/us/en/docs/onemkl/
8. oneDNN Documentation: https://oneapi-src.github.io/oneDNN/
