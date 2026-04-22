
# eml-pytorch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-Triton-6677CC?logo=nvidia&logoColor=white)](https://github.com/openai/triton)
[![Colab](https://img.shields.io/badge/Colab-T4-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

Hardware-efficient Exp-Minus-Log (EML) operator for PyTorch. A unified activation primitive for neuro-symbolic AI and edge deployment.

---

## Quick Start

### Installation

```bash
pip install git+https://github.com/UyNewNas/eml-pytorch.git
```

For GPU acceleration with Triton (recommended for Linux/WSL2/Colab):

```bash
pip install triton
```

### Single Node Example

```python
from eml_pytorch import EMLNode
...
```

Training output:

```text
Epoch   0, Loss: 5.311534
...
最终损失: 0.051967
```

### Two-Node Network Example

```python
from eml_pytorch import TinyEMLNet
...
```

Training output:

```text
Epoch   0, Loss: 6.100719
...
最终损失: 0.176003
```

## Performance

The EML operator is accelerated by a custom **Triton fused kernel** on CUDA-enabled GPUs. By fusing `exp` and `log` into a single kernel, memory access overhead is significantly reduced.

Benchmarked on an NVIDIA T4 GPU (Google Colab) with PyTorch 2.5.1, float32 tensors.

| Tensor Size | Native PyTorch Custom Op | Triton Fused Kernel | **Speedup** |
| :--- | :--- | :--- | :--- |
| 1024×1024 | 0.0168 s | 0.0057 s | **2.93×** |
| 2048×2048 | 0.0638 s | 0.0212 s | **3.01×** |
| 4096×4096 | 0.2538 s | 0.0844 s | **3.01×** |

> 💡 The Triton kernel automatically falls back to the native implementation when CUDA or Triton is unavailable (e.g., on CPU or Windows).

### Reproduce the Benchmark

Run the provided benchmark script:

```bash
python examples/benchmark_triton.py
```

Make sure you have Triton installed and a CUDA-capable GPU available.

> **Note for Windows users**: Official Triton binaries are not yet available for Windows. The fused GPU kernel can be tested via **WSL2** or cloud environments like **Google Colab**. On native Windows, the operator gracefully falls back to CPU execution.
