# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-25

### Added
- **Symbolic Regression Application**: Full end-to-end symbolic regression pipeline using the Feynman dataset (`examples/symbolic_regression.py`). Includes `EMLExpressionTree`, progressive STE training, weight snapshot, and symbolic expression extraction via sympy.
- **`torch.compile` Compatibility Benchmark**: New benchmark script (`examples/benchmark_compile.py`) comparing eager mode, `torch.compile`, Triton fused kernel, and `torch.compile(Triton)` across multiple tensor sizes.
- **CI Enhancement**: Added `ruff` linting step to GitHub Actions CI pipeline (`.github/workflows/test.yml`).

### Fixed
- Resolved `test_eml_node_backward` gradient assertion failure by preserving leaf tensor status for `y` input.
- Added `numpy` to CI workflow dependencies to suppress initialization warnings.

## [0.2.0] - 2026-04-24

### Added
- **Performance Baseline**: New benchmark script (`examples/benchmark_activations.py`) comparing EML against ReLU and GELU on a synthetic regression task.
- **Gradient Audit**: `gradgradcheck` test verifying correct second-order gradients for input `x` (`tests/test_ops.py`).
- **Mixed Precision Validation**: Comprehensive AMP stability tests (`tests/test_amp_stability.py`) covering normal range, boundary range, and gradient flow under `autocast`.

### Fixed
- Fixed `EMLActivation` to correctly call `torch.ops.eml_pytorch.eml`, ensuring benchmarks truly test the project's core operator.

## [0.1.0] - 2026-04-22

### Added
- Initial release of the `eml-pytorch` package.
- Core `eml(x, y) = exp(x) - log(y)` operator as a PyTorch custom op with full autograd support.
- `EMLNode`, `EMLActivation`, and `TinyEMLNet` modules in `eml_pytorch.nn`.
- Optional Triton fused kernel for GPU acceleration (`eml_pytorch.triton_kernel`).
- Basic unit tests including `gradcheck` for custom op gradients.
- Single-node and two-node training demos.

[0.3.0]: https://github.com/UyNewNas/eml-pytorch/releases/tag/v0.3.0
[0.2.0]: https://github.com/UyNewNas/eml-pytorch/releases/tag/v0.2.0
[0.1.0]: https://github.com/UyNewNas/eml-pytorch/releases/tag/v0.1.0
