"""
Triton fused kernel for Exp-Minus-Log (EML) operator.
Provides hardware acceleration on CUDA-capable GPUs.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    # 定义占位符，避免后续代码报错
    class triton:
        @staticmethod
        def jit(*args, **kwargs):
            return lambda *args, **kwargs: None
    tl = None


if TRITON_AVAILABLE:
    @triton.jit
    def _eml_fused_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        # 数值稳定性处理
        y_safe = tl.where(y <= 0.0, 1e-8, y)

        output = tl.exp(x) - tl.log(y_safe)
        tl.store(out_ptr + offsets, output, mask=mask)


def eml_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Triton accelerated EML operator.
    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
    Returns:
        torch.Tensor: output = exp(x) - log(y)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Please install triton.")
    if not x.is_cuda or not y.is_cuda:
        raise RuntimeError("Triton kernel requires CUDA tensors.")
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")

    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()

    # 定义网格和块大小
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _eml_fused_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out