import torch
import pytest
from eml_pytorch import eml

def test_eml_forward_cpu():
    """测试 CPU 上前向传播的数值正确性"""
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([2.0, 4.0], dtype=torch.float32)
    expected = torch.exp(x) - torch.log(y)
    out = eml(x, y)
    assert torch.allclose(out, expected, rtol=1e-5)

def test_eml_backward_cpu():
    """测试 CPU 上反向传播梯度正确性（通过 gradcheck）"""
    from torch.autograd import gradcheck
    x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
    y = torch.rand(5, 5, dtype=torch.double, requires_grad=True) + 0.1
    assert gradcheck(eml, (x, y), eps=1e-6, atol=1e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_eml_forward_cuda():
    """测试 GPU 上前向传播（如果可用）"""
    device = torch.device("cuda")
    x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    y = torch.tensor([2.0, 4.0], dtype=torch.float32, device=device)
    expected = torch.exp(x) - torch.log(y)
    out = eml(x, y)
    assert torch.allclose(out, expected, rtol=1e-5)

def test_numerical_stability():
    """测试数值稳定性：当 y 接近零或为负数时的 clamp 行为"""
    x = torch.tensor([0.0])
    y_zero = torch.tensor([0.0])
    y_neg = torch.tensor([-1.0])
    # 不应抛出异常或产生 NaN
    out_zero = eml(x, y_zero)
    out_neg = eml(x, y_neg)
    assert not torch.isnan(out_zero).any()
    assert not torch.isnan(out_neg).any()