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

def test_eml_gradgradcheck_x():
    """
    【梯度审计】验证 eml 算子对输入 x 的二阶梯度正确性。
    这是衡量算子是否具备工业级数值稳定性的关键测试。
    """
    # 确保输入 x 需要计算二阶梯度，y 只计算一阶梯度
    x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
    y = torch.rand(5, 5, dtype=torch.double) + 0.2  # 确保 y > 0，且不需要二阶梯度

    # gradgradcheck 会对 x 计算二阶梯度
    # eps 和 atol 可适当放宽，因为二阶导数的有限差分极易引入误差
    test_result = torch.autograd.gradgradcheck(
        lambda x_input: eml(x_input, y),  # 固定 y，仅将 x 作为变量输入
        (x,),
        eps=1e-5,
        atol=1e-3
    )
    
    assert test_result, "❌ eml 算子对 x 的二阶梯度检查失败！"
    print("✅ 二阶梯度审计通过：eml 算子对 x 的二阶导数正确。")

def test_eml_gradcheck_full():
    """
    【回归测试】验证 eml 算子的一阶梯度（同时测试 x 和 y）。
    这是对之前工作的回归保护。
    """
    x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
    y = torch.rand(5, 5, dtype=torch.double, requires_grad=True) + 0.2
    
    test_result = torch.autograd.gradcheck(eml, (x, y), eps=1e-6, atol=1e-4)
    assert test_result, "❌ eml 算子一阶梯度检查失败！"
    print("✅ 回归测试通过：eml 算子一阶梯度正确。")