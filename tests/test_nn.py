import torch
import pytest
from eml_pytorch import EMLNode, TinyEMLNet

def test_eml_node_forward():
    node = EMLNode(input_dim=5)
    x = torch.randn(3, 5)   # batch=3
    y = torch.rand(3, 5) + 0.1
    out = node(x, y)
    assert out.shape == (3,)
    assert not torch.isnan(out).any()

def test_eml_node_backward():
    node = EMLNode(input_dim=5)
    x = torch.randn(3, 5, requires_grad=True)
    y = torch.rand(3, 5, requires_grad=True) + 0.1
    out = node(x, y)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert y.grad is not None

def test_tiny_eml_net():
    model = TinyEMLNet(input_dim=5)
    x = torch.randn(10, 5)
    y = torch.rand(10, 5) + 0.1
    out = model(x, y)
    assert out.shape == (10, 1)

def test_training_step():
    """验证模型能否完成一个简单的训练步骤"""
    model = TinyEMLNet(input_dim=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(20, 5)
    y = torch.rand(20, 5) + 0.1
    target = torch.randn(20, 1)
    out = model(x, y)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    optimizer.step()
    # 只要不报错就算通过