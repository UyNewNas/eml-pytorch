import torch
import pytest
from eml_pytorch.nn import EMLNode, EMLActivation, TinyEMLNet

def test_eml_node_forward():
    """测试 EMLNode 的前向传播"""
    node = EMLNode(input_dim=5)
    x = torch.randn(3, 5)
    y = torch.rand(3, 5) + 0.1
    out = node(x, y)
    assert out.shape == (3,)

def test_eml_node_backward():
    """测试 EMLNode 的反向传播"""
    node = EMLNode(input_dim=5)
    x = torch.randn(3, 5, requires_grad=True)
    # 修复：创建叶子张量 y，然后原地加上 0.1，这样 y 仍然是叶子节点
    y = torch.rand(3, 5, requires_grad=True)
    y.data.add_(0.1)
    
    out = node(x, y)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert y.grad is not None  # 现在断言可以通过了

def test_eml_activation_forward():
    """测试 EMLActivation 的前向传播"""
    activation = EMLActivation(c_init=2.0)
    x = torch.randn(3, 5)
    out = activation(x)
    assert out.shape == (3, 5)

def test_tiny_eml_net_forward():
    """测试 TinyEMLNet 的前向传播"""
    net = TinyEMLNet(input_dim=5)
    x = torch.randn(3, 5)
    y = torch.rand(3, 5) + 0.1
    out = net(x, y)
    assert out.shape == (3, 1)
