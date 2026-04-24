import torch
import torch.nn as nn
from .ops import eml

class EMLNode(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim))
        self.b1 = nn.Parameter(torch.zeros(1))
        self.w2 = nn.Parameter(torch.randn(input_dim))
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        u = torch.matmul(x, self.w1) + self.b1
        v = torch.matmul(y, self.w2) + self.b2
        v = torch.clamp(v, min=1e-8)
        return eml(u, v)

class EMLActivation(nn.Module):
    """
    将 EML 算子封装为标准的单输入激活函数。
    内部执行 eml(x, c)，其中 c 是可学习的标量参数。
    """
    def __init__(self, c_init=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))

    def forward(self, x):
        c_expanded = self.c.expand_as(x)
        return torch.ops.eml_pytorch.eml(x, c_expanded)

class TinyEMLNet(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.node1 = EMLNode(input_dim)
        self.node2 = EMLNode(1)

    def forward(self, x, y):
        out1 = self.node1(x, y)
        out1_2d = out1.unsqueeze(1)
        out2 = self.node2(out1_2d, out1_2d)
        return out2.unsqueeze(1)