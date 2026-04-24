import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from eml_pytorch import EMLActivation  # 从 eml-pytorch 导入 EMLActivation

# 设置随机种子，确保可复现
torch.manual_seed(42)
np.random.seed(42)

# ------------------- 1. 生成合成回归数据 -------------------
def generate_data(n_samples=1000, n_features=10):
    x = torch.randn(n_samples, n_features)
    # 构建一个非线性目标：y = sin(x1) + cos(x2) + 0.5*x3^2 + noise
    y = (torch.sin(x[:, 0]) +
         torch.cos(x[:, 1]) +
         0.5 * x[:, 2]**2 +
         0.1 * torch.randn(n_samples))
    return x, y.unsqueeze(1)

x_train, y_train = generate_data()

# ------------------- 2. 定义统一的微型网络结构 -------------------
class MiniNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1, activation_fn=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = activation_fn()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # 输出层无激活函数
        return x

# ------------------- 3. 训练函数 -------------------
def train_model(model, x, y, epochs=500, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# ------------------- 4. 执行对比实验 -------------------
activations = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'EML': EMLActivation
}

epochs = 500
results = {}

for name, act_fn in activations.items():
    model = MiniNet(activation_fn=act_fn)
    losses = train_model(model, x_train, y_train, epochs=epochs)
    results[name] = losses
    print(f"{name} 最终损失: {losses[-1]:.6f}")

# ------------------- 5. 绘制损失曲线对比图 -------------------
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.title('Convergence Comparison: EML vs ReLU vs GELU')
plt.legend()
plt.grid(True)
plt.savefig('activation_benchmark.png', dpi=150)
plt.show()