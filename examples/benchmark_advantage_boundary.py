"""
优势边界探索实验：寻找 EML 算子优于 ReLU/GELU 的场景。

测试三类合成任务：
  1. 指数过程回归 (Exponential)
  2. 对数过程回归 (Logarithmic)
  3. 复合过程回归 (Exponential + Logarithmic)

结论会诚实地展示 EML 是否具有明确的性能优势或边界。
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_pytorch import EMLActivation

torch.manual_seed(42)
np.random.seed(42)

class MiniNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, activation_fn=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = activation_fn()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

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

def generate_exponential_data(n_samples=500):
    """目标函数：y = exp(0.3x)"""
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.exp(0.3 * x)
    return x, y

def generate_logarithmic_data(n_samples=500):
    """目标函数：y = log(2 + x^2)"""
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.log(2 + x.pow(2))
    return x, y

def generate_compound_data(n_samples=500):
    """目标函数：y = exp(0.2x) - log(1.5 + x^2)"""
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.exp(0.2 * x) - torch.log(1.5 + x.pow(2))
    return x, y

activations = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'EML': EMLActivation
}

task_generators = {
    "Exponential Process (exp)": generate_exponential_data,
    "Logarithmic Process (log)": generate_logarithmic_data,
    "Compound Process (exp-log)": generate_compound_data,
}

results = {}

for task_name, gen_func in task_generators.items():
    print(f"\n{'='*50}")
    print(f"Testing Task: {task_name}")
    x, y = gen_func()
    
    results[task_name] = {}
    
    for act_name, act_fn in activations.items():
        model = MiniNet(activation_fn=act_fn)
        loss_curve = train_model(model, x, y, epochs=500, lr=0.001)
        final_loss = loss_curve[-1]
        results[task_name][act_name] = {
            'final_loss': final_loss,
            'curve': loss_curve
        }
        print(f"  {act_name}: Final MSE = {final_loss:.6f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = {'ReLU': 'blue', 'GELU': 'orange', 'EML': 'green'}

for idx, (task_name, data) in enumerate(results.items()):
    ax = axes[idx]
    for act_name, metrics in data.items():
        ax.plot(metrics['curve'], label=act_name, color=colors[act_name])
    ax.set_title(task_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'advantage_boundary_experiment.png')
plt.savefig(output_path, dpi=150)
plt.show()

print("\n\n============ 最终结论汇总 ============")
for task_name, data in results.items():
    best_act = min(data, key=lambda k: data[k]['final_loss'])
    best_loss = data[best_act]['final_loss']
    eml_loss = data['EML']['final_loss']
    print(f"{task_name}: EML loss = {eml_loss:.6f}. Best ({best_act}) = {best_loss:.6f}")
    if best_act == 'EML':
        print(f"  >>> EML 在此任务上表现最佳，验证了潜在优势！")
    else:
        print(f"  --- EML 在此任务上未占优。")

print(f"\n实验完成。结果图像已保存为 {output_path}")
