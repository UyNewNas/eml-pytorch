import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from eml_pytorch.ops import eml

torch.manual_seed(42)
np.random.seed(42)

n_samples = 1000
n_features = 10
X = torch.randn(n_samples, n_features)
y = (torch.sin(X[:, 0]) +
     torch.cos(X[:, 1]) +
     0.5 * X[:, 2] ** 2 +
     0.1 * torch.randn(n_samples))
y = y.unsqueeze(1)


class EMLActivation(nn.Module):
    def __init__(self, c_init=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))

    def forward(self, x):
        c_expanded = self.c.expand_as(x)
        return eml(x, c_expanded)


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
        x = self.fc3(x)
        return x


def train_model(model, X, y, epochs=500, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


activations = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'EML': EMLActivation,
}

results = {}
for name, act_fn in activations.items():
    torch.manual_seed(42)
    model = MiniNet(activation_fn=act_fn)
    losses = train_model(model, X, y, epochs=500, lr=0.001)
    results[name] = losses
    print(f"{name} 最终损失: {losses[-1]:.6f}")

plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.title('Convergence Comparison: EML vs ReLU vs GELU')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=150)
plt.show()
