import torch
import torch.nn as nn
from eml_pytorch import TinyEMLNet

def target_func(x, y):
    return (torch.exp(x.mean(dim=1)) - torch.log(y.mean(dim=1) + 1e-8)) / 10.0

def main():
    input_dim = 5
    num_samples = 200
    X = torch.randn(num_samples, input_dim) * 0.5
    Y = torch.rand(num_samples, input_dim) + 0.2
    targets = target_func(X, Y).unsqueeze(1)

    model = TinyEMLNet(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X, Y)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")

    print("训练完成。")

if __name__ == "__main__":
    main()