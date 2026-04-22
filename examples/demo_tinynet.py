'''
Author: slava
Date: 2026-04-22 08:36:15
LastEditTime: 2026-04-22 08:36:40
LastEditors: ch4nslava@gmail.com
Description: 

'''
import torch
import torch.nn as nn
from eml_pytorch import TinyEMLNet

def main():
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    # 目标函数缩放至 [-1.5, 1.5] 左右
    y_true = (torch.sin(2 * x.squeeze()) + 0.5 * x.squeeze()) / 2.0

    # 两个输入不同，增加表达空间
    x_input = x
    y_input = torch.ones_like(x)   # 常数输入

    model = TinyEMLNet(input_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    print("训练 TinyEMLNet (改进版)...")
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(x_input, y_input)
        loss = criterion(out, y_true.unsqueeze(1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.5f}")

    with torch.no_grad():
        final_loss = criterion(model(x_input, y_input), y_true.unsqueeze(1))
        print(f"最终损失: {final_loss.item():.6f}")

if __name__ == "__main__":
    main()