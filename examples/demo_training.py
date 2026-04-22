"""
demo_training.py

演示如何使用 eml-pytorch 包中的 EMLNode 进行简单函数拟合。
目标函数：f(x) = exp(0.5 * x) - log(1 + x^2)，展示了 EML 算子的非线性表达能力。
"""

import torch
import torch.nn as nn
from eml_pytorch import EMLNode

def main():
    # ------------------- 1. 生成合成数据 -------------------
    torch.manual_seed(42)
    x = torch.linspace(-1, 1, 100).unsqueeze(1)          # [100, 1]

    # 目标函数：一个合理的非线性组合，让 EMLNode 有机会拟合
    # 注意：目标值不要过大，保持在 [-5, 5] 范围内，避免数值问题
    y_true = torch.exp(0.5 * x.squeeze()) - torch.log(1 + x.squeeze()**2)

    # EMLNode 需要两个输入。这里将第二个输入固定为常数 1，以避免 log(0)
    # 在实际应用中，第二个输入可以来自数据特征
    x_input = x
    y_input = torch.ones_like(x)   # 常数 1

    # ------------------- 2. 构建模型 -------------------
    model = EMLNode(input_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ------------------- 3. 训练循环 -------------------
    num_epochs = 500
    print("开始训练单节点 EML 网络...")
    print(f"{'Epoch':<8} {'Loss':<12}")
    print("-" * 20)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(x_input, y_input)
        loss = criterion(out, y_true)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"{epoch:<8} {loss.item():<12.6f}")

    # ------------------- 4. 输出最终结果 -------------------
    print("-" * 20)
    with torch.no_grad():
        final_out = model(x_input, y_input)
        final_loss = criterion(final_out, y_true)
        print(f"最终损失: {final_loss.item():.6f}")

        # 展示前5个样本的预测对比
        print("\n示例预测 vs 真实值:")
        print(f"{'预测值':<12} {'真实值':<12}")
        for i in range(5):
            print(f"{final_out[i].item():<12.4f} {y_true[i].item():<12.4f}")

if __name__ == "__main__":
    main()