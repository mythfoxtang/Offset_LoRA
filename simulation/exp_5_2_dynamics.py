import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# 将 src 路径加入环境变量，确保可以导入通用模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import OffsetLoRAModel


def run_experiment_5_2():
    # --- 1. 配置参数 ---
    d, r, n = 512, 8, 1024
    steps = 500
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Exp 5.2 on device: {device}")

    # --- 2. 模拟数据生成 ---
    torch.manual_seed(42)
    X = torch.randn(n, d).to(device)
    A_star = torch.randn(r, d).to(device)
    B_star = torch.randn(d, r).to(device)
    W_target = B_star @ A_star
    Y = X @ W_target + torch.randn(n, d).to(device) * 0.1

    # --- 3. 运行实验循环 ---
    modes = ['standard', 'non_zero_li', 'offset_gaussian', 'offset_orthogonal']
    results = {m: {'loss': [], 'ratio': []} for m in modes}

    for mode in modes:
        print(f"Running mode: {mode}...")
        model = OffsetLoRAModel(d, r, mode).to(device)
        optimizer = optim.SGD([model.A, model.B], lr=lr, momentum=0.9)
        criterion = nn.MSELoss()

        for step in range(steps):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()

            # 计算梯度范数比例 ||gA||/||gB||
            grad_A_norm = model.A.grad.norm().item()
            grad_B_norm = model.B.grad.norm().item()
            ratio = grad_A_norm / (grad_B_norm + 1e-8)

            results[mode]['loss'].append(loss.item())
            results[mode]['ratio'].append(ratio)
            optimizer.step()

    # --- 4. 可视化 ---
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss 收敛曲线
    plt.subplot(1, 2, 1)
    colors = {'standard': '#1f77b4', 'non_zero_li': '#ff7f0e',
              'offset_gaussian': '#2ca02c', 'offset_orthogonal': '#d62728'}
    for mode in modes:
        plt.plot(results[mode]['loss'], label=mode, color=colors[mode], alpha=0.8)
    plt.yscale('log')
    plt.title("Training Loss Convergence (Exp 5.2)")
    plt.xlabel("Steps")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    # Plot 2: 梯度范数比例
    plt.subplot(1, 2, 2)
    for mode in modes:
        plt.plot(results[mode]['ratio'], label=mode, color=colors[mode], alpha=0.8)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Optimal (1.0)')
    plt.title("Gradient Norm Ratio Stability")
    plt.xlabel("Steps")
    plt.ylabel("Ratio ||gA||/||gB||")
    plt.ylim(0, 5)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 自动创建保存目录
    os.makedirs('results/figures', exist_ok=True)
    save_path = 'results/figures/exp_5_2_dynamics.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    run_experiment_5_2()