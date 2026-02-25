import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# 将上一级目录或当前目录加入路径，确保能导入 model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import OffsetLoRAModel


def run_experiment_5_3():
    # --- 1. 实验配置 ---
    dims = [512, 4096]  # 对比基础维度 vs 大模型常用维度 (Llama-3 等)
    r = 8
    steps = 200
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Running Exp 5.3: Scaling Stability ===")
    print(f"Device: {device}")

    def run_single_session(d, mode):
        print(f"  -> Testing d={d}, mode={mode}")
        # 生成模拟数据
        torch.manual_seed(42)
        X = torch.randn(1024, d).to(device)
        # 目标权重
        W_target = (torch.randn(d, r).to(device) @ torch.randn(r, d).to(device)) * 0.1
        Y = X @ W_target

        # 初始化模型 (调用统一的 model.py)
        model = OffsetLoRAModel(d, r, mode).to(device)
        optimizer = optim.SGD([model.A, model.B], lr=lr, momentum=0.9)
        criterion = nn.MSELoss()

        ratios = []
        for _ in range(steps):
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()

            gA = model.A.grad.norm().item()
            gB = model.B.grad.norm().item()
            # 记录梯度范数比例：这是衡量“梯度锁定效应”的关键指标
            ratios.append(gA / (gB + 1e-8))
            optimizer.step()
        return ratios

    # --- 2. 收集数据 ---
    results = {}
    for d in dims:
        results[f"Standard (d={d})"] = run_single_session(d, 'standard')
        results[f"Offset-LoRA (d={d})"] = run_single_session(d, 'offset_orthogonal')

    # --- 3. 结果可视化 ---
    plt.figure(figsize=(10, 6))

    # 颜色与样式配置
    styles = {
        f"Standard (d=512)": ("#1f77b4", "-"),
        f"Standard (d=4096)": ("#aec7e8", "--"),
        f"Offset-LoRA (d=512)": ("#d62728", "-"),
        f"Offset-LoRA (d=4096)": ("#ff9896", "-."),
    }

    for label, data in results.items():
        color, ls = styles.get(label, (None, "-"))
        lw = 2.5 if 'Offset' in label else 1.2
        plt.plot(data, label=label, color=color, linestyle=ls, linewidth=lw)

    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal Ratio (1.0)')

    # 核心：使用 Log 轴，因为 Standard LoRA 在高维下梯度极不平衡
    plt.yscale('log')
    plt.title("Gradient Norm Ratio Stability Across Dimensions (Exp 5.3)", fontsize=12)
    plt.xlabel("Training Steps")
    plt.ylabel("Ratio ||gA|| / ||gB|| (Log Scale)")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, which="both", alpha=0.3)

    # 自动保存结果
    os.makedirs('results/figures', exist_ok=True)
    image_path = 'results/figures/exp_5_3_scaling_comparison.png'
    plt.savefig(image_path, dpi=300)
    print(f"\n[Success] Scaling experiment completed. Plot saved at: {image_path}")
    plt.show()


if __name__ == "__main__":
    run_experiment_5_3()