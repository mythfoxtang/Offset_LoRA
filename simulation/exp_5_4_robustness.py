import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# 确保可以导入同级目录下的 model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import OffsetLoRAModel


def run_experiment_5_4():
    # --- 1. 实验配置 ---
    d, r, n = 512, 8, 1024
    steps = 300
    learning_rates = [1e-4, 1e-2, 1e-1]  # 指数级跨度测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Running Exp 5.4: LR Robustness Stress Test ===")
    print(f"Device: {device}")

    # --- 2. 模拟数据生成 ---
    torch.manual_seed(42)
    X = torch.randn(n, d).to(device)
    # 目标低秩矩阵
    W_target = (torch.randn(d, r).to(device) @ torch.randn(r, d).to(device)) * 0.1
    Y = X @ W_target

    # --- 3. 运行应力测试循环 ---
    results = {}

    for lr in learning_rates:
        for mode in ['standard', 'offset_orthogonal']:
            label = f"{mode}_lr{lr}"
            print(f"  -> Testing hyper-parameters: {label}")

            model = OffsetLoRAModel(d, r, mode).to(device)
            optimizer = optim.SGD([model.A, model.B], lr=lr)
            criterion = nn.MSELoss()

            losses = []
            diverged = False
            for step in range(steps):
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, Y)

                # 核心：发散检测逻辑
                if torch.isnan(loss) or loss.item() > 1e10:
                    print(f"     [Alert] {label} diverged at step {step}")
                    diverged = True
                    break

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # 如果发散，用 None 填充剩余步数以便绘图对齐
            if diverged:
                losses.extend([None] * (steps - len(losses)))
            results[label] = losses

    # --- 4. 可视化与保存 ---
    plt.figure(figsize=(12, 7))
    colors = {'standard': '#1f77b4', 'offset_orthogonal': '#d62728'}  # 蓝红对比
    linestyles = {1e-4: ':', 1e-2: '--', 1e-1: '-'}

    for label, losses in results.items():
        mode = 'standard' if 'standard' in label else 'offset_orthogonal'
        lr = float(label.split('lr')[1])

        # 过滤发散点
        valid_losses = [l for l in losses if l is not None]
        plt.plot(valid_losses,
                 label=f"{mode} (lr={lr})",
                 color=colors[mode],
                 linestyle=linestyles[lr],
                 linewidth=2.5 if lr == 1e-1 else 1.5)

    plt.yscale('log')
    plt.title("Learning Rate Robustness Stress Test (Exp 5.4)", fontsize=14)
    plt.xlabel("Training Steps")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend(ncol=2, loc='upper right', frameon=True)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    # 自动保存
    os.makedirs('results/figures', exist_ok=True)
    save_path = 'results/figures/exp_5_4_lr_robustness.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Robustness test completed. Plot saved at: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_experiment_5_4()