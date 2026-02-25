import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# 确保可以导入同级目录下的 model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import OffsetLoRAModel


def run_experiment_5_5():
    # --- 1. 核心参数设置 ---
    d, r, n = 512, 8, 1024
    steps = 150
    lr = 0.01  # 统一学习率以突出收敛速度差异
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Running Exp 5.5: Breaking the Symmetry Barrier ===")
    print(f"Device: {device}")

    # --- 2. 模拟高难度目标数据 ---
    # 模拟一个“远在天边”的目标，强制模型必须快速打破对称性才能收敛
    torch.manual_seed(42)
    X = torch.randn(n, d).to(device)
    # 显著增大目标权重的范数，制造严重的对称性陷阱
    W_target = torch.randn(d, d).to(device) * 5.0
    Y = X @ W_target

    # --- 3. 运行对比实验 ---
    results = {}
    for mode in ['standard', 'offset_orthogonal']:
        print(f"  -> Training mode: {mode}")
        model = OffsetLoRAModel(d, r, mode).to(device)
        optimizer = optim.SGD([model.A, model.B], lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # 结果映射到易读标签
        label = 'Standard LoRA' if mode == 'standard' else 'Offset-LoRA'
        results[label] = losses

    # --- 4. 绘图与分析 ---
    plt.figure(figsize=(10, 6))

    plt.plot(results['Standard LoRA'], label='Standard LoRA (B=0 Initialization)',
             color='royalblue', linewidth=2, linestyle='--')
    plt.plot(results['Offset-LoRA'], label='Offset-LoRA (Proposed)',
             color='crimson', linewidth=3)

    plt.yscale('log')
    plt.title("Convergence Analysis: Breaking the Symmetry Barrier (Exp 5.5)", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("MSE Loss (Log Scale)", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    # 5. 视觉引导标注 (对应你代码中的标注逻辑)
    # 标注出训练初期的“冷启动”延迟与“快速启动”对比
    idx = 10
    plt.annotate('Offset-LoRA: Faster Start',
                 xy=(idx, results['Offset-LoRA'][idx]),
                 xytext=(idx + 20, results['Offset-LoRA'][idx] * 5),
                 arrowprops=dict(facecolor='crimson', shrink=0.05, width=1, headwidth=5))

    plt.annotate('Standard LoRA: Cold-start Delay',
                 xy=(idx, results['Standard LoRA'][idx]),
                 xytext=(idx + 20, results['Standard LoRA'][idx] * 2),
                 arrowprops=dict(facecolor='royalblue', shrink=0.05, width=1, headwidth=5))

    # 自动保存
    os.makedirs('results/figures', exist_ok=True)
    save_path = 'results/figures/exp_5_5_convergence_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Symmetry experiment completed. Plot saved at: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_experiment_5_5()