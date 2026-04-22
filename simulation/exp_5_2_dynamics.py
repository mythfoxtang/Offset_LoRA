import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def orthogonal_matrix(rows, cols, rng):
    raw = rng.standard_normal((rows, cols))
    q, _ = np.linalg.qr(raw.T)
    return q[:, :rows].T


def make_problem(n_samples, d_in, d_out, rank, rng):
    x = rng.standard_normal((n_samples, d_in)) / np.sqrt(d_in)
    u = orthogonal_matrix(rank, d_out, rng).T
    v = orthogonal_matrix(rank, d_in, rng)
    singular_values = np.linspace(1.2, 0.6, rank)
    delta_w_star = u @ np.diag(singular_values) @ v
    y = x @ delta_w_star.T
    return x, y


def step_loss(x, y, delta_w):
    residual = x @ delta_w.T - y
    loss = 0.5 * np.mean(np.sum(residual * residual, axis=1))
    grad_w = residual.T @ x / x.shape[0]
    return loss, grad_w


def run_standard(x, y, rank, steps, lr, init_scale, rng):
    d_out = y.shape[1]
    d_in = x.shape[1]
    a = rng.standard_normal((rank, d_in)) * init_scale
    b = np.zeros((d_out, rank))

    losses = []
    grad_a_norms = []
    grad_b_norms = []

    for _ in range(steps):
        delta_w = b @ a
        loss, grad_w = step_loss(x, y, delta_w)
        grad_b = grad_w @ a.T
        grad_a = b.T @ grad_w

        losses.append(loss)
        grad_a_norms.append(np.linalg.norm(grad_a))
        grad_b_norms.append(np.linalg.norm(grad_b))

        a -= lr * grad_a
        b -= lr * grad_b

    return np.array(losses), np.array(grad_a_norms), np.array(grad_b_norms)


def run_offset(x, y, rank, steps, lr, offset_scale, rng):
    d_out = y.shape[1]
    d_in = x.shape[1]
    a0 = orthogonal_matrix(rank, d_in, rng) * offset_scale
    b0 = orthogonal_matrix(rank, d_out, rng).T * offset_scale
    a = a0.copy()
    b = b0.copy()

    losses = []
    grad_a_norms = []
    grad_b_norms = []

    for _ in range(steps):
        delta_w = b @ a - b0 @ a0
        loss, grad_w = step_loss(x, y, delta_w)
        grad_b = grad_w @ a.T
        grad_a = b.T @ grad_w

        losses.append(loss)
        grad_a_norms.append(np.linalg.norm(grad_a))
        grad_b_norms.append(np.linalg.norm(grad_b))

        a -= lr * grad_a
        b -= lr * grad_b

    return np.array(losses), np.array(grad_a_norms), np.array(grad_b_norms)


def summarize_curve(losses):
    initial = float(losses[0])
    best_10 = float(np.min(losses[:10]))
    best_20 = float(np.min(losses[:20]))
    plateau_step = int(np.argmax(losses <= initial * 0.95))
    if not np.any(losses <= initial * 0.95):
        plateau_step = int(len(losses))
    return {
        "initial_loss": initial,
        "best_10": best_10,
        "best_20": best_20,
        "relative_drop_10": float((initial - best_10) / initial),
        "relative_drop_20": float((initial - best_20) / initial),
        "steps_to_5pct_drop": plateau_step,
    }


def run_experiment_5_2():
    config = {
        "n_samples": 1024,
        "d_in": 96,
        "d_out": 96,
        "rank": 8,
        "steps": 80,
        "lr": 0.9,
        "standard_init_scale": 3e-3,
        "offset_scale": 0.55,
        "num_seeds": 24,
    }
    print("Starting Exp 5.2 controlled low-rank experiment")

    all_standard_losses = []
    all_offset_losses = []
    all_standard_ratio = []
    all_offset_ratio = []

    for seed in range(config["num_seeds"]):
        rng = np.random.default_rng(seed)
        x, y = make_problem(
            config["n_samples"],
            config["d_in"],
            config["d_out"],
            config["rank"],
            rng,
        )
        std_losses, std_grad_a, std_grad_b = run_standard(
            x,
            y,
            config["rank"],
            config["steps"],
            config["lr"],
            config["standard_init_scale"],
            rng,
        )
        off_losses, off_grad_a, off_grad_b = run_offset(
            x,
            y,
            config["rank"],
            config["steps"],
            config["lr"],
            config["offset_scale"],
            rng,
        )
        all_standard_losses.append(std_losses)
        all_offset_losses.append(off_losses)
        all_standard_ratio.append(std_grad_a / (std_grad_b + 1e-12))
        all_offset_ratio.append(off_grad_a / (off_grad_b + 1e-12))

    standard_losses = np.stack(all_standard_losses)
    offset_losses = np.stack(all_offset_losses)
    standard_ratio = np.stack(all_standard_ratio)
    offset_ratio = np.stack(all_offset_ratio)

    metrics = {
        "config": config,
        "standard": summarize_curve(standard_losses.mean(axis=0)),
        "offset_orthogonal": summarize_curve(offset_losses.mean(axis=0)),
        "standard_seedwise_steps_to_5pct_drop_mean": float(
            np.mean([summarize_curve(curve)["steps_to_5pct_drop"] for curve in standard_losses])
        ),
        "offset_orthogonal_seedwise_steps_to_5pct_drop_mean": float(
            np.mean([summarize_curve(curve)["steps_to_5pct_drop"] for curve in offset_losses])
        ),
    }

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    figure_path = "results/figures/exp_5_2_dynamics.png"
    metrics_path = "results/metrics/exp_5_2_dynamics_metrics.json"

    steps = np.arange(config["steps"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)

    std_mean = standard_losses.mean(axis=0)
    std_std = standard_losses.std(axis=0)
    off_mean = offset_losses.mean(axis=0)
    off_std = offset_losses.std(axis=0)

    axes[0].plot(steps, std_mean, label="Standard LoRA", color="#d55e00", linewidth=2.0)
    axes[0].fill_between(steps, std_mean - std_std, std_mean + std_std, color="#d55e00", alpha=0.18)
    axes[0].plot(steps, off_mean, label="Offset-LoRA", color="#0072b2", linewidth=2.0)
    axes[0].fill_between(steps, off_mean - off_std, off_mean + off_std, color="#0072b2", alpha=0.18)
    axes[0].set_title("Controlled Early-Stage Convergence (Exp 5.2)")
    axes[0].set_xlabel("Optimization Step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, standard_ratio.mean(axis=0), label="Standard LoRA", color="#d55e00", linewidth=2.0)
    axes[1].plot(steps, offset_ratio.mean(axis=0), label="Offset-LoRA", color="#0072b2", linewidth=2.0)
    axes[1].axhline(1.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[1].set_title("Gradient-Norm Ratio ||g_A|| / ||g_B||")
    axes[1].set_xlabel("Optimization Step")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Section 5.2 Reproduction with Seed Averaging", fontsize=14)
    fig.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[Success] Figure saved to {figure_path}")
    print(f"[Success] Metrics saved to {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_experiment_5_2()
