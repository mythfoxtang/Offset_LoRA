import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from curve_io import load_curve


def rolling_std(values, window):
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.std(arr[start : i + 1])
    return out


def summarize(curve):
    arr = np.asarray(curve, dtype=float)
    peak_step = int(np.argmax(arr))
    return {
        "peak": float(np.max(arr)),
        "peak_step": peak_step,
        "mean": float(np.mean(arr)),
        "variance": float(np.var(arr)),
        "std": float(np.std(arr)),
        "final": float(arr[-1]),
        "drop_from_start": float(arr[0] - arr[-1]),
        "best_in_window": float(np.min(arr)),
    }


def save_metrics(standard, offset, output_dir):
    std_metrics = summarize(standard)
    off_metrics = summarize(offset)
    metrics = {
        "standard": std_metrics,
        "offset": off_metrics,
        "delta": {
            "peak_gap": std_metrics["peak"] - off_metrics["peak"],
            "variance_gap": std_metrics["variance"] - off_metrics["variance"],
            "cumulative_gain": float(np.sum(np.asarray(standard) - np.asarray(offset))),
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_early_compare(steps, standard, offset, std_label, off_label, output_dir, tag):
    plt.figure(figsize=(10, 5.6), dpi=180)
    plt.plot(steps, standard, color="#d55e00", linewidth=2.0, label=std_label)
    plt.plot(steps, offset, color="#0072b2", linewidth=2.0, label=off_label)
    plt.scatter(int(np.argmax(standard)), float(np.max(standard)), color="#d55e00", s=22)
    plt.scatter(int(np.argmax(offset)), float(np.max(offset)), color="#0072b2", s=22)
    plt.title(f"Early-Stage Loss Comparison: {tag}")
    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "early_compare.png", bbox_inches="tight")
    plt.close()


def plot_variance_focus(steps, standard, offset, rolling_window, std_label, off_label, output_dir, tag):
    std_roll = rolling_std(standard, rolling_window)
    off_roll = rolling_std(offset, rolling_window)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=180, gridspec_kw={"width_ratios": [3.2, 1.2]})

    axes[0].plot(steps, standard, color="#d55e00", linewidth=1.8, label=std_label)
    axes[0].plot(steps, offset, color="#0072b2", linewidth=1.8, label=off_label)
    axes[0].fill_between(steps, standard - std_roll, standard + std_roll, color="#d55e00", alpha=0.18)
    axes[0].fill_between(steps, offset - off_roll, offset + off_roll, color="#0072b2", alpha=0.18)
    axes[0].set_title(f"Early-Window Stability: {tag}")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    variances = [float(np.var(offset)), float(np.var(standard))]
    bars = axes[1].bar(["Offset", "Standard"], variances, color=["#4f8bd6", "#e05a4f"])
    axes[1].set_title("Window Variance")
    axes[1].set_ylabel("Variance")
    for bar, value in zip(bars, variances):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4g}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "variance_focus.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", required=True, help="Path to standard curve file")
    parser.add_argument("--offset", required=True, help="Path to offset curve file")
    parser.add_argument("--tag", required=True, help="Output tag")
    parser.add_argument("--window", type=int, default=50, help="Only analyze the first N steps")
    parser.add_argument("--rolling", type=int, default=8, help="Rolling std window")
    parser.add_argument("--standard-label", default="Standard LoRA")
    parser.add_argument("--offset-label", default="Offset-LoRA")
    args = parser.parse_args()

    standard = load_curve(args.standard)
    offset = load_curve(args.offset)

    window = min(args.window, len(standard), len(offset))
    standard = np.asarray(standard[:window], dtype=float)
    offset = np.asarray(offset[:window], dtype=float)
    steps = np.arange(window)

    output_dir = Path("outputs") / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_early_compare(steps, standard, offset, args.standard_label, args.offset_label, output_dir, args.tag)
    plot_variance_focus(steps, standard, offset, args.rolling, args.standard_label, args.offset_label, output_dir, args.tag)
    save_metrics(standard, offset, output_dir)

    print(f"Saved plots and metrics to: {output_dir}")


if __name__ == "__main__":
    main()
