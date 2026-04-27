import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize_values(values):
    arr = np.asarray(values, dtype=float)
    return {
        "peak": float(np.max(arr)),
        "peak_step": int(np.argmax(arr)),
        "variance": float(np.var(arr)),
        "std": float(np.std(arr)),
        "drop_from_start": float(arr[0] - arr[-1]),
        "final_loss": float(arr[-1]),
    }


def rolling_std(values, window):
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.std(arr[start : i + 1])
    return out


def load_losses(path_str):
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    return np.asarray(payload["losses"], dtype=float), payload


def save_metrics(output_dir, standard_values, offset_values):
    standard = summarize_values(standard_values)
    offset = summarize_values(offset_values)
    metrics = {
        "standard": standard,
        "offset": offset,
        "delta": {
            "peak_gap": standard["peak"] - offset["peak"],
            "variance_gap": standard["variance"] - offset["variance"],
            "drop_gap": offset["drop_from_start"] - standard["drop_from_start"],
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", required=True)
    parser.add_argument("--offset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--rolling", type=int, default=8)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--standard-label", default="Standard LoRA")
    parser.add_argument("--offset-label", default="Offset-LoRA")
    args = parser.parse_args()

    standard, std_payload = load_losses(args.standard)
    offset, off_payload = load_losses(args.offset)
    window = min(args.window, len(standard), len(offset))
    standard = standard[:window]
    offset = offset[:window]
    steps = np.arange(window)

    output_dir = Path(args.output_root) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5.6), dpi=180)
    plt.plot(steps, standard, color="#d55e00", linewidth=2.0, label=args.standard_label)
    plt.plot(steps, offset, color="#0072b2", linewidth=2.0, label=args.offset_label)
    plt.scatter(int(np.argmax(standard)), float(np.max(standard)), color="#d55e00", s=24)
    plt.scatter(int(np.argmax(offset)), float(np.max(offset)), color="#0072b2", s=24)
    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.title(args.tag)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "early_compare.png", bbox_inches="tight")
    plt.close()

    std_roll = rolling_std(standard, args.rolling)
    off_roll = rolling_std(offset, args.rolling)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=180, gridspec_kw={"width_ratios": [3.2, 1.2]})
    axes[0].plot(steps, standard, color="#d55e00", linewidth=1.8, label=args.standard_label)
    axes[0].plot(steps, offset, color="#0072b2", linewidth=1.8, label=args.offset_label)
    axes[0].fill_between(steps, standard - std_roll, standard + std_roll, color="#d55e00", alpha=0.18)
    axes[0].fill_between(steps, offset - off_roll, offset + off_roll, color="#0072b2", alpha=0.18)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    variances = [float(np.var(offset)), float(np.var(standard))]
    bars = axes[1].bar(["Offset", "Standard"], variances, color=["#4f8bd6", "#e05a4f"])
    axes[1].set_title("Early Variance")
    axes[1].set_ylabel("Variance")
    for bar, value in zip(bars, variances):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4g}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "variance_focus.png", bbox_inches="tight")
    plt.close(fig)

    save_metrics(output_dir, standard, offset)


if __name__ == "__main__":
    main()
