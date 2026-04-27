import json
from pathlib import Path
from statistics import mean, pstdev, pvariance


def first_step_below(values, threshold):
    for idx, value in enumerate(values):
        if value <= threshold:
            return idx
    return None


def rolling_std(values, window):
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(pstdev(chunk) if len(chunk) > 1 else 0.0)
    return out


def summarize_losses(losses, early_window=40, rolling_window=8):
    if not losses:
        raise ValueError("losses must not be empty")

    early = losses[: min(early_window, len(losses))]
    peak = max(losses)
    peak_step = losses.index(peak)
    early_peak = max(early)
    early_peak_step = early.index(early_peak)
    start_loss = losses[0]
    final_loss = losses[-1]
    early_best = min(early)
    early_best_step = early.index(early_best)
    roll = rolling_std(early, rolling_window)

    return {
        "num_points": len(losses),
        "start_loss": start_loss,
        "final_loss": final_loss,
        "peak": peak,
        "peak_step": peak_step,
        "early_window": len(early),
        "early_peak": early_peak,
        "early_peak_step": early_peak_step,
        "early_best": early_best,
        "early_best_step": early_best_step,
        "mean": mean(losses),
        "variance": pvariance(losses) if len(losses) > 1 else 0.0,
        "std": pstdev(losses) if len(losses) > 1 else 0.0,
        "early_mean": mean(early),
        "early_variance": pvariance(early) if len(early) > 1 else 0.0,
        "early_std": pstdev(early) if len(early) > 1 else 0.0,
        "drop_from_start": start_loss - final_loss,
        "first_below_99pct": first_step_below(losses, start_loss * 0.99),
        "first_below_95pct": first_step_below(losses, start_loss * 0.95),
        "first_below_90pct": first_step_below(losses, start_loss * 0.90),
        "rolling_std_window": rolling_window,
        "rolling_std_early": roll,
        "max_rolling_std_early": max(roll) if roll else 0.0,
    }


def safe_tag(value):
    text = str(value)
    return text.replace(".", "p").replace("-", "m")


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

