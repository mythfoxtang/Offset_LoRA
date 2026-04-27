import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_runs(raw_dir):
    runs = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def build_pairs(runs):
    buckets = defaultdict(dict)
    for run in runs:
        key = (run["task"], run["lr"], run["seed"])
        buckets[key][run["mode"]] = run
    return buckets


def compare_pair(offset_run, standard_run):
    off = offset_run["metrics"]
    std = standard_run["metrics"]
    return {
        "task": offset_run["task"],
        "lr": offset_run["lr"],
        "seed": offset_run["seed"],
        "offset_early_peak": off["early_peak"],
        "standard_early_peak": std["early_peak"],
        "offset_early_variance": off["early_variance"],
        "standard_early_variance": std["early_variance"],
        "offset_drop_from_start": off["drop_from_start"],
        "standard_drop_from_start": std["drop_from_start"],
        "offset_first_below_95pct": off["first_below_95pct"],
        "standard_first_below_95pct": std["first_below_95pct"],
        "offset_wins_peak": off["early_peak"] < std["early_peak"],
        "offset_wins_variance": off["early_variance"] < std["early_variance"],
        "offset_wins_drop": off["drop_from_start"] > std["drop_from_start"],
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="results/chapter6_rerun_roberta/raw")
    parser.add_argument("--out-dir", default="results/chapter6_rerun_roberta/summary")
    args = parser.parse_args()

    runs = load_runs(args.raw_dir)
    pairs = build_pairs(runs)
    paired_rows = []
    task_summary = defaultdict(lambda: {"count": 0, "peak_wins": 0, "variance_wins": 0, "drop_wins": 0})

    for _, pair in sorted(pairs.items()):
        if "offset" not in pair or "standard" not in pair:
            continue
        row = compare_pair(pair["offset"], pair["standard"])
        paired_rows.append(row)
        task_key = f"{row['task']}_lr{row['lr']}"
        task_summary[task_key]["count"] += 1
        task_summary[task_key]["peak_wins"] += int(row["offset_wins_peak"])
        task_summary[task_key]["variance_wins"] += int(row["offset_wins_variance"])
        task_summary[task_key]["drop_wins"] += int(row["offset_wins_drop"])

    summary_rows = []
    for task_key, stats in sorted(task_summary.items()):
        count = stats["count"]
        summary_rows.append(
            {
                "task_lr": task_key,
                "num_pairs": count,
                "peak_win_rate": stats["peak_wins"] / count if count else 0.0,
                "variance_win_rate": stats["variance_wins"] / count if count else 0.0,
                "drop_win_rate": stats["drop_wins"] / count if count else 0.0,
            }
        )

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "paired_runs.csv", paired_rows)
    write_csv(out_dir / "task_summary.csv", summary_rows)
    print(f"saved: {out_dir / 'paired_runs.csv'}")
    print(f"saved: {out_dir / 'task_summary.csv'}")


if __name__ == "__main__":
    main()

