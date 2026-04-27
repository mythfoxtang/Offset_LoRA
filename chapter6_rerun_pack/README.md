# Chapter 6 Rerun Pack

This folder is meant to be copied into the training repo root:

```powershell
C:\Users\Administrator\Desktop\本科内容\毕设\Offset_LoRA 代码
```

The point is not to rerun the old scripts unchanged. The point is to produce stronger evidence for the thesis:

- save every run as structured JSON
- support explicit `mode`, `lr`, and `seed`
- measure early-window peak and variance directly
- compare offset vs standard in paired runs

## Groups

- `roberta/`
  RoBERTa-related run notes and example commands.

- `llama/`
  Llama-related run notes and example commands.

## Shared Files

- `exp_roberta_mrpc_multiseed.py`
  Manual per-step MRPC runner. This is the most important script for Chapter 6.

- `exp_roberta_sst2_manual.py`
  Manual SST2 runner for RoBERTa. Use this as support evidence.

- `exp_llama_sst2_multiseed.py`
  Llama SST2 runner with 4-bit loading and structured output.

- `metrics.py`
  Shared metrics logic.

- `aggregate_runs.py`
  Merges all raw JSON files and writes summary CSVs.

## Why this is better than the old scripts

The old experiment scripts have three problems:

1. They hardcode `mode` and `lr`.
2. They mostly print raw losses to the console instead of saving structured data.
3. They do not support multi-seed paired comparison, so it is hard to argue robustness.

This rerun pack fixes all three.

## Recommended order

1. Run the `roberta` group first.
2. Run the `llama` group separately.
3. After each group, run `aggregate_runs.py`.

## Summary step

After all runs finish:

```powershell
python .\chapter6_rerun_pack\aggregate_runs.py
```

Outputs:

- `results/chapter6_rerun/summary/paired_runs.csv`
- `results/chapter6_rerun/summary/task_summary.csv`

## What to send back

Bring back:

- the `results/chapter6_rerun/raw/*.json` files
- the `results/chapter6_rerun/summary/*.csv` files

With those files, the Chapter 6 wording can be rewritten around paired win rates instead of a single lucky curve.
