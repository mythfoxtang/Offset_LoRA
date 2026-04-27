# Offset-LoRA

Offset-LoRA is a LoRA initialization method for improving the early-stage
optimization stability of large language model adaptation.

The core parameterization is:

```math
\Delta W = BA - B_0A_0
```

`A0` and `B0` are fixed offset buffers. `A` and `B` are initialized from the
same values, so the initial effective update is still zero, while the trainable
branch starts from a non-degenerate point.

This repository contains the code, reproduction scripts, rerun utilities, and
experiment artifacts used for the undergraduate thesis:

**Offset-LoRA: Gradient-based Initialization Optimization for Large Language
Model Adaptation**

## What Is Included

```text
src/offset_lora/              Core Offset-LoRA layers and initialization helpers
src/utils/                    Matrix utility functions
simulation/                   Chapter 5 controlled matrix experiments
experiments/                  Original downstream task scripts
scripts/                      Small reproduction scripts used by the thesis
chapter6_rerun_pack/          Current Chapter 6 rerun pack
chapter6_rerun_roberta/       RoBERTa rerun scripts
chapter6_rerun_llama/         Llama rerun scripts
chapter6_replot_code/         Plotting code for Chapter 6 loss curves
results/                      Figures, metrics, raw JSON runs, and summaries
```

The newest code from the thesis working directory is organized under
`chapter6_rerun_pack/`, `chapter6_rerun_roberta/`, `chapter6_rerun_llama/`, and
`chapter6_replot_code/`.

## Environment

Python 3.8+ is recommended.

```bash
pip install -r requirements.txt
```

For Llama 4-bit experiments, make sure the local machine has a compatible CUDA,
PyTorch, and `bitsandbytes` setup.

## Chapter 5 Reproduction

Run the controlled low-rank simulation:

```bash
python simulation/exp_5_2_dynamics.py
```

Run the thesis-side reproduction script:

```bash
python scripts/reproduce_section_5_2.py
```

Main outputs:

```text
results/figures/exp_5_2_dynamics.png
results/metrics/exp_5_2_dynamics_metrics.json
results/figures/section_5_2_reproduction.png
results/metrics/section_5_2_reproduction_metrics.json
```

## Chapter 6 Reruns

The current rerun pack is designed to save structured JSON for every run and
then aggregate paired comparisons between `standard` and `offset` modes.

Run RoBERTa examples:

```powershell
.\chapter6_rerun_pack\roberta\run_roberta_examples.ps1
```

Run Llama examples:

```powershell
.\chapter6_rerun_pack\llama\run_llama_examples.ps1
```

Aggregate all raw runs:

```bash
python chapter6_rerun_pack/aggregate_runs.py
```

Expected summary outputs:

```text
results/chapter6_rerun/raw/*.json
results/chapter6_rerun/summary/paired_runs.csv
results/chapter6_rerun/summary/task_summary.csv
```

The repository already includes the latest raw JSON results and generated
summary CSV files.

## Replotting Chapter 6 Curves

Use `chapter6_replot_code/` to generate early-window comparison plots from loss
curves.

Example:

```bash
python chapter6_replot_code/make_chapter6_plots.py \
  --standard chapter6_replot_code/inputs/mrpc_1e3_standard.json \
  --offset chapter6_replot_code/inputs/mrpc_1e3_offset.json \
  --tag mrpc_1e3 \
  --window 50 \
  --rolling 8
```

Outputs are written to:

```text
chapter6_replot_code/outputs/<tag>/
```

## Main Experiment Results

Current Chapter 6 paired summary:

```text
results/chapter6_rerun/summary/task_summary.csv
```

Current Chapter 6 plotted curves:

```text
chapter6_replot_code/outputs/
```

Current Chapter 5 figures:

```text
results/figures/
```

## Citation

```bibtex
@article{tang2026offsetlora,
  title={Offset-LoRA: Gradient-based Initialization Optimization for Large Language Model Adaptation},
  author={Tang, Zhaoyi},
  school={School of Mathematical Sciences, Fudan University},
  year={2026}
}
```

## License

This project is released under the MIT License.
