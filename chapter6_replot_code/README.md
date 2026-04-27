# Chapter 6 Replot Code

这套代码用于重画论文第六章的曲线图，重点强调初始化带来的前期差异，而不是把全部训练步数都混在一起。

适用场景：
- 你已经在原始实验环境里跑出了 `standard` 和 `offset` 的 loss 序列。
- 你想重画“前 20 到 80 步”的对比图。
- 你想把“全局方差”改成“前期窗口方差”或“滚动标准差”。

## 输入格式

支持三种输入文件：
- `json`：内容是一个数字数组，例如 `[0.71, 0.69, 0.73, ...]`
- `txt`：内容是 Python 列表字符串，例如 `[0.71, 0.69, 0.73, ...]`
- `csv`：单列数字，每行一个 loss

推荐目录结构：

```text
chapter6_replot_code/
  inputs/
    mrpc_1e3_standard.json
    mrpc_1e3_offset.json
    roberta_5e4_standard.json
    roberta_5e4_offset.json
    llama_5e4_standard.json
    llama_5e4_offset.json
  outputs/
```

## 快速开始

先把原实验环境里导出的两条 loss 曲线放进 `inputs/`，然后运行：

```bash
python make_chapter6_plots.py ^
  --standard inputs/mrpc_1e3_standard.json ^
  --offset inputs/mrpc_1e3_offset.json ^
  --tag mrpc_1e3 ^
  --window 50 ^
  --rolling 8
```

输出会写到 `outputs/mrpc_1e3/`，包括：
- `early_compare.png`
- `variance_focus.png`
- `metrics.json`

## 参数说明

- `--window`
  只分析前多少步，默认 `50`
- `--rolling`
  滚动标准差窗口，默认 `8`
- `--tag`
  输出子目录名
- `--standard-label`
  标准方法图例名
- `--offset-label`
  Offset 方法图例名

## 建议用法

- `MRPC 1e-3`
  建议 `--window 40` 或 `50`
- `MRPC 5e-4`
  建议 `--window 60`
- `RoBERTa 1e-3` 稳定性图
  建议 `--window 25` 或 `30`
- `Llama 5e-4` 方差图
  建议 `--window 30` 或 `40`

## 指标定义

`metrics.json` 会给出这些值：
- `peak`
- `peak_step`
- `mean`
- `variance`
- `std`
- `final`
- `drop_from_start`
- `best_in_window`
- `cumulative_gain`

其中：
- `cumulative_gain = sum(standard_loss - offset_loss)`
- 如果这个值为正，表示 Offset 在该窗口内整体更优

## 依赖

```bash
pip install numpy matplotlib
```
