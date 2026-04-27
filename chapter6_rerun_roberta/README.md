# Chapter 6 Rerun RoBERTa

把整个文件夹复制到 RoBERTa 那台开发机的训练仓库根目录。

这个文件夹是完全独立的，不依赖其他 rerun 包。

## 包含内容

- `exp_roberta_mrpc_multiseed.py`
- `exp_roberta_sst2_manual.py`
- `metrics.py`
- `aggregate_runs.py`
- `run_examples.ps1`

## 先跑什么

先跑 `MRPC`，因为论文里最需要补证据的是这部分。

```powershell
python .\chapter6_rerun_roberta\exp_roberta_mrpc_multiseed.py --mode offset --lr 1e-3 --seed 1
python .\chapter6_rerun_roberta\exp_roberta_mrpc_multiseed.py --mode standard --lr 1e-3 --seed 1
python .\chapter6_rerun_roberta\exp_roberta_mrpc_multiseed.py --mode offset --lr 5e-4 --seed 1
python .\chapter6_rerun_roberta\exp_roberta_mrpc_multiseed.py --mode standard --lr 5e-4 --seed 1
```

然后把 `seed` 改成 `2/3/4` 继续跑。

## 输出目录

```text
results/chapter6_rerun_roberta/raw
results/chapter6_rerun_roberta/summary
```

