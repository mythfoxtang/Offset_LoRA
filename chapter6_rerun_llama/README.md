# Chapter 6 Rerun Llama

把整个文件夹复制到 Llama 那台开发机的训练仓库根目录。

这个文件夹是完全独立的，不依赖其他 rerun 包。

## 包含内容

- `exp_llama_sst2_multiseed.py`
- `metrics.py`
- `aggregate_runs.py`
- `run_examples.ps1`

## 先跑什么

```powershell
python .\chapter6_rerun_llama\exp_llama_sst2_multiseed.py --mode offset --lr 1e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_llama\exp_llama_sst2_multiseed.py --mode standard --lr 1e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_llama\exp_llama_sst2_multiseed.py --mode offset --lr 5e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_llama\exp_llama_sst2_multiseed.py --mode standard --lr 5e-4 --seed 1 --max-steps 40
```

然后把 `seed` 改成 `2/3/4` 继续跑。

## 输出目录

```text
results/chapter6_rerun_llama/raw
results/chapter6_rerun_llama/summary
```

