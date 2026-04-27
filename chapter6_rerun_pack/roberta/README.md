# RoBERTa Group

This group contains the RoBERTa-side rerun plan for Chapter 6.

## Scripts

- `..\exp_roberta_mrpc_multiseed.py`
- `..\exp_roberta_sst2_manual.py`

## Priority

Run `MRPC` first. That is the part currently least well supported by the old figures.

Recommended first batch:

```powershell
python .\chapter6_rerun_pack\exp_roberta_mrpc_multiseed.py --mode offset --lr 1e-3 --seed 1
python .\chapter6_rerun_pack\exp_roberta_mrpc_multiseed.py --mode standard --lr 1e-3 --seed 1
python .\chapter6_rerun_pack\exp_roberta_mrpc_multiseed.py --mode offset --lr 5e-4 --seed 1
python .\chapter6_rerun_pack\exp_roberta_mrpc_multiseed.py --mode standard --lr 5e-4 --seed 1
```

Then repeat with `--seed 2`, `--seed 3`, `--seed 4`.
