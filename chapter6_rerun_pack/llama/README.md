# Llama Group

This group contains the Llama-side rerun plan for Chapter 6.

## Script

- `..\exp_llama_sst2_multiseed.py`

Recommended first batch:

```powershell
python .\chapter6_rerun_pack\exp_llama_sst2_multiseed.py --mode offset --lr 1e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_pack\exp_llama_sst2_multiseed.py --mode standard --lr 1e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_pack\exp_llama_sst2_multiseed.py --mode offset --lr 5e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_pack\exp_llama_sst2_multiseed.py --mode standard --lr 5e-4 --seed 1 --max-steps 40
```

Then repeat with more seeds if the machine budget allows it.
