$ErrorActionPreference = "Stop"

python .\chapter6_rerun_llama\exp_llama_sst2_multiseed.py --mode offset --lr 1e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_llama\exp_llama_sst2_multiseed.py --mode standard --lr 1e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_llama\exp_llama_sst2_multiseed.py --mode offset --lr 5e-4 --seed 1 --max-steps 40
python .\chapter6_rerun_llama\exp_llama_sst2_multiseed.py --mode standard --lr 5e-4 --seed 1 --max-steps 40

python .\chapter6_rerun_llama\aggregate_runs.py

