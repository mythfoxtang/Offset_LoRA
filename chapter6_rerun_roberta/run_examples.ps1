$ErrorActionPreference = "Stop"

python .\chapter6_rerun_roberta\exp_roberta_mrpc_multiseed.py --mode offset --lr 1e-3 --seed 1
python .\chapter6_rerun_roberta\exp_roberta_mrpc_multiseed.py --mode standard --lr 1e-3 --seed 1
python .\chapter6_rerun_roberta\exp_roberta_mrpc_multiseed.py --mode offset --lr 5e-4 --seed 1
python .\chapter6_rerun_roberta\exp_roberta_mrpc_multiseed.py --mode standard --lr 5e-4 --seed 1

python .\chapter6_rerun_roberta\exp_roberta_sst2_manual.py --mode offset --lr 1e-3 --seed 1 --max-steps 80
python .\chapter6_rerun_roberta\exp_roberta_sst2_manual.py --mode standard --lr 1e-3 --seed 1 --max-steps 80

python .\chapter6_rerun_roberta\aggregate_runs.py

