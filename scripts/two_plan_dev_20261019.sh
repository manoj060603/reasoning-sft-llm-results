#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
[[ "$#" -eq 1 && ( "$1" = parent || "$1" = stage2 ) ]]
[[ "${SLURM_JOB_PARTITION}" = b200_normal_q ]]
export HF_HOME="$PWD/artifacts/hf" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD"
PY="$PWD/.venv-arc/bin/python"
[[ -s artifacts/two-plans-20261019/additional_budget_authorization.json ]]
"$PY" -c 'import os,torch; assert int(os.environ["SLURM_GPUS_ON_NODE"]) == 1; assert torch.cuda.device_count() == 1; assert "B200" in torch.cuda.get_device_name(0)'
exec "$PY" -m reasoning.two_plan_dev_20261019 "--run-$1"
