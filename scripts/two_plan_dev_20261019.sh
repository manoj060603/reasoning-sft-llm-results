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
PY="$PWD/.venv-lmf-a30-ds0169/bin/python"
[[ -s artifacts/two-plans-20261019/additional_budget_authorization.json ]]
nvidia-smi --query-gpu=name --format=csv,noheader | "$PY" -c 'import sys; s=sys.stdin.read().splitlines(); assert len(s)==1 and "B200" in s[0]'
exec "$PY" -m reasoning.two_plan_dev_20261019 "--run-$1"
