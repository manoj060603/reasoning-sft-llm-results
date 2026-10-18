#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-early-b200-${SLURM_JOB_ID}"
PY="$PWD/.venv-lmf-a30-ds0169/bin/python"
[[ "${SLURM_JOB_PARTITION}" = b200_normal_q || "${SLURM_JOB_PARTITION}" = b200_preemptable_q ]]
nvidia-smi --query-gpu=name --format=csv,noheader | "$PY" -c 'import sys; assert "B200" in sys.stdin.read()'
"$PY" -m reasoning.early_b200 --register
RUN=runs/lmf-full-length-15k-16k-3epoch-early-b200
"$PY" scripts/train_memory_ce.py "$RUN/effective_config.yaml"
"$PY" -m reasoning.early_b200 --verify-final
# Training only. The unchanged single-B200 evaluation is a separate job.
