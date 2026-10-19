#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
PLAN="${1:?Specify assistance_coverage or stage2}"
case "$PLAN" in
 assistance_coverage) RUN=runs/lmf-assistance-coverage-20261019 ; GPUS=4 ;;
 stage2) RUN=runs/stage2-hard3k-20261019 ; GPUS=2 ;;
 *) exit 64 ;;
esac
export HF_HOME="$PWD/artifacts/hf"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD"
[[ "${SLURM_JOB_PARTITION}" = b200_normal_q ]]
# This requires a separate explicit budget authorization JSON. It never creates one.
AUTH=artifacts/two-plans-20261019/additional_budget_authorization.json
[[ -s "$AUTH" ]]
PY="$PWD/.venv-lmf-a30-ds0169/bin/python"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-two-plan-train-${SLURM_JOB_ID}"
"$PY" -c 'import os,sys,torch; n=int(sys.argv[1]); assert int(os.environ["SLURM_GPUS_ON_NODE"])==n; assert torch.cuda.device_count()==n; assert all("B200" in torch.cuda.get_device_name(i) for i in range(n))' "$GPUS"
"$PY" -m reasoning.two_plan_execution_20261019 --plan "$PLAN" --register-train --authorization "$AUTH"
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPUS" scripts/train_memory_ce.py "$RUN/effective_config.yaml"
"$PY" -m reasoning.two_plan_execution_20261019 --plan "$PLAN" --verify-final
# No automatic requeue, continuation or evaluation submission.
