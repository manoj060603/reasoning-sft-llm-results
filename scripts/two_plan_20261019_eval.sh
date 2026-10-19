#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
PLAN="${1:?Specify assistance_coverage or stage2}"
case "$PLAN" in
 assistance_coverage) RUN=runs/lmf-assistance-coverage-20261019 ;;
 stage2) RUN=runs/stage2-hard3k-20261019 ;;
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
PY="$PWD/.venv-arc/bin/python"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_WORKSPACE_BASE="$PWD/artifacts/flashinfer-runtime"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/reasoning-two-plan-eval-${SLURM_JOB_ID}"
"$PY" -c 'import os,sys,torch; n=int(sys.argv[1]); assert int(os.environ["SLURM_GPUS_ON_NODE"])==n; assert torch.cuda.device_count()==n; assert all("B200" in torch.cuda.get_device_name(i) for i in range(n))' 1
"$PY" -m reasoning.two_plan_execution_20261019 --plan "$PLAN" --register-eval --authorization "$AUTH"
RESULT="results/two-plan-${PLAN}-20261019-suite"
"$PY" -m reasoning.evaluate --model "$RUN" --data-dir artifacts/benchmarks_extended \
 --benchmarks amc aime25 aime math cn_math_2024 kaoyan minerva olympiadbench gpqa \
 --output "$RESULT" --batch-size 1024 --max-num-seqs 128 \
 --max-new-tokens 32768 --record-hardware --gpu-memory-utilization .85
"$PY" -m reasoning.compare --baseline results/full-base-suite-v2 --candidate "$RESULT" \
 --output "${RESULT}-comparison.json"
set +e
"$PY" -m reasoning.domination "${RESULT}-comparison.json" --output "${RESULT}-domination.json"
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then exit "$status"; fi
