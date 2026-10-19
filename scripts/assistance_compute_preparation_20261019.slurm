#!/usr/bin/env bash
#SBATCH --job-name=reasoning_assistance_cpu_prep
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/assistance_compute_preparation_%j.log
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_PARTITION:-}" = normal_q ]]
case "$(hostname -s)" in owl1|owl2|owl3) echo 'Refusing computation on login node' >&2; exit 64 ;; esac
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
export HF_HOME="$PWD/artifacts/hf"
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD"
PY="$PWD/.venv-lmf-a30-ds0169/bin/python"
printf 'CPU_PREPARATION_COMPUTE_JOB=%s HOST=%s\n' "$SLURM_JOB_ID" "$(hostname)"
OUT=artifacts/datasets-assistance-coverage-20261019
# Keep the interrupted login-node attempt; never silently adopt partial output.
if [[ -d "$OUT" ]]; then
 [[ ! -e "$OUT/selection_manifest.json" && ! -e "$OUT/advanced.json" ]]
 mv "$OUT" "${OUT}-interrupted-login-before-${SLURM_JOB_ID}"
fi
"$PY" research/assistance_coverage_20261019.py \
 --saved-plan artifacts/demonstration-response/plan.json \
 --ledger artifacts/demonstration-response/generations.jsonl \
 --negative-receipt artifacts/demonstration-response/negative_preparation_receipt.json \
 --candidate-a artifacts/datasets-full-length-15k/selection_manifest.json \
 --output "$OUT"
"$PY" -m reasoning.two_plan_execution_20261019 --plan assistance_coverage --prepare
# The already-authorized single GPU job is submitted only after complete audits.
# Both validation/hashing and guarded admission run here on a CPU compute node.
"$PY" -m reasoning.two_plan_execution_20261019 --plan assistance_coverage --submit-train
