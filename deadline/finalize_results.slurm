#!/bin/bash
#SBATCH --job-name=sft-finalize-results
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/finalize-results-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/finalize-results-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/home/manoj663/.conda/envs/reasoning/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=""
DEADLINE_PYTHON="$PWD/.venv/bin/python"
"$DEADLINE_PYTHON" -m deadline.pipeline guard-start --reserve-hours 1
"$DEADLINE_PYTHON" -m deadline.pipeline report
"$DEADLINE_PYTHON" -m deadline.compare_uncertainty \
  --original deadline/results/original \
  --random deadline/results/random \
  --advanced deadline/results/advanced \
  --output deadline/report/uncertainty.json
"$DEADLINE_PYTHON" -m deadline.package_deliverables \
  --root deadline --output deadline/publication-metadata
