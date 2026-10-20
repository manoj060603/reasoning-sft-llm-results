#!/bin/bash
#SBATCH --job-name=sft-transfer-finalize
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/finalize-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/finalize-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD"
.venv/bin/python -m unittest discover -s tests -p 'test_deadline_*.py' -v
.venv/bin/python -m deadline.finalize_preparation
.venv/bin/python -m deadline.pipeline submit-initial
