#!/bin/bash
#SBATCH --job-name=sft-publication-check
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/publication-check-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/publication-check-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
DEADLINE_PYTHON="$PWD/.venv/bin/python"
"$DEADLINE_PYTHON" -c 'from importlib.metadata import version; print("huggingface-hub", version("huggingface-hub"))'
"$DEADLINE_PYTHON" -m unittest discover -s tests -p 'test_deadline_package.py' -v
"$DEADLINE_PYTHON" -m unittest discover -s tests -p 'test_deadline_publish.py' -v
