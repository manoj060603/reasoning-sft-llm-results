#!/bin/bash
#SBATCH --job-name=sft-transfer-prepare
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/prepare-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/prepare-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/deadline/hf" PYTHONPATH="$PWD"
TRAIN_PYTHON=/home/manoj663/.conda/envs/reasoning/bin/python
if [[ ! -x .venv/bin/python ]]; then
  "$TRAIN_PYTHON" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --no-deps latex2sympy2==1.9.1 antlr4-python3-runtime==4.7.2 word2number==1.1 Pebble==5.1.1 timeout-decorator==0.5.0 jsonlines==4.0.0
.venv/bin/python -m reasoning.grader_preflight
.venv/bin/python -m deadline.prepare_transfer --model /home/manoj663/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
