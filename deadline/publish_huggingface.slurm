#!/bin/bash
#SBATCH --job-name=sft-publish-hf
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/publish-hf-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/deadline/logs/publish-hf-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PATH="$PWD/.venv/bin:/home/manoj663/.conda/envs/reasoning/bin:$PATH"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_ETAG_TIMEOUT=45 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DEBUG=0 HF_HUB_VERBOSITY=error
DEADLINE_PYTHON="$PWD/.venv/bin/python"
"$DEADLINE_PYTHON" -m deadline.pipeline guard-start --reserve-hours 2
"$DEADLINE_PYTHON" -m deadline.publish_huggingface --expected-owner adversarialbandit --timeout-seconds 5400
