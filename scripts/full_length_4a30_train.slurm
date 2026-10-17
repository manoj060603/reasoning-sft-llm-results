#!/usr/bin/env bash
#SBATCH --job-name=reasoning_full_length_4a30
#SBATCH --account=tml_2026
#SBATCH --partition=a30_preemptable_q
#SBATCH --qos=fal_a30_preemptable_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/full_length_4a30_%j.log
set -euo pipefail
exec bash /projects/tealab/rohit/reasoning-sft-20261011/scripts/arc_full_length_4a30_train.sh
