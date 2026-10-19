#!/usr/bin/env bash
#SBATCH --job-name=reasoning_two_plan_train
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/two_plan_train_%j.log
set -euo pipefail
# Positional plan must be assistance_coverage or stage2. No budget authorization is bundled.
exec bash /projects/tealab/rohit/reasoning-sft-20261011/scripts/two_plan_20261019_train.sh "$@"
