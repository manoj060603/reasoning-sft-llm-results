#!/usr/bin/env bash
#SBATCH --job-name=reasoning_two_plan_dev
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/two_plan_dev_%j.log
set -euo pipefail
[[ "$#" -eq 1 && ( "$1" = parent || "$1" = stage2 ) ]]
exec bash /projects/tealab/rohit/reasoning-sft-20261011/scripts/two_plan_dev_20261019.sh "$1"
