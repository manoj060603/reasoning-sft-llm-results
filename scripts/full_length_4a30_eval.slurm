#!/usr/bin/env bash
#SBATCH --job-name=reasoning_4a30_final_eval
#SBATCH --account=tml_2026
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/full_length_4a30_eval_%j.log
set -euo pipefail
exec bash /projects/tealab/rohit/reasoning-sft-20261011/scripts/arc_full_length_4a30_eval.sh
