#!/usr/bin/env bash
set -euo pipefail
cd /projects/tealab/rohit/reasoning-sft-20261011
export HF_HOME="$PWD/artifacts/hf" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_HOME=/apps/common/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/home/rohits25/.conda/envs/sata-vllm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/LLaMA-Factory/src:$PWD"
PY="$PWD/.venv-lmf-a30-ds0169/bin/python"
[[ "${SLURM_JOB_PARTITION}" = b200_normal_q ]]
[[ -s artifacts/two-plans-20261019/additional_budget_authorization.json ]]
"$PY" -c 'import os,sys,torch; n=int(sys.argv[1]); assert int(os.environ["SLURM_GPUS_ON_NODE"])==n; assert torch.cuda.device_count()==n; assert all("B200" in torch.cuda.get_device_name(i) for i in range(n))' 1
"$PY" -m reasoning.stage2_convergence_20261019 --run
