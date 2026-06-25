#!/bin/bash
# ============================================================================
# SIMPLE SLURM job: run the 4 core arms one after another on ONE GPU.
# Submit from inside this folder:   sbatch submit.slurm
# Watch:   squeue -u $USER       tail -f runs/slurm-<jobid>.out
# ============================================================================
#SBATCH --job-name=posembed
#SBATCH --output=runs/slurm-%j.out          # combined stdout+stderr
#SBATCH --gres=gpu:1                         # 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00                      # 4 arms sequential; raise if needed
#SBATCH --partition=gpu                      # <<< CHANGE to your cluster's GPU partition
## #SBATCH --account=YOUR_ACCOUNT            # <<< uncomment + set if your cluster requires it
## #SBATCH --qos=YOUR_QOS                    # <<< uncomment if required

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                        # the dir you ran sbatch from
mkdir -p runs

# ---- cluster environment: edit ONE of these to get python+torch+timm+CUDA ----
# module load cuda/12.1 anaconda3 ; source activate posembed
# source ~/miniconda3/bin/activate posembed
# export PY=/path/to/venv/bin/python
export PY="${PY:-python}"

# ---- paths ----
# Dataset is BUNDLED in ./data (train/ + test/); env.sh defaults DATA_DIR there,
# so nothing to set. To use data elsewhere instead, uncomment:
# export DATA_DIR=/path/to/jinyinsai_data
export NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1             # weights prefetched on login node

echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES data=$DATA_DIR"
bash 00_smoke.sh                              # fail fast before the long runs
bash run_all.sh core                          # -> runs/submissions/*_tta_balanced.zip
