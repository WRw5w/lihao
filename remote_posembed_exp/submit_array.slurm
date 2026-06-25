#!/bin/bash
# ============================================================================
# JOB ARRAY: run each arm on its OWN GPU in parallel (the efficient way).
# Each array task picks the (TASK_ID+1)-th data line of arms.tsv.
#   arms.tsv data-line order: 0=b224_native 1=b448_default 2=r416_default
#                             3=r416_aligned 4=b448_aligned 5=r608_aligned
# Submit the 4 core arms:        sbatch --array=0-3 submit_array.slurm
# All six arms:                  sbatch --array=0-5 submit_array.slurm
# Just the 2 stretch arms:       sbatch --array=4-5 submit_array.slurm
# Limit to 2 GPUs at a time:     sbatch --array=0-3%2 submit_array.slurm
# ============================================================================
#SBATCH --job-name=posembed
#SBATCH --output=runs/slurm-%A_%a.out        # %A=array job id, %a=task id
#SBATCH --gres=gpu:1                          # 1 GPU PER ARM
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00                       # PER ARM (r608 is heaviest)
#SBATCH --partition=gpu                       # <<< CHANGE to your GPU partition
## #SBATCH --account=YOUR_ACCOUNT             # <<< uncomment + set if required

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

# ---- cluster environment (same as submit.slurm) ----
# module load cuda/12.1 anaconda3 ; source activate posembed
export PY="${PY:-python}"
# Dataset is BUNDLED in ./data; env.sh defaults DATA_DIR there. To relocate:
# export DATA_DIR=/path/to/jinyinsai_data
export NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# pick this task's arm = the (TASK_ID+1)-th non-comment, non-blank line of arms.tsv
line=$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' arms.tsv | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r name img pos bs scales tier _rest <<< "$line"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID arm=$name img=$img pos=$pos bs=$bs"

bash run_arm.sh "$name" "$img" "$pos" "$bs" "$scales"
