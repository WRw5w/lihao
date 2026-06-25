#!/usr/bin/env bash
# Shared config for the position-embedding resampling comparison.
# Edit the values below for your server, or override any inline, e.g.:
#     PY=python3.11 DATA_DIR=/data/jinyinsai BATCH defaults are per-arm bash run_all.sh
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PKG_ROOT
export CODE="$PKG_ROOT/code"

# ---- EDIT THESE FOR YOUR SERVER ----
export PY="${PY:-python}"                          # a python with torch + timm + CUDA
export DATA_DIR="${DATA_DIR:-$PKG_ROOT/data}"      # must contain train/<class>/*.jpg and test/*.jpg
export OUT_ROOT="${OUT_ROOT:-$PKG_ROOT/runs}"      # checkpoints + logs + submissions land here
export CACHE_DIR="${CACHE_DIR:-$PKG_ROOT/cache}"   # bundled 224 feature cache (resolution-independent)
export EPOCHS="${EPOCHS:-12}"                       # winning recipe = 12ep trajectory
export SWA_START="${SWA_START:-4}"                  # same-trajectory SWA averages ep<SWA_START>..ep<EPOCHS>
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SEED="${SEED:-42}"
# Keep the CLIP weight download self-contained inside the package, so a SLURM
# compute node (often no internet) can read what the login node prefetched.
export HF_HOME="${HF_HOME:-$PKG_ROOT/hf_cache}"
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1    # set by 00_prefetch / slurm once weights are cached

# Frozen winning recipe (cleanlab denoise + mixup0.2 + EMA + RandAug + rank32 attn_mlp,
# trained on 100% data with per-epoch snapshots for SWA). The ONLY things that vary
# across arms are --img-size / --pos-resample / --batch-size (set per-arm in arms.tsv).
export COMMON="--full --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
--keep-ratio 0.90 --ema-decay 0.999 --randaug --pseudo-thresh 0.6 --pseudo-margin 0.05 \
--label-smoothing 0.1 --denoise cleanlab --mixup-alpha 0.2 --save-every 1 --no-pin"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/submissions"
