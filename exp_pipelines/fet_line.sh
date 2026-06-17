#!/usr/bin/env bash
# FET line (autonomous): train FETLoraClassifier 90/10 -> full-recipe inference
# (multi-scale 448/512/576 + balance 1.0, the proven winning inference) -> a
# leaderboard candidate to A/B vs soup_uniform=77.69. Isolated in outputs_fet_c448/.
# batch=4x8=32 fits 8GB (smoke: 6.2GB peak). Reuses outputs/cache + our denoising.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe

echo "[fet] === 1. train FET 90/10 seed42 6ep (batch 4x8=32, 448) ==="
$PY -u finetune_fet.py \
  --work-dir outputs_fet_c448 --cache-dir outputs/cache --img-size 448 \
  --epochs 6 --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
  --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --pfi-weight 0.5 --pfi-classes 4 --pfi-images 8 \
  --local-depth 2 --num-parts 8 --part-channels 16 --local-scale 0.5 --gaussian-ksize 15 \
  --num-workers 2 --no-pin --save-every 1 --snapshot-after 3 > exp_pipelines/fet_c448_train.log 2>&1
echo "[fet] train rc=$?  best mid: $(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/fet_c448_train.log | sort -t= -k2 -rn | head -1)"

if [ ! -f outputs_fet_c448/lora/best.pt ]; then echo "[fet] NO best.pt -- training failed"; exit 1; fi

echo "[fet] === 2. full-recipe inference (448/512/576 + balance 1.0) on best.pt ==="
$PY -u tools/tta_predict_fet.py --work-dir outputs_fet_c448 --checkpoint best \
  --out-prefix submissions/pred_results_fet_c448 --scales 448,512,576 --balance-strength 1.0 \
  --batch-size 96 --num-workers 2 --no-pin >> exp_pipelines/fet_c448_train.log 2>&1

echo "[fet] === 3. validate submission ==="
$PY check_submission.py --csv submissions/pred_results_fet_c448_tta_balanced.csv \
  --zip submissions/pred_results_fet_c448_tta_balanced.zip 2>&1 | grep -E 'RESULT|ERROR'

echo "[fet] === 4. commit+push ==="
git add tools/tta_predict_fet.py exp_pipelines/fet_line.sh exp_pipelines/fet_c448_train.log \
  submissions/pred_results_fet_c448_tta_balanced.zip submissions/pred_results_fet_c448_tta_balanced.csv 2>/dev/null
BEST=$(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/fet_c448_train.log | sort -t= -k2 -rn | head -1)
git commit -q -m "FET line opened: FETLoraClassifier 90/10 + full-recipe inference ($BEST)

CLIP ViT-B/32 frozen + LoRA + LocalBranch(CBAM+part-Transformer) + PFI aux loss.
Reuses our denoising/cache. batch 4x8=32 fits 8GB (6.2GB peak). Full TTA(448/512/576)
+balance1.0 candidate for leaderboard A/B vs soup_uniform=77.69. tools/tta_predict_fet.py
adds FET-compatible multi-scale+balance inference (fair A/B).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -2
echo "[fet] FET LINE DONE -- val best=$BEST ; candidate=pred_results_fet_c448_tta_balanced.zip"
