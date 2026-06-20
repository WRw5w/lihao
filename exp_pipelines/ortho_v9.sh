#!/usr/bin/env bash
# Wave 9 -- escalate the WINNER clmix(=cleanlab+mixup=77.13) with a LONG-trajectory
# within-run soup (the champion's +1.6 came from many same-recipe trajectory points,
# not 3 epochs). Train clmix 12 epochs saving every epoch, then SWA-soup the post-
# warmup trajectory (ep06-12). 跨seed已死 -> 只在这一条轨迹内平均. -> 冲破 77.73.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v9.log; : > "$MASTER"

echo "[clmix_long] TRAIN 12ep save-every1 (cleanlab+mixup0.2)" | tee -a "$MASTER"
$PY -u finetune_lora.py --work-dir outputs_ortho_clmix_long --cache-dir outputs/cache \
  --img-size 448 --epochs 12 --batch-size 32 --lora-rank 32 --lora-alpha 64 \
  --lora-target attn_mlp --lora-blocks 12 --keep-ratio 0.90 --ema-decay 0.999 --randaug \
  --pseudo-thresh 0.6 --pseudo-margin 0.05 --label-smoothing 0.1 --num-workers 2 --no-pin \
  --denoise cleanlab --mixup-alpha 0.2 --save-every 1 > exp_pipelines/ortho_clmix_long.log 2>&1
echo "[clmix_long] best $(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/ortho_clmix_long.log|sort -t= -k2 -rn|head -1)" | tee -a "$MASTER"
ls outputs_ortho_clmix_long/lora/ep12.pt >/dev/null 2>&1 || { echo "[clmix_long] training incomplete"; tail -8 exp_pipelines/ortho_clmix_long.log | tee -a "$MASTER"; exit 1; }

echo "[clmixsoup2] === long-trajectory SWA (ep06-12) ===" | tee -a "$MASTER"
$PY tools/swa_soup.py --out outputs_ortho_clmixsoup2/lora/full.pt --checkpoints \
  outputs_ortho_clmix_long/lora/ep06.pt outputs_ortho_clmix_long/lora/ep07.pt \
  outputs_ortho_clmix_long/lora/ep08.pt outputs_ortho_clmix_long/lora/ep09.pt \
  outputs_ortho_clmix_long/lora/ep10.pt outputs_ortho_clmix_long/lora/ep11.pt \
  outputs_ortho_clmix_long/lora/ep12.pt >> "$MASTER" 2>&1
$PY -u tools/tta_predict.py --work-dir outputs_ortho_clmixsoup2 \
  --out-prefix submissions/pred_results_ortho_clmixsoup2 --scales 448,512,576 \
  --balance-strength 1.0 --batch-size 64 --num-workers 2 --no-pin >> "$MASTER" 2>&1
$PY check_submission.py --csv submissions/pred_results_ortho_clmixsoup2_tta_balanced.csv \
  --zip submissions/pred_results_ortho_clmixsoup2_tta_balanced.zip 2>&1 | grep -aE 'RESULT|ERROR' | tee -a "$MASTER"
cp -f submissions/pred_results_ortho_clmixsoup2_tta_balanced.zip \
      submissions/pred_results_ortho_clmixsoup2_tta_balanced.csv submissions/next_queue/ 2>/dev/null
git add submissions/pred_results_ortho_clmixsoup2_tta_balanced.* exp_pipelines/ortho_clmix_long.log exp_pipelines/ortho_v9.sh 2>/dev/null
git commit -q -m "clmixsoup2: clmix 12ep长轨迹SWA(ep06-12) -> next_queue (冲77.73)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -1
echo "[ortho_v9] ALL DONE" | tee -a "$MASTER"