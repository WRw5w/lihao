#!/usr/bin/env bash
# Wave 10 -- push past the NEW CHAMPION clmixsoup2=78.26. That was a 7-point SWA
# (clmix_long ep06-12). SWA improves with a longer, more spread-out trajectory ->
# train clmix 18 epochs, stream-SWA the post-warmup span (ep08-18, 11 points).
# 同轨迹内SWA(跨seed已死). 制胜配方 = cleanlab去噪 + mixup0.2 + 长轨迹SWA.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v10.log; : > "$MASTER"

echo "[clmix_xl] TRAIN 18ep save-every1 (cleanlab+mixup0.2)" | tee -a "$MASTER"
$PY -u finetune_lora.py --work-dir outputs_ortho_clmix_xl --cache-dir outputs/cache \
  --img-size 448 --epochs 18 --batch-size 32 --lora-rank 32 --lora-alpha 64 \
  --lora-target attn_mlp --lora-blocks 12 --keep-ratio 0.90 --ema-decay 0.999 --randaug \
  --pseudo-thresh 0.6 --pseudo-margin 0.05 --label-smoothing 0.1 --num-workers 2 --no-pin \
  --denoise cleanlab --mixup-alpha 0.2 --save-every 1 > exp_pipelines/ortho_clmix_xl.log 2>&1
echo "[clmix_xl] best $(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/ortho_clmix_xl.log|sort -t= -k2 -rn|head -1)" | tee -a "$MASTER"
ls outputs_ortho_clmix_xl/lora/ep18.pt >/dev/null 2>&1 || { echo "[clmix_xl] incomplete"; tail -8 exp_pipelines/ortho_clmix_xl.log | tee -a "$MASTER"; exit 1; }

echo "[clmixsoup3] === long-trajectory SWA (ep08-18, 11pts, streamed) ===" | tee -a "$MASTER"
CK=""
for e in 08 09 10 11 12 13 14 15 16 17 18; do CK="$CK outputs_ortho_clmix_xl/lora/ep${e}.pt"; done
$PY tools/swa_soup.py --out outputs_ortho_clmixsoup3/lora/full.pt --checkpoints $CK >> "$MASTER" 2>&1
$PY -u tools/tta_predict.py --work-dir outputs_ortho_clmixsoup3 \
  --out-prefix submissions/pred_results_ortho_clmixsoup3 --scales 448,512,576 \
  --balance-strength 1.0 --batch-size 64 --num-workers 2 --no-pin >> "$MASTER" 2>&1
$PY check_submission.py --csv submissions/pred_results_ortho_clmixsoup3_tta_balanced.csv \
  --zip submissions/pred_results_ortho_clmixsoup3_tta_balanced.zip 2>&1 | grep -aE 'RESULT|ERROR' | tee -a "$MASTER"
cp -f submissions/pred_results_ortho_clmixsoup3_tta_balanced.zip \
      submissions/pred_results_ortho_clmixsoup3_tta_balanced.csv submissions/next_queue/ 2>/dev/null
git add submissions/pred_results_ortho_clmixsoup3_tta_balanced.* exp_pipelines/ortho_clmix_xl.log exp_pipelines/ortho_v10.sh tools/swa_soup.py 2>/dev/null
git commit -q -m "clmixsoup3: clmix 18ep长轨迹SWA(ep08-18) 冲>78.26 (新冠军clmixsoup2=78.26)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -1
echo "[ortho_v10] ALL DONE" | tee -a "$MASTER"