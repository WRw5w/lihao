#!/usr/bin/env bash
# Wave 7 -- improve the winner: cleanlab_relabel (回收置信判错样本,改标为CL预测类).
# cleanlab(drop) 单76.61; relabel 把丢掉的~8%带纠正标签加回 -> 用满数据, 应 >= cleanlab。
# HF_HUB_OFFLINE 防 timm 拉权重的网络抖动(s3 崩过)。候选 -> next_queue。
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v7.log; : > "$MASTER"

echo "[clrelabel] TRAIN" | tee -a "$MASTER"
$PY -u finetune_lora.py --work-dir outputs_ortho_clrelabel --cache-dir outputs/cache \
  --img-size 448 --epochs 6 --batch-size 32 --lora-rank 32 --lora-alpha 64 \
  --lora-target attn_mlp --lora-blocks 12 --keep-ratio 0.90 --ema-decay 0.999 --randaug \
  --pseudo-thresh 0.6 --pseudo-margin 0.05 --label-smoothing 0.1 --num-workers 2 --no-pin \
  --snapshot-after 3 --denoise cleanlab_relabel > exp_pipelines/ortho_clrelabel.log 2>&1
best=$(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/ortho_clrelabel.log | sort -t= -k2 -rn | head -1)
echo "[clrelabel] best $best (cleanlab=0.9229->榜76.61)" | tee -a "$MASTER"
[ -f outputs_ortho_clrelabel/lora/best.pt ] || { echo "[clrelabel] NO best.pt"; tail -8 exp_pipelines/ortho_clrelabel.log | tee -a "$MASTER"; exit 1; }
cp -f outputs_ortho_clrelabel/lora/best.pt outputs_ortho_clrelabel/lora/full.pt

$PY -u tools/tta_predict.py --work-dir outputs_ortho_clrelabel \
  --out-prefix submissions/pred_results_ortho_clrelabel --scales 448,512,576 \
  --balance-strength 1.0 --batch-size 64 --num-workers 2 --no-pin >> "$MASTER" 2>&1
$PY check_submission.py --csv submissions/pred_results_ortho_clrelabel_tta_balanced.csv \
  --zip submissions/pred_results_ortho_clrelabel_tta_balanced.zip 2>&1 | grep -aE 'RESULT|ERROR' | tee -a "$MASTER"
git add submissions/pred_results_ortho_clrelabel_tta_balanced.zip \
  submissions/pred_results_ortho_clrelabel_tta_balanced.csv exp_pipelines/ortho_clrelabel.log exp_pipelines/ortho_v7.sh 2>/dev/null
git commit -q -m "ortho v7 clrelabel: val $best -> next_queue

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
cp -f submissions/pred_results_ortho_clrelabel_tta_balanced.zip \
      submissions/pred_results_ortho_clrelabel_tta_balanced.csv submissions/next_queue/ 2>/dev/null
git push 2>&1 | tail -1
echo "[ortho_v7] ALL DONE  clrelabel=$best" | tee -a "$MASTER"