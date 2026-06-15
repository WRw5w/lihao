#!/usr/bin/env bash
# Five-signal experiment (fully autonomous):
#   1. extract augmentation-consistency for the train set (~1h, one-time, cached)
#   2. val-mode retrain with COMPOSITE reliability
#        = kNN agreement x teacher conf x margin x visual-prototype x aug-consistency
#      select best by TRUSTED-val (high-agreement & teacher-consensus subset)
#   3. predict from best.pt with LEAN TTA (448+flip only) + adjustable balance (0.5)
#   4. validate
# batch 32 for 8GB headroom; single model, compliant.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
WD=exp_pipelines/five_signal
PRED=outputs_five_signal
COMMON="--num-workers 2 --no-pin --cache-dir outputs/cache --lora-target attn_mlp --lora-rank 32 --lora-alpha 64 --ema-decay 0.999 --randaug --img-size 448 --batch-size 32 --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05"

echo "[5sig] === 1. 增强一致性提取 (若无缓存) ==="
if [ ! -f outputs/cache/aug_consistency.pt ]; then
  $PY -u tools/extract_aug_features.py
fi
[ -f outputs/cache/aug_consistency.pt ] || { echo "[5sig] FAILED: aug_consistency.pt 未生成"; exit 1; }

echo "[5sig] === 2. 5合一综合分 val-mode 重训 (trusted-val 选型, 8轮) ==="
rm -rf $WD; mkdir -p $WD/lora
$PY -u finetune_lora.py --epochs 8 $COMMON \
  --proto-power 0.5 --aug-consist-power 0.5 --trusted-agree 0.7 --early-stop-patience 0 \
  --work-dir $WD > $WD/train.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f $WD/lora/best.pt ]; then echo "[5sig] TRAIN FAILED rc=$rc"; exit 1; fi

echo "[5sig] === 3. best.pt 出提交 (精简TTA 448+翻转, balance 0.5) ==="
rm -rf $PRED; mkdir -p $PRED/lora
cp $WD/lora/best.pt $PRED/lora/full.pt
$PY -u tools/tta_predict.py --work-dir $PRED \
  --out-prefix submissions/pred_results_5sig --scales 448 --balance-strength 0.5 \
  --num-workers 2 --no-pin

echo "[5sig] === 4. 校验 ==="
$PY check_submission.py --csv submissions/pred_results_5sig_tta_balanced.csv \
  --zip submissions/pred_results_5sig_tta_balanced.zip 2>&1 | grep RESULT
echo "[5sig] FIVE SIGNAL DONE"
