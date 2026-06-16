#!/usr/bin/env bash
# 李洋's 3 "make the model conservative" changes, with FULL inference (lesson:
# full TTA + full balance won 76.7; lean TTA + balance0.5 lost).
#   A. balance-lambda sweep on the 76.7 soup (isolate #3 on the winning model)
#   B. conservative full-data retrain: #1 reliability=weighted-sum (not product)
#      + #2 soft pseudo-fusion (keep 0.5 of original label vs hard-replace)
#   C/D. predict with FULL TTA(448/512/576) + FULL balance(1.0), validate
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe

echo "[cons] === A. balance-λ 扫描 (在 76.7 的 soup_uniform 上, λ=0.25/0.5/0.75; λ=1 已是76.7) ==="
if [ -f outputs_soup_uniform/lora/full.pt ]; then
  $PY -u tools/tta_predict.py --work-dir outputs_soup_uniform \
    --out-prefix submissions/pred_results_soup_sweep --scales 448,512,576 \
    --balance-strength 0.25,0.5,0.75 --num-workers 2 --no-pin
else
  echo "[cons] (跳过A: outputs_soup_uniform/lora/full.pt 不存在)"
fi

WD=outputs_full_conservative
echo "[cons] === B. 保守化满数据重训 (reliability=sum + soft-pseudo0.5, 6轮, batch32) ==="
rm -rf $WD; mkdir -p $WD/lora
$PY -u finetune_lora.py --full --epochs 6 --img-size 448 --batch-size 32 \
  --lora-target attn_mlp --lora-rank 32 --lora-alpha 64 --ema-decay 0.999 --randaug \
  --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --reliability-mode sum --pseudo-soft-alpha 0.5 \
  --work-dir $WD --cache-dir outputs/cache > $WD/train.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f $WD/lora/full.pt ]; then echo "[cons] TRAIN FAILED rc=$rc"; exit 1; fi

echo "[cons] === C. predict (满TTA 448/512/576 + 满均衡1.0) ==="
$PY -u tools/tta_predict.py --work-dir $WD --out-prefix submissions/pred_results_conservative \
  --scales 448,512,576 --balance-strength 1.0 --num-workers 2 --no-pin

echo "[cons] === D. 校验 ==="
$PY check_submission.py --csv submissions/pred_results_conservative_tta_balanced.csv \
  --zip submissions/pred_results_conservative_tta_balanced.zip 2>&1 | grep RESULT
echo "[cons] CONSERVATIVE DONE"
