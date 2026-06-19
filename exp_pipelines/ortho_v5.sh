#!/usr/bin/env bash
# Wave 5 -- escalate the WINNER. ortho榜分: cleanlab=76.61 是唯一破单模型天花板(76.1)的机制。
# 用已证明 +1.5pt 的 SOUP 杠杆把它推过 soup 冠军 77.73:
#   cleanlabsoup : cleanlab 单run内快照 soup (便宜, 先打)
#   clmix        : cleanlab + mixup02 叠加(两个各自≈帮忙的正交杠杆), 单候选
#   cleanlab_s2  : cleanlab 第二seed(123) 训练 -> 给跨seed soup 增多样性
#   cl2soup      : {cleanlab(s42), cleanlab_s2(s123)} 跨seed soup -> 破77.73 的主攻
# 同冠军配方 + 满TTA + 均衡; 候选自动投 next_queue。在 v4 后链式跑(GPU/RAM 空)。
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v5.log
: > "$MASTER"

CKPT_ARGS="--img-size 448 --epochs 6 --batch-size 32 --lora-rank 32 --lora-alpha 64 \
  --lora-target attn_mlp --lora-blocks 12 --keep-ratio 0.90 --ema-decay 0.999 --randaug \
  --pseudo-thresh 0.6 --pseudo-margin 0.05 --label-smoothing 0.1 --num-workers 2 --no-pin --snapshot-after 3"

drop() {  # commit + copy to next_queue
  name=$1
  git add "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
    "submissions/pred_results_ortho_${name}_tta_balanced.csv" "exp_pipelines/ortho_v5.log" 2>/dev/null
  git commit -q -m "ortho v5 $name -> next_queue

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
  cp -f "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
        "submissions/pred_results_ortho_${name}_tta_balanced.csv" submissions/next_queue/ 2>/dev/null
  echo "[$name] -> next_queue" | tee -a "$MASTER"
}

infer() {  # infer a work-dir's full.pt -> candidate
  name=$1; wd=$2
  $PY -u tools/tta_predict.py --work-dir "$wd" \
    --out-prefix "submissions/pred_results_ortho_$name" --scales 448,512,576 \
    --balance-strength 1.0 --batch-size 64 --num-workers 2 --no-pin >> "$MASTER" 2>&1
  $PY check_submission.py --csv "submissions/pred_results_ortho_${name}_tta_balanced.csv" \
    --zip "submissions/pred_results_ortho_${name}_tta_balanced.zip" 2>&1 | grep -aE 'RESULT|ERROR' | tee -a "$MASTER"
}

train() {  # train champion recipe + extra into outputs_ortho_<name>
  name=$1; shift; extra="$*"; wd="outputs_ortho_$name"
  echo "[$name] TRAIN $extra" | tee -a "$MASTER"
  $PY -u finetune_lora.py --work-dir "$wd" --cache-dir outputs/cache $CKPT_ARGS $extra \
    > "exp_pipelines/ortho_${name}.log" 2>&1
  best=$(grep -aoE 'mid_03_06=[0-9.]+' "exp_pipelines/ortho_${name}.log" | sort -t= -k2 -rn | head -1)
  echo "[$name] best $best" | tee -a "$MASTER"
}

# 1) cleanlab 单run快照 soup (便宜先打)
echo "[v5] === cleanlabsoup ===" | tee -a "$MASTER"
$PY tools/swa_soup.py --out outputs_ortho_cleanlabsoup/lora/full.pt --checkpoints \
  outputs_ortho_cleanlab/lora/best.pt outputs_ortho_cleanlab/lora/best_ep04.pt \
  outputs_ortho_cleanlab/lora/best_ep03.pt outputs_ortho_cleanlab/lora/last.pt >> "$MASTER" 2>&1
infer cleanlabsoup outputs_ortho_cleanlabsoup && drop cleanlabsoup

# 2) cleanlab + mixup 叠加
train clmix --denoise cleanlab --mixup-alpha 0.2
cp -f outputs_ortho_clmix/lora/best.pt outputs_ortho_clmix/lora/full.pt 2>/dev/null
infer clmix outputs_ortho_clmix && drop clmix

# 3) cleanlab 第二seed
train cleanlab_s2 --denoise cleanlab --seed 123

# 4) 跨seed cleanlab soup (主攻 77.73)
echo "[v5] === cl2soup (cross-seed) ===" | tee -a "$MASTER"
$PY tools/swa_soup.py --out outputs_ortho_cl2soup/lora/full.pt --checkpoints \
  outputs_ortho_cleanlab/lora/best.pt outputs_ortho_cleanlab/lora/last.pt \
  outputs_ortho_cleanlab_s2/lora/best.pt outputs_ortho_cleanlab_s2/lora/last.pt >> "$MASTER" 2>&1
infer cl2soup outputs_ortho_cl2soup && drop cl2soup

echo "[ortho_v5] ALL DONE" | tee -a "$MASTER"
git push 2>&1 | tail -1