#!/usr/bin/env bash
# Wave 6 -- push the cleanlab winner harder with a richer cross-seed SOUP.
# 证据: fet_soup 榜77.10(单76->汤77.1,soup杠杆在榜上确认+1.1); cleanlab单76.61.
# 跨seed多样性是更大soup增益的已证路径(冠军汤靠多checkpoint)。
#   cleanlab_s2(infer) : 第二seed单模型上榜 -> 确认 cleanlab 76.6 可复现(非偶然)
#   cleanlab_s3/s4     : 第3、4个seed训练, 给4-seed soup供料
#   cl4soup            : {s42,s123,s7,s2024} 4-seed soup -> 冲破 77.73 的更强一发
# 同冠军配方; 候选自动投 next_queue。GPU空闲时跑。
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v6.log
: > "$MASTER"
CKPT_ARGS="--img-size 448 --epochs 6 --batch-size 32 --lora-rank 32 --lora-alpha 64 \
  --lora-target attn_mlp --lora-blocks 12 --keep-ratio 0.90 --ema-decay 0.999 --randaug \
  --pseudo-thresh 0.6 --pseudo-margin 0.05 --label-smoothing 0.1 --num-workers 2 --no-pin --snapshot-after 3 \
  --denoise cleanlab"

drop() { name=$1
  git add "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
    "submissions/pred_results_ortho_${name}_tta_balanced.csv" exp_pipelines/ortho_v6.log 2>/dev/null
  git commit -q -m "ortho v6 $name -> next_queue

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
  cp -f "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
        "submissions/pred_results_ortho_${name}_tta_balanced.csv" submissions/next_queue/ 2>/dev/null
  echo "[$name] -> next_queue" | tee -a "$MASTER"; }
infer() { name=$1; wd=$2
  $PY -u tools/tta_predict.py --work-dir "$wd" --out-prefix "submissions/pred_results_ortho_$name" \
    --scales 448,512,576 --balance-strength 1.0 --batch-size 64 --num-workers 2 --no-pin >> "$MASTER" 2>&1
  $PY check_submission.py --csv "submissions/pred_results_ortho_${name}_tta_balanced.csv" \
    --zip "submissions/pred_results_ortho_${name}_tta_balanced.zip" 2>&1 | grep -aE 'RESULT|ERROR' | tee -a "$MASTER"; }
train() { name=$1; shift; echo "[$name] TRAIN $*" | tee -a "$MASTER"
  $PY -u finetune_lora.py --work-dir "outputs_ortho_$name" --cache-dir outputs/cache $CKPT_ARGS "$@" \
    > "exp_pipelines/ortho_${name}.log" 2>&1
  echo "[$name] best $(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/ortho_${name}.log|sort -t= -k2 -rn|head -1)" | tee -a "$MASTER"; }

# 1) 第二seed单模型上榜(复现性确认)
cp -f outputs_ortho_cleanlab_s2/lora/best.pt outputs_ortho_cleanlab_s2/lora/full.pt 2>/dev/null
echo "[v6] === cleanlab_s2 single ===" | tee -a "$MASTER"
infer cleanlab_s2 outputs_ortho_cleanlab_s2 && drop cleanlab_s2

# 2) 第3、4个seed
train cleanlab_s3 --seed 7
train cleanlab_s4 --seed 2024

# 3) 4-seed soup (主攻; best.pt×4 控RAM)
echo "[v6] === cl4soup (4-seed) ===" | tee -a "$MASTER"
$PY tools/swa_soup.py --out outputs_ortho_cl4soup/lora/full.pt --checkpoints \
  outputs_ortho_cleanlab/lora/best.pt outputs_ortho_cleanlab_s2/lora/best.pt \
  outputs_ortho_cleanlab_s3/lora/best.pt outputs_ortho_cleanlab_s4/lora/best.pt >> "$MASTER" 2>&1
infer cl4soup outputs_ortho_cl4soup && drop cl4soup

echo "[ortho_v6] ALL DONE" | tee -a "$MASTER"
git push 2>&1 | tail -1