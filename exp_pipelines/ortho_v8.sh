#!/usr/bin/env bash
# Wave 8 -- breadth +5 directions x 2 (2026-06-19), focused on PRODUCTIVE families
# (label-noise correction / balanced-prior / SSL) + untested augmentation:
#   OT(#28)      : ot / otknn      - Sinkhorn-balance teacher probs to uniform prior, then CL
#   LabelProp(#27): lp08 / lp095   - propagate labels on feature kNN graph
#   DivideMix(#10): divmix5/divmix7- GMM loss-split + co-refine (semi-sup use of noisy)
#   ClassBal(#35): cb05 / cb10     - inverse-freq weighting vs noise imbalance
#   Aug(#32)     : cutmix / erase  - CutMix / RandomErasing (new aug family)
# Champion recipe + each direction's flags; per-run GPU smoke-guard (HF offline,
# 免疫拉权重网络抖动); full TTA+balance; auto-drop next_queue. Soup只同轨迹内做(教训).
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v8.log; : > "$MASTER"

run_one() {
  name=$1; shift; extra="$*"
  wd="outputs_ortho_$name"; log="exp_pipelines/ortho_${name}.log"
  echo "[$name] smoke-guard ($extra)..." | tee -a "$MASTER"
  $PY -u finetune_lora.py --smoke --work-dir outputs_tmp --cache-dir outputs/cache \
    --num-workers 2 --no-pin --batch-size 64 $extra > "exp_pipelines/ortho_${name}_smoke.log" 2>&1
  if [ $? -ne 0 ]; then echo "[$name] SMOKE FAILED -- skip"; tail -5 "exp_pipelines/ortho_${name}_smoke.log" | tee -a "$MASTER"; return; fi
  echo "[$name] ===== TRAIN $extra =====" | tee -a "$MASTER"
  $PY -u finetune_lora.py --work-dir "$wd" --cache-dir outputs/cache \
    --img-size 448 --epochs 6 --batch-size 32 --lora-rank 32 --lora-alpha 64 \
    --lora-target attn_mlp --lora-blocks 12 --keep-ratio 0.90 --ema-decay 0.999 --randaug \
    --pseudo-thresh 0.6 --pseudo-margin 0.05 --label-smoothing 0.1 --num-workers 2 --no-pin \
    --snapshot-after 3 $extra > "$log" 2>&1
  best=$(grep -aoE 'mid_03_06=[0-9.]+' "$log" | sort -t= -k2 -rn | head -1)
  echo "[$name] best $best (cleanlab 0.9229->76.61, ref 0.9233->76.14)" | tee -a "$MASTER"
  [ -f "$wd/lora/best.pt" ] || { echo "[$name] NO best.pt"; tail -6 "$log" | tee -a "$MASTER"; return; }
  cp -f "$wd/lora/best.pt" "$wd/lora/full.pt"
  $PY -u tools/tta_predict.py --work-dir "$wd" --out-prefix "submissions/pred_results_ortho_$name" \
    --scales 448,512,576 --balance-strength 1.0 --batch-size 64 --num-workers 2 --no-pin >> "$log" 2>&1
  $PY check_submission.py --csv "submissions/pred_results_ortho_${name}_tta_balanced.csv" \
    --zip "submissions/pred_results_ortho_${name}_tta_balanced.zip" 2>&1 | grep -aE 'RESULT|ERROR' | tee -a "$MASTER"
  git add "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
    "submissions/pred_results_ortho_${name}_tta_balanced.csv" "$log" 2>/dev/null
  git commit -q -m "ortho v8 $name: val $best [$extra]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
  cp -f "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
        "submissions/pred_results_ortho_${name}_tta_balanced.csv" submissions/next_queue/ 2>/dev/null
  echo "[$name] DONE $best -> next_queue" | tee -a "$MASTER"
}

run_one ot       --denoise ot
run_one otknn    --denoise otknn
run_one lp08     --denoise labelprop --lp-alpha 0.8
run_one lp095    --denoise labelprop --lp-alpha 0.95
run_one divmix5  --denoise divmix --divmix-thresh 0.5
run_one divmix7  --denoise divmix --divmix-thresh 0.7
run_one cb05     --class-balance --cb-power 0.5
run_one cb10     --class-balance --cb-power 1.0
run_one cutmix   --cutmix 1.0
run_one erase    --random-erasing
echo "[ortho_v8] ALL DONE" | tee -a "$MASTER"
git push 2>&1 | tail -1