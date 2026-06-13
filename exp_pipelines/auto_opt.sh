#!/usr/bin/env bash
# Auto-optimization queue (bounded, run-to-completion).
# Builds on bake-off winner C(448, mid_03_06=0.9091).
# Per candidate: 90/10 seed42 val run -> validate precision (best mid_03_06)
#   -> write per-epoch result.csv + RESULT.md -> commit to a NEW branch -> push.
# Every candidate pushes its own branch (even if it does not beat C), per user choice.
# Checkpoints (*.pt) stay gitignored; only logs/history/result are committed.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
COMMON="--num-workers 2 --no-pin --cache-dir outputs/cache --lora-target attn_mlp --ema-decay 0.999 --randaug"

run_candidate() {
  name=$1; shift; extra="$*"
  wd="exp_pipelines/auto_${name}"
  mkdir -p "${wd}/lora"
  echo "[auto-opt] START ${name} :: ${extra}"
  $PY -u finetune_lora.py $COMMON --work-dir "${wd}" ${extra} > "${wd}/train.log" 2>&1
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "[auto-opt] ${name} TRAIN FAILED rc=${rc}, skipping git"; return; fi

  best=$($PY - "${wd}/lora/history.json" "${wd}/result.csv" <<'PYEOF'
import json, sys, csv
hist_path, out_csv = sys.argv[1], sys.argv[2]
h = json.load(open(hist_path))
rows = [r for r in h if "mid_03_06" in r]
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "loss", "noisy_all", "low_lt03", "mid_03_06", "high_ge06"])
    for r in rows:
        w.writerow([r.get("epoch"), r.get("loss"), r["noisy_all"], r["low_lt03"], r["mid_03_06"], r["high_ge06"]])
b = max(rows, key=lambda r: r["mid_03_06"])
print(f"{b['mid_03_06']:.4f}|ep{b.get('epoch')}|noisy={b['noisy_all']}|low={b['low_lt03']}|high={b['high_ge06']}")
PYEOF
)
  mid=${best%%|*}
  {
    echo "# auto-opt: ${name}"
    echo ""
    echo "- config: \`${extra}\`"
    echo "- best: ${best}"
    echo "- baseline to beat: C(448) mid_03_06=0.9091"
  } > "${wd}/RESULT.md"

  br="auto-opt/${name}"
  git checkout -B "${br}" >/dev/null 2>&1
  git add -f "${wd}/RESULT.md" "${wd}/result.csv" "${wd}/lora/history.json" "${wd}/train.log"
  git commit -q -m "auto-opt ${name}: mid_03_06=${mid} | ${extra}"
  git push -q -u origin "${br}" 2>&1 | tail -1
  git checkout -q main >/dev/null 2>&1
  echo "[auto-opt] ${name} DONE mid=${mid} -> pushed ${br}"
}

# --- candidate queue (edit/extend here) ---
run_candidate c448_drecall --img-size 448 --batch-size 64 --keep-ratio 0.85 --pseudo-thresh 0.6 --pseudo-margin 0.05 --epochs 8
run_candidate c512         --img-size 512 --batch-size 32 --epochs 8
run_candidate c448_rank32  --img-size 448 --batch-size 64 --lora-rank 32 --lora-alpha 64 --epochs 8

echo "[auto-opt] QUEUE COMPLETE"
