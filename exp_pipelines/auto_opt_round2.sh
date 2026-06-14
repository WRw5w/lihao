#!/usr/bin/env bash
# Auto-optimization round 2: build on champion c448_drecall (mid_03_06=0.9184).
# Stack capacity (rank32) and/or more data recall (keep0.90) onto the winner.
# Per candidate: 90/10 seed42 val -> result.csv -> commit to auto-opt/<name> -> push.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
COMMON="--num-workers 2 --no-pin --cache-dir outputs/cache --lora-target attn_mlp --ema-decay 0.999 --randaug"

run_candidate() {
  name=$1; shift; extra="$*"
  wd="exp_pipelines/auto_${name}"
  rm -rf "${wd}"; mkdir -p "${wd}/lora"
  echo "[auto-opt2] START ${name} :: ${extra}"
  $PY -u finetune_lora.py $COMMON --work-dir "${wd}" ${extra} > "${wd}/train.log" 2>&1
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "[auto-opt2] ${name} TRAIN FAILED rc=${rc}, skipping git"; return; fi

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
    echo "# auto-opt round2: ${name}"
    echo ""
    echo "- config: \`${extra}\`"
    echo "- best: ${best}"
    echo "- baseline to beat: c448_drecall mid_03_06=0.9184"
  } > "${wd}/RESULT.md"

  br="auto-opt/${name}"
  git checkout -B "${br}" >/dev/null 2>&1
  git add -f "${wd}/RESULT.md" "${wd}/result.csv" "${wd}/lora/history.json" "${wd}/train.log"
  git commit -q -m "auto-opt2 ${name}: mid_03_06=${mid} | ${extra}"
  git push -q -u origin "${br}" 2>&1 | tail -1
  git checkout -q main >/dev/null 2>&1
  echo "[auto-opt2] ${name} DONE mid=${mid} -> pushed ${br}"
}

run_candidate c448_dr_rank32        --img-size 448 --batch-size 64 --keep-ratio 0.85 --pseudo-thresh 0.6 --pseudo-margin 0.05 --lora-rank 32 --lora-alpha 64 --epochs 8
run_candidate c448_dr_keep90        --img-size 448 --batch-size 64 --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 --epochs 8
run_candidate c448_dr_rank32_keep90 --img-size 448 --batch-size 64 --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 --lora-rank 32 --lora-alpha 64 --epochs 8
echo "[auto-opt2] ROUND2 QUEUE COMPLETE"
