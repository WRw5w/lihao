#!/usr/bin/env bash
# Build cache (if missing) then run all arms of the chosen tier, sequentially.
#   usage: bash run_all.sh [core|stretch|all]      (default: core)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
TIER="${1:-core}"

bash "$HERE/01_build_cache.sh"

ran=0
# fields: name img pos bs scales tier
while read -r name img pos bs scales tier _rest; do
  case "$name" in ''|\#*) continue;; esac          # skip blanks / comments
  [ "$TIER" = "all" ] || [ "$tier" = "$TIER" ] || continue
  bash "$HERE/run_arm.sh" "$name" "$img" "$pos" "$bs" "$scales"
  ran=$((ran+1))
done < "$HERE/arms.tsv"

echo "============================================================"
echo "ALL DONE  ($ran arms, tier=$TIER)"
echo "Submit ONLY these (balanced is the +2.9pt lever):"
ls -1 "$OUT_ROOT"/submissions/*_tta_balanced.zip 2>/dev/null || echo "  (none found — check logs)"
