# Pos-embed resolution / resample study — leaderboard results

Submitted to aicomp (team **swpu_1**, AIC-2026-58579595) on **2026-06-25/26**, TTA-balanced
prediction zips. Recipe is **identical across arms** (frozen winning recipe in `env.sh` COMMON:
cleanlab denoise + mixup0.2 + EMA + RandAug + rank32 attn_mlp + pseudo-label, 12ep,
same-trajectory SWA ep4–12, seed 42 unless noted); the only variables are **input resolution**
and **pos-embed resample method**. Grid = img_size / 32 (CLIP ViT-B/32; native pretrained grid
is 7×7 @224). `aligned` = bilinear + `align_corners=True`; `default` = timm default interpolation.

| resolution | grid | parity | default (timm) | aligned | aligned − default |
|---|---|---|---|---|---|
| 224 | 7×7 (native) | — | 73.2847 | — | — |
| 416 | 13×13 | odd | 78.0711 | 78.1071 | +0.0360 |
| 448 | 14×14 | even | 78.7359 | 78.8561 | +0.1202 |
| 480 | 15×15 | odd | — | 78.7439 (seed42) / **78.9082 (seed123)** | — |
| 512 | 16×16 | even | 78.5357 | **78.9122** 🏆 | **+0.3765** |
| 544 | 17×17 | odd | — | 78.5557 | — |
| 576 | 18×18 | even | — | 78.7920 | — |
| 608 | 19×19 | odd | — | 78.6438 | — |

**Champion: b512_aligned = 78.9122** (b480_aligned seed123 = 78.9082 essentially ties it).
Prior whole-project best was clmixsoup5 = 78.6318; this study added +0.28, purely from
resolution≈512 + aligned pos-embed, same recipe otherwise.

## Conclusions

1. **`aligned` (align_corners=True) beats timm default, and the gain GROWS with resolution.**
   +0.036 @416, +0.120 @448, **+0.377 @512**. This is the headline, robust result (the +0.38
   at 512 is far above the noise floor). Mechanism: the pos-embed must be extrapolated from the
   native 7×7 grid; the further you go (higher resolution), the more the interpolation method
   matters. timm default degrades fast; align_corners degrades slowly.

2. **`aligned` shifts the optimal resolution rightward.** The default(timm) series peaks at 448
   (78.7359) and is already *down* by 512 (78.5357). The aligned series is still climbing at
   512 (78.9122 > 448's 78.8561). Better extrapolation lets you cash in higher resolution.

3. **Resolution is a steep climb to ~448, then a noisy plateau (448–608 ≈ 78.6–78.9).**
   224 ≪ 416 ≪ {448 … 608 plateau}. Within the plateau the differences are dominated by noise
   (see 4), so "the peak is exactly 512" is NOT a reliable claim — 448/480/512/576 are
   statistically indistinguishable.

4. **Run-to-run (seed) noise is large, ~±0.15.** b480_aligned: seed42 = 78.7439, seed123 =
   78.9082 — a 0.164 swing from the seed alone, at one resolution. This single control reshapes
   the reading of everything in the plateau.

5. **The "even-grid beats odd-grid" parity pattern was NOISE, not a law.** Mid-experiment the
   even grids {448,512,576} all sat high and odd grids {480(s42),544,608} all sat low, a clean
   3-vs-3 split that looked real and *contradicted* the original "odd grids preserve anchors
   exactly so they win" hypothesis. The b480 reseed killed it: seed123 jumped 480 from the
   "odd-low" band straight into the "even-high" band. So neither the parity pattern NOR the
   original odd-grid-anchor hypothesis holds — both were over-readings of single-run noise.

6. **Operating point: aligned at 448–512.** 512 is the best single point (78.9122); 448 is the
   cheapest near-equal (bs32, fastest). Pushing past ~512 (544/576/608) does not help.

## Notes

- This experiment iteratively corrected itself three times: "448 is a sharp peak" → "448–512
  plateau, peak 512" → "noisy plateau, parity was noise". The reseed and the b512_default
  control were what prevented shipping noise as findings.
- The earlier whole-project leaderboard recipe (clmixsoup, 78.63) was at 448 with timm default;
  switching to 512 + aligned is a clean +0.28 with no recipe change.

## Operational notes (for re-runs)

- **TTA must force HF offline** (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; weights cached from
  training). The first r608 TTA crashed in `build_lora_model` on a transient HF-Hub call.
- **`--pos-resample` valid values are `timm` and `aligned`** (NOT `default` — that arg errors out).
- **Submission infra**: 打榜 Chrome tabs freeze after long idle → CDP `Runtime.enable` timeout;
  fix = close+recreate tabs and probe with `aicomp_cdp.mjs heartbeat` before submitting. The
  platform scores within ~1 min but the public leaderboard "发布时间" only refreshes hourly and
  occasionally lags (the midnight-Beijing publish was ~1h late) — if the runner exhausts its
  capture attempts it sets a blocking `aicomp_active_submission.json` lock; clear it with
  `aicomp_submit_queue.mjs skip-score <idx>` after manually reading the score off the leaderboard.
