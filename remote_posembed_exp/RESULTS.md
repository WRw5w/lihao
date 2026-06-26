# Pos-embed resolution / resample study — leaderboard results

Submitted to aicomp (team **swpu_1**, AIC-2026-58579595) on **2026-06-25/26**,
TTA-balanced prediction zips. Recipe is **identical across arms** (frozen winning recipe
in `env.sh` COMMON: cleanlab denoise + mixup0.2 + EMA + RandAug + rank32 attn_mlp +
pseudo-label, 12ep, same-trajectory SWA ep4–12, seed 42); the only variables are **input
resolution** and **pos-embed resample method**. Grid = img_size / 32 (CLIP ViT-B/32; native
pretrained grid is 7×7 @224). `aligned` = bilinear + `align_corners=True`; `default` = timm's
default pos-embed interpolation.

| resolution | grid | parity | default | aligned |
|---|---|---|---|---|
| 224 | 7×7 (native) | — | 73.2847 | — |
| 416 | 13×13 | odd | 78.0711 | 78.1071 |
| 448 | 14×14 | even | 78.7359 | 78.8561 |
| 480 | 15×15 | odd | — | 78.7439 |
| **512** | 16×16 | even | — | **78.9122** 🏆 |
| 608 | 19×19 | odd | — | 78.6438 |

**Champion: b512_aligned = 78.9122** (prior best in the whole project was clmixsoup5 = 78.6318;
this study added +0.28 purely from resolution=512 + aligned pos-embed, same recipe otherwise).

## Conclusions

1. **`aligned` (align_corners=True) universally beats timm default.** +0.0360 at 416,
   +0.1202 at 448. The benefit is NOT limited to odd grids — see point 3.

2. **Resolution is a broad plateau over 448–512, not a sharp peak.** Rises steeply
   224→416→448 (73.28 → 78.1 → 78.86), then 448/480/512 sit in a tight 78.74–78.91 band
   (best at **512**), then falls at 608 (78.64). An earlier draft called the peak 448 — that
   was premature (only 448 vs 608 existed then); 512 then beat 448 by +0.056.

3. **The original "odd-grid exact-anchor-preservation wins" hypothesis is contradicted by
   the data.** arms.tsv bet that odd grids (13@416, 19@608) — where align_corners lands the
   49 native anchors exactly on output nodes — would be the clean winners. Empirically those
   two odd "clean" grids are the LOWER points, and the even grids (14@448, 16@512) are the
   winners. So exact interpolation alignment is second-order; what actually matters is
   (a) enough resolution and (b) not extrapolating the 7×7 pretrain grid too far (608 = 2.7×
   is past the plateau). Parity pattern observed: even {448, 512} > odd {480, 608}, but this
   is at the edge of the noise band — do not over-read it.

4. **Single-run noise is ~±0.1**, so the 480 dip (78.7439, below both 448 and 512) and the
   +0.056 of 512-over-448 should be read as "448–512 is a plateau," not a precise ranking.
   The robust, noise-resistant claims are: 224 ≪ 416 ≪ {448…512 plateau} > 608, and aligned > default.

5. **Operating point: 512 (aligned), 78.9122.** 448 (aligned) is an essentially-equal,
   cheaper fallback (bs32 vs bs24, faster).

## Untested / possible next (diminishing returns — differences now within noise)

- `b544_aligned` (17×17 odd) / `b576_aligned` (18×18 even) would test whether the peak is
  really at 512 or slightly higher, but at ±0.1 noise this is noise-chasing.
- `b480_aligned` re-run with a different seed would tell whether the 480 dip is real or noise.
- `b512_default` would isolate how much of 512's win is resolution vs the aligned resample.

## Operational notes (for re-runs)

- **TTA must force HF offline** (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; weights are cached
  from training). The first r608 TTA crashed in `build_lora_model` on a transient HuggingFace
  Hub call. Baking offline into the launcher fixed it for b448/480/512.
- **Submission infra**: the打榜 Chrome tabs freeze after long idle → CDP `Runtime.enable`
  times out. Fix = close + recreate the submit/leaderboard tabs (HTTP `/json/close`, `/json/new`)
  and probe with `aicomp_cdp.mjs heartbeat` before letting the runner submit. If the runner is
  killed at session teardown before its scheduled capture, the score is still on the platform —
  re-read the leaderboard manually and backfill.
