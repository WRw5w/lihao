# Pos-embed resolution / resample study — leaderboard results

Submitted to aicomp (team **swpu_1**, AIC-2026-58579595) one-per-hour on **2026-06-25**,
TTA-balanced prediction zips. Recipe is **identical across arms** (frozen winning recipe
in `env.sh` COMMON); the only variables are **input resolution** and **pos-embed resample
method**. Grid = img_size / 32 (CLIP ViT-B/32; native pretrained grid is 7×7 @224).

| arm | img | grid | resample | LB score | Δ vs 448 |
|---|---|---|---|---|---|
| **b448_default** | 448 | 14×14 (even) | timm default | **78.7359** 🏆 | — |
| r416_aligned | 416 | 13×13 (odd, lossless) | aligned (align_corners) | 78.1071 | −0.6288 |
| r416_default | 416 | 13×13 (odd) | timm default | 78.0711 | −0.6648 |
| b224_native | 224 | 7×7 (native) | timm default | 73.2847 | −5.4512 |

## Conclusions

1. **Resolution dominates, monotonically: 448 > 416 > 224.** Dropping resolution is pure
   loss — 224 collapses −5.45 vs 448. Fine-grained recognition is resolution-hungry; the
   14×14 token grid carries materially more signal than 13×13, and far more than 7×7.
   **→ The "go to 416" hypothesis is falsified. 448 is the better operating point.**

2. **Aligned (lossless) interpolation helps, but only marginally.** On the odd 13×13 grid
   `align_corners=True` lands the original 49 anchors exactly on output nodes
   (anchor_err=0.0000 vs ~0.0573 for timm default), and it does buy **+0.0360** (78.1071 vs
   78.0711). The effect is real and directionally correct, but ~17× too small to offset the
   −0.63 resolution penalty of going 448→416.

3. **Unexpected: b448_default set a new overall best (78.7359)**, beating the prior champion
   clmixsoup5 = 78.6318 (+0.10) — even without cleanlab/mixup/long-trajectory soup. The
   posembed base recipe at 448 is already very strong.

## Not run (stretch arms in arms.tsv)

- `b448_aligned` (448, even grid — aligned has no exact-preserve advantage there).
- **`r608_aligned`** (608, 19×19 odd grid = lossless AND higher resolution than 448). By
  the monotonic-resolution result + the small positive aligned effect, this is the most
  promising untested point — likely ≥ 448. Cost: bs=16, ~expensive. **Recommended next.**
