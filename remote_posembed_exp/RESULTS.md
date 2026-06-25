# Pos-embed resolution / resample study — leaderboard results

Submitted to aicomp (team **swpu_1**, AIC-2026-58579595) on **2026-06-25/26**,
TTA-balanced prediction zips. Recipe is **identical across arms** (frozen winning recipe
in `env.sh` COMMON: cleanlab denoise + mixup0.2 + EMA + RandAug + rank32 attn_mlp +
pseudo-label, 12ep, same-trajectory SWA ep4–12); the only variables are **input
resolution** and **pos-embed resample method**. Grid = img_size / 32 (CLIP ViT-B/32;
native pretrained grid is 7×7 @224).

| arm | img | grid | resample | LB score | Δ vs 448 |
|---|---|---|---|---|---|
| **b448_default** | 448 | 14×14 (even) | timm default | **78.7359** 🏆 | — |
| r608_aligned | 608 | 19×19 (odd, lossless) | aligned (align_corners) | 78.6438 | −0.0921 |
| r416_aligned | 416 | 13×13 (odd, lossless) | aligned (align_corners) | 78.1071 | −0.6288 |
| r416_default | 416 | 13×13 (odd) | timm default | 78.0711 | −0.6648 |
| b224_native | 224 | 7×7 (native) | timm default | 73.2847 | −5.4512 |

## Conclusions

1. **Resolution is an inverted-U, peaking at 448 — NOT monotonic.**
   224 (73.28) → 416 (78.07–78.11) → **448 (78.74, peak)** → 608 (78.64, −0.09). Going up
   helps a lot until 448, then 608 gives it back. (An earlier draft, before the 608 point
   existed, wrongly called this monotonic.)

2. **Pos-embed extrapolation distance — not interpolation cleanliness — is the binding
   constraint.** 608 has BOTH theoretical advantages over 448 (higher resolution AND a clean
   odd 19×19 grid where `align_corners` preserves all 49 native anchors exactly) yet still
   loses. The native grid is 7×7; 448 is 2.0× native, 608 is 2.7×. Stretching the learned
   positional relationships to 2.7× distorts the backbone's spatial priors enough to outweigh
   the extra resolution. 448's "imperfect" even-grid interpolation wins anyway. **Takeaway:
   how far you extrapolate from the 7×7 pretrain grid matters more than whether the
   interpolation is lossless.**

3. **Aligned (lossless) interpolation helps only marginally, and cannot rescue an
   off-sweet-spot resolution.** At 416 it bought +0.0360 (78.1071 vs 78.0711, odd grid);
   at 608 the lossless 19×19 grid was not enough to reach 448.

4. **b448_default also set a new overall best (78.7359)**, edging the prior champion
   clmixsoup5 = 78.6318 (+0.10) — same recipe family, so within the noise/recipe-window band,
   not a free lunch. r608_aligned (78.6438) likewise essentially ties that prior champion.

5. **Operating point: 448. Do not increase resolution further.**

## Untested / possible next

- `b448_aligned` (448, even 14×14 grid). Aligned has no exact-preserve advantage on an even
  grid, but at 416 it still beat default by +0.036; if a similar micro-gain appears at the
  peak resolution it could edge above 78.7359. Cheapest remaining shot at a new best.
- `r608_default` would isolate how much of 608's gap is resample vs resolution, but given the
  448 peak it is low-value.

## Operational notes (for re-runs)

- **TTA step must force HF offline** (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, weights are
  already cached from training): the 608 run's first TTA attempt crashed in `build_lora_model`
  on a transient HuggingFace Hub call (`RuntimeError: Cannot send a request, as the client has
  been closed`). Training + SWA were fine; only TTA reloads the backbone and hit the network.
  Re-running TTA-only offline produced the zip in ~6.5 min (no retrain).
