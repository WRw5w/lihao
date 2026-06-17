"""Model soup lab: evaluate same-architecture checkpoints on the held-out 90/10
val, then build a Uniform Soup and a Greedy Soup (Wortsman et al. 2022).

Constraints for a fair experiment:
- Only checkpoints with identical architecture (lora_rank=32, attn_mlp, 448) are
  poolable (weight shapes must match to average).
- Only 90/10-trained checkpoints (same seed-42 split) are used, so the held-out
  val is genuinely unseen for all of them -> val ranking & greedy accept/reject
  are fair. Full-data models (trained on 100%) are excluded (their val is biased).

Memory trick: the frozen backbone is byte-identical across checkpoints, so we
only keep/average the trainable keys (lora* + head*) and reuse one backbone copy.

Selection metric: mid_03_06 (kNN-agreement mid band) — more leaderboard-aligned
than noisy_all (which rewards fitting noisy val labels). Both reported.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from robustft.data import IndexedImageDataset, build_finetune_transforms
from robustft.denoise import knn_agreement
from robustft.engine import stratified_split
from robustft.models import build_lora_model
from finetune_lora import evaluate_images

SEL = "mid_03_06"
CAND_GLOBS = [
    "exp_pipelines/run60_c448_dr_rank32_keep90/lora/ep*.pt",
    "exp_pipelines/run60_c448_dr_rank32_keep90/lora/best_ep*.pt",
    "exp_pipelines/auto_c448_rank32/lora/best.pt",
    "exp_pipelines/auto_c448_dr_rank32/lora/best.pt",
    "exp_pipelines/auto_c448_dr_rank32_keep90/lora/best.pt",
    # v4: add fresh STRONG breadth models (val-gated: weak ones won't crack top-10)
    "exp_pipelines/breadth_gce/lora/best.pt",
    "exp_pipelines/breadth_keep85/lora/best.pt",
    "exp_pipelines/breadth_keep95/lora/best.pt",
    "exp_pipelines/breadth_aug06/lora/best.pt",
    "exp_pipelines/breadth_ema9995/lora/best.pt",
    "exp_pipelines/auto_c448_drecall/lora/best.pt",
]


def compatible(a):
    return a.get("lora_rank") == 32 and a.get("lora_target") == "attn_mlp" and a.get("img_size") == 448


def trainable_keys(sd):
    return [k for k in sd if ("lora" in k.lower()) or k.startswith("head.")]


def sig(sd):
    ks = trainable_keys(sd)
    return round(sum(float(sd[k].float().abs().sum()) for k in ks[:8]), 2)


def avg_subs(subs):
    return {k: torch.stack([s[k].float() for s in subs]).mean(0).to(subs[0][k].dtype) for k in subs[0]}


@torch.inference_mode()
def main():
    device = torch.device("cuda")
    cache = torch.load("outputs/cache/train_features.pt", map_location="cpu")
    labels, class_names, feats = cache["labels"], cache["class_names"], cache["features"]
    num_classes = len(class_names)
    tr_idx, va_idx = stratified_split(labels, 0.1, 42)

    f16 = feats.to(device)
    y = labels.to(device)
    tr, va = tr_idx.to(device), va_idx.to(device)
    val_agree = knn_agreement(f16[va], y[va], f16[tr], y[tr], k=16, exclude_self=False).to(device)
    val_labels = y[va]

    paths = [p for p, _ in ImageFolder(str(config.TRAIN_DIR)).samples]
    val_paths = [paths[i] for i in va_idx.tolist()]

    cps = sorted({p for g in CAND_GLOBS for p in glob.glob(g)})

    eval_tf = loader = backbone_ref = ref_a = None
    cand, seen = [], set()
    print(f"=== 评测 {len(cps)} 个候选文件(同架构 rank32/attn_mlp/448, 90/10 held-out val) ===", flush=True)
    for cp in cps:
        ck = torch.load(cp, map_location="cpu")
        a = ck.get("args", {})
        if not compatible(a):
            continue
        sd = ck["model"]
        s = sig(sd)
        if s in seen:
            continue
        seen.add(s)
        model = build_lora_model(num_classes, 32, a.get("lora_alpha", 64.0), 0.0, None, device,
                                 lora_blocks=a.get("lora_blocks", 12), lora_target="attn_mlp", img_size=448)
        model.load_state_dict(sd)
        if eval_tf is None:
            ref_a = a
            _, eval_tf = build_finetune_transforms(model.backbone, a.get("crop_min_scale", 0.8), img_size=448)
            loader = DataLoader(IndexedImageDataset(val_paths, eval_tf), batch_size=128,
                                shuffle=False, num_workers=2, pin_memory=False)
            backbone_ref = {k: v.clone() for k, v in sd.items()}
        bands = evaluate_images(model, loader, val_labels, val_agree, device)
        cand.append((cp, {k: sd[k].clone() for k in trainable_keys(sd)}, bands))
        print(f"  {bands[SEL]:.4f}  noisy={bands['noisy_all']:.4f}  {Path(cp).parent.parent.name}/{Path(cp).name}", flush=True)
        del model
        torch.cuda.empty_cache()

    cand.sort(key=lambda x: x[2][SEL], reverse=True)
    top = cand[:10]
    print(f"\n=== top-10 by {SEL} (共 {len(cand)} 候选) ===", flush=True)
    for cp, _, b in top:
        print(f"  {b[SEL]:.4f}  noisy={b['noisy_all']:.4f}  {Path(cp).parent.parent.name}/{Path(cp).name}", flush=True)

    def eval_soup(subs):
        merged = dict(backbone_ref)
        merged.update(avg_subs(subs))
        m = build_lora_model(num_classes, 32, ref_a.get("lora_alpha", 64.0), 0.0, None, device,
                             lora_blocks=ref_a.get("lora_blocks", 12), lora_target="attn_mlp", img_size=448)
        m.load_state_dict(merged)
        b = evaluate_images(m, loader, val_labels, val_agree, device)
        del m
        torch.cuda.empty_cache()
        return b, merged

    # uniform soup of top-10
    b_uni, sd_uni = eval_soup([s for _, s, _ in top])

    # greedy soup: sort desc, add if it does not hurt SEL
    print(f"\n=== Greedy Soup ===", flush=True)
    greedy = [top[0][1]]
    best = top[0][2][SEL]
    print(f"  start: {Path(top[0][0]).name}  {SEL}={best:.4f}", flush=True)
    for cp, sub, _ in top[1:]:
        cb, _ = eval_soup(greedy + [sub])
        if cb[SEL] >= best - 1e-9:
            greedy.append(sub)
            best = cb[SEL]
            print(f"  + 接受 {Path(cp).name} -> {SEL}={best:.4f}", flush=True)
        else:
            print(f"  - 拒绝 {Path(cp).name} ({cb[SEL]:.4f} < {best:.4f})", flush=True)
    b_greedy, sd_greedy = eval_soup(greedy)

    for tag, sd in [("v4_uniform", sd_uni), ("v4_greedy", sd_greedy)]:
        out = Path(f"outputs_soup_{tag}/lora/full.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": sd, "class_names": class_names, "args": ref_a, "epoch": f"soup_{tag}"}, out)

    print(f"\n=== 结果汇总（val, {SEL} / noisy_all）===", flush=True)
    print(f"  最优单模型:        {top[0][2][SEL]:.4f} / {top[0][2]['noisy_all']:.4f}", flush=True)
    print(f"  Uniform Soup(10):  {b_uni[SEL]:.4f} / {b_uni['noisy_all']:.4f}", flush=True)
    print(f"  Greedy Soup({len(greedy)}):   {b_greedy[SEL]:.4f} / {b_greedy['noisy_all']:.4f}", flush=True)
    print("SOUP LAB DONE", flush=True)


if __name__ == "__main__":
    main()
