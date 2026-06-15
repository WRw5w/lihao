"""Weight-average (SWA / model soup) several same-architecture checkpoints into
ONE model. The output is a single set of weights -> single forward pass at
inference, i.e. still a single model (competition-compliant; this is a training
technique, not an output-level ensemble/vote/fusion).

Only valid for checkpoints with identical architecture (same lora-rank/target/
img-size). Backbone (frozen) weights are identical across ckpts; LoRA + head
weights get averaged.

Usage:
  python tools/swa_soup.py --checkpoints a.pt b.pt c.pt --out work/lora/full.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    states = [torch.load(c, map_location="cpu") for c in args.checkpoints]
    base = states[0]
    keys = list(base["model"].keys())
    assert all(set(s["model"].keys()) == set(keys) for s in states), "arch mismatch across checkpoints"

    avg = {}
    for k in keys:
        tensors = [s["model"][k] for s in states]
        if tensors[0].is_floating_point():
            avg[k] = (sum(t.float() for t in tensors) / len(tensors)).to(tensors[0].dtype)
        else:
            avg[k] = tensors[0]  # integer buffers: keep first
    base["model"] = avg
    base["epoch"] = "swa"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    print(f"souped {len(states)} checkpoints -> {out}")
    print("  sources:", ", ".join(Path(c).name for c in args.checkpoints))


if __name__ == "__main__":
    main()
