"""Extract a trained FET model's predictions over the FULL train set, for
iterative relabeling (self-distillation).

A linear teacher head on frozen CLIP features is weak; a fully-trained FET/LoRA
model is a much stronger label cleaner. We run flip-TTA inference over every
train image (in ImageFolder order, matching outputs/cache/train_features.pt's
image_names) and save softmax probabilities. finetune_fet's prepare_targets can
then fold these in via --teacher-preds-path to recover/relabel more of the ~37%
noisy labels than the linear teacher could -> cleaner targets -> stronger model.

Usage:
  python tools/extract_model_preds.py --work-dir outputs_fet_c448_elr \
      --checkpoint best --out outputs/cache/fet_preds_round0.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from robustft.data import IndexedImageDataset, build_finetune_transforms
from robustft.fet_model import build_fet_model
from robustft.robust_utils import choose_checkpoint


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--checkpoint", choices=("best", "last", "full"), default="best")
    p.add_argument("--train-dir", default=str(config.TRAIN_DIR))
    p.add_argument("--out", required=True, help="save path for probs tensor (.pt)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--no-flip-tta", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    ckpt_path = choose_checkpoint(Path(args.work_dir) / "lora", args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    t = ckpt.get("args", {})
    img_size = t.get("img_size", 448)
    model = build_fet_model(
        len(class_names), t.get("lora_rank", 32), t.get("lora_alpha", 64.0),
        0.0, None, device,
        lora_blocks=t.get("lora_blocks", 12), lora_target=t.get("lora_target", "attn_mlp"),
        img_size=img_size,
        num_parts=t.get("num_parts", 8), part_channels=t.get("part_channels", 16),
        local_depth=t.get("local_depth", 2), local_scale=t.get("local_scale", 0.5),
        use_pfi=False, gaussian_ksize=t.get("gaussian_ksize", 15),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not [k for k in missing if not k.startswith("pfi_")], f"missing {missing}"
    model.eval()

    base = ImageFolder(args.train_dir)
    paths = [pp for pp, _ in base.samples]
    image_names = [Path(pp).name for pp in paths]
    _, eval_tf = build_finetune_transforms(model.backbone,
                                           t.get("crop_min_scale", 0.8), img_size=img_size)
    loader = DataLoader(IndexedImageDataset(paths, eval_tf), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=not args.no_pin)
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')}); predicting {len(paths)} train imgs", flush=True)

    probs = torch.zeros(len(paths), len(class_names), dtype=torch.float32)
    done = 0
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            if not args.no_flip_tta:
                logits = logits + model(torch.flip(images, dims=[3]))
        probs[idx] = logits.float().softmax(1).cpu()
        done += images.size(0)
        if done % (args.batch_size * 50) < args.batch_size:
            print(f"  {done}/{len(paths)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"probs": probs, "image_names": image_names,
                "class_names": class_names, "src": str(ckpt_path)}, args.out)
    pmax = probs.max(1).values
    print(f"saved {args.out}  probs{tuple(probs.shape)}  mean_conf={pmax.mean():.3f} "
          f"p90_conf={pmax.kthvalue(int(0.9*len(pmax))).values:.3f}", flush=True)


if __name__ == "__main__":
    main()
