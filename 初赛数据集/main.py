from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image, ImageFile
from timm.data import create_transform, resolve_data_config
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_NAME = "vit_base_patch32_clip_224.openai"
DEFAULT_OUTPUT_NAME = "pred_results.csv"


class LabeledImageDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), label


class UnlabeledImageDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), Path(path).name


class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.1, init_scale: float = 10.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * (x @ w.t())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def build_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def sample_items(items: list, max_items: int | None, seed: int) -> list:
    if max_items is None or max_items >= len(items):
        return items
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), max_items))
    return [items[i] for i in indices]


def load_train_samples(train_dir: Path, max_items: int | None, seed: int) -> tuple[list[tuple[str, int]], list[str]]:
    base = ImageFolder(str(train_dir))
    samples = sample_items(list(base.samples), max_items, seed)
    return samples, list(base.classes)


def load_test_paths(test_dir: Path, max_items: int | None, seed: int) -> list[str]:
    paths = sorted(
        str(p)
        for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    return sample_items(paths, max_items, seed)


def build_transform(model: nn.Module):
    cfg = resolve_data_config(model=model, use_test_size=True)
    return create_transform(
        input_size=cfg["input_size"],
        is_training=False,
        interpolation=cfg["interpolation"],
        mean=cfg["mean"],
        std=cfg["std"],
        crop_pct=cfg.get("crop_pct"),
        crop_mode=cfg.get("crop_mode"),
    )


def build_backbone(device: torch.device) -> nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model


def extract_features(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta_flip: bool = True,
    desc: str = "extract",
):
    if len(dataset) == 0:
        raise ValueError(f"{desc}: dataset is empty")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feat_dim = getattr(model, "num_features", None)
    if feat_dim is None:
        raise RuntimeError("backbone does not expose num_features")
    features = torch.empty((len(dataset), feat_dim), dtype=torch.float16)
    labels = None
    names: list[str] | None = None
    offset = 0
    model.eval()
    with torch.inference_mode():
        for images, meta in tqdm(loader, desc=desc, total=len(loader)):
            images = images.to(device, non_blocking=True)
            with maybe_autocast(device):
                feat = model(images)
                if tta_flip:
                    flipped = torch.flip(images, dims=[3])
                    feat_flip = model(flipped)
                    feat = 0.5 * (F.normalize(feat.float(), dim=-1) + F.normalize(feat_flip.float(), dim=-1))
                else:
                    feat = feat.float()
            feat = F.normalize(feat, dim=-1).cpu().to(torch.float16)
            bs = feat.size(0)
            features[offset : offset + bs] = feat
            if torch.is_tensor(meta):
                if labels is None:
                    labels = torch.empty((len(dataset),), dtype=torch.long)
                labels[offset : offset + bs] = meta
            else:
                if names is None:
                    names = []
                names.extend(list(meta))
            offset += bs
    if labels is not None:
        return features, labels
    return features, names or []


def save_tensor_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_tensor_cache(path: Path):
    return torch.load(path, map_location="cpu")


def cache_matches(payload: dict, expected_names: list[str], name_key: str) -> bool:
    if "features" not in payload or name_key not in payload:
        return False
    if int(payload["features"].size(0)) != len(expected_names):
        return False
    return list(payload[name_key]) == expected_names


def train_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    indices: torch.Tensor | None,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    dropout: float,
    device: torch.device,
    init_state: dict | None = None,
    log_prefix: str = "stage",
) -> CosineClassifier:
    features = features.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    if indices is None:
        indices = torch.arange(labels.size(0), device=device)
    else:
        indices = indices.to(device=device, dtype=torch.long)
    model = CosineClassifier(features.size(1), num_classes, dropout=dropout).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    n = indices.numel()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = indices[torch.randperm(n, device=device)]
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb = features[batch_idx]
            yb = labels[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += yb.size(0)
        scheduler.step()
        print(
            f"{log_prefix} epoch {epoch:02d}/{epochs} "
            f"loss={total_loss / max(1, total_seen):.4f} acc={total_correct / max(1, total_seen):.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
    return model


def select_clean_indices(
    model: CosineClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        probs = logits.softmax(dim=1)
        conf = probs[torch.arange(labels.size(0), device=labels.device), labels]
    num_classes = logits.size(1)
    keep_chunks = []
    class_stats = {}
    for cls in range(num_classes):
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).squeeze(1)
        if cls_idx.numel() == 0:
            continue
        k = max(1, int(cls_idx.numel() * keep_ratio))
        k = min(k, cls_idx.numel())
        topk = torch.topk(conf[cls_idx], k=k, largest=True).indices
        keep_chunks.append(cls_idx[topk])
        class_stats[int(cls)] = (int(cls_idx.numel()), int(k))
    keep_idx = torch.cat(keep_chunks) if keep_chunks else torch.empty(0, dtype=torch.long, device=labels.device)
    keep_idx = keep_idx[torch.argsort(keep_idx)]
    return keep_idx, conf


def save_predictions(
    output_csv: Path,
    filenames: list[str],
    pred_indices: list[int],
    idx_to_class: list[str],
) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for name, pred in zip(filenames, pred_indices):
            writer.writerow([name, idx_to_class[pred]])
    return output_csv


def zip_submission(csv_path: Path) -> Path:
    zip_path = csv_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
    return zip_path


def run_pipeline(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = build_device(args.device)
    print(f"device: {device}")
    print(f"model: {MODEL_NAME}")

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train_features.pt"
    test_cache = cache_dir / "test_features.pt"

    backbone = build_backbone(device)
    transform = build_transform(backbone)

    train_samples, class_names = load_train_samples(train_dir, args.max_train_samples, args.seed)
    test_paths = load_test_paths(test_dir, args.max_test_samples, args.seed)
    train_names = [Path(p).name for p, _ in train_samples]
    test_expected_names = [Path(p).name for p in test_paths]
    print(f"train samples: {len(train_samples)}")
    print(f"test samples: {len(test_paths)}")
    print(f"classes: {len(class_names)}")

    train_cache_ok = False
    if not args.rebuild_cache and train_cache.exists():
        cached = load_tensor_cache(train_cache)
        train_cache_ok = cache_matches(cached, train_names, "image_names") and cached.get("class_names") == class_names
        if train_cache_ok:
            train_features = cached["features"]
            train_labels = cached["labels"]
            class_names = cached.get("class_names", class_names)
        else:
            print("train feature cache does not match current data; rebuilding")

    if not train_cache_ok:
        train_dataset = LabeledImageDataset(train_samples, transform)
        train_features, train_labels = extract_features(
            backbone,
            train_dataset,
            device,
            batch_size=args.extract_batch_size,
            num_workers=args.num_workers,
            tta_flip=not args.no_tta_flip,
            desc="train features",
        )
        save_tensor_cache(
            train_cache,
            {
                "features": train_features,
                "labels": train_labels,
                "class_names": class_names,
                "image_names": train_names,
            },
        )

    test_cache_ok = False
    if not args.rebuild_cache and test_cache.exists():
        cached = load_tensor_cache(test_cache)
        test_cache_ok = cache_matches(cached, test_expected_names, "names")
        if test_cache_ok:
            test_features = cached["features"]
            test_names = cached["names"]
        else:
            print("test feature cache does not match current data; rebuilding")

    if not test_cache_ok:
        test_dataset = UnlabeledImageDataset(test_paths, transform)
        test_features, test_names = extract_features(
            backbone,
            test_dataset,
            device,
            batch_size=args.extract_batch_size,
            num_workers=args.num_workers,
            tta_flip=not args.no_tta_flip,
            desc="test features",
        )
        save_tensor_cache(
            test_cache,
            {
                "features": test_features,
                "names": test_names,
            },
        )

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_features = train_features.contiguous()
    test_features = test_features.contiguous()
    train_labels = train_labels.contiguous()
    train_idx = torch.arange(train_labels.size(0), device=device)
    train_features = train_features.to(device=device, dtype=torch.float32)
    test_features = test_features.to(device=device, dtype=torch.float32)
    train_labels = train_labels.to(device=device, dtype=torch.long)

    print("stage 1: noisy-label warmup")
    stage1 = train_head(
        train_features,
        train_labels,
        len(class_names),
        train_idx,
        epochs=args.stage1_epochs,
        batch_size=args.head_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.stage1_label_smoothing,
        dropout=args.dropout,
        device=device,
        init_state=None,
        log_prefix="stage1",
    )

    print("selecting clean samples")
    keep_idx, conf = select_clean_indices(stage1, train_features, train_labels, args.keep_ratio)
    keep_count = int(keep_idx.numel())
    print(f"kept {keep_count}/{train_labels.numel()} samples ({keep_count / max(1, train_labels.numel()):.2%})")

    print("stage 2: clean subset refinement")
    stage2 = train_head(
        train_features,
        train_labels,
        len(class_names),
        keep_idx,
        epochs=args.stage2_epochs,
        batch_size=args.head_batch_size,
        lr=args.lr * args.stage2_lr_mult,
        weight_decay=args.weight_decay,
        label_smoothing=args.stage2_label_smoothing,
        dropout=args.dropout,
        device=device,
        init_state=stage1.state_dict(),
        log_prefix="stage2",
    )

    artifact_dir = work_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": stage1.state_dict(),
            "num_features": int(train_features.size(1)),
            "num_classes": len(class_names),
            "class_names": class_names,
            "stage": 1,
        },
        artifact_dir / "stage1_head.pt",
    )
    torch.save(
        {
            "state_dict": stage2.state_dict(),
            "num_features": int(train_features.size(1)),
            "num_classes": len(class_names),
            "class_names": class_names,
            "stage": 2,
            "keep_ratio": args.keep_ratio,
        },
        artifact_dir / "stage2_head.pt",
    )

    print("predicting test set")
    stage2.eval()
    pred_indices: list[int] = []
    with torch.inference_mode():
        for start in tqdm(range(0, test_features.size(0), args.head_batch_size), desc="predict"):
            xb = test_features[start : start + args.head_batch_size]
            logits = stage2(xb)
            pred_indices.extend(logits.argmax(dim=1).tolist())

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = Path.cwd() / output_csv
    save_predictions(output_csv, test_names, pred_indices, class_names)
    zip_path = zip_submission(output_csv)

    meta = {
        "model": MODEL_NAME,
        "device": str(device),
        "train_samples": int(train_labels.numel()),
        "test_samples": int(test_features.size(0)),
        "classes": len(class_names),
        "keep_ratio": args.keep_ratio,
        "output_csv": str(output_csv),
        "output_zip": str(zip_path),
    }
    with (artifact_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"saved: {output_csv}")
    print(f"saved: {zip_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP ViT-B/32 baseline for the preliminary round")
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--test-dir", default="test")
    parser.add_argument("--work-dir", default="outputs")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--head-batch-size", type=int, default=8192)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--stage2-lr-mult", type=float, default=0.6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stage1-label-smoothing", type=float, default=0.1)
    parser.add_argument("--stage2-label-smoothing", type=float, default=0.05)
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--max-train-samples", type=int, default=None, help="debug helper")
    parser.add_argument("--max-test-samples", type=int, default=None, help="debug helper")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-tta-flip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
