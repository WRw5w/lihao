"""Model components: cosine classifier head and LoRA adapters for the ViT."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "vit_base_patch32_clip_224.openai"


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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        out = out + self.dropout(x) @ self.lora_a.t() @ self.lora_b.t() * self.scaling
        return out


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float, last_blocks: int = 12,
                target: str = "attn", peft: str = "lora") -> list[str]:
    """target: "attn" adapts qkv/proj only; "attn_mlp" also adapts mlp fc1/fc2.
    peft: "lora" (additive low-rank) or "dora" (weight-decomposed, strictly generalises lora)."""
    adapter = DoRALinear if peft == "dora" else LoRALinear
    replaced = []
    first_block = max(0, len(model.blocks) - last_blocks)
    for blk_i, blk in enumerate(model.blocks):
        if blk_i < first_block:
            continue
        attn = blk.attn
        attn.qkv = adapter(attn.qkv, rank, alpha, dropout)
        attn.proj = adapter(attn.proj, rank, alpha, dropout)
        replaced += [f"blocks.{blk_i}.attn.qkv", f"blocks.{blk_i}.attn.proj"]
        if target == "attn_mlp":
            blk.mlp.fc1 = adapter(blk.mlp.fc1, rank, alpha, dropout)
            blk.mlp.fc2 = adapter(blk.mlp.fc2, rank, alpha, dropout)
            replaced += [f"blocks.{blk_i}.mlp.fc1", f"blocks.{blk_i}.mlp.fc2"]
    return replaced


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).

    Decomposes the pretrained weight into a per-output-row magnitude and a
    direction; LoRA updates the direction while a separate learnable magnitude
    vector rescales each row. Init is an exact identity (lora_b=0 -> delta=0,
    magnitude=||W0||), so it strictly generalises LoRA at zero accuracy risk.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.magnitude = nn.Parameter(base.weight.detach().norm(p=2, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_b @ self.lora_a) * self.scaling
        weight = self.base.weight + delta
        norm = weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        weight = self.magnitude.unsqueeze(1) * (weight / norm)
        return F.linear(self.dropout(x), weight, self.base.bias)


class LoraClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: CosineClassifier):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = F.normalize(feat.float(), dim=-1)
        return self.head(feat)


def build_frozen_backbone(device: torch.device) -> nn.Module:
    """Frozen CLIP visual tower for feature extraction."""
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model


def build_lora_model(num_classes: int, rank: int, alpha: float, lora_dropout: float,
                     head_state: dict | None, device: torch.device,
                     lora_blocks: int = 12, lora_target: str = "attn",
                     img_size: int = 224, peft: str = "lora") -> LoraClassifier:
    # img_size != 224 makes timm resample the CLIP position embeddings; the
    # pretrained weights themselves are unchanged (competition-compliant).
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=img_size)
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, rank, alpha, lora_dropout, last_blocks=lora_blocks, target=lora_target, peft=peft)
    head = CosineClassifier(backbone.num_features, num_classes, dropout=0.0)
    if head_state is not None:
        head.load_state_dict(head_state)
    model = LoraClassifier(backbone, head).to(device)
    return model
