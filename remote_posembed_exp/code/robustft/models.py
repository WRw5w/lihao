"""Model components: cosine classifier head and LoRA adapters for the ViT."""

from __future__ import annotations

import contextlib
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "vit_base_patch32_clip_224.openai"


def _aligned_resample_abs_pos_embed(posemb, new_size, old_size=None, num_prefix_tokens=1,
                                    interpolation="bicubic", antialias=True, verbose=False):
    """Drop-in for timm's ``resample_abs_pos_embed`` using bilinear + align_corners=True
    + antialias=False instead of the default bicubic + antialias + align_corners=False.

    For the 7x7 CLIP-B/32 grid, an odd target side N = 2k-1 (13 @416px, 19 @608px)
    maps every original anchor exactly onto an output node, so the pretrained
    position embeddings are preserved losslessly; the in-between nodes are exact
    linear means of their neighbours. timm's default blurs *all* anchors (verified:
    max |anchor - original| = 0.057 default vs 0.000 aligned at 416)."""
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb
    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = (hw, hw)
    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix = None
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float().reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode="bilinear", align_corners=True, antialias=False)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim).to(orig_dtype)
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb


@contextlib.contextmanager
def patched_pos_resample(mode: str):
    """Temporarily swap timm's abs-pos-embed resampler at every call site it is
    looked up from. mode='aligned' installs the lossless bilinear+align_corners=True
    variant; any other value is a no-op (timm default bicubic). Restored on exit so
    nothing else in the process is affected."""
    if mode != "aligned":
        yield
        return
    import timm.layers as _layers
    import timm.layers.pos_embed as _pe
    import timm.models.vision_transformer as _vt
    targets = [m for m in (_pe, _layers, _vt) if hasattr(m, "resample_abs_pos_embed")]
    saved = [(m, m.resample_abs_pos_embed) for m in targets]
    for m in targets:
        m.resample_abs_pos_embed = _aligned_resample_abs_pos_embed
    try:
        yield
    finally:
        for m, fn in saved:
            m.resample_abs_pos_embed = fn


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
    def __init__(self, backbone: nn.Module, head: CosineClassifier, feat_fuse: int = 0,
                 attn_pool: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # feat_fuse>0: fuse the CLS tokens of the last K transformer blocks with a
        # learnable softmax-weighted sum (uniform init = genuine multi-layer fusion).
        # Tests whether earlier-layer features lift the frozen-B/32 ceiling. Compliant
        # (single model, single inference; dim stays D so the head is unchanged).
        self.feat_fuse = feat_fuse
        if feat_fuse > 0:
            self.fuse_weight = nn.Parameter(torch.zeros(feat_fuse))
        # attn_pool: pool the last-block patch tokens with a learned attention query
        # (init zeros -> mean-pool of patches) instead of taking the CLS token. Tests
        # whether the frozen patch tokens carry usable signal the CLS misses.
        self.attn_pool = attn_pool
        if attn_pool:
            self.pool_query = nn.Parameter(torch.zeros(backbone.num_features))

    def _attn_pool_feat(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.patch_embed(x)
        x = b._pos_embed(x)
        x = b.patch_drop(x)
        x = b.norm_pre(x)
        for blk in b.blocks:
            x = blk(x)
        x = b.norm(x)
        patches = x[:, 1:]  # (B, L, D) drop CLS
        scores = (patches @ self.pool_query) / (patches.size(-1) ** 0.5)  # (B, L)
        w = scores.float().softmax(-1).to(patches.dtype)
        return (w.unsqueeze(-1) * patches).sum(1)  # (B, D)

    def _fused_feat(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.patch_embed(x)
        x = b._pos_embed(x)
        x = b.patch_drop(x)
        x = b.norm_pre(x)
        start = len(b.blocks) - self.feat_fuse
        cls = []
        for i, blk in enumerate(b.blocks):
            x = blk(x)
            if i >= start:
                cls.append(x[:, 0])
        feats = b.norm(torch.stack(cls, dim=1))  # (B, K, D), frozen final LN per token
        w = self.fuse_weight.softmax(0).to(feats.dtype)
        return (feats * w[None, :, None]).sum(1)  # (B, D)

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_pool:
            return self._attn_pool_feat(x)
        if self.feat_fuse > 0:
            return self._fused_feat(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.normalize(self.extract_feat(x).float(), dim=-1)
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
                     img_size: int = 224, peft: str = "lora", feat_fuse: int = 0,
                     attn_pool: bool = False, pos_resample: str = "timm") -> LoraClassifier:
    # img_size != 224 makes timm resample the CLIP position embeddings; the
    # pretrained weights themselves are unchanged (competition-compliant).
    # pos_resample='aligned' uses lossless bilinear+align_corners=True resampling
    # (preserves the pretrained anchors exactly on odd grids, e.g. 13x13 @416px).
    with patched_pos_resample(pos_resample):
        backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=img_size)
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, rank, alpha, lora_dropout, last_blocks=lora_blocks, target=lora_target, peft=peft)
    head = CosineClassifier(backbone.num_features, num_classes, dropout=0.0)
    if head_state is not None:
        head.load_state_dict(head_state)
    model = LoraClassifier(backbone, head, feat_fuse=feat_fuse, attn_pool=attn_pool).to(device)
    return model
