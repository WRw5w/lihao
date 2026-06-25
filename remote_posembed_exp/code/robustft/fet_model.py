"""FET-enhanced LoRA classifier for CLIP ViT-B/32.

Architecture:
  CLIP ViT-B/32 (frozen) + LoRA adapters
  + LocalBranch (CBAM + part-Transformer)
  + fused feature -> CosineClassifier

Forward path at train time optionally applies PFI (Pair Feature Interaction)
from FET: intra-class closest / inter-class closest pairs exchange attention
features to enrich the global embedding.

PFI requires at least 2 classes in the batch (use BalancedBatchSampler or
--pfi-classes / --pfi-images flags in finetune_fet.py). When the batch is too
small or contains only 1 class, PFI is silently skipped.
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from robustft.models import CosineClassifier, LoRALinear, inject_lora, MODEL_NAME
from robustft.fet_modules import LocalBranch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdist(v: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance matrix (B, B)."""
    return (
        -2 * v @ v.t()
        + v.pow(2).sum(1, keepdim=True)
        + v.pow(2).sum(1, keepdim=True).t()
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FETLoraClassifier(nn.Module):
    """CLIP ViT-B/32 + LoRA + FET LocalBranch.

    Args:
        backbone:      timm CLIP model (LoRA already injected, frozen base)
        head:          CosineClassifier
        num_patches:   spatial patch count (49 @ 224, 196 @ 448)
        num_parts:     part groups for LocalBranch (default 8)
        part_channels: CBAM output channels per part (default 16)
        local_depth:   PartTransformer layers (default 2)
        local_scale:   how much to add local features (default 0.5)
        use_pfi:       enable Pair Feature Interaction at training time
        gaussian_ksize: smoothing kernel size for decision mask
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: CosineClassifier,
        num_patches: int,
        num_parts: int = 8,
        part_channels: int = 16,
        local_depth: int = 2,
        local_scale: float = 0.5,
        use_pfi: bool = True,
        gaussian_ksize: int = 15,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.local_scale = local_scale
        self.use_pfi = use_pfi

        embed_dim: int = backbone.num_features  # 768 for ViT-B/32

        self.local_branch = LocalBranch(
            embed_dim=embed_dim,
            num_patches=num_patches,
            num_parts=num_parts,
            part_channels=part_channels,
            depth=local_depth,
            gaussian_ksize=gaussian_ksize,
        )
        # project local feat into the same space if needed (identity here since dims match)
        self.local_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # PFI attention weighting (requires attention values from ViT)
        if use_pfi:
            self.pfi_norm = nn.LayerNorm(embed_dim)
            self.pfi_pool = nn.AdaptiveAvgPool1d(1)

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _get_patch_features(self, x: torch.Tensor):
        """Run backbone and return (cls_feat, patch_feats, attn_weights).

        patch_feats: (B, L, D)  — spatial token features before final norm
        attn_weights: (B, L, D) — last-layer attention * values (approximation)

        We hook the last block's output to get patch features.
        """
        # timm CLIP ViT forward_features returns the CLS token after norm.
        # We need intermediate patch features: run block by block.
        backbone = self.backbone

        x = backbone.patch_embed(x)
        x = backbone._pos_embed(x)
        x = backbone.patch_drop(x)
        x = backbone.norm_pre(x)

        # iterate over transformer blocks
        for blk in backbone.blocks:
            x = blk(x)

        # x: (B, 1+L, D)  — CLS token is index 0
        cls_token = backbone.norm(x[:, 0])          # (B, D)
        patch_tokens = x[:, 1:]                     # (B, L, D) — un-normed spatial tokens

        return cls_token, patch_tokens

    def _pfi_attention_feature(self, patch_tokens: torch.Tensor, attn_ref: torch.Tensor) -> torch.Tensor:
        """Attention-guided global feature (simplified; uses spatial avg weighting)."""
        # attn_ref: (B, L, D) — used as soft spatial weights
        weight = attn_ref.abs().mean(dim=-1, keepdim=True).softmax(dim=1)  # (B, L, 1)
        weighted = (patch_tokens * weight)  # (B, L, D)
        feat = self.pfi_norm(weighted)
        feat = self.pfi_pool(feat.transpose(1, 2)).squeeze(-1)  # (B, D)
        return feat

    # ------------------------------------------------------------------
    # PFI pair selection (identical logic to FET GlobalBranch.get_pairs)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pairs(embeddings: torch.Tensor, labels: torch.Tensor):
        dist = _pdist(embeddings)
        B = embeddings.size(0)
        labels_col = labels.unsqueeze(1)
        same = (labels_col == labels_col.t()).clone()

        dist_same = dist.clone()
        same_no_diag = same.clone().fill_diagonal_(False)
        dist_same[~same_no_diag] = float("inf")
        intra_idxs = dist_same.argmin(dim=1)

        dist_diff = dist.clone()
        same_diag = same.clone().fill_diagonal_(True)
        dist_diff[same_diag] = float("inf")
        inter_idxs = dist_diff.argmin(dim=1)

        arange = torch.arange(B, device=embeddings.device)
        intra_pairs = torch.stack([arange, intra_idxs], dim=1)
        inter_pairs = torch.stack([arange, inter_idxs], dim=1)
        intra_labels = torch.stack([labels, labels[intra_idxs]], dim=1)
        inter_labels = torch.stack([labels, labels[inter_idxs]], dim=1)
        return intra_pairs, inter_pairs, intra_labels, inter_labels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, images: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor | tuple:
        cls_feat, patch_feats = self._get_patch_features(images)  # (B,D), (B,L,D)

        # decision_mask: uniform keep — CLIP ViT-B/32 has no dynamic pruning;
        # we pass all-ones so CBAM spatial gate is the sole selector.
        B, L, D = patch_feats.shape
        decision_mask = torch.ones(B, L, 1, device=images.device)

        local_feat, _ = self.local_branch(decision_mask, patch_feats.detach())
        local_feat = self.local_proj(local_feat)

        fused = cls_feat + self.local_scale * local_feat      # (B, D)
        fused_norm = F.normalize(fused.float(), dim=-1)
        logits = self.head(fused_norm)

        if not self.training or targets is None:
            return logits

        # ---- PFI (only when batch has ≥ 2 classes) --------------------
        if not self.use_pfi:
            return logits

        unique_classes = targets.unique()
        if unique_classes.numel() < 2:
            return logits

        with torch.no_grad():
            intra_pairs, inter_pairs, intra_labels, inter_labels = self._get_pairs(
                cls_feat.detach(), targets)

        def _pfi_feat(idx_a: torch.Tensor, idx_b: torch.Tensor, local_cache: torch.Tensor) -> torch.Tensor:
            # local_cache: pre-computed local_feat for ALL samples in this batch (B, D)
            pa = patch_feats[idx_a]                              # (n, L, D)
            pb = patch_feats[idx_b]                              # (n, L, D)
            fa = cls_feat[idx_a]                                 # (n, D)
            enriched = self._pfi_attention_feature(pa, pb)       # (n, D)
            fused_pfi = fa + enriched + self.local_scale * local_cache[idx_a]
            return F.normalize(fused_pfi.float(), dim=-1)

        # intra (same class) pairs
        f_self    = _pfi_feat(intra_pairs[:, 0], intra_pairs[:, 0], local_feat)
        f_intra   = _pfi_feat(intra_pairs[:, 0], intra_pairs[:, 1], local_feat)
        # inter (diff class) pairs
        f_inter_s = _pfi_feat(inter_pairs[:, 0], inter_pairs[:, 0], local_feat)
        f_inter_o = _pfi_feat(inter_pairs[:, 0], inter_pairs[:, 1], local_feat)

        # concatenate for auxiliary CE
        pfi_feats   = torch.cat([f_self, f_intra, f_inter_s, f_inter_o], dim=0)
        pfi_targets = torch.cat([
            intra_labels[:, 0], intra_labels[:, 1],
            inter_labels[:, 0], inter_labels[:, 1],
        ], dim=0)
        pfi_logits = self.head(pfi_feats)

        return logits, pfi_logits, pfi_targets


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_fet_model(
    num_classes: int,
    rank: int,
    alpha: float,
    lora_dropout: float,
    head_state: dict | None,
    device: torch.device,
    lora_blocks: int = 12,
    lora_target: str = "attn_mlp",
    img_size: int = 448,
    num_parts: int = 8,
    part_channels: int = 16,
    local_depth: int = 2,
    local_scale: float = 0.5,
    use_pfi: bool = True,
    gaussian_ksize: int = 15,
) -> FETLoraClassifier:
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=img_size)
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, rank, alpha, lora_dropout, last_blocks=lora_blocks, target=lora_target)

    head = CosineClassifier(backbone.num_features, num_classes, dropout=0.0)
    if head_state is not None:
        head.load_state_dict(head_state)

    # num_patches from timm
    grid = backbone.patch_embed.grid_size  # e.g. (14,14) for 448px with patch32
    num_patches = grid[0] * grid[1]

    model = FETLoraClassifier(
        backbone, head,
        num_patches=num_patches,
        num_parts=num_parts,
        part_channels=part_channels,
        local_depth=local_depth,
        local_scale=local_scale,
        use_pfi=use_pfi,
        gaussian_ksize=gaussian_ksize,
    ).to(device)
    return model
