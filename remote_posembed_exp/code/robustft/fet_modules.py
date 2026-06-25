"""FET-style local branch for CLIP ViT-B/32.

Ported from NICE-FUTURE/FET-FGVC (Pattern Recognition 2024).

Key changes from the original:
- Dropped torch_geometric dependency: GCNBlock replaced by a 1-layer
  Transformer encoder operating over num_parts part tokens.
- CBAM and SpatialGate kept verbatim.
- Works with CLIP ViT-B/32 patch feature dim (D=768, num_patches depends
  on img_size: 49 for 224, 196 for 448).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CBAM (Channel + Spatial attention)
# ---------------------------------------------------------------------------

class _ChannelGate(nn.Module):
    def __init__(self, gate_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.avg_pool2d(x, (x.size(2), x.size(3))).squeeze(-1).squeeze(-1)
        mx  = F.max_pool2d(x, (x.size(2), x.size(3))).squeeze(-1).squeeze(-1)
        scale = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * scale.unsqueeze(2).unsqueeze(3)


class _SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = lambda x: torch.cat(
            [x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)], dim=1)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor, decision_mask: torch.Tensor):
        compressed = self.compress(x)
        scale = self.conv(compressed) * decision_mask
        scale = torch.sigmoid(scale)
        return x * scale, scale.detach()


class CBAM(nn.Module):
    def __init__(self, gate_channels: int, part_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channel_gate = _ChannelGate(gate_channels, reduction_ratio)
        self.spatial_gate = _SpatialGate()
        self.fc = nn.Conv2d(gate_channels, part_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, decision_mask: torch.Tensor):
        x = self.channel_gate(x)
        x, parts_mask = self.spatial_gate(x, decision_mask)
        x = self.fc(x)
        return x, parts_mask


# ---------------------------------------------------------------------------
# Gaussian kernel helpers (same logic as FET utils)
# ---------------------------------------------------------------------------

def _generate_gaussian_kernel(kernel_size: int, sigma: float = 7.0) -> torch.Tensor:
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)


def _smooth_decision_mask(kernel: torch.Tensor, ksize: int, mask: torch.Tensor) -> torch.Tensor:
    padding = ksize // 2
    B, L, C = mask.shape
    s = int(math.sqrt(L))
    mask = mask.permute(0, 2, 1).reshape(B * C, 1, s, s)
    mask = F.conv2d(mask, kernel, padding=padding)
    mask = mask.reshape(B, C, s, s).permute(0, 2, 3, 1).reshape(B, L, C)
    return mask


# ---------------------------------------------------------------------------
# Part-relation module: lightweight Transformer (replaces GCN)
# ---------------------------------------------------------------------------

class _PartTransformer(nn.Module):
    """Single Transformer encoder layer over num_parts tokens."""
    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# LocalBranch
# ---------------------------------------------------------------------------

class LocalBranch(nn.Module):
    """Extracts and enriches representations of discriminative image regions.

    Args:
        embed_dim:    patch feature channels from CLIP backbone (e.g. 768)
        num_patches:  spatial tokens from backbone (49 @ 224, 196 @ 448)
        num_parts:    how many part groups to split features into (default 8)
        part_channels: output channels per part after CBAM projection
        depth:        number of PartTransformer layers for part relation
        gaussian_ksize: smoothing kernel for decision mask (None to disable)
    """
    def __init__(
        self,
        embed_dim: int,
        num_patches: int,
        num_parts: int = 8,
        part_channels: int = 16,
        depth: int = 2,
        gaussian_ksize: int | None = 15,
    ):
        super().__init__()
        assert embed_dim % num_parts == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_parts ({num_parts})"
        self.num_parts = num_parts
        self.part_channels = part_channels
        self.parts_dim = embed_dim // num_parts

        if gaussian_ksize:
            kernel = _generate_gaussian_kernel(gaussian_ksize)
            self.register_buffer("kernel", kernel)
            self.ksize = gaussian_ksize
        else:
            self.kernel = None
            self.ksize = None

        self.cbam_list = nn.ModuleList([
            CBAM(self.parts_dim, part_channels) for _ in range(num_parts)
        ])

        # aggregate part-region features -> part token of size embed_dim
        gcn_dim = part_channels * num_patches
        self.proj_in = (
            nn.Linear(gcn_dim, embed_dim)
            if gcn_dim != embed_dim
            else nn.Identity()
        )

        self.transformer_blocks = nn.ModuleList([
            _PartTransformer(embed_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def locate_parts(self, decision_mask: torch.Tensor, x: torch.Tensor):
        """Apply CBAM to each channel partition.

        Args:
            decision_mask: (B, L, 1) soft keep-mask from the global branch
            x:             (B, L, D) patch features

        Returns:
            (x_out, parts_masks)
            x_out:       (B, num_parts * part_channels, sqrt(L), sqrt(L))
            parts_masks: (B, num_parts, sqrt(L), sqrt(L))
        """
        if self.kernel is not None:
            decision_mask = _smooth_decision_mask(self.kernel, self.ksize, decision_mask)

        B, L, D = x.shape
        s = int(math.sqrt(L))
        x2d = x.permute(0, 2, 1).reshape(B, D, s, s)
        mask2d = decision_mask.permute(0, 2, 1).reshape(B, 1, s, s)

        parts_list, masks_list = [], []
        for chunk, cbam in zip(x2d.split(self.parts_dim, dim=1), self.cbam_list):
            out, m = cbam(chunk, mask2d)
            parts_list.append(out)     # (B, part_channels, s, s)
            masks_list.append(m)       # (B, 1, s, s)

        x_out = torch.cat(parts_list, dim=1)       # (B, num_parts*part_channels, s, s)
        parts_masks = torch.cat(masks_list, dim=1) # (B, num_parts, s, s)
        return x_out, parts_masks

    # ------------------------------------------------------------------
    def forward(
        self, decision_mask: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decision_mask: (B, L, 1)  keep-mask from backbone
            x:             (B, L, D)  patch features

        Returns:
            local_feat: (B, D)  enriched local representation
            parts_masks:(B, num_parts, s, s)
        """
        shortcut = x.mean(dim=1)  # (B, D)
        B, L, _ = x.shape

        x_out, parts_masks = self.locate_parts(decision_mask, x)  # (B, P*Cp, s, s)

        # Reshape: treat each part as a token -> (B, num_parts, embed_dim)
        s = int(math.sqrt(L))
        x_tokens = (
            x_out.view(B, self.num_parts, self.part_channels, s * s)
                 .reshape(B, self.num_parts, self.part_channels * s * s)
        )
        x_tokens = self.proj_in(x_tokens)  # (B, num_parts, embed_dim)

        for blk in self.transformer_blocks:
            x_tokens = blk(x_tokens)
        x_tokens = self.norm(x_tokens)

        local_feat = x_tokens.mean(dim=1) + shortcut  # (B, D)
        return local_feat, parts_masks
