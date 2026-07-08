"""HRNet feature backbone for the two-channel spot detector (build stage 3).

A compact HRNetV2-style backbone: it maintains several **parallel branches at
different spatial resolutions** and repeatedly **fuses** them, so the
high-resolution branch never loses its spatial detail (critical for resolving
close, diffraction-limited spots in the high-overlap regime). The final
representation upsamples every branch back to full input resolution and
concatenates them, so all prediction heads operate at full ``H x W``.

Design notes
------------
* Input is ``[B, 2, H, W]`` (the two PMT channels). The stem stays at full
  resolution (stride 1) -- we deliberately do **not** downsample at the stem the
  way classification HRNets do, because the heads need pixel-accurate heatmaps.
* Branch ``i`` runs at resolution ``H / 2**i`` with ``base_channels * 2**i``
  channels. ``num_branches`` (default 3 -> {1, 1/2, 1/4}) and ``base_channels``
  set the size; this is intentionally a *small* HRNet -- phase 1 prioritises
  correctness and resolution over parameter count (see CLAUDE.md build order).
* Normalisation is GroupNorm, not BatchNorm: training batches are small and
  inference is often a single image, so batch statistics would be unreliable;
  GroupNorm is batch-size independent and behaves identically in train/eval.

The backbone returns a single full-resolution feature map ``[B, C, H, W]`` with
``C = sum(base_channels * 2**i)``; :attr:`out_channels` exposes ``C`` for the
heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HRNetBackbone", "build_backbone"]


def _norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with a sensible group count (largest power-of-two divisor <= 8)."""
    groups = 8
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class BasicBlock(nn.Module):
    """Residual block: two 3x3 convs at a fixed resolution and channel count."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm1 = _norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm2 = _norm(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + x)


class TransitionLayer(nn.Module):
    """Grow the branch set by one: append a new, half-resolution branch.

    The existing branches pass through unchanged; the new (lowest-resolution)
    branch is produced by a stride-2 3x3 conv from the previous last branch.
    """

    def __init__(self, prev_channels: int, new_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(prev_channels, new_channels, 3, 2, 1, bias=False),
            _norm(new_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, branches: list[torch.Tensor]) -> list[torch.Tensor]:
        return list(branches) + [self.downsample(branches[-1])]


class FuseModule(nn.Module):
    """Multi-resolution fusion: every output branch sums resampled copies of all
    input branches (HRNet's exchange unit).

    For output branch ``i`` and input branch ``j``:
      * ``j == i``: identity.
      * ``j  > i`` (input is lower-res): 1x1 conv to match channels, then
        bilinear upsample to branch ``i``'s size.
      * ``j  < i`` (input is higher-res): a chain of stride-2 3x3 convs that
        downsamples by ``2**(i-j)`` and maps to branch ``i``'s channels.
    """

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.k = len(channels)
        self.paths = nn.ModuleList()
        for i in range(self.k):
            row = nn.ModuleList()
            for j in range(self.k):
                if j == i:
                    row.append(nn.Identity())
                elif j > i:
                    row.append(
                        nn.Sequential(
                            nn.Conv2d(channels[j], channels[i], 1, bias=False),
                            _norm(channels[i]),
                        )
                    )
                else:  # j < i: downsample by 2**(i-j)
                    ops: list[nn.Module] = []
                    cin = channels[j]
                    for step in range(i - j):
                        last = step == i - j - 1
                        cout = channels[i] if last else channels[j]
                        ops += [nn.Conv2d(cin, cout, 3, 2, 1, bias=False), _norm(cout)]
                        if not last:
                            ops.append(nn.ReLU(inplace=True))
                        cin = cout
                    row.append(nn.Sequential(*ops))
            self.paths.append(row)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for i in range(self.k):
            acc = None
            for j in range(self.k):
                contrib = self.paths[i][j](xs[j])
                if j > i:  # upsample lower-res branch to branch i's size
                    contrib = F.interpolate(
                        contrib, size=xs[i].shape[-2:], mode="bilinear", align_corners=False
                    )
                acc = contrib if acc is None else acc + contrib
            out.append(self.relu(acc))
        return out


class HRModule(nn.Module):
    """One HRNet stage: per-branch residual blocks followed by a fusion."""

    def __init__(self, channels: list[int], blocks_per_branch: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(*[BasicBlock(c) for _ in range(blocks_per_branch)]) for c in channels
        )
        self.fuse = FuseModule(channels)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        return self.fuse(xs)


class HRNetBackbone(nn.Module):
    """Small HRNetV2 backbone returning a full-resolution feature map.

    Parameters
    ----------
    in_channels : input channels (2 for the two PMT channels).
    base_channels : channels of the highest-resolution branch; branch ``i`` has
        ``base_channels * 2**i``.
    num_branches : number of parallel resolution branches (resolutions
        ``1, 1/2, ... 1/2**(num_branches-1)``).
    blocks_per_branch : residual blocks per branch within each stage.

    Input ``H`` and ``W`` must be divisible by ``2**(num_branches-1)`` so the
    branch resolutions stay integer and fuse exactly.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 16,
        num_branches: int = 3,
        blocks_per_branch: int = 2,
    ) -> None:
        super().__init__()
        if num_branches < 1:
            raise ValueError("num_branches must be >= 1")
        self.num_branches = num_branches
        self.channels = [base_channels * (2 ** i) for i in range(num_branches)]
        self._divisor = 2 ** (num_branches - 1)

        c0 = self.channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, 3, 1, 1, bias=False),
            _norm(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c0, 3, 1, 1, bias=False),
            _norm(c0),
            nn.ReLU(inplace=True),
        )
        # Stage 1: blocks on the single full-resolution branch.
        self.stage1 = nn.Sequential(*[BasicBlock(c0) for _ in range(blocks_per_branch)])

        # Incrementally add a branch (transition) and run a fused stage.
        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        for b in range(2, num_branches + 1):
            self.transitions.append(TransitionLayer(self.channels[b - 2], self.channels[b - 1]))
            self.stages.append(HRModule(self.channels[:b], blocks_per_branch))

        self.out_channels = sum(self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % self._divisor or w % self._divisor:
            raise ValueError(
                f"input H,W=({h},{w}) must be divisible by {self._divisor} "
                f"(= 2**(num_branches-1)) for {self.num_branches}-branch HRNet"
            )
        x = self.stem(x)
        branches = [self.stage1(x)]
        for transition, stage in zip(self.transitions, self.stages):
            branches = transition(branches)
            branches = stage(branches)

        # HRNetV2 representation head: upsample all branches to full resolution
        # and concatenate.
        target = branches[0].shape[-2:]
        feats = [branches[0]] + [
            F.interpolate(b, size=target, mode="bilinear", align_corners=False)
            for b in branches[1:]
        ]
        return torch.cat(feats, dim=1)


def build_backbone(config: dict | None = None) -> HRNetBackbone:
    """Construct the HRNet backbone from a ``model:`` config block."""
    cfg = config or {}
    return HRNetBackbone(
        in_channels=int(cfg.get("in_channels", 2)),
        base_channels=int(cfg.get("base_channels", 16)),
        num_branches=int(cfg.get("num_branches", 3)),
        blocks_per_branch=int(cfg.get("blocks_per_branch", 2)),
    )
