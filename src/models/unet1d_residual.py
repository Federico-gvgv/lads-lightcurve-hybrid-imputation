from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = x.shape[-1]
    if current_len == target_len:
        return x
    if current_len < target_len:
        return F.pad(x, (0, target_len - current_len))
    return x[..., :target_len]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.block = _ConvBlock(channels, channels, kernel_size=kernel_size, dropout=dropout)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _match_length(self.block(x), x.shape[-1])
        return self.act(self.norm(x + y))


class UNet1DResidualImputer(nn.Module):
    """
    Non-causal 1D U-Net residual imputer.

    Standard mode:
        delta = model(y_in, obs_mask)
        y_in, obs_mask: (B, L)

    Pre-built channel mode, used by Fourier-aware training:
        delta = model(x)
        x: (B, input_channels, L), e.g. [y_obs_zeroed, y_fourier_full,
           obs_mask, residual_visible] for input_channels=4.

    Returns:
        delta: (B, L)
    """

    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        # Cap the doubling schedule to keep the default model reasonably sized.
        max_channels = base_channels * 8
        skip_channels = [min(base_channels * (2 ** i), max_channels) for i in range(depth)]
        down_channels = [min(base_channels * (2 ** (i + 1)), max_channels) for i in range(depth)]

        self.stem = _ConvBlock(input_channels, skip_channels[0], kernel_size, dropout)

        downs = []
        in_ch = skip_channels[0]
        for out_ch in down_channels:
            downs.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(_group_count(out_ch), out_ch),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    _ConvBlock(out_ch, out_ch, kernel_size, dropout),
                )
            )
            in_ch = out_ch
        self.downs = nn.ModuleList(downs)

        self.bottleneck = _ResidualConvBlock(in_ch, kernel_size=kernel_size, dropout=dropout)

        ups = []
        dec_blocks = []
        for skip_ch in reversed(skip_channels):
            ups.append(nn.ConvTranspose1d(in_ch, skip_ch, kernel_size=2, stride=2))
            dec_blocks.append(_ConvBlock(skip_ch * 2, skip_ch, kernel_size, dropout))
            in_ch = skip_ch
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.head = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, y_in: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if obs_mask is None:
            x = y_in
            if x.ndim != 3:
                raise ValueError(f"Expected pre-built input with shape (B, C, L), got {tuple(x.shape)}")
            if x.shape[1] != self.input_channels:
                raise ValueError(f"Expected C={self.input_channels}, got {x.shape[1]}")
        else:
            if y_in.ndim != 2 or obs_mask.ndim != 2:
                raise ValueError("Two-argument forward expects y_in and obs_mask with shape (B, L)")
            if y_in.shape != obs_mask.shape:
                raise ValueError(f"y_in and obs_mask shapes differ: {tuple(y_in.shape)} vs {tuple(obs_mask.shape)}")
            if self.input_channels != 2:
                raise ValueError("Two-argument forward requires input_channels=2")
            x = torch.stack([y_in, obs_mask], dim=1)

        input_len = x.shape[-1]
        h = self.stem(x)
        skips = []
        for down in self.downs:
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, dec_block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = _match_length(h, skip.shape[-1])
            h = dec_block(torch.cat([h, skip], dim=1))

        delta = self.head(_match_length(h, input_len)).squeeze(1)
        return _match_length(delta, input_len)
