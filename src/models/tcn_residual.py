# src/models/tcn_residual.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class _TCNBlock(nn.Module):
    """
    Dilated causal-ish conv block (non-causal, uses same padding) with residual connection.
    Works well for inpainting because it can use both sides (non-causal).
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size // 2) * dilation  # "same" padding for odd k
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L)
        y = self.net(x)
        # Conv1d with "same" padding can increase length by a few when dilation/pad mismatch for even k
        # Our pad choice keeps length stable for odd kernel_size. Still, safe-crop if needed.
        if y.shape[-1] != x.shape[-1]:
            L = x.shape[-1]
            y = y[..., :L]
        return self.norm(x + y)


class TCNResidualImputer(nn.Module):
    """
    Input per time step: [y_in, obs_mask]  -> (B,L,2)
    Outputs: delta (residual) per time step -> (B,L)
    Final in notebook: y_hat = y_in + delta*(1-obs_mask)
    """
    def __init__(
        self,
        seq_len: int,
        channels: int = 128,
        num_blocks: int = 10,
        kernel_size: int = 9,   # use odd number
        dropout: float = 0.1,
        input_channels: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels

        self.in_proj = nn.Conv1d(input_channels, channels, kernel_size=1)

        # dilation schedule: 1,2,4,... repeats if num_blocks > log2(L)
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** (i % 10)  # repeats every 10; you can change 10 to control max dilation
            blocks.append(_TCNBlock(channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, y_in: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        y_in: (B,L) with obs_mask, or pre-built x: (B,C,L)
        obs_mask: (B,L), optional for standard two-channel mode
        returns delta: (B,L)
        """
        if obs_mask is None:
            x = y_in
            B, C, L = x.shape
            assert C == self.input_channels, f"Expected C={self.input_channels}, got {C}"
        else:
            B, L = y_in.shape
            x = torch.stack([y_in, obs_mask], dim=1)  # (B,2,L) for Conv1d
            assert self.input_channels == 2, "Two-argument forward requires input_channels=2"
        assert L == self.seq_len, f"Expected L={self.seq_len}, got {L}"

        h = self.in_proj(x)                       # (B,C,L)
        h = self.blocks(h)                        # (B,C,L)
        delta = self.head(h).squeeze(1)           # (B,L)
        return delta
