# src/models/transformer_residual.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class TransformerResidualImputer(nn.Module):
    """
    Input per time step: [y_in, obs_mask]
    Outputs: delta (residual) per time step
    Final: y_hat = y_in + delta * (1-obs_mask)
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        input_channels: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.in_proj = nn.Linear(input_channels, d_model)

        # Learned positional embedding
        self.pos_emb = nn.Embedding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, y_in: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        y_in: (B,L) with obs_mask, or pre-built x: (B,C,L)
        obs_mask: (B,L) float 1 observed / 0 missing, optional
        returns delta: (B,L)
        """
        if obs_mask is None:
            x_ch = y_in
            B, C, L = x_ch.shape
            assert C == self.input_channels, f"Expected C={self.input_channels}, got {C}"
            x = x_ch.transpose(1, 2).contiguous()  # (B,L,C)
        else:
            B, L = y_in.shape
            x = torch.stack([y_in, obs_mask], dim=-1)  # (B,L,2)
            assert self.input_channels == 2, "Two-argument forward requires input_channels=2"
        assert L == self.seq_len, f"Expected L={self.seq_len}, got {L}"

        h = self.in_proj(x)  # (B,L,d)

        pos = torch.arange(L, device=y_in.device)
        h = h + self.pos_emb(pos)[None, :, :]  # (B,L,d)

        h = self.encoder(h)  # (B,L,d)
        delta = self.head(h).squeeze(-1)  # (B,L)
        return delta
