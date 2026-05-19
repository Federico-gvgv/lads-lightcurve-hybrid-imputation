# src/models/conv_transformer_residual.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class ConvTransformerResidualImputer(nn.Module):
    """
    Conv frontend + Transformer encoder residual imputer.

    Input per time step: [y_in, obs_mask]
    Output: delta (residual) per time step
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        conv_kernel: int = 9,      # odd
        conv_layers: int = 2,
        input_channels: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels

        convs = []
        in_ch = input_channels
        ch = d_model
        for i in range(conv_layers):
            convs.append(nn.Conv1d(in_ch, ch, kernel_size=conv_kernel, padding=conv_kernel//2))
            convs.append(nn.GELU())
            convs.append(nn.Dropout(dropout))
            in_ch = ch
        self.conv_frontend = nn.Sequential(*convs)

        # back to (B,L,d_model)
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
        obs_mask: (B,L), optional for standard two-channel mode
        returns delta: (B,L)
        """
        if obs_mask is None:
            x = y_in
            B, C, L = x.shape
            assert C == self.input_channels, f"Expected C={self.input_channels}, got {C}"
        else:
            B, L = y_in.shape
            x = torch.stack([y_in, obs_mask], dim=1)  # (B,2,L)
            assert self.input_channels == 2, "Two-argument forward requires input_channels=2"
        assert L == self.seq_len, f"Expected L={self.seq_len}, got {L}"

        h = self.conv_frontend(x)                 # (B,d_model,L)

        # to (B,L,d)
        h = h.transpose(1, 2).contiguous()        # (B,L,d)

        pos = torch.arange(L, device=y_in.device)
        h = h + self.pos_emb(pos)[None, :, :]     # (B,L,d)

        h = self.encoder(h)                       # (B,L,d)
        delta = self.head(h).squeeze(-1)          # (B,L)
        return delta
