"""Baseline forecaster F0 — encoder–decoder LSTM.

The encoder ingests the lookback window, the decoder emits the multi-horizon
forecast in one shot from the encoder's final hidden state. This is the
"F0" of §3 of the experiment spec; the physics-guided variant F1 layers
soft constraints on top of the same architecture and lives in a companion
module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class LSTMEncDecConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    horizons: int = 5
    dropout: float = 0.0


class LSTMEncoderDecoder(nn.Module):
    """Simple LSTM encoder + linear decoder over the final hidden state.

    The decoder is a single Linear that emits ``H`` values in one shot. This
    keeps the baseline deliberately small: the paper's headline claim is
    about *transfer* (Path A) and *calibration* (Path B), not about the
    absolute skill of a bespoke sequence-to-sequence architecture.
    """

    def __init__(self, cfg: LSTMEncDecConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, F]
        _, (h_n, _) = self.encoder(x)
        # h_n: [num_layers, B, hidden]; take the last layer.
        context = h_n[-1]
        return self.head(context)                       # [B, horizons]
