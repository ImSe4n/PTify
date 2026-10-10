"""The rhythm model: a small Transformer encoder with four per-note heads.

Small on purpose (about 4M parameters at the defaults): the training set is 176
scores, and a bigger model would memorise them rather than learn rhythm.
Encoder-only, so every note sees its whole window in both directions: a note's
written value depends on what comes AFTER it as much as before.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rhythm.features import (IGNORE, N_INPUT, N_DELTA_CLASSES,
                             N_DURATION_CLASSES, N_KEYS)

HEADS = {"delta": N_DELTA_CLASSES, "duration": N_DURATION_CLASSES,
         "downbeat": 2, "hand": 2}


class RhythmTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           dropout=dropout, max_len=max_len)
        self.pitch = nn.Embedding(N_KEYS, d_model)
        self.cont = nn.Linear(N_INPUT, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout,
            batch_first=True, norm_first=True)
        # nested-tensor fast path is skipped with norm_first anyway; disabling it
        # silences the warning rather than hiding anything.
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict({k: nn.Linear(d_model, n)
                                    for k, n in HEADS.items()})

    def forward(self, pitch: torch.Tensor, cont: torch.Tensor,
                pad: torch.Tensor) -> dict[str, torch.Tensor]:
        """pitch (B,T) long, cont (B,T,4), pad (B,T) bool True=padding."""
        pos = torch.arange(pitch.shape[1], device=pitch.device)
        x = self.pitch(pitch) + self.cont(cont) + self.pos(pos)[None]
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        return {k: head(x) for k, head in self.heads.items()}


def rhythm_loss(logits: dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """Sum of per-head cross-entropies over labelled notes only.

    A head with no labelled note in the batch contributes 0, not NaN
    (`cross_entropy` averages over zero items otherwise).
    """
    total = torch.zeros((), device=next(iter(logits.values())).device)
    parts = {}
    for k, out in logits.items():
        target = batch[k]
        if (target != IGNORE).any():
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                   target.reshape(-1), ignore_index=IGNORE)
            total = total + loss
            parts[k] = float(loss.detach())
        else:
            parts[k] = 0.0
    return total, parts
