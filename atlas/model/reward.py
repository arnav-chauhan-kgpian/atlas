"""Reward model for preference learning.

A bidirectional Transformer encoder that maps a token sequence to a scalar
reward. Used for RLHF: the reward model scores completions so the policy
can be optimized toward higher-scoring outputs.

Includes Bradley-Terry and margin ranking loss functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """Transformer encoder → masked mean pool → scalar reward.

    Uses ``nn.TransformerEncoder`` for bidirectional attention (reward models
    score complete sequences, so causal masking is unnecessary).

    Args:
        vocab_size: Vocabulary size.
        block_size: Maximum sequence length.
        n_layer: Number of encoder layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
        pad_id: Token id used for padding (masked out in pooling).
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        dropout: float = 0.1,
        pad_id: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of token sequences.

        Args:
            x: ``(B, T)`` token ids.

        Returns:
            ``(B,)`` scalar rewards.
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        pad_mask = x == self.pad_id
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln(h)

        # Masked mean pooling (ignore padding positions)
        mask = (~pad_mask).float().unsqueeze(-1)  # (B, T, 1)
        h_sum = (h * mask).sum(dim=1)
        length = mask.sum(dim=1).clamp_min(1.0)
        pooled = h_sum / length

        return self.head(pooled).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def bradley_terry_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry preference loss.

    ``-log σ(r_pos - r_neg) = softplus(-(r_pos - r_neg))``

    Args:
        r_pos: ``(B,)`` rewards for preferred completions.
        r_neg: ``(B,)`` rewards for rejected completions.

    Returns:
        Scalar mean loss.
    """
    return F.softplus(-(r_pos - r_neg)).mean()


def margin_ranking_loss(
    r_pos: torch.Tensor,
    r_neg: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Margin ranking loss for reward modeling.

    Args:
        r_pos: ``(B,)`` rewards for preferred completions.
        r_neg: ``(B,)`` rewards for rejected completions.
        margin: Target margin.

    Returns:
        Scalar mean loss.
    """
    y = torch.ones_like(r_pos)
    return F.margin_ranking_loss(r_pos, r_neg, y, margin=margin)
