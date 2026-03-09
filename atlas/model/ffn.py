"""Feed-forward network variants.

Provides SwiGLU and standard GELU FFN, plus a factory for config-driven
selection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Computes ``(x @ W1) ⊗ SiLU(x @ W2) @ W3`` with expansion factor ``mult``.
    SwiGLU is the de-facto standard FFN in modern LLMs (LLaMA, Mistral, etc.).

    Reference: https://arxiv.org/abs/2002.05202

    Args:
        dim: Input/output dimension.
        mult: Expansion factor for the hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = mult * dim
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(self.w1(x) * self.act(self.w2(x))))


class FeedForward(nn.Module):
    """Standard GELU feed-forward network.

    Two-layer MLP: ``Linear → GELU → Linear → Dropout``.

    Args:
        dim: Input/output dimension.
        mult: Expansion factor for the hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim),
            nn.GELU(),
            nn.Linear(mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_ffn(ffn_type: str, dim: int, mult: int = 4, dropout: float = 0.0) -> nn.Module:
    """Factory for feed-forward network variants.

    Args:
        ffn_type: ``"swiglu"`` or ``"gelu"``.
        dim: Input/output dimension.
        mult: Expansion factor.
        dropout: Dropout probability.

    Returns:
        An instantiated FFN module.
    """
    if ffn_type == "swiglu":
        return SwiGLU(dim, mult=mult, dropout=dropout)
    elif ffn_type == "gelu":
        return FeedForward(dim, mult=mult, dropout=dropout)
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type!r}")
