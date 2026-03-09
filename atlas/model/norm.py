"""Normalization layers.

Provides RMSNorm and a factory for selecting between RMSNorm and LayerNorm
via configuration.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Compared to LayerNorm, RMSNorm skips the mean-centering step, making it
    ~15 % faster while matching or exceeding LayerNorm on convergence quality.

    Reference: https://arxiv.org/abs/1910.07467

    Formula:
        y = x * g / rms(x)
        rms(x) = sqrt(mean(x²) + ε)
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def build_norm(norm_type: str, dim: int) -> nn.Module:
    """Factory for normalization layers.

    Args:
        norm_type: ``"rmsnorm"`` or ``"layernorm"``.
        dim: Feature dimension.

    Returns:
        An instantiated normalization module.
    """
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm type: {norm_type!r}")
