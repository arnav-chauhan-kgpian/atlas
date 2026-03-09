"""Mixture-of-Experts (MoE) layers.

Implements top-k token-level expert routing with Switch-style load-balancing
auxiliary loss, and an optional hybrid dense+MoE blending layer.

Reference: https://arxiv.org/abs/2101.03961 (Switch Transformers)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from atlas.model.ffn import SwiGLU, FeedForward


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


class TopKGate(nn.Module):
    """Top-k softmax gating with load-balancing auxiliary loss.

    Args:
        dim: Input hidden size.
        n_expert: Number of experts.
        k: Experts selected per token (1 or 2 typical).
    """

    def __init__(self, dim: int, n_expert: int, k: int = 1) -> None:
        super().__init__()
        if k < 1 or k > n_expert:
            raise ValueError(f"k={k} must be in [1, n_expert={n_expert}]")
        self.n_expert = n_expert
        self.k = k
        self.w_g = nn.Linear(dim, n_expert, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute expert assignments and load-balancing loss.

        Args:
            x: ``(S, C)`` flattened token tensor (S = B * T).

        Returns:
            Tuple of ``(indices, weights, aux_loss)`` where:
            - ``indices``: ``(S, k)`` selected expert ids.
            - ``weights``: ``(S, k)`` gate weights.
            - ``aux_loss``: Scalar load-balancing penalty.
        """
        logits = self.w_g(x)  # (S, E)
        probs = torch.softmax(logits, dim=-1)  # (S, E)
        topk_vals, topk_idx = torch.topk(probs, k=self.k, dim=-1)  # (S, k)

        # Load-balancing auxiliary loss (Switch-style)
        S, E = probs.shape
        importance = probs.mean(dim=0)  # (E,)
        hard1 = topk_idx[:, 0]  # (S,)
        load = torch.zeros(E, device=x.device)
        load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype))
        load = load / max(S, 1)
        aux_loss = E * (importance * load).sum()

        return topk_idx, topk_vals, aux_loss


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------


class ExpertMLP(nn.Module):
    """Single expert MLP (SwiGLU or GELU).

    Args:
        dim: Input/output dimension.
        mult: Expansion factor.
        swiglu: Use SwiGLU (True) or GELU (False).
        dropout: Dropout probability.
    """

    def __init__(
        self, dim: int, mult: int = 4, swiglu: bool = True, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if swiglu:
            self.ffn = SwiGLU(dim, mult=mult, dropout=dropout)
        else:
            self.ffn = FeedForward(dim, mult=mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with token-wise top-k routing.

    Single-GPU friendly (loops over experts for clarity; can be replaced with
    a fused kernel for production multi-GPU setups).

    Args:
        dim: Input dimension.
        n_expert: Number of experts.
        k: Experts per token.
        mult: FFN expansion factor.
        swiglu: Use SwiGLU in experts.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        n_expert: int,
        k: int = 1,
        mult: int = 4,
        swiglu: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_expert = n_expert
        self.k = k
        self.gate = TopKGate(dim, n_expert, k=k)
        self.experts = nn.ModuleList(
            [ExpertMLP(dim, mult=mult, swiglu=swiglu, dropout=dropout) for _ in range(n_expert)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, T, C)`` input.

        Returns:
            Tuple of ``(output, aux_loss)``.
        """
        B, T, C = x.shape
        S = B * T
        x_flat = x.reshape(S, C)
        idx, w, aux = self.gate(x_flat)  # (S, k), (S, k), scalar

        y = torch.zeros_like(x_flat)
        for e in range(self.n_expert):
            for slot in range(self.k):
                sel = idx[:, slot] == e
                if sel.any():
                    y[sel] += w[sel, slot : slot + 1] * self.experts[e](x_flat[sel])

        return y.view(B, T, C), aux


# ---------------------------------------------------------------------------
# Hybrid FFN (Dense + MoE blend)
# ---------------------------------------------------------------------------


class HybridFFN(nn.Module):
    """Blended dense FFN + MoE output: ``y = α * Dense(x) + (1-α) * MoE(x)``.

    Use α ∈ [0, 1] to trade between stability (dense) and capacity (MoE).

    Args:
        dim: Input dimension.
        alpha: Blending weight for dense path.
        mult: FFN expansion factor.
        swiglu: Use SwiGLU activation.
        n_expert: Number of experts.
        k: Experts per token.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 0.5,
        mult: int = 4,
        swiglu: bool = True,
        n_expert: int = 4,
        k: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.dense = FeedForward(dim, mult=mult, dropout=dropout)
        self.moe = MoELayer(dim, n_expert=n_expert, k=k, mult=mult, swiglu=swiglu, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of ``(output, aux_loss)``.
        """
        y_dense = self.dense(x)
        y_moe, aux = self.moe(x)
        y = self.alpha * y_dense + (1.0 - self.alpha) * y_moe
        return y, aux
