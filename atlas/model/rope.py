"""Rotary Positional Embeddings (RoPE).

Precomputes cos/sin frequency tables and applies rotary transformations
to query/key tensors. Supports automatic table growth for sequences exceeding
the initially configured ``max_pos``.

Reference: https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch


class RoPECache:
    """Cache of precomputed cos/sin tables for RoPE.

    Lazily grows the tables when sequences exceed current ``max_pos``.

    Args:
        head_dim: Per-head dimension (must be even).
        max_pos: Initial maximum supported sequence length.
        base: Frequency base (default 10 000).
        device: Torch device for the tables.
    """

    def __init__(
        self,
        head_dim: int,
        max_pos: int,
        base: float = 10_000.0,
        device: torch.device | None = None,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self.max_pos: int = 0  # will be set by _build
        self.cos: torch.Tensor = torch.empty(0)
        self.sin: torch.Tensor = torch.empty(0)
        self._build(max_pos)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` tables indexed by *positions*.

        Args:
            positions: 1-D ``(T,)`` or ``(1, T)`` tensor of absolute positions.

        Returns:
            Tuple of ``(T, head_dim/2)`` cosine and sine tensors.
        """
        if positions.dim() == 2:
            positions = positions[0]
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            self._build(max(need, self.max_pos * 2))
        return self.cos[positions], self.sin[positions]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self, max_pos: int) -> None:
        """(Re)build cos/sin tables."""
        self.max_pos = max_pos
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim)
        )
        t = torch.arange(max_pos, device=self.device).float()
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embedding to *x*.

    Rotates consecutive pairs along the last dimension::

        x_out[..., 2i]   = x[..., 2i]   * cos - x[..., 2i+1] * sin
        x_out[..., 2i+1] = x[..., 2i]   * sin + x[..., 2i+1] * cos

    Args:
        x: ``(B, H, T, D)`` tensor (queries or keys).
        cos: ``(T, D/2)`` cosine table.
        sin: ``(T, D/2)`` sine table.

    Returns:
        Rotated tensor of the same shape as *x*.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out
