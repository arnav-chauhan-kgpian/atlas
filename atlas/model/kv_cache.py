"""KV cache structures for autoregressive generation.

Provides a simple ``KVCache`` dataclass for standard caching and a
``RollingKV`` buffer that retains attention sinks + a sliding window.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class KVCache:
    """Key-value cache for a single attention layer.

    Stores the accumulated keys and values across generation steps.

    Attributes:
        k: ``(B, H, T_cached, D)`` key tensor.
        v: ``(B, H, T_cached, D)`` value tensor.
    """

    k: torch.Tensor
    v: torch.Tensor

    @property
    def seq_len(self) -> int:
        """Number of cached positions."""
        return self.k.size(2)


class RollingKV:
    """Rolling KV buffer with optional attention sink.

    Retains the first ``sink`` tokens (attention sinks) plus the most recent
    ``window`` tokens, evicting everything in between. This enables
    constant-memory streaming inference.

    Args:
        window: Number of recent tokens to retain.
        sink: Number of initial tokens to always keep.
    """

    def __init__(self, window: int, sink: int = 0) -> None:
        self.window = window
        self.sink = sink
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def step(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and evict stale positions.

        Args:
            k_new: ``(B, H, T_new, D)`` new key tensor.
            v_new: ``(B, H, T_new, D)`` new value tensor.

        Returns:
            Tuple of cropped ``(k, v)`` tensors.
        """
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

        # Evict middle tokens when exceeding capacity
        if self.k.size(2) > self.window + self.sink:
            sink_k = self.k[:, :, : self.sink, :]
            sink_v = self.v[:, :, : self.sink, :]
            tail_k = self.k[:, :, -self.window :, :]
            tail_v = self.v[:, :, -self.window :, :]
            self.k = torch.cat([sink_k, tail_k], dim=2)
            self.v = torch.cat([sink_v, tail_v], dim=2)

        return self.k, self.v
