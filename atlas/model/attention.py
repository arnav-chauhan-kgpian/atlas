"""Causal self-attention with modern features.

Supports:
- **Grouped Query Attention (GQA)**: fewer KV heads than query heads
- **Rotary Positional Embeddings (RoPE)**: position-aware via rotations
- **KV cache**: incremental decoding for efficient generation
- **Sliding-window attention + attention sinks**: bounded memory for streaming

Uses ``F.scaled_dot_product_attention`` (FlashAttention backend when available).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.model.rope import RoPECache, apply_rope
from atlas.model.kv_cache import KVCache


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with GQA, RoPE, and KV cache.

    Args:
        n_embd: Model embedding dimension.
        n_head: Number of query heads.
        n_kv_head: Number of KV heads (``None`` = same as *n_head*, i.e., MHA).
        dropout: Attention dropout probability.
        rope: Whether to apply RoPE.
        max_pos: Maximum sequence length for RoPE tables.
        sliding_window: If set, limits attention span to this many recent tokens.
        attention_sink: Number of initial tokens to always attend to.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        dropout: float = 0.0,
        rope: bool = True,
        max_pos: int = 4096,
        sliding_window: int | None = None,
        attention_sink: int = 0,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({n_head}) must be a multiple of n_kv_head ({self.n_kv_head})"
            )
        self.group_size = self.n_head // self.n_kv_head
        self.d_head = n_embd // n_head

        # Separate Q / K / V projections for GQA compatibility
        self.wq = nn.Linear(n_embd, self.n_head * self.d_head, bias=False)
        self.wk = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.use_rope = rope
        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos

        # Sliding-window + attention sinks
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache]:
        """Forward pass.

        Args:
            x: ``(B, T, C)`` input tensor.
            kv_cache: Previous step's cache (``None`` = prefill / training).
            start_pos: Absolute position of the first token in *x*.

        Returns:
            Tuple of ``(output, new_kv_cache)``.
        """
        B, T, C = x.shape
        self._ensure_rope(x.device)

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)       # (B,H,T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)    # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)    # (B,Hk,T,D)

        # Apply RoPE to current tokens (cached keys were already rotated)
        if self.use_rope and self.rope_cache is not None:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Concatenate past KV cache
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Sliding-window + attention-sink eviction
        if (
            self.sliding_window is not None
            and k_all.size(2) > (self.sliding_window + self.attention_sink)
        ):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window :, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window :, :]], dim=2)

        # GQA: expand KV heads to match Q heads
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)
        else:
            k_attn, v_attn = k_all, v_all

        # Scaled dot-product attention (Flash backend when available)
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(
            q,
            k_attn,
            v_attn,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Merge heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # Build new cache (store compact KV heads, not expanded)
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = KVCache(k_new, v_new)

        return y, new_cache

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_rope(self, device: torch.device) -> None:
        """Lazily initialize the RoPE cache on the correct device."""
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)
