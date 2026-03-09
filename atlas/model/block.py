"""Transformer block with config-driven component selection.

A single block composes pre-norm → attention → residual → pre-norm → FFN → residual.
The normalization, FFN, and MoE variants are all selectable via configuration.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from atlas.model.norm import build_norm
from atlas.model.attention import CausalSelfAttention
from atlas.model.ffn import SwiGLU, FeedForward
from atlas.model.moe import MoELayer, HybridFFN
from atlas.model.kv_cache import KVCache
from atlas.config import ModelConfig


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

    Component selection is driven entirely by :class:`ModelConfig`:

    - **Norm**: RMSNorm or LayerNorm (``use_rmsnorm``).
    - **FFN**: SwiGLU, GELU, MoE, or HybridFFN (``use_swiglu``, ``moe_*``).
    - **Attention**: GQA, RoPE, sliding-window, attention sinks.

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        norm_type = "rmsnorm" if config.use_rmsnorm else "layernorm"
        self.ln1 = build_norm(norm_type, config.n_embd)
        self.attn = CausalSelfAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            dropout=config.dropout,
            rope=config.rope,
            max_pos=config.max_pos,
            sliding_window=config.sliding_window,
            attention_sink=config.attention_sink,
        )
        self.ln2 = build_norm(norm_type, config.n_embd)
        self.ffn: nn.Module = _build_ffn_from_config(config)

        # Bookkeeping for MoE aux-loss propagation
        self._has_moe = isinstance(self.ffn, (MoELayer, HybridFFN))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, T, C)`` hidden states.
            kv_cache: Previous cache (``None`` for training / prefill).
            start_pos: Absolute position offset for RoPE.

        Returns:
            Tuple of ``(hidden_states, new_kv_cache, aux_loss)``.
        """
        # Self-attention
        a, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + a

        # Feed-forward
        if self._has_moe:
            ffn_out, aux = self.ffn(self.ln2(x))
        else:
            ffn_out = self.ffn(self.ln2(x))
            aux = torch.tensor(0.0, device=x.device)
        x = x + ffn_out

        return x, kv_cache, aux


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_ffn_from_config(cfg: ModelConfig) -> nn.Module:
    """Instantiate the correct FFN variant based on config."""
    # MoE takes priority
    if cfg.moe_num_experts is not None:
        if cfg.moe_hybrid_alpha is not None:
            return HybridFFN(
                dim=cfg.n_embd,
                alpha=cfg.moe_hybrid_alpha,
                mult=cfg.moe_mult,
                swiglu=cfg.use_swiglu,
                n_expert=cfg.moe_num_experts,
                k=cfg.moe_top_k,
                dropout=cfg.dropout,
            )
        return MoELayer(
            dim=cfg.n_embd,
            n_expert=cfg.moe_num_experts,
            k=cfg.moe_top_k,
            mult=cfg.moe_mult,
            swiglu=cfg.use_swiglu,
            dropout=cfg.dropout,
        )
    # Dense FFN
    if cfg.use_swiglu:
        return SwiGLU(cfg.n_embd, mult=4, dropout=cfg.dropout)
    return FeedForward(cfg.n_embd, mult=4, dropout=cfg.dropout)
