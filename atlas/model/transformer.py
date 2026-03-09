"""Transformer language model.

The main ``Transformer`` class composes token embeddings, stacked blocks,
final normalization, and a language model head. It supports:

- Config-driven architecture (RMSNorm, SwiGLU, RoPE, GQA, MoE)
- KV cache for efficient autoregressive generation
- Optional weight tying between embeddings and LM head
-``generate()`` with temperature, top-k, top-p, and early EOS stopping
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.config import ModelConfig
from atlas.model.block import TransformerBlock
from atlas.model.norm import build_norm
from atlas.model.kv_cache import KVCache


class Transformer(nn.Module):
    """Decoder-only Transformer language model.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # Token embedding (no position embedding when RoPE is active)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Final normalization
        if config.use_rmsnorm:
            self.ln_f = build_norm("rmsnorm", config.n_embd)
        else:
            self.ln_f = build_norm("layernorm", config.n_embd)

        # Language model head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Optional weight tying
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-style weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache_list: list[KVCache | None] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KVCache], torch.Tensor]:
        """Forward pass.

        Args:
            idx: ``(B, T)`` token indices.
            targets: ``(B, T)`` target indices for loss computation (optional).
            kv_cache_list: Per-layer KV caches (``None`` for training).
            start_pos: Absolute position offset for KV-cache generation.

        Returns:
            Tuple of ``(logits, loss, new_caches, total_aux_loss)`` where:
            - ``logits``: ``(B, T, V)`` output logits.
            - ``loss``: Cross-entropy loss (``None`` if *targets* not given).
            - ``new_caches``: Updated per-layer KV caches.
            - ``total_aux_loss``: Sum of MoE auxiliary losses across layers.
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block_size {self.block_size}"
        )

        x = self.tok_emb(idx)
        x = self.drop(x)

        new_caches: list[KVCache] = []
        total_aux = torch.tensor(0.0, device=idx.device)

        for i, blk in enumerate(self.blocks):
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache, aux = blk(x, kv_cache=cache, start_pos=start_pos)
            new_caches.append(cache)
            total_aux = total_aux + aux

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_caches, total_aux

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache.

        Args:
            prompt: ``(B, T)`` prompt token ids.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k filtering.
            top_p: Nucleus (top-p) filtering.
            eos_id: Stop token id (``None`` = no early stopping).

        Returns:
            ``(B, T + generated)`` full sequence tensor.
        """
        self.eval()
        idx = prompt
        kvs: list[KVCache | None] = [None] * len(self.blocks)

        for _ in range(max_new_tokens):
            # First step: full prompt; subsequent: single token
            idx_cond = idx[:, -self.block_size :] if kvs[0] is None else idx[:, -1:]
            start_pos = 0 if kvs[0] is None else kvs[0].seq_len

            logits, _, kvs, _ = self(idx_cond, kv_cache_list=kvs, start_pos=start_pos)

            # Sample next token
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            next_logits = _top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)

            if temperature == 0.0:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

            # Early stopping
            if eos_id is not None and (next_id == eos_id).all():
                break

        return idx

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Count parameters, optionally excluding embeddings."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p) sampling.

    Args:
        logits: ``(B, V)`` logit tensor.
        top_k: Keep only top-k logits.
        top_p: Keep smallest set with cumulative probability ≥ top_p.

    Returns:
        Filtered logits with ``-inf`` for masked entries.
    """
    B, V = logits.shape
    filtered = logits.clone()

    if top_k is not None and top_k < V:
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)
        kth = topk_vals[:, -1].unsqueeze(-1)
        filtered[filtered < kth] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum > top_p
        mask[..., 0] = False  # keep at least one token
        sorted_logits[mask] = float("-inf")
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(1, sorted_idx, sorted_logits)

    return filtered
