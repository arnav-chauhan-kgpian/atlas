"""Tests for atlas.model — core Transformer components.

Covers: RMSNorm, RoPE, attention shapes, KV cache, config-driven block,
full Transformer forward + generate.
"""

from __future__ import annotations

import pytest
import torch

from atlas.config import ModelConfig
from atlas.model.norm import RMSNorm, build_norm
from atlas.model.rope import RoPECache, apply_rope
from atlas.model.ffn import SwiGLU, FeedForward, build_ffn
from atlas.model.kv_cache import KVCache, RollingKV
from atlas.model.attention import CausalSelfAttention
from atlas.model.block import TransformerBlock
from atlas.model.transformer import Transformer


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_rmsnorm_unit_rms(self):
        """After RMSNorm, the RMS of the output should be ~ 1."""
        norm = RMSNorm(128, eps=1e-8)
        x = torch.randn(4, 16, 128)
        y = norm(x)
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.3)


class TestBuildNorm:
    def test_rmsnorm(self):
        norm = build_norm("rmsnorm", 32)
        assert isinstance(norm, RMSNorm)

    def test_layernorm(self):
        norm = build_norm("layernorm", 32)
        assert isinstance(norm, torch.nn.LayerNorm)

    def test_invalid(self):
        with pytest.raises(ValueError):
            build_norm("batchnorm", 32)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_rope_cache_shape(self):
        cache = RoPECache(head_dim=32, max_pos=128)
        assert cache.cos.shape == (128, 16)

    def test_apply_rope_preserves_shape(self):
        cache = RoPECache(head_dim=32, max_pos=128)
        x = torch.randn(2, 4, 16, 32)
        pos = torch.arange(16)
        cos, sin = cache.get(pos)
        y = apply_rope(x, cos, sin)
        assert y.shape == x.shape

    def test_rope_auto_growth(self):
        cache = RoPECache(head_dim=16, max_pos=64)
        pos = torch.arange(128)  # exceeds initial 64
        cos, sin = cache.get(pos)
        assert cos.shape[0] == 128


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------


class TestFFN:
    def test_swiglu_shape(self):
        ffn = SwiGLU(64)
        x = torch.randn(2, 10, 64)
        assert ffn(x).shape == x.shape

    def test_feedforward_shape(self):
        ffn = FeedForward(64)
        x = torch.randn(2, 10, 64)
        assert ffn(x).shape == x.shape

    def test_build_ffn(self):
        assert isinstance(build_ffn("swiglu", 64), SwiGLU)
        assert isinstance(build_ffn("gelu", 64), FeedForward)
        with pytest.raises(ValueError):
            build_ffn("relu", 64)


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


class TestKVCache:
    def test_kv_cache_seq_len(self):
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache = KVCache(k, v)
        assert cache.seq_len == 10

    def test_rolling_kv_eviction(self):
        rk = RollingKV(window=5, sink=2)
        k = torch.randn(1, 2, 10, 16)
        v = torch.randn(1, 2, 10, 16)
        k_out, v_out = rk.step(k, v)
        # 2 sink + 5 window = 7
        assert k_out.size(2) == 7

    def test_rolling_kv_small(self):
        rk = RollingKV(window=10, sink=0)
        k = torch.randn(1, 2, 5, 16)
        v = torch.randn(1, 2, 5, 16)
        k_out, v_out = rk.step(k, v)
        assert k_out.size(2) == 5  # no eviction yet


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class TestAttention:
    def test_mha_output_shape(self):
        attn = CausalSelfAttention(n_embd=64, n_head=4, rope=True, max_pos=128)
        x = torch.randn(2, 16, 64)
        y, cache = attn(x)
        assert y.shape == (2, 16, 64)
        assert cache.seq_len == 16

    def test_gqa_output_shape(self):
        attn = CausalSelfAttention(n_embd=64, n_head=8, n_kv_head=2, rope=True, max_pos=128)
        x = torch.randn(2, 12, 64)
        y, cache = attn(x)
        assert y.shape == (2, 12, 64)

    def test_kv_cache_incremental(self):
        attn = CausalSelfAttention(n_embd=64, n_head=4, rope=True, max_pos=128)
        x = torch.randn(1, 10, 64)
        _, cache = attn(x)
        assert cache.seq_len == 10

        x2 = torch.randn(1, 1, 64)
        _, cache2 = attn(x2, kv_cache=cache, start_pos=10)
        assert cache2.seq_len == 11


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class TestBlock:
    def test_dense_block(self):
        cfg = ModelConfig(n_embd=64, n_head=4, n_layer=1, vocab_size=100, block_size=32)
        block = TransformerBlock(cfg)
        x = torch.randn(2, 8, 64)
        y, cache, aux = block(x)
        assert y.shape == x.shape
        assert aux.item() == 0.0  # no MoE

    def test_moe_block(self):
        cfg = ModelConfig(
            n_embd=64, n_head=4, n_layer=1, vocab_size=100, block_size=32,
            moe_num_experts=4, moe_top_k=1
        )
        block = TransformerBlock(cfg)
        x = torch.randn(2, 8, 64)
        y, cache, aux = block(x)
        assert y.shape == x.shape
        assert aux.item() > 0  # MoE aux loss present


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class TestTransformer:
    @pytest.fixture
    def cfg(self):
        return ModelConfig(
            vocab_size=128, block_size=32, n_layer=2, n_head=4, n_embd=64
        )

    def test_forward_shape(self, cfg):
        model = Transformer(cfg)
        x = torch.randint(0, 128, (2, 16))
        logits, loss, caches, aux = model(x)
        assert logits.shape == (2, 16, 128)
        assert loss is None
        assert len(caches) == 2

    def test_forward_with_targets(self, cfg):
        model = Transformer(cfg)
        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        logits, loss, _, _ = model(x, y)
        assert loss is not None
        assert loss.item() > 0

    def test_generate(self, cfg):
        model = Transformer(cfg)
        prompt = torch.randint(0, 128, (1, 8))
        out = model.generate(prompt, max_new_tokens=10)
        assert out.shape[1] == 18  # 8 + 10

    def test_num_parameters(self, cfg):
        model = Transformer(cfg)
        n = model.num_parameters()
        assert n > 0
