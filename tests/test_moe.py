"""Tests for atlas.model.moe — Mixture of Experts.

Covers: TopKGate shapes, MoELayer forward, HybridFFN blending.
"""

from __future__ import annotations

import torch

from atlas.model.moe import TopKGate, ExpertMLP, MoELayer, HybridFFN


class TestTopKGate:
    def test_output_shapes(self):
        gate = TopKGate(dim=64, n_expert=4, k=2)
        x = torch.randn(32, 64)
        idx, w, aux = gate(x)
        assert idx.shape == (32, 2)
        assert w.shape == (32, 2)
        assert aux.dim() == 0  # scalar

    def test_aux_loss_positive(self):
        gate = TopKGate(dim=64, n_expert=4, k=1)
        x = torch.randn(64, 64)
        _, _, aux = gate(x)
        assert aux.item() > 0


class TestMoELayer:
    def test_forward_shape(self):
        moe = MoELayer(dim=64, n_expert=4, k=1)
        x = torch.randn(2, 8, 64)
        y, aux = moe(x)
        assert y.shape == x.shape
        assert aux.dim() == 0


class TestHybridFFN:
    def test_forward_shape(self):
        ffn = HybridFFN(dim=64, alpha=0.5, n_expert=4, k=1)
        x = torch.randn(2, 8, 64)
        y, aux = ffn(x)
        assert y.shape == x.shape

    def test_alpha_blending(self):
        """Alpha=0 should be pure MoE, alpha=1 should be pure dense."""
        ffn_moe = HybridFFN(dim=32, alpha=0.0, n_expert=2, k=1)
        ffn_dense = HybridFFN(dim=32, alpha=1.0, n_expert=2, k=1)
        x = torch.randn(1, 4, 32)
        y_moe, _ = ffn_moe(x)
        y_dense, _ = ffn_dense(x)
        # They should produce different outputs (probabilistically)
        assert not torch.allclose(y_moe, y_dense, atol=1e-6)
