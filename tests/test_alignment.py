"""Tests for atlas.alignment — loss functions, reward model, policy.

Covers: Bradley-Terry loss, PPO loss, GRPO loss, reward model forward,
PolicyWithValue forward.
"""

from __future__ import annotations

import pytest
import torch

from atlas.config import ModelConfig
from atlas.model.reward import RewardModel, bradley_terry_loss, margin_ranking_loss
from atlas.model.policy import PolicyWithValue
from atlas.alignment.ppo import ppo_losses
from atlas.alignment.grpo import grpo_losses
from atlas.alignment.rollout import shift_labels, gather_logprobs


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class TestBradleyTerryLoss:
    def test_bt_preferred(self):
        """When r_pos >> r_neg the loss should be near 0."""
        r_pos = torch.tensor([5.0, 4.0])
        r_neg = torch.tensor([0.0, 0.0])
        loss = bradley_terry_loss(r_pos, r_neg)
        assert loss.item() < 0.1

    def test_bt_equal(self):
        """When r_pos == r_neg the loss should be log(2)."""
        r = torch.tensor([1.0, 1.0])
        loss = bradley_terry_loss(r, r)
        assert loss.item() == pytest.approx(0.6931, abs=0.01)


class TestMarginLoss:
    def test_margin_satisfied(self):
        r_pos = torch.tensor([3.0])
        r_neg = torch.tensor([0.0])
        loss = margin_ranking_loss(r_pos, r_neg, margin=1.0)
        assert loss.item() == 0.0


class TestPPOLoss:
    def test_zero_advantage(self):
        """Zero advantage should yield near-zero policy loss."""
        N = 16
        lp = torch.randn(N)
        adv = torch.zeros(N)
        vals = torch.randn(N)
        ret = vals.clone()
        out = ppo_losses(lp, lp, adv, vals, vals, ret)
        assert out.policy_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_total_is_sum(self):
        N = 8
        new_lp = torch.randn(N)
        old_lp = new_lp.detach().clone()
        adv = torch.randn(N)
        vals = torch.randn(N)
        ret = torch.randn(N)
        out = ppo_losses(new_lp, old_lp, adv, vals, vals, ret, vf_coef=0.5)
        expected = out.policy_loss + 0.5 * out.value_loss
        assert out.total_loss.item() == pytest.approx(expected.item(), abs=1e-5)


class TestGRPOLoss:
    def test_empty_input(self):
        empty = torch.zeros(0)
        out = grpo_losses(empty, empty, empty)
        assert out.total_loss.item() == 0.0

    def test_with_kl(self):
        N = 8
        lp = torch.randn(N)
        adv = torch.randn(N)
        kl = torch.tensor(0.1)
        out = grpo_losses(lp, lp, adv, kl_coef=0.01, kl_mean=kl)
        assert out.kl_ref.item() == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class TestRewardModel:
    def test_forward_shape(self):
        rm = RewardModel(vocab_size=128, block_size=32, n_layer=1, n_head=2, n_embd=32)
        x = torch.randint(0, 128, (4, 16))
        r = rm(x)
        assert r.shape == (4,)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class TestPolicyWithValue:
    def test_forward_shape(self):
        cfg = ModelConfig(vocab_size=128, block_size=32, n_layer=1, n_head=2, n_embd=32)
        pol = PolicyWithValue(cfg)
        x = torch.randint(0, 128, (2, 16))
        logits, values, loss = pol(x)
        assert logits.shape == (2, 16, 128)
        assert values.shape == (2, 16)
        assert loss is None

    def test_forward_with_loss(self):
        cfg = ModelConfig(vocab_size=128, block_size=32, n_layer=1, n_head=2, n_embd=32)
        pol = PolicyWithValue(cfg)
        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        _, _, loss = pol(x, y)
        assert loss is not None


# ---------------------------------------------------------------------------
# Rollout utilities
# ---------------------------------------------------------------------------


class TestRolloutUtils:
    def test_shift_labels(self):
        x = torch.tensor([[1, 2, 3, 4, 5]])
        y = shift_labels(x)
        assert y.tolist() == [[2, 3, 4, 5]]

    def test_gather_logprobs(self):
        logits = torch.randn(2, 10, 128)
        labels = torch.randint(0, 128, (2, 10))
        lp = gather_logprobs(logits, labels)
        assert lp.shape == (2, 10)
        assert (lp <= 0).all()  # log-probs are ≤ 0
