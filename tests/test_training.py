"""Tests for atlas.training — scheduler, checkpointing, optimizer.

Covers: WarmupCosineLR step values, checkpoint save/load roundtrip,
architecture verification, AmpGrad accumulation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from atlas.config import ModelConfig
from atlas.model.transformer import Transformer
from atlas.training.scheduler import WarmupCosineLR
from atlas.training.checkpointing import save_checkpoint, load_checkpoint
from atlas.training.optimizer import AmpGrad, build_optimizer


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_warmup_phase(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        sched = WarmupCosineLR(opt, warmup_steps=10, total_steps=100, base_lr=1.0)
        lrs = [sched.step() for _ in range(10)]
        # LR should ramp linearly
        assert lrs[0] == pytest.approx(0.1, abs=1e-6)
        assert lrs[9] == pytest.approx(1.0, abs=1e-6)

    def test_cosine_decay(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        sched = WarmupCosineLR(opt, warmup_steps=1, total_steps=100, base_lr=1.0, min_lr=0.0)
        lrs = [sched.step() for _ in range(100)]
        # Peak at step 1, decay to ~0 at step 100
        assert lrs[0] == pytest.approx(1.0, abs=1e-6)
        assert lrs[-1] < 0.05

    def test_state_dict_roundtrip(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        sched = WarmupCosineLR(opt, warmup_steps=5, total_steps=50, base_lr=1.0)
        for _ in range(20):
            sched.step()
        state = sched.state_dict()
        sched2 = WarmupCosineLR(opt, warmup_steps=5, total_steps=50, base_lr=1.0)
        sched2.load_state_dict(state)
        assert sched2.step_num == 20


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    @pytest.fixture
    def model_and_config(self):
        cfg = ModelConfig(vocab_size=128, block_size=16, n_layer=1, n_head=2, n_embd=32)
        model = Transformer(cfg)
        return model, cfg

    def test_save_load_roundtrip(self, model_and_config, tmp_path):
        model, cfg = model_and_config
        opt = build_optimizer(model, lr=1e-3)
        save_checkpoint(model, opt, None, None, 42, str(tmp_path), cfg.to_dict())

        model2 = Transformer(cfg)
        step = load_checkpoint(model2, str(tmp_path / "model_last.pt"), device="cpu")
        assert step == 42

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_architecture_mismatch_raises(self, model_and_config, tmp_path):
        model, cfg = model_and_config
        save_checkpoint(model, None, None, None, 0, str(tmp_path), cfg.to_dict())

        cfg2 = ModelConfig(vocab_size=128, block_size=16, n_layer=3, n_head=2, n_embd=32)
        model2 = Transformer(cfg2)
        with pytest.raises(RuntimeError, match="mismatch"):
            load_checkpoint(model2, str(tmp_path / "model_last.pt"))


# ---------------------------------------------------------------------------
# AmpGrad
# ---------------------------------------------------------------------------


class TestAmpGrad:
    def test_accumulation(self):
        m = nn.Linear(4, 2)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        amp = AmpGrad(opt, accum=4, amp=False)

        for i in range(4):
            loss = m(torch.randn(1, 4)).sum()
            amp.backward(loss)
            if i < 3:
                assert not amp.should_step()
        assert amp.should_step()
