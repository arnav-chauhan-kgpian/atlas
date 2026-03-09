"""Learning rate schedulers.

Provides ``WarmupCosineLR``: linear warmup followed by cosine decay.
This is the standard schedule used in GPT-style pretraining.
"""

from __future__ import annotations

import math


class WarmupCosineLR:
    """Linear warmup → cosine decay learning rate scheduler.

    Manually updates optimizer parameter groups each step (no
    ``torch.optim.lr_scheduler`` dependency).

    Args:
        optimizer: Optimizer whose LR to control.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        base_lr: Peak learning rate (reached at end of warmup).
        min_lr: Minimum learning rate at end of decay.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self) -> float:
        """Advance one step and return the new learning rate."""
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = state.get("step_num", 0)
