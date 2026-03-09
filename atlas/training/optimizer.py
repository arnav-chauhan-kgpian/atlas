"""Optimizer utilities.

Provides ``AmpGrad`` — a thin wrapper that handles mixed-precision scaling
and gradient accumulation — and a factory for building optimizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AmpGrad:
    """Automatic mixed precision + gradient accumulation wrapper.

    Handles scaled backward passes and deferred optimizer steps. Typical usage::

        amp = AmpGrad(optimizer, accum=4, amp=True)
        for xb, yb in loader:
            with torch.cuda.amp.autocast(enabled=amp.amp):
                loss = model(xb, yb)
            amp.backward(loss)
            if amp.should_step():
                amp.step()
                amp.zero_grad()

    Args:
        optimizer: PyTorch optimizer instance.
        accum: Gradient accumulation steps.
        amp: Enable AMP (automatically disabled on CPU).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        accum: int = 1,
        amp: bool = True,
    ) -> None:
        self.optim = optimizer
        self.accum = max(1, accum)
        self.amp = amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self._n = 0

    def backward(self, loss: torch.Tensor) -> None:
        """Scale and accumulate gradients."""
        loss = loss / self.accum
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._n += 1

    def should_step(self) -> bool:
        """Check if we've accumulated enough micro-batches."""
        return self._n % self.accum == 0

    def step(self) -> None:
        """Apply one optimizer step (with optional AMP unscaling)."""
        if self.amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        self.optim.zero_grad(set_to_none=True)


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Build an AdamW optimizer with sensible defaults.

    Args:
        model: Model whose parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.

    Returns:
        Configured ``AdamW`` optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
