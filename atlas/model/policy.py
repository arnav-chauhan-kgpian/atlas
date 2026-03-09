"""Policy wrapper for RLHF.

Wraps the base ``Transformer`` language model with a value head for PPO.
GRPO ignores the value head (policy-only optimization).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from atlas.model.transformer import Transformer
from atlas.config import ModelConfig


class PolicyWithValue(nn.Module):
    """Policy network = Transformer LM + scalar value head.

    The value head projects from vocabulary logits → scalar, keeping the
    architecture simple and avoiding dependence on hidden-state internals.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.lm = Transformer(config)
        self.val_head = nn.Linear(config.vocab_size, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: ``(B, T)`` token ids.
            targets: ``(B, T)`` targets for loss (optional).

        Returns:
            Tuple of ``(logits, values, loss)`` where:
            - ``logits``: ``(B, T, V)``
            - ``values``: ``(B, T)`` scalar values per position.
            - ``loss``: Cross-entropy loss or ``None``.
        """
        logits, loss, _, _ = self.lm(x, targets)
        values = self.val_head(logits).squeeze(-1)  # (B, T)
        return logits, values, loss

    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Delegate to the underlying LM's generate method."""
        return self.lm.generate(*args, **kwargs)
