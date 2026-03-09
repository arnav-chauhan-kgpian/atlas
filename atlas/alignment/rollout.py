"""Shared RL utilities for RLHF alignment.

Provides token-level log-probability computation, label shifting,
and approximate KL divergence calculation used by both PPO and GRPO.
"""

from __future__ import annotations

import torch


def shift_labels(x: torch.Tensor) -> torch.Tensor:
    """Shift labels for causal LM: predict ``x[t+1]`` from ``x[:t]``.

    Args:
        x: ``(B, T)`` token tensor.

    Returns:
        ``(B, T-1)`` shifted labels.
    """
    return x[:, 1:].contiguous()


def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather per-token log-probabilities of the given labels.

    Args:
        logits: ``(B, T, V)`` output logits.
        labels: ``(B, T)`` target token ids (same T as logits).

    Returns:
        ``(B, T)`` log-probabilities.
    """
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def model_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    """Compute ``log p(x[t+1] | x[:t])`` for all positions.

    Works with both ``Transformer`` and ``PolicyWithValue`` (auto-detects
    the ``.lm`` attribute).

    Args:
        model: A model with a forward method returning ``(logits, ...)``.
        x: ``(B, T)`` token tensor.

    Returns:
        ``(B, T-1)`` per-position log-probabilities.
    """
    # Support PolicyWithValue or raw Transformer
    if hasattr(model, "lm"):
        logits = model.lm(x, None)[0]
    else:
        logits = model(x, None)[0]
    labels = shift_labels(x)
    return gather_logprobs(logits[:, :-1, :], labels)


def approx_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Approximate KL divergence: ``E[log π - log π_ref]``.

    Args:
        policy_logp: ``(N,)`` policy log-probs.
        ref_logp: ``(N,)`` reference log-probs.

    Returns:
        Scalar KL estimate.
    """
    return (policy_logp - ref_logp).mean()
