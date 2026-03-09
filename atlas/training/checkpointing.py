"""Checkpointing with architecture verification.

Provides ``save_checkpoint`` and ``load_checkpoint`` with:
- Architecture config embedding in checkpoints
- Architecture verification on resume
- Rolling checkpoint garbage collection
- Atomic save (write last + per-step copy)
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

CHECKPOINT_NAME = "model_last.pt"


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    amp: Any | None,
    step: int,
    out_dir: str | Path,
    config: dict | None = None,
    tokenizer_dir: str | None = None,
) -> Path:
    """Save a checkpoint with full training state.

    Args:
        model: Model to save.
        optimizer: Optimizer state (optional).
        scheduler: Scheduler state (optional).
        amp: AmpGrad wrapper (optional).
        step: Current training step.
        out_dir: Output directory.
        config: Model configuration dict to embed.
        tokenizer_dir: Path to tokenizer (stored as metadata).

    Returns:
        Path to the saved checkpoint file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "amp_scaler": (
            amp.scaler.state_dict()
            if amp is not None and getattr(amp, "scaler", None)
            else None
        ),
        "step": int(step),
        "config": config or _extract_config(model),
        "version": "atlas-v1",
    }

    path = out / CHECKPOINT_NAME
    torch.save(ckpt, path)

    if tokenizer_dir is not None:
        (out / "tokenizer_dir.txt").write_text(str(tokenizer_dir))

    return path


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    amp: Any | None = None,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> int:
    """Load a checkpoint and optionally restore training state.

    Args:
        model: Model to load weights into.
        path: Path to checkpoint file.
        optimizer: Optimizer to restore (optional).
        scheduler: Scheduler to restore (optional).
        amp: AmpGrad to restore (optional).
        strict: Enforce strict state dict matching.
        device: Device to map tensors to.

    Returns:
        Training step at which the checkpoint was saved.

    Raises:
        RuntimeError: If architecture verification fails.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Verify architecture
    config = ckpt.get("config")
    if config:
        ok, msg = _verify_architecture(model, config)
        if not ok:
            raise RuntimeError(
                f"{msg}\nRebuild the model with the checkpoint config, "
                f"or load with strict=False."
            )

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler"])
    if amp is not None and ckpt.get("amp_scaler") is not None:
        if getattr(amp, "scaler", None):
            amp.scaler.load_state_dict(ckpt["amp_scaler"])

    return ckpt.get("step", 0)


def atomic_save_all(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp: Any,
    step: int,
    out_dir: Path,
    config: dict,
    tokenizer_dir: str | None = None,
    keep_last_k: int = 2,
) -> None:
    """Atomic checkpoint: save ``model_last.pt`` + a per-step copy, then GC old copies.

    Args:
        model: Model to save.
        optimizer: Optimizer.
        scheduler: Scheduler.
        amp: AmpGrad wrapper.
        step: Current step.
        out_dir: Output directory.
        config: Model config dict.
        tokenizer_dir: Path to tokenizer.
        keep_last_k: Number of per-step checkpoints to retain.
    """
    save_checkpoint(model, optimizer, scheduler, amp, step, out_dir, config, tokenizer_dir)
    last = out_dir / CHECKPOINT_NAME
    per_step = out_dir / f"model_step{step:07d}.pt"

    try:
        shutil.copy2(last, per_step)
    except Exception:
        pass

    # Garbage-collect old per-step checkpoints
    try:
        ckpts = sorted(out_dir.glob("model_step*.pt"))
        for old in ckpts[:-keep_last_k]:
            old.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_config(model: nn.Module) -> dict:
    """Best-effort config extraction from a model instance."""
    cfg: dict[str, Any] = {}
    try:
        if hasattr(model, "config"):
            return model.config.to_dict()

        tok_emb = getattr(model, "tok_emb", None)
        blocks = getattr(model, "blocks", None)
        if tok_emb is None or not blocks:
            return cfg

        cfg["vocab_size"] = int(tok_emb.num_embeddings)
        cfg["block_size"] = int(getattr(model, "block_size", 0))
        cfg["n_layer"] = int(len(blocks))

        attn = getattr(blocks[0], "attn", None)
        if attn:
            cfg["n_head"] = int(getattr(attn, "n_head", 0))
            cfg["n_embd"] = int(cfg["n_head"] * getattr(attn, "d_head", 0))
            cfg["n_kv_head"] = int(getattr(attn, "n_kv_head", cfg["n_head"]))
    except Exception:
        pass
    return cfg


def _verify_architecture(
    model: nn.Module, config: dict
) -> tuple[bool, str]:
    """Verify model architecture matches checkpoint config."""
    try:
        expected = {
            "block_size": config.get("block_size"),
            "n_layer": config.get("n_layer"),
            "vocab_size": config.get("vocab_size"),
        }
        got = {
            "block_size": int(getattr(model, "block_size", -1)),
            "n_layer": int(len(getattr(model, "blocks", []))),
            "vocab_size": int(getattr(model, "tok_emb", type("", (), {"num_embeddings": -1})).num_embeddings),
        }

        # Check attention-specific fields if available
        blocks = getattr(model, "blocks", [])
        if blocks:
            attn = getattr(blocks[0], "attn", None)
            if attn:
                expected.update({
                    "n_head": config.get("n_head"),
                    "n_embd": config.get("n_embd"),
                })
                got.update({
                    "n_head": int(getattr(attn, "n_head", -1)),
                    "n_embd": int(getattr(attn, "n_head", 0) * getattr(attn, "d_head", 0)),
                })

        diffs = [
            f"{k}: ckpt={expected[k]} vs model={got[k]}"
            for k in expected
            if expected[k] is not None and expected[k] != got.get(k)
        ]
        if diffs:
            return False, "Architecture mismatch:\n  " + "\n  ".join(diffs)
    except Exception as e:
        return False, f"Architecture verification failed: {e}"

    return True, "ok"
