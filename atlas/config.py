"""Unified configuration system for Atlas.

All hyperparameters live in typed dataclasses. Configs can be loaded from YAML
files and overridden via CLI arguments.
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the Transformer language model."""

    vocab_size: int = 32_000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    n_kv_head: Optional[int] = None  # None → MHA; set < n_head for GQA
    dropout: float = 0.0

    # Architecture switches
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    rope: bool = True
    max_pos: int = 4096

    # Sliding-window attention
    sliding_window: Optional[int] = None
    attention_sink: int = 0

    # Mixture-of-Experts (None = dense FFN)
    moe_num_experts: Optional[int] = None
    moe_top_k: int = 1
    moe_mult: int = 4
    moe_hybrid_alpha: Optional[float] = None  # None = pure MoE; float = hybrid blend

    # Weight tying
    tie_embeddings: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        # Filter out keys that are not fields of this dataclass
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Configuration for pretraining."""

    # Data
    data_path: str = ""
    out_dir: str = "runs/pretrain"

    # Tokenizer
    tokenizer_type: str = "bpe"  # "bpe" or "byte"
    tokenizer_dir: Optional[str] = None  # path to load existing BPE tokenizer

    # Optimization
    batch_size: int = 32
    steps: int = 2000
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 20
    grad_accum_steps: int = 4
    mixed_precision: bool = False

    # Logging
    log_backend: str = "tensorboard"  # "tensorboard", "wandb", "none"
    log_every: int = 50

    # Checkpointing
    save_every: int = 50
    keep_last_k: int = 2
    resume_from: Optional[str] = None

    # Sampling during training
    sample_every: int = 200
    sample_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 50
    top_p: Optional[float] = None

    # Misc
    compile: bool = False
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""

    data_source: str = "huggingface"  # "huggingface" or path to JSONL
    out_dir: str = "runs/sft"
    checkpoint: Optional[str] = None  # pretrained model to load

    # Tokenizer
    tokenizer_type: str = "bpe"
    tokenizer_dir: Optional[str] = None

    # Model (used only when no checkpoint)
    vocab_size: int = 32_000
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256

    # Optimization
    batch_size: int = 8
    steps: int = 200
    lr: float = 3e-4

    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reward Modeling
# ---------------------------------------------------------------------------

@dataclass
class RMConfig:
    """Configuration for reward model training."""

    out_dir: str = "runs/rm"
    loss_type: str = "bt"  # "bt" (Bradley-Terry) or "margin"

    # Tokenizer
    tokenizer_type: str = "byte"
    tokenizer_dir: Optional[str] = None

    # Model
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256

    # Optimization
    batch_size: int = 8
    steps: int = 500
    lr: float = 1e-4

    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# RL (PPO / GRPO)
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Configuration for RLHF alignment (PPO or GRPO)."""

    algorithm: str = "ppo"  # "ppo" or "grpo"
    out_dir: str = "runs/rl"
    policy_ckpt: str = ""
    reward_ckpt: str = ""

    # Tokenizer
    tokenizer_type: str = "byte"
    tokenizer_dir: Optional[str] = None

    block_size: int = 256
    batch_size: int = 4
    resp_len: int = 64
    steps: int = 100

    # RL
    kl_coef: float = 0.01
    gamma: float = 1.0
    lam: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    lr: float = 1e-5

    # GRPO-specific
    group_size: int = 4  # completions per prompt

    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path, config_cls: type) -> Any:
    """Load a config from a YAML or JSON file.

    Args:
        path: Path to the config file.
        config_cls: The dataclass type to instantiate.

    Returns:
        An instance of *config_cls* populated from the file.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text) or {}
    elif p.suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")

    # Support nested keys: model.*, train.*, etc. — flatten if needed
    flat: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v

    valid = {f.name for f in config_cls.__dataclass_fields__.values()}
    return config_cls(**{k: v for k, v in flat.items() if k in valid})
