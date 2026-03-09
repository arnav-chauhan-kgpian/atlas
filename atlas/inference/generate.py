"""Standalone generation from a saved checkpoint.

Handles model reconstruction from checkpoint config, tokenizer setup,
and text generation with KV cache.
"""

from __future__ import annotations

from pathlib import Path

import torch

from atlas.config import ModelConfig
from atlas.model.transformer import Transformer
from atlas.data.tokenizer import build_tokenizer, ByteTokenizer, BPETokenizer


def generate_from_checkpoint(
    ckpt_path: str | Path,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    tokenizer_dir: str | None = None,
    device: str = "auto",
) -> str:
    """Generate text from a saved Atlas checkpoint.

    Args:
        ckpt_path: Path to ``model_last.pt``.
        prompt: Text prompt (empty = unconditional).
        max_new_tokens: Tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        top_p: Nucleus sampling threshold.
        tokenizer_dir: Path to BPE tokenizer (if applicable).
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        Generated text string.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg_dict = ckpt.get("config", {})

    # Build config
    mcfg = ModelConfig.from_dict(cfg_dict)

    # Determine tokenizer
    tok_dir = tokenizer_dir
    if tok_dir is None:
        # Check for saved tokenizer dir reference
        ckpt_parent = Path(ckpt_path).parent
        ref = ckpt_parent / "tokenizer_dir.txt"
        if ref.exists():
            tok_dir = ref.read_text().strip()

    # Use byte tokenizer if vocab ≤ 256 or no BPE dir
    if mcfg.vocab_size <= 256 or tok_dir is None:
        tok = ByteTokenizer()
    else:
        tok = build_tokenizer("bpe", vocab_size=mcfg.vocab_size, tokenizer_dir=tok_dir)

    # Build model
    model = Transformer(mcfg).to(dev)
    
    # Strip "lm." prefix if loaded from PPO PolicyWithValue
    state_dict = ckpt["model"]
    if any(k.startswith("lm.") for k in state_dict.keys()):
        # RLHF model has 'lm.' prefix for Transformer and a 'val_head'
        state_dict = {
            k.replace("lm.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("lm.")
        }
    
    model.load_state_dict(state_dict)
    model.eval()

    # Encode prompt
    if prompt:
        ids = tok.encode(prompt)
    else:
        ids = [0]  # Start token

    idx = torch.tensor([ids], dtype=torch.long, device=dev)

    # Generate
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return tok.decode(out[0].cpu().tolist())
