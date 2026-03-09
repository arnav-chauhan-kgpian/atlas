"""Supervised Fine-Tuning (SFT) trainer.

Trains a Transformer on instruction-response pairs with prompt-masked loss.
"""

from __future__ import annotations

from pathlib import Path

import torch

from atlas.config import ModelConfig, SFTConfig
from atlas.model.transformer import Transformer
from atlas.data.tokenizer import build_tokenizer
from atlas.data.sft import load_sft_data, SFTCollator, LengthCurriculum
from atlas.training.checkpointing import load_checkpoint, save_checkpoint


class SFTTrainer:
    """Supervised fine-tuning loop.

    Args:
        model_config: Model architecture (used only if no checkpoint loaded).
        sft_config: SFT hyperparameters.
    """

    def __init__(self, model_config: ModelConfig, sft_config: SFTConfig) -> None:
        self.mcfg = model_config
        self.scfg = sft_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> None:
        """Execute the SFT training loop."""
        torch.manual_seed(self.scfg.seed)

        # If loading from a checkpoint, extract its saved config first
        # so that tokenizer and model are built with matching settings.
        ckpt = None
        if self.scfg.checkpoint:
            ckpt = torch.load(self.scfg.checkpoint, map_location=self.device, weights_only=False)
            saved_cfg = ckpt.get("config", {})
            if saved_cfg:
                self.mcfg = ModelConfig.from_dict(saved_cfg)
                print(f"[sft] Using model config from checkpoint")

        # Tokenizer — use checkpoint vocab size when available
        tok_type = self.scfg.tokenizer_type
        tok_vocab = self.mcfg.vocab_size
        if tok_vocab <= 256:
            tok_type = "byte"
        tok = build_tokenizer(
            tokenizer_type=tok_type,
            vocab_size=tok_vocab,
            tokenizer_dir=self.scfg.tokenizer_dir,
        )

        # Data
        items = load_sft_data(split="train[:200]", use_hf=True)
        tuples = [(it.prompt, it.response) for it in items]
        curriculum = list(LengthCurriculum(tuples))

        collator = SFTCollator(tok, block_size=self.mcfg.block_size)

        # Model
        self.mcfg.vocab_size = tok.vocab_size
        model = Transformer(self.mcfg).to(self.device)

        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            print(f"[sft] Loaded pretrained model from {self.scfg.checkpoint}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.scfg.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        model.train()

        step = 0
        i = 0
        while step < self.scfg.steps:
            batch = curriculum[i : i + self.scfg.batch_size]
            if not batch:
                i = 0
                continue
            xb, yb = collator.collate(batch)
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits, loss, _, _ = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            i += self.scfg.batch_size

            if step % 20 == 0:
                print(f"[sft] step {step}: loss={loss.item():.4f}")

        # Save
        out = Path(self.scfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        config = self.mcfg.to_dict()
        save_checkpoint(model, optimizer, None, None, step, str(out), config=config)
        print(f"[sft] Saved model → {out / 'model_last.pt'}")
