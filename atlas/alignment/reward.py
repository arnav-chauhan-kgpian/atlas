"""Reward model trainer.

Trains a ``RewardModel`` on pairwise preference data using Bradley-Terry
or margin ranking loss.
"""

from __future__ import annotations

from pathlib import Path

import torch

from atlas.config import RMConfig
from atlas.model.reward import RewardModel, bradley_terry_loss, margin_ranking_loss
from atlas.data.tokenizer import build_tokenizer
from atlas.data.preferences import load_preferences, PairCollator
from atlas.training.checkpointing import save_checkpoint


class RewardTrainer:
    """Reward model training loop.

    Args:
        config: Reward model training configuration.
    """

    def __init__(self, config: RMConfig) -> None:
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> None:
        """Execute the reward model training loop."""
        torch.manual_seed(self.cfg.seed)

        # Tokenizer
        tok = build_tokenizer(
            tokenizer_type=self.cfg.tokenizer_type,
            vocab_size=self.cfg.vocab_size,
            tokenizer_dir=self.cfg.tokenizer_dir,
        )

        # Data
        items = load_preferences(split="train[:80]", use_hf=True)
        triples = [(it.prompt, it.chosen, it.rejected) for it in items]
        collator = PairCollator(tok, block_size=self.cfg.block_size)

        # Model
        model = RewardModel(
            vocab_size=tok.vocab_size,
            block_size=self.cfg.block_size,
            n_layer=self.cfg.n_layer,
            n_head=self.cfg.n_head,
            n_embd=self.cfg.n_embd,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.lr, betas=(0.9, 0.999)
        )

        loss_fn = (
            bradley_terry_loss if self.cfg.loss_type == "bt" else margin_ranking_loss
        )

        # Train
        step = 0
        i = 0
        while step < self.cfg.steps:
            batch = triples[i : i + self.cfg.batch_size]
            if not batch:
                i = 0
                continue
            pos, neg = collator.collate(batch)
            pos, neg = pos.to(self.device), neg.to(self.device)

            r_pos = model(pos)
            r_neg = model(neg)
            loss = loss_fn(r_pos, r_neg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            i += self.cfg.batch_size

            if step % 25 == 0:
                acc = (r_pos > r_neg).float().mean().item()
                print(f"[rm] step {step}: loss={loss.item():.4f} acc={acc:.2f}")

        # Save
        out = Path(self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "vocab_size": tok.vocab_size,
            "block_size": self.cfg.block_size,
            "n_layer": self.cfg.n_layer,
            "n_head": self.cfg.n_head,
            "n_embd": self.cfg.n_embd,
        }
        save_checkpoint(model, optimizer, None, None, step, str(out), config=config)
        print(f"[rm] Saved reward model → {out / 'model_last.pt'}")
