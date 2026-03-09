"""Main Trainer for pretraining.

Combines model, optimizer, scheduler, AMP, checkpointing, logging,
and signal handling into a single, self-contained training loop.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path

import torch
import torch.nn as nn

from atlas.config import ModelConfig, TrainConfig
from atlas.model.transformer import Transformer
from atlas.data.tokenizer import build_tokenizer
from atlas.data.dataset import TextDataset, make_loader
from atlas.training.optimizer import AmpGrad, build_optimizer
from atlas.training.scheduler import WarmupCosineLR
from atlas.training.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    atomic_save_all,
)
from atlas.training.logger import init_logger


class Trainer:
    """Pretraining loop for Atlas Transformer models.

    Handles the full lifecycle: tokenizer setup, model construction,
    optimizer/scheduler creation, checkpoint resume, training, evaluation,
    sampling, and graceful shutdown.

    Args:
        model_config: Model architecture configuration.
        train_config: Training hyperparameters configuration.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ) -> None:
        self.mcfg = model_config
        self.tcfg = train_config

        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Will be initialized in setup()
        self.tok = None
        self.model: Transformer | None = None
        self.optimizer = None
        self.scheduler = None
        self.amp = None
        self.logger = None
        self.step = 0

        # Graceful shutdown
        self._save_requested = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize all components: tokenizer, model, optimizer, etc."""
        # Set seed
        torch.manual_seed(self.tcfg.seed)

        # Tokenizer
        self.tok = build_tokenizer(
            tokenizer_type=self.tcfg.tokenizer_type,
            vocab_size=self.mcfg.vocab_size,
            tokenizer_dir=self.tcfg.tokenizer_dir,
        )

        # Train BPE if needed
        if (
            self.tcfg.tokenizer_type == "bpe"
            and self.tcfg.tokenizer_dir is None
            and hasattr(self.tok, "train")
        ):
            out_dir = Path(self.tcfg.out_dir) / "tokenizer"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.tok.train(self.tcfg.data_path)
            self.tok.save(str(out_dir))
            self.tcfg.tokenizer_dir = str(out_dir)
            print(f"[init] Trained BPE tokenizer → {out_dir}")

        # Update vocab size from tokenizer
        self.mcfg.vocab_size = self.tok.vocab_size

        # Model
        self.model = Transformer(self.mcfg).to(self.device)

        # Compile
        if self.tcfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Optimizer, scheduler, AMP
        self.optimizer = build_optimizer(
            self.model, lr=self.tcfg.lr, weight_decay=self.tcfg.weight_decay
        )
        total_steps = self.tcfg.steps
        warmup = min(self.tcfg.warmup_steps, max(total_steps // 10, 1))
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=warmup,
            total_steps=total_steps,
            base_lr=self.tcfg.lr,
        )
        self.amp = AmpGrad(
            self.optimizer,
            accum=self.tcfg.grad_accum_steps,
            amp=self.tcfg.mixed_precision,
        )

        # Resume checkpoint
        if self.tcfg.resume_from:
            ckpt_path = Path(self.tcfg.resume_from)
            if ckpt_path.exists():
                self.step = load_checkpoint(
                    self.model,
                    str(ckpt_path),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    amp=self.amp,
                    device=self.device,
                )
                print(f"[resume] Loaded checkpoint at step {self.step}")

        # Logger
        self.logger = init_logger(self.tcfg.log_backend, out_dir=self.tcfg.out_dir)

        # Graceful shutdown on interrupt
        self._save_requested = False
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        print(
            f"[init] Model: {self.model.num_parameters():,} params | "
            f"Device: {self.device} | Steps: {self.tcfg.steps}"
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the pretraining loop."""
        assert self.model is not None, "Call setup() before train()"

        loader = make_loader(
            self.tcfg.data_path,
            self.tok,
            block_size=self.mcfg.block_size,
            batch_size=self.tcfg.batch_size,
        )

        self.model.train()
        out_dir = Path(self.tcfg.out_dir)

        while self.step < self.tcfg.steps:
            for xb, yb in loader:
                if self.step >= self.tcfg.steps:
                    break
                if self._save_requested:
                    self._checkpoint(out_dir)
                    print(f"[signal] Saved at step {self.step}. Exiting.")
                    return

                t0 = time.time()
                xb, yb = xb.to(self.device), yb.to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self.amp.amp
                ):
                    logits, loss, _, aux = self.model(xb, yb)
                    if aux.item() > 0:
                        loss = loss + 0.01 * aux  # MoE load-balance penalty

                self.amp.backward(loss)

                if self.amp.should_step():
                    if self.tcfg.grad_clip > 0:
                        if self.amp.amp:
                            self.amp.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.tcfg.grad_clip
                        )
                    self.amp.step()
                    self.amp.zero_grad()
                    lr = self.scheduler.step()
                    self.step += 1

                    # Logging
                    if self.step % self.tcfg.log_every == 0:
                        dt = time.time() - t0
                        toks_per_s = int(xb.numel()) / max(dt, 1e-6)
                        self.logger.log(
                            step=self.step,
                            loss=float(loss.item()),
                            lr=float(lr),
                            **{"sys/tokens_per_s": toks_per_s},
                        )
                        print(
                            f"step {self.step:5d} | loss {loss.item():.4f} "
                            f"| lr {lr:.2e} | {toks_per_s:.0f} tok/s"
                        )

                    # Checkpointing
                    if self.step % self.tcfg.save_every == 0:
                        self._checkpoint(out_dir)

                    # Sampling
                    if (
                        self.tcfg.sample_every > 0
                        and self.step % self.tcfg.sample_every == 0
                    ):
                        self._sample(xb)

        # Final save
        self._checkpoint(out_dir)
        self.logger.close()
        print(f"[done] Training complete. Checkpoint → {out_dir / 'model_last.pt'}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _checkpoint(self, out_dir: Path) -> None:
        atomic_save_all(
            self.model,
            self.optimizer,
            self.scheduler,
            self.amp,
            self.step,
            out_dir,
            config=self.mcfg.to_dict(),
            tokenizer_dir=self.tcfg.tokenizer_dir,
            keep_last_k=self.tcfg.keep_last_k,
        )

    def _sample(self, xb: torch.Tensor) -> None:
        """Generate a sample during training."""
        self.model.eval()
        try:
            with torch.no_grad():
                seed = xb[:1, : self.mcfg.block_size // 2]
                out = self.model.generate(
                    seed,
                    max_new_tokens=self.tcfg.sample_tokens,
                    temperature=self.tcfg.temperature,
                    top_k=self.tcfg.top_k,
                    top_p=self.tcfg.top_p,
                )
                if self.tok:
                    text = self.tok.decode(out[0].cpu().tolist())
                    print(f"\n{'='*40} SAMPLE {'='*40}\n{text[-512:]}\n{'='*88}\n")
                    self.logger.text("samples/generation", text, self.step)
        except Exception as e:
            print(f"[sample] error: {e}")
        self.model.train()

    def _on_signal(self, sig, frame) -> None:
        self._save_requested = True
