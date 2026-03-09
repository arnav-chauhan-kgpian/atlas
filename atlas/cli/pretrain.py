"""``atlas-train`` — pretraining CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from atlas.config import ModelConfig, TrainConfig, load_config
from atlas.training.trainer import Trainer


def main() -> None:
    p = argparse.ArgumentParser(description="Atlas: pretrain a Transformer LM")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--data", type=str, default=None, help="Path to training text file")
    p.add_argument("--out", type=str, default=None, help="Output directory")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    p.add_argument("--tokenizer", type=str, default=None, choices=["bpe", "byte"])
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--amp", action="store_true")
    # New Tuning Knobs
    p.add_argument("--n-kv-head", type=int, default=None, help="GQA: number of KV heads")
    p.add_argument("--accum", type=int, default=None, help="Gradient accumulation steps")
    p.add_argument("--moe-experts", type=int, default=None, help="Number of MoE experts")
    p.add_argument("--moe-topk", type=int, default=None, help="MoE top-k experts")
    p.add_argument("--temp", type=float, default=None, help="Sampling temperature")
    p.add_argument("--sample-tokens", type=int, default=None)
    p.add_argument("--sample-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--keep-last-k", type=int, default=None)
    args = p.parse_args()

    # Load from YAML or use defaults
    if args.config:
        mcfg = load_config(args.config, ModelConfig)
        tcfg = load_config(args.config, TrainConfig)
    else:
        mcfg = ModelConfig()
        tcfg = TrainConfig()

    # CLI overrides
    if args.data:
        tcfg.data_path = args.data
    if args.out:
        tcfg.out_dir = args.out
    if args.steps:
        tcfg.steps = args.steps
    if args.lr:
        tcfg.lr = args.lr
    if args.batch:
        tcfg.batch_size = args.batch
    if args.resume:
        tcfg.resume_from = args.resume
    if args.tokenizer:
        tcfg.tokenizer_type = args.tokenizer
    if args.tokenizer_dir:
        tcfg.tokenizer_dir = args.tokenizer_dir
    if args.compile:
        tcfg.compile = True
    if args.amp:
        tcfg.mixed_precision = True
    if args.block_size:
        mcfg.block_size = args.block_size
    if args.n_layer:
        mcfg.n_layer = args.n_layer
    if args.n_head:
        mcfg.n_head = args.n_head
    if args.n_embd:
        mcfg.n_embd = args.n_embd
    if args.n_kv_head:
        mcfg.n_kv_head = args.n_kv_head
    if args.moe_experts:
        mcfg.moe_num_experts = args.moe_experts
    if args.moe_topk:
        mcfg.moe_top_k = args.moe_topk

    # Tuning overrides
    if args.accum:
        tcfg.grad_accum_steps = args.accum
    if args.temp:
        tcfg.temperature = args.temp
    if args.sample_tokens:
        tcfg.sample_tokens = args.sample_tokens
    if args.sample_every:
        tcfg.sample_every = args.sample_every
    if args.save_every:
        tcfg.save_every = args.save_every
    if args.log_every:
        tcfg.log_every = args.log_every
    if args.keep_last_k:
        tcfg.keep_last_k = args.keep_last_k

    if not tcfg.data_path:
        p.error("--data is required (path to training text file)")

    trainer = Trainer(mcfg, tcfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
