"""``atlas-sft`` — supervised fine-tuning CLI."""

from __future__ import annotations

import argparse

from atlas.config import ModelConfig, SFTConfig, load_config
from atlas.alignment.sft import SFTTrainer


def main() -> None:
    p = argparse.ArgumentParser(description="Atlas: supervised fine-tuning")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--ckpt", type=str, default=None, help="Pretrained checkpoint")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tokenizer-dir", type=str, default=None)
    args = p.parse_args()

    if args.config:
        mcfg = load_config(args.config, ModelConfig)
        scfg = load_config(args.config, SFTConfig)
    else:
        mcfg = ModelConfig()
        scfg = SFTConfig()

    if args.ckpt:
        scfg.checkpoint = args.ckpt
    if args.out:
        scfg.out_dir = args.out
    if args.steps:
        scfg.steps = args.steps
    if args.lr:
        scfg.lr = args.lr
    if args.tokenizer_dir:
        scfg.tokenizer_dir = args.tokenizer_dir

    trainer = SFTTrainer(mcfg, scfg)
    trainer.run()


if __name__ == "__main__":
    main()
