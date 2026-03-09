"""``atlas-rm`` — reward model training CLI."""

from __future__ import annotations

import argparse

from atlas.config import RMConfig, load_config
from atlas.alignment.reward import RewardTrainer


def main() -> None:
    p = argparse.ArgumentParser(description="Atlas: reward model training")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--loss", type=str, default=None, choices=["bt", "margin"])
    args = p.parse_args()

    if args.config:
        cfg = load_config(args.config, RMConfig)
    else:
        cfg = RMConfig()

    if args.out:
        cfg.out_dir = args.out
    if args.steps:
        cfg.steps = args.steps
    if args.loss:
        cfg.loss_type = args.loss

    trainer = RewardTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
