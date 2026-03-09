"""``atlas-ppo`` and ``atlas-grpo`` — RL alignment CLIs."""

from __future__ import annotations

import argparse

from atlas.config import RLConfig, load_config


def _build_parser(algo: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Atlas: RLHF alignment ({algo.upper()})"
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--policy-ckpt", type=str, default=None)
    p.add_argument("--reward-ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--kl-coef", type=float, default=None)
    p.add_argument("--group-size", type=int, default=None)
    return p


def main_ppo() -> None:
    p = _build_parser("ppo")
    args = p.parse_args()

    cfg = load_config(args.config, RLConfig) if args.config else RLConfig()
    cfg.algorithm = "ppo"

    if args.policy_ckpt:
        cfg.policy_ckpt = args.policy_ckpt
    if args.reward_ckpt:
        cfg.reward_ckpt = args.reward_ckpt
    if args.out:
        cfg.out_dir = args.out
    if args.steps:
        cfg.steps = args.steps
    if args.kl_coef is not None:
        cfg.kl_coef = args.kl_coef

    if not cfg.policy_ckpt or not cfg.reward_ckpt:
        p.error("--policy-ckpt and --reward-ckpt are required")

    from atlas.alignment.ppo import PPOTrainer
    PPOTrainer(cfg).run()


def main_grpo() -> None:
    p = _build_parser("grpo")
    args = p.parse_args()

    cfg = load_config(args.config, RLConfig) if args.config else RLConfig()
    cfg.algorithm = "grpo"

    if args.policy_ckpt:
        cfg.policy_ckpt = args.policy_ckpt
    if args.reward_ckpt:
        cfg.reward_ckpt = args.reward_ckpt
    if args.out:
        cfg.out_dir = args.out
    if args.steps:
        cfg.steps = args.steps
    if args.kl_coef is not None:
        cfg.kl_coef = args.kl_coef
    if args.group_size is not None:
        cfg.group_size = args.group_size

    if not cfg.policy_ckpt or not cfg.reward_ckpt:
        p.error("--policy-ckpt and --reward-ckpt are required")

    from atlas.alignment.grpo import GRPOTrainer
    GRPOTrainer(cfg).run()


if __name__ == "__main__":
    main_ppo()
