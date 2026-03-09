"""PPO trainer for RLHF alignment.

Implements Proximal Policy Optimization with:
- Clipped policy loss
- Value function loss (MSE)
- KL penalty against a frozen reference policy
- Entropy bonus (optional)

Reference: https://arxiv.org/abs/1707.06347
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from atlas.config import ModelConfig, RLConfig
from atlas.model.policy import PolicyWithValue
from atlas.model.reward import RewardModel
from atlas.data.tokenizer import build_tokenizer
from atlas.data.sft import format_example, format_prompt_only, Example
from atlas.alignment.rollout import model_logprobs


# ---------------------------------------------------------------------------
# PPO Loss
# ---------------------------------------------------------------------------


@dataclass
class PPOLossOut:
    """Container for PPO loss components."""
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    total_loss: torch.Tensor


def ppo_losses(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
) -> PPOLossOut:
    """Compute clipped PPO policy loss + value loss.

    Args:
        new_logp: ``(N,)`` current policy log-probs.
        old_logp: ``(N,)`` old policy log-probs (from rollout).
        advantages: ``(N,)`` advantage estimates.
        new_values: ``(N,)`` current value predictions.
        old_values: ``(N,)`` old value predictions.
        returns: ``(N,)`` return targets.
        clip_ratio: PPO clipping threshold.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy bonus coefficient.

    Returns:
        ``PPOLossOut`` with all loss components.
    """
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    value_loss = F.mse_loss(new_values, returns)
    entropy = -new_logp.mean()
    approx_kl = torch.mean(old_logp - new_logp)

    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return PPOLossOut(policy_loss, value_loss, entropy, approx_kl, total)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """PPO alignment trainer.

    Loads an SFT policy and a reward model, then optimizes the policy
    via PPO with KL penalty against the frozen SFT reference.

    Args:
        config: RL training configuration.
    """

    def __init__(self, config: RLConfig) -> None:
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> None:
        """Execute the PPO training loop."""
        torch.manual_seed(self.cfg.seed)
        cfg = self.cfg

        # Tokenizer
        tok = build_tokenizer(
            tokenizer_type=cfg.tokenizer_type,
            vocab_size=256,
            tokenizer_dir=cfg.tokenizer_dir,
        )

        # Load SFT policy
        ckpt = torch.load(cfg.policy_ckpt, map_location=self.device, weights_only=False)
        pcfg = ckpt.get("config", {})
        mcfg = ModelConfig.from_dict(pcfg)
        mcfg.block_size = cfg.block_size

        policy = PolicyWithValue(mcfg).to(self.device)
        policy.lm.load_state_dict(ckpt["model"])

        # Frozen reference
        ref = PolicyWithValue(mcfg).to(self.device)
        ref.lm.load_state_dict(ckpt["model"])
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()

        # Reward model
        rckpt = torch.load(cfg.reward_ckpt, map_location=self.device, weights_only=False)
        rcfg = rckpt.get("config", {})
        rm = RewardModel(
            vocab_size=rcfg.get("vocab_size", tok.vocab_size),
            block_size=rcfg.get("block_size", cfg.block_size),
            n_layer=rcfg.get("n_layer", 4),
            n_head=rcfg.get("n_head", 4),
            n_embd=rcfg.get("n_embd", 256),
        ).to(self.device)
        rm.load_state_dict(rckpt["model"])
        rm.eval()

        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

        # Prompt pool
        prompts = _sample_prompts(16)

        for step in range(1, cfg.steps + 1):
            # Select prompts
            batch_prompts = _select_batch(prompts, step, cfg.batch_size)
            texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
            in_ids = [tok.encode(t) for t in texts]

            # Generate rollouts
            with torch.no_grad():
                out_ids = []
                for x in in_ids:
                    idx = torch.tensor([x], dtype=torch.long, device=self.device)
                    out = policy.generate(idx, max_new_tokens=cfg.resp_len, temperature=0.2, top_k=3)
                    out_ids.append(out[0].tolist())

            # Build training batch
            data = []
            for i, prompt in enumerate(batch_prompts):
                full = out_ids[i]
                boundary = len(in_ids[i][-mcfg.block_size:])
                resp_ids = full[boundary:]
                resp_text = tok.decode(resp_ids)
                r = _compute_reward(rm, tok, prompt, resp_text, mcfg.block_size, self.device)
                data.append((torch.tensor(full, dtype=torch.long), boundary, r))

            # Pad and build tensors
            max_len = min(mcfg.block_size, max(t[0].numel() for t in data))
            B = len(data)
            seq = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
            mask = torch.zeros(B, max_len, dtype=torch.bool, device=self.device)
            rewards = torch.zeros(B, max_len, dtype=torch.float, device=self.device)

            for i, (ids, bnd, r) in enumerate(data):
                L = min(ids.numel(), max_len)
                drop = ids.numel() - L
                b = max(0, bnd - drop)
                seq[i, :L] = ids[-L:]
                if L < max_len:
                    seq[i, L:] = 2
                mask[i, b:L] = True
                rewards[i, L - 1] = r

            # Rollout logprobs
            pol_lp = model_logprobs(policy, seq)
            ref_lp = model_logprobs(ref, seq)

            with torch.no_grad():
                _, values, _ = policy(seq, None)
            values = values[:, :-1]

            act_mask = mask[:, 1:]
            old_logp = pol_lp[act_mask].detach()
            ref_logp = ref_lp[act_mask].detach()
            old_values = values[act_mask].detach()

            kl = old_logp - ref_logp
            shaped_r = rewards[:, 1:][act_mask] - cfg.kl_coef * kl
            returns = shaped_r
            adv = returns - old_values
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)

            # PPO update
            policy.train()
            logits_new, values_new, _ = policy(seq, None)
            logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
            labels = seq[:, 1:]
            new_logp = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[act_mask]
            new_values = values_new[:, :-1][act_mask]

            out_loss = ppo_losses(
                new_logp, old_logp, adv, new_values, old_values, returns,
                clip_ratio=cfg.clip_ratio, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
            )

            optimizer.zero_grad(set_to_none=True)
            out_loss.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            policy.eval()

            if step % 10 == 0:
                print(
                    f"[ppo] step {step} | loss {out_loss.total_loss.item():.4f}"
                    f" | value_loss {out_loss.value_loss.item():.4f}"
                )

        # Save
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": policy.state_dict(), "config": mcfg.to_dict()},
            str(out / "model_last.pt"),
        )
        print(f"[ppo] Saved policy → {out / 'model_last.pt'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_reward(rm, tok, prompt, resp_text, block_size, device):
    text = format_example(Example(prompt, resp_text))
    ids = tok.encode(text)
    x = torch.tensor([ids[:block_size]], dtype=torch.long, device=device)
    with torch.no_grad():
        r = rm(x)
    return float(r[0].item())


def _sample_prompts(n: int) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train[:24]")
        arr = []
        for r in ds:
            inst = (r.get("instruction") or "").strip()
            inp = (r.get("input") or "").strip()
            if inp:
                inst = inst + "\n" + inp
            if inst:
                arr.append(inst)
            if len(arr) >= n:
                break
        if arr:
            return arr
    except Exception:
        pass
    base = [
        "Explain the purpose of attention in transformers.",
        "Give two pros and cons of BPE tokenization.",
        "Summarize why PPO is used in RLHF.",
        "Write a tiny Python function that reverses a list.",
    ]
    return (base * ((n + len(base) - 1) // len(base)))[:n]


def _select_batch(prompts, step, batch_size):
    batch = prompts[((step - 1) * batch_size) % len(prompts): (step * batch_size) % len(prompts)]
    if len(batch) < batch_size:
        batch += prompts[: batch_size - len(batch)]
    return batch
