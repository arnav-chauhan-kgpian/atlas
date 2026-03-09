"""GRPO trainer for RLHF alignment.

Group Relative Policy Optimization — a value-free variant of PPO that uses
group-relative baselines instead of a learned value function.

Key differences from PPO:
- No value head is used (policy-only optimization)
- Multiple completions per prompt; advantage = reward - group mean reward
- KL penalty is added as an explicit loss term

Reference: https://arxiv.org/abs/2402.03300
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from atlas.config import ModelConfig, RLConfig
from atlas.model.policy import PolicyWithValue
from atlas.model.reward import RewardModel
from atlas.data.tokenizer import build_tokenizer
from atlas.data.sft import format_example, format_prompt_only, Example
from atlas.alignment.rollout import model_logprobs


# ---------------------------------------------------------------------------
# GRPO Loss
# ---------------------------------------------------------------------------


@dataclass
class GRPOLossOut:
    """Container for GRPO loss components."""
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    kl_ref: torch.Tensor
    total_loss: torch.Tensor


def grpo_losses(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    ent_coef: float = 0.0,
    kl_coef: float = 0.0,
    kl_mean: torch.Tensor | None = None,
) -> GRPOLossOut:
    """GRPO policy-only clipped loss + explicit KL penalty.

    Args:
        new_logp: ``(N,)`` current policy log-probs on action tokens.
        old_logp: ``(N,)`` old policy log-probs.
        advantages: ``(N,)`` group-relative advantages.
        clip_ratio: PPO clipping threshold.
        ent_coef: Entropy bonus coefficient.
        kl_coef: KL penalty coefficient.
        kl_mean: Scalar mean KL(π || π_ref).

    Returns:
        ``GRPOLossOut`` with all loss components.
    """
    device = new_logp.device
    if new_logp.numel() == 0:
        zero = torch.tensor(0.0, device=device)
        return GRPOLossOut(zero, zero, zero, zero, zero)

    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    entropy = -new_logp.mean() if ent_coef != 0.0 else new_logp.new_tensor(0.0)
    approx_kl = torch.mean(old_logp - new_logp)
    kl_ref = kl_mean if kl_mean is not None else new_logp.new_tensor(0.0)

    total = policy_loss - ent_coef * entropy + kl_coef * kl_ref
    return GRPOLossOut(policy_loss, entropy, approx_kl, kl_ref, total)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """GRPO alignment trainer.

    Samples multiple completions per prompt, computes group-relative
    advantages, and optimizes the policy with a clipped objective + KL penalty.

    Args:
        config: RL training configuration.
    """

    def __init__(self, config: RLConfig) -> None:
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> None:
        """Execute the GRPO training loop."""
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
        policy.eval()

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
        prompts_pool = _sample_prompts(16)
        G = cfg.group_size
        pool_idx = 0

        for step in range(1, cfg.steps + 1):
            # Select prompts
            P = max(1, cfg.batch_size)
            if pool_idx + P > len(prompts_pool):
                pool_idx = 0
            batch_prompts = prompts_pool[pool_idx: pool_idx + P]
            pool_idx += P

            prompt_texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
            prompt_in_ids = [tok.encode(t) for t in prompt_texts]

            # Generate G completions per prompt
            seq_list, boundary_list, prompt_id_of, raw_rewards = [], [], [], []

            with torch.no_grad():
                for pid, p_ids in enumerate(prompt_in_ids):
                    for g in range(G):
                        idx = torch.tensor([p_ids], dtype=torch.long, device=self.device)
                        out = policy.generate(idx, max_new_tokens=cfg.resp_len, temperature=2, top_k=3)
                        full_ids = out[0].tolist()
                        boundary = len(p_ids[-mcfg.block_size:])
                        resp_ids = full_ids[boundary:]
                        r = _compute_reward(rm, tok, batch_prompts[pid], resp_ids, mcfg.block_size, self.device)
                        seq_list.append(torch.tensor(full_ids, dtype=torch.long))
                        boundary_list.append(boundary)
                        prompt_id_of.append(pid)
                        raw_rewards.append(r)

            # Pad to batch
            B = len(seq_list)
            max_len = min(mcfg.block_size, max(s.numel() for s in seq_list))
            seq = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
            mask = torch.zeros(B, max_len, dtype=torch.bool, device=self.device)

            for i, (ids, bnd) in enumerate(zip(seq_list, boundary_list)):
                L = min(ids.numel(), max_len)
                drop = ids.numel() - L
                b = max(0, bnd - drop)
                seq[i, :L] = ids[-L:]
                if L < max_len:
                    seq[i, L:] = 2
                mask[i, b:L] = True

            # Logprobs
            with torch.no_grad():
                pol_lp = model_logprobs(policy, seq)
                ref_lp = model_logprobs(ref, seq)

            act_mask = mask[:, 1:]
            old_logp = pol_lp[act_mask].detach()
            ref_logp = ref_lp[act_mask].detach()

            # Group-relative advantages
            raw_r = torch.tensor(raw_rewards, dtype=torch.float, device=self.device)
            group_mean = torch.zeros(B, dtype=torch.float, device=self.device)
            for pid in range(P):
                idxs = [i for i in range(B) if prompt_id_of[i] == pid]
                if idxs:
                    idxs_t = torch.tensor(idxs, dtype=torch.long, device=self.device)
                    group_mean[idxs_t] = raw_r[idxs_t].mean()

            traj_adv = raw_r - group_mean

            # Map trajectory advantages to action tokens
            traj_id_for_token = []
            for i in range(B):
                n_i = int(act_mask[i].sum().item())
                traj_id_for_token.extend([i] * n_i)
            traj_ids = torch.tensor(traj_id_for_token, dtype=torch.long, device=self.device)
            adv_flat = traj_adv[traj_ids] if traj_ids.numel() > 0 else torch.zeros(0, device=self.device)
            if adv_flat.numel() > 1:
                adv_flat = (adv_flat - adv_flat.mean()) / adv_flat.std().clamp_min(1e-6)

            # GRPO update
            policy.train()
            logits_new, _, _ = policy(seq, None)
            logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
            labels = seq[:, 1:]
            new_logp = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[act_mask]

            kl_now = (new_logp - ref_logp).mean() if new_logp.numel() > 0 else torch.tensor(0.0, device=self.device)

            out_loss = grpo_losses(
                new_logp, old_logp, adv_flat,
                clip_ratio=cfg.clip_ratio, ent_coef=cfg.ent_coef,
                kl_coef=cfg.kl_coef, kl_mean=kl_now,
            )

            optimizer.zero_grad(set_to_none=True)
            out_loss.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            policy.eval()

            if step % 10 == 0:
                print(
                    f"[grpo] step {step} | loss {out_loss.total_loss.item():.4f}"
                    f" | kl_ref {out_loss.kl_ref.item():.6f}"
                )

        # Save
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": policy.state_dict(), "config": mcfg.to_dict()},
            str(out / "model_last.pt"),
        )
        print(f"[grpo] Saved policy → {out / 'model_last.pt'}")


# ---------------------------------------------------------------------------
# Helpers (shared with PPO — could be factored out)
# ---------------------------------------------------------------------------


def _compute_reward(rm, tok, prompt, resp_ids, block_size, device):
    resp_text = tok.decode(resp_ids)
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
    return [
        "Explain the purpose of attention in transformers.",
        "Give two pros and cons of BPE tokenization.",
        "Summarize why PPO is used in RLHF.",
        "Write a tiny Python function that reverses a list.",
    ] * ((n + 3) // 4)
