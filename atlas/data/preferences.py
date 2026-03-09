"""Preference data pipeline for reward model training.

Loads pairwise preference data (chosen vs. rejected completions) and provides
a collator that encodes both alternatives for side-by-side scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atlas.data.tokenizer import ByteTokenizer, BPETokenizer
from atlas.data.sft import format_example, Example


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PrefItem:
    """A single preference pair: chosen > rejected for a given prompt."""
    prompt: str
    chosen: str
    rejected: str


def load_preferences(
    split: str = "train[:80]",
    use_hf: bool = True,
) -> list[PrefItem]:
    """Load preference data from HuggingFace or built-in fallback.

    Attempts to use ``Anthropic/hh-rlhf`` as a reference dataset.

    Args:
        split: HuggingFace split string.
        use_hf: Whether to attempt loading from HuggingFace.

    Returns:
        List of ``PrefItem`` instances.
    """
    items: list[PrefItem] = []

    if use_hf:
        try:
            from datasets import load_dataset
            ds = load_dataset("Anthropic/hh-rlhf", split=split)
            for row in ds:
                chosen = (row.get("chosen", "") or "").strip()
                rejected = (row.get("rejected", "") or "").strip()
                if chosen and rejected:
                    # Use first sentence as prompt approximation
                    prompt = chosen.split("\n")[0][:200]
                    items.append(PrefItem(prompt=prompt, chosen=chosen, rejected=rejected))
        except Exception:
            pass

    if not items:
        # Minimal fallback
        items = [
            PrefItem("Explain AI", "AI is a field of computer science...", "AI means robots"),
            PrefItem("What is 2+2?", "2+2 equals 4.", "It's 5."),
            PrefItem("Capital of France?", "Paris is the capital of France.", "London"),
        ]

    return items


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class PairCollator:
    """Encode preference pairs for reward model training.

    Encodes chosen and rejected completions as separate sequences,
    returning two tensors ready for the reward model.

    Args:
        tokenizer: Tokenizer instance.
        block_size: Maximum sequence length.
    """

    def __init__(
        self,
        tokenizer: ByteTokenizer | BPETokenizer,
        block_size: int = 256,
    ) -> None:
        self.tok = tokenizer
        self.block_size = block_size

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, "vocab_size", 256)

    def _encode(self, text: str) -> list[int]:
        ids = self.tok.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    def collate(
        self, batch: list[tuple[str, str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate a batch of ``(prompt, chosen, rejected)`` triples.

        Args:
            batch: List of ``(prompt, chosen, rejected)`` string tuples.

        Returns:
            Tuple of ``(pos_ids, neg_ids)`` each ``(B, block_size)``.
        """
        pos_list: list[list[int]] = []
        neg_list: list[list[int]] = []

        for prompt, chosen, rejected in batch:
            pos_text = format_example(Example(prompt, chosen))
            neg_text = format_example(Example(prompt, rejected))
            pos_ids = self._encode(pos_text)[: self.block_size]
            neg_ids = self._encode(neg_text)[: self.block_size]
            pos_list.append(pos_ids)
            neg_list.append(neg_ids)

        def _pad(seq: list[int], val: int = 2) -> list[int]:
            if len(seq) < self.block_size:
                seq = seq + [val] * (self.block_size - len(seq))
            return seq[: self.block_size]

        pos = torch.tensor([_pad(s) for s in pos_list], dtype=torch.long)
        neg = torch.tensor([_pad(s) for s in neg_list], dtype=torch.long)
        return pos, neg
