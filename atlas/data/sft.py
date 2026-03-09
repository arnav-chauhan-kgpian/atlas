"""Supervised fine-tuning (SFT) data pipeline.

Includes prompt/response formatting, dataset loading from HuggingFace,
collation with prompt label masking, and a length-based curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from atlas.data.tokenizer import ByteTokenizer, BPETokenizer, build_tokenizer


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

TEMPLATE = (
    "<s>\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}</s>"
)


@dataclass
class Example:
    """A single instruction-response pair."""
    instruction: str
    response: str


def format_example(ex: Example) -> str:
    """Format an instruction-response pair into prompt text."""
    return TEMPLATE.format(
        instruction=ex.instruction.strip(),
        response=ex.response.strip(),
    )


def format_prompt_only(instruction: str) -> str:
    """Format an instruction as a prompt (empty response)."""
    return TEMPLATE.format(instruction=instruction.strip(), response="")


# ---------------------------------------------------------------------------
# SFT data types
# ---------------------------------------------------------------------------


@dataclass
class SFTItem:
    """A single SFT training example."""
    prompt: str
    response: str


def load_sft_data(
    split: str = "train[:200]",
    use_hf: bool = True,
) -> list[SFTItem]:
    """Load instruction data from HuggingFace or a built-in fallback.

    Uses ``tatsu-lab/alpaca`` as a familiar instruction-following dataset.

    Args:
        split: HuggingFace split string.
        use_hf: Whether to attempt loading from HuggingFace.

    Returns:
        List of ``SFTItem`` instances.
    """
    items: list[SFTItem] = []

    if use_hf:
        try:
            from datasets import load_dataset
            ds = load_dataset("tatsu-lab/alpaca", split=split)
            for row in ds:
                instr = (row.get("instruction", "") or "").strip()
                inp = (row.get("input", "") or "").strip()
                out = (row.get("output", "") or "").strip()
                if inp:
                    instr = instr + "\n" + inp
                if instr and out:
                    items.append(SFTItem(prompt=instr, response=out))
        except Exception:
            pass

    if not items:
        # Built-in fallback examples
        seeds = [
            ("First prime number", "2"),
            ("What are the three primary colors?", "red, blue, yellow"),
            ("Device name which points to direction?", "compass"),
        ]
        items = [SFTItem(prompt=p, response=r) for p, r in seeds]

    return items


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class SFTCollator:
    """Turn ``(instruction, response)`` pairs into token ids with masked labels.

    Labels for the prompt portion are set to ``-100`` so they don't contribute
    to the cross-entropy loss (causal LM loss on response tokens only).

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
        self, batch: list[tuple[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate a batch of (prompt, response) pairs.

        Args:
            batch: List of ``(prompt_str, response_str)`` tuples.

        Returns:
            Tuple of ``(input_ids, labels)`` each ``(B, block_size)``.
        """
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []

        for prompt, response in batch:
            prefix_text = format_prompt_only(prompt).replace("</s>", "")
            full_text = format_example(Example(prompt, response))
            ids = self._encode(full_text)[: self.block_size]
            prompt_ids = self._encode(prefix_text)[: self.block_size]
            n_prompt = min(len(prompt_ids), len(ids))

            x = ids[:]
            y = ids[:]
            # Shift labels: y[t] = ids[t+1]
            for t in range(len(y) - 1):
                y[t] = ids[t + 1]
            y[-1] = -100
            # Mask prompt positions
            for i in range(n_prompt - 1):
                y[i] = -100

            input_ids.append(x)
            labels.append(y)

        # Pad to block_size
        def _pad(seq: list[int], pad_val: int) -> list[int]:
            if len(seq) < self.block_size:
                seq = seq + [pad_val] * (self.block_size - len(seq))
            return seq[: self.block_size]

        x_t = torch.tensor([_pad(s, 2) for s in input_ids], dtype=torch.long)
        y_t = torch.tensor([_pad(s, -100) for s in labels], dtype=torch.long)
        return x_t, y_t


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


class LengthCurriculum:
    """Iterate examples from short → long prompts for curriculum learning.

    Args:
        items: List of ``(prompt, response)`` tuples.
    """

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self.items = sorted(items, key=lambda p: len(p[0]))
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self) -> tuple[str, str]:
        if self._i >= len(self.items):
            raise StopIteration
        item = self.items[self._i]
        self._i += 1
        return item
