"""Tokenizers for Atlas.

Provides a byte-level tokenizer, a BPE tokenizer (backed by HuggingFace
``tokenizers``), and a factory for config-driven selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import torch


# ---------------------------------------------------------------------------
# Byte-level tokenizer
# ---------------------------------------------------------------------------


class ByteTokenizer:
    """Ultra-simple byte-level tokenizer.

    - ``encode(str) → LongTensor [N]``
    - ``decode(ids) → str``
    - ``vocab_size = 256``
    """

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(ids).decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# BPE tokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """BPE tokenizer wrapper using HuggingFace ``tokenizers``.

    Can train on text files or load a pre-trained tokenizer from disk.

    Args:
        vocab_size: Target vocabulary size for training.
        special_tokens: List of special tokens.
    """

    def __init__(
        self,
        vocab_size: int = 32_000,
        special_tokens: list[str] | None = None,
    ) -> None:
        try:
            from tokenizers import ByteLevelBPETokenizer, Tokenizer  # noqa: F401
        except ImportError:
            raise ImportError("Install `tokenizers` for BPE: pip install tokenizers")

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self, data_path: str | Path) -> None:
        """Train a new BPE tokenizer on text file(s).

        Args:
            data_path: Path to a ``.txt`` file or directory of ``.txt`` files.
        """
        from tokenizers import ByteLevelBPETokenizer

        p = Path(data_path)
        files = [str(fp) for fp in p.glob("**/*.txt")] if p.is_dir() else [str(p)]
        if not files:
            raise FileNotFoundError(f"No .txt files found at {data_path}")

        tok = ByteLevelBPETokenizer()
        tok.train(
            files=files,
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
        )
        self._tok = tok

    def save(self, out_dir: str | Path) -> None:
        """Save the trained tokenizer to disk."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        assert self._tok is not None, "Train or load the tokenizer before saving."
        self._tok.save_model(str(out))
        self._tok.save(str(out / "tokenizer.json"))
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        (out / "bpe_meta.json").write_text(json.dumps(meta))

    def load(self, dir_path: str | Path) -> None:
        """Load a trained tokenizer from disk.

        Args:
            dir_path: Directory containing ``tokenizer.json`` and ``bpe_meta.json``.
        """
        from tokenizers import Tokenizer

        dirp = Path(dir_path)
        tok_file = dirp / "tokenizer.json"
        if not tok_file.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {dirp}")

        self._tok = Tokenizer.from_file(str(tok_file))
        meta_file = dirp / "bpe_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.special_tokens = meta.get("special_tokens", self.special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        assert self._tok is not None, "Train or load the tokenizer first."
        return self._tok.encode(text).ids

    def decode(self, ids) -> str:
        """Decode token ids to text."""
        assert self._tok is not None, "Train or load the tokenizer first."
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tok.decode(ids)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_tokenizer(
    tokenizer_type: str = "byte",
    vocab_size: int = 32_000,
    tokenizer_dir: str | None = None,
) -> ByteTokenizer | BPETokenizer:
    """Build a tokenizer from configuration.

    Args:
        tokenizer_type: ``"byte"`` or ``"bpe"``.
        vocab_size: Target vocab size (BPE only).
        tokenizer_dir: Path to load existing BPE tokenizer (optional).

    Returns:
        Instantiated tokenizer.
    """
    if tokenizer_type == "byte":
        return ByteTokenizer()
    elif tokenizer_type == "bpe":
        tok = BPETokenizer(vocab_size=vocab_size)
        if tokenizer_dir is not None:
            tok.load(tokenizer_dir)
        return tok
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type!r}")
