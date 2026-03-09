"""Pretraining datasets for next-token prediction.

Provides both a simple in-memory ``TextDataset`` for byte-level data and a
``BPEDataset`` with PyTorch ``DataLoader`` support for BPE-tokenized data.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from atlas.data.tokenizer import ByteTokenizer, BPETokenizer


# ---------------------------------------------------------------------------
# Byte-level dataset (in-memory)
# ---------------------------------------------------------------------------


class TextDataset:
    """In-memory byte-level dataset for next-token prediction.

    Loads the entire file into memory as raw bytes and yields random
    ``(x, y)`` blocks where ``y = x`` shifted by one position.

    Args:
        path: Path to a text file.
        block_size: Context window length.
        split_ratio: Fraction of data used for training.
    """

    def __init__(
        self, path: str | Path, block_size: int = 256, split_ratio: float = 0.9
    ) -> None:
        raw = Path(path).read_bytes()
        data = torch.tensor(list(raw), dtype=torch.long)
        n = int(len(data) * split_ratio)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(
        self, split: str, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch.

        Args:
            split: ``"train"`` or ``"val"``.
            batch_size: Number of sequences.
            device: Target device.

        Returns:
            Tuple of ``(x, y)`` each ``(batch_size, block_size)``.
        """
        buf = self.train if split == "train" else self.val
        assert len(buf) > self.block_size + 1, "Data too small for given block_size"
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
        x = torch.stack([buf[i : i + self.block_size] for i in ix])
        y = torch.stack([buf[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# BPE dataset (DataLoader-compatible)
# ---------------------------------------------------------------------------


class BPEDataset(Dataset):
    """BPE-tokenized dataset with PyTorch DataLoader support.

    Tokenizes the entire text file once and stores the result as a flat tensor.
    Each ``__getitem__`` returns a consecutive ``(x, y)`` chunk.

    Args:
        path: Path to a text file.
        tokenizer: A tokenizer with an ``.encode()`` method.
        block_size: Context window length.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer | BPETokenizer,
        block_size: int = 256,
    ) -> None:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(1, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def make_loader(
    path: str | Path,
    tokenizer: ByteTokenizer | BPETokenizer | None,
    block_size: int = 256,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for pretraining.

    If *tokenizer* is ``None``, falls back to byte-level encoding.

    Args:
        path: Path to a text file.
        tokenizer: Tokenizer instance (or ``None`` for byte-level).
        block_size: Context window length.
        batch_size: Batch size.
        shuffle: Shuffle flag.
        num_workers: DataLoader workers.

    Returns:
        A PyTorch ``DataLoader``.
    """
    tok = tokenizer or ByteTokenizer()
    ds = BPEDataset(path, tok, block_size=block_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
