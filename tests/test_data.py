"""Tests for atlas.data — tokenizers, datasets, SFT data.

Covers: ByteTokenizer encode/decode, BPEDataset shift, SFT collator masking,
formatter template, config loader.
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path

from atlas.data.tokenizer import ByteTokenizer, BPETokenizer, build_tokenizer
from atlas.data.sft import (
    SFTCollator,
    format_example,
    format_prompt_only,
    Example,
    load_sft_data,
    LengthCurriculum,
)
from atlas.config import ModelConfig, load_config


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestByteTokenizer:
    def test_encode_decode_roundtrip(self):
        tok = ByteTokenizer()
        text = "Hello, Atlas!"
        assert tok.decode(tok.encode(text)) == text

    def test_vocab_size(self):
        assert ByteTokenizer().vocab_size == 256

    def test_encode_returns_list(self):
        tok = ByteTokenizer()
        ids = tok.encode("abc")
        assert isinstance(ids, list)
        assert ids == [97, 98, 99]


class TestBuildTokenizer:
    def test_byte_factory(self):
        tok = build_tokenizer("byte")
        assert isinstance(tok, ByteTokenizer)

    def test_bpe_factory(self):
        tok = build_tokenizer("bpe", vocab_size=1000)
        assert isinstance(tok, BPETokenizer)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            build_tokenizer("sentencepiece")


# ---------------------------------------------------------------------------
# Dataset shift
# ---------------------------------------------------------------------------


class TestDatasetShift:
    def test_shift(self):
        """x[t+1] should equal y[t] in the byte dataset."""
        tmp = Path("__test_dataset_shift.txt")
        tmp.write_text("abcdefghijklmnop" * 100, encoding="utf-8")
        try:
            from atlas.data.dataset import TextDataset
            ds = TextDataset(str(tmp), block_size=16)
            x, y = ds.get_batch("train", 4, torch.device("cpu"))
            assert torch.all(x[:, 1:] == y[:, :-1])
        finally:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


class TestFormatter:
    def test_format_example(self):
        ex = Example("What is 1+1?", "2")
        text = format_example(ex)
        assert "### Instruction:" in text
        assert "### Response:" in text
        assert "What is 1+1?" in text
        assert "2" in text
        assert text.startswith("<s>")
        assert text.endswith("</s>")

    def test_format_prompt_only(self):
        text = format_prompt_only("Hello")
        assert "Hello" in text
        assert "### Response:" in text


class TestSFTCollator:
    def test_masking(self):
        """Prompt tokens should have label -100."""
        tok = ByteTokenizer()
        collator = SFTCollator(tok, block_size=128)
        batch = [("What is 1+1?", "2")]
        x, y = collator.collate(batch)
        assert x.shape == (1, 128)
        assert y.shape == (1, 128)
        # First few positions (prompt) should be -100
        assert (y[0, :5] == -100).all()


class TestSFTData:
    def test_fallback_loads(self):
        items = load_sft_data(use_hf=False)
        assert len(items) >= 1
        assert items[0].prompt != ""


class TestCurriculum:
    def test_sorted_order(self):
        items = [("long prompt text here", "resp"), ("hi", "resp"), ("medium len", "resp")]
        cur = LengthCurriculum(items)
        result = list(cur)
        lengths = [len(r[0]) for r in result]
        assert lengths == sorted(lengths)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_load_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model:\n  n_layer: 12\n  n_head: 8\n")
        cfg = load_config(str(cfg_file), ModelConfig)
        assert cfg.n_layer == 12
        assert cfg.n_head == 8

    def test_from_dict(self):
        d = {"n_layer": 3, "n_head": 4, "n_embd": 128, "non_existent_key": 99}
        cfg = ModelConfig.from_dict(d)
        assert cfg.n_layer == 3
        assert cfg.n_head == 4
