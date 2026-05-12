"""Test the SFT tokenization helper: labels must mask prompt tokens with -100."""

from __future__ import annotations

import pytest

from lora_recipes.train import _tokenize_for_sft


class _StubTokenizer:
    """Tiny char-level stub. Treats each char as a token. eos_token = '<E>'."""

    eos_token = "<E>"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(c) for c in text]}


def test_labels_mask_prompt_tokens() -> None:
    tok = _StubTokenizer()
    out = _tokenize_for_sft(tok, prompt="abc", target=" xy", max_seq_length=50)
    # prompt "abc" → 3 token ids, then target " xy<E>" → 4 ids ([32, 120, 121, 60])
    assert out["input_ids"][:3] == [ord("a"), ord("b"), ord("c")]
    # First 3 label tokens must be -100 (prompt masked).
    assert out["labels"][:3] == [-100, -100, -100]
    # Target tokens must NOT be -100.
    assert all(v != -100 for v in out["labels"][3:])


def test_truncation_keeps_target() -> None:
    tok = _StubTokenizer()
    out = _tokenize_for_sft(tok, prompt="abcdefghij", target=" xy", max_seq_length=6)
    # Truncated to max_seq_length (6 tokens).
    assert len(out["input_ids"]) == 6
    # The target tail (' xy<E>' → 6 chars) should be the visible tail; the
    # very last character of the EOS marker ('>') is at the end.
    assert out["input_ids"][-1] == ord(">")
