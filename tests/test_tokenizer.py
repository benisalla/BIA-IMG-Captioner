"""Tests for tokenizer.py (offline: a word-level tokenizer stands in for GPT-2)."""

import pytest
from conftest import SHAPE_WORDS

from bia_captioner.tokenizer import (
    IGNORE_INDEX,
    PAD_TOKEN,
    build_tokenizer,
    clean_caption,
    encode_caption,
    vocab_size,
)


@pytest.mark.usefixtures("offline_tokenizer")
def test_build_tokenizer_adds_a_dedicated_pad_token() -> None:
    tokenizer = build_tokenizer("any/id")
    assert tokenizer.pad_token == PAD_TOKEN
    # The trap this avoids: pad == eos would make the loss ignore EOS.
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    assert tokenizer.bos_token_id == tokenizer.eos_token_id  # GPT-2 style
    assert vocab_size(tokenizer) == len(SHAPE_WORDS) + 2  # words + endoftext + pad
    assert tokenizer.model_max_length > 10**6  # we truncate ourselves


def test_clean_caption_normalises_whitespace() -> None:
    assert clean_caption("  a  big\tred\n circle ") == "a big red circle"


@pytest.mark.usefixtures("offline_tokenizer")
def test_encode_caption_round_trips() -> None:
    tokenizer = build_tokenizer("any/id")
    caption = "a big red circle on the left and a small blue square on the right"
    ids = encode_caption(tokenizer, caption, max_tokens=64)
    assert ids[0] == tokenizer.bos_token_id and ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(ids, skip_special_tokens=True) == caption
    # Truncation keeps the start and drops EOS.
    short = encode_caption(tokenizer, caption, max_tokens=4)
    assert len(short) == 4 and tokenizer.eos_token_id not in short[1:]


def test_ignore_index_is_skipped_by_cross_entropy() -> None:
    import torch
    import torch.nn.functional as F

    logits = torch.randn(3, 5)
    labels = torch.tensor([1, IGNORE_INDEX, 2])
    expected = F.cross_entropy(logits[[0, 2]], labels[[0, 2]])
    assert torch.allclose(F.cross_entropy(logits, labels), expected)
