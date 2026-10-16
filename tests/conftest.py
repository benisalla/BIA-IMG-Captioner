"""Shared test helpers: a tiny offline tokenizer that behaves like the real one."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from bia_captioner.data import SHAPE_COLORS, SHAPE_KINDS, SHAPE_SIZES

# Every word of the synthetic shapes captions, so they round-trip exactly.
SHAPE_WORDS = [
    "a", "on", "the", "left", "and", "right",
    *SHAPE_SIZES, *SHAPE_COLORS, *SHAPE_KINDS,
]  # fmt: skip


def make_word_tokenizer(
    words: list[str], with_pad: bool = True, eos_is_bos: bool = True
) -> PreTrainedTokenizerFast:
    """Word-level tokenizer saved / loaded like GPT-2's, no network needed.

    Like GPT-2: one <|endoftext|> token serves as BOS and EOS, and (with
    with_pad=False) there is no pad token yet.
    """
    specials = ["<|endoftext|>"] + ([] if eos_is_bos else ["<bos>"])
    vocab = {word: i for i, word in enumerate(specials + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token=words[0]))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    kwargs = {"eos_token": "<|endoftext|>", "bos_token": "<|endoftext|>" if eos_is_bos else "<bos>"}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, **kwargs)
    if with_pad:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


@pytest.fixture
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make build_tokenizer() return the shapes word tokenizer instead of downloading GPT-2."""
    from transformers import AutoTokenizer

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: make_word_tokenizer(SHAPE_WORDS, with_pad=False),
    )
