"""Tokenizer setup shared by the dataset, the model and generation.

We reuse GPT-2's byte-level BPE vocabulary but give it a *dedicated* padding
token. GPT-2 has no pad token, and the common trick `pad = eos` is harmful
here: the loss ignores padding, so it would also ignore EOS and the model
would never learn when a caption ends.

Every sequence looks like:  <|endoftext|> caption tokens ... <|endoftext|>
                             (BOS)                            (EOS)
"""

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

PAD_TOKEN = "<|pad|>"
IGNORE_INDEX = -100  # label value that cross-entropy skips (used for padding)


def build_tokenizer(tokenizer_id: str, cache_dir: Path | None = None) -> PreTrainedTokenizerBase:
    """Load the base tokenizer and add a dedicated pad token."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, cache_dir=str(cache_dir) if cache_dir else None
    )
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token  # GPT-2 style: same token starts and ends
    # We truncate ourselves in encode_caption, so silence the "sequence too long" warning.
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def clean_caption(caption: str) -> str:
    """Normalise whitespace; the model learns the casing and punctuation as written."""
    return " ".join(caption.split())


def encode_caption(tokenizer: PreTrainedTokenizerBase, caption: str, max_tokens: int) -> list[int]:
    """Turn one caption into [BOS, ...caption..., EOS], truncated to max_tokens.

    When the caption is too long we keep the start and drop EOS, so the model
    is never taught to stop in the middle of a sentence.
    """
    body = tokenizer.encode(clean_caption(caption), add_special_tokens=False)
    ids = [tokenizer.bos_token_id, *body, tokenizer.eos_token_id]
    return ids[:max_tokens]


def vocab_size(tokenizer: PreTrainedTokenizerBase) -> int:
    """Size including added tokens (`tokenizer.vocab_size` excludes them)."""
    return len(tokenizer)


__all__ = [
    "IGNORE_INDEX",
    "PAD_TOKEN",
    "build_tokenizer",
    "clean_caption",
    "encode_caption",
    "vocab_size",
]
