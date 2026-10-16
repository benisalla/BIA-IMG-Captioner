"""Offline tests for the data pipeline (no network, no real tokenizer)."""

from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from bia_captioner.config import DataConfig
from bia_captioner.data import (
    CaptionCollator,
    FlickrDataset,
    as_caption_list,
    build_image_transform,
    to_rgb_image,
)

BOS, EOS, PAD = 1, 2, 0


class FakeTokenizer:
    """Minimal tokenizer: one token per character (ids start at 10)."""

    bos_token_id = BOS
    eos_token_id = EOS
    pad_token_id = PAD

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [10 + ord(char) % 50 for char in text]


class FakeSplit:
    """Tiny stand-in for a Hugging Face dataset split."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, str):  # column access, like datasets.Dataset
            return [row[key] for row in self.rows]
        return self.rows[key]

    def select(self, indices: range) -> "FakeSplit":
        return FakeSplit([self.rows[i] for i in indices])


def make_loader(captions: list[list[str] | str]) -> Any:
    """Return a fake `load_dataset` that also records its arguments."""
    rows = [
        {"image": Image.new("RGB", (40, 30), "white"), "original_alt_text": caps}
        for caps in captions
    ]
    calls: list[dict[str, Any]] = []

    def loader(dataset_id: str, split: str, cache_dir: str | None) -> FakeSplit:
        calls.append({"dataset_id": dataset_id, "split": split, "cache_dir": cache_dir})
        return FakeSplit(rows)

    loader.calls = calls  # type: ignore[attr-defined]
    return loader


def make_dataset(
    captions: list[list[str] | str], max_tokens: int = 64, **kwargs: Any
) -> FlickrDataset:
    config = DataConfig(image_size=32, max_tokens=max_tokens)
    return FlickrDataset(
        config, FakeTokenizer(), split="train", dataset_loader=make_loader(captions), **kwargs
    )


def test_item_shapes_and_dtypes() -> None:
    item = make_dataset([["a dog"]])[0]
    assert item["pixel_values"].shape == (3, 32, 32)
    assert item["pixel_values"].dtype == torch.float32
    assert item["input_ids"].dtype == torch.long
    assert item["input_ids"].shape == (len("a dog") + 2,)  # unpadded: BOS + caption + EOS
    assert item["index"].item() == 0


def test_sequence_starts_with_bos_and_ends_with_eos() -> None:
    ids = make_dataset([["a dog"]])[0]["input_ids"].tolist()
    assert ids[0] == BOS
    assert ids[-1] == EOS


def test_truncation_keeps_start_and_drops_eos() -> None:
    ids = make_dataset([["x" * 100]], max_tokens=8)[0]["input_ids"].tolist()
    assert len(ids) == 8
    assert ids[0] == BOS
    assert EOS not in ids


def test_validation_uses_the_first_caption() -> None:
    dataset = make_dataset([["ab", "abcdef", "abcdefgh"]])
    lengths = {len(dataset[0]["input_ids"]) for _ in range(10)}
    assert lengths == {4}  # always "ab" -> BOS + 2 + EOS


def test_training_samples_every_caption() -> None:
    dataset = make_dataset([["ab", "abcdef", "abcdefgh"]], train=True)
    lengths = {len(dataset[0]["input_ids"]) for _ in range(60)}
    assert lengths == {4, 8, 10}


def test_references_are_all_captions_cleaned() -> None:
    dataset = make_dataset([["  A dog  runs. ", "", "A cat"], "single caption"])
    assert dataset.references(0) == ["A dog runs.", "A cat"]  # empty caption dropped
    assert dataset.references(1) == ["single caption"]  # a plain string becomes a list


def test_image_without_caption_is_rejected() -> None:
    with pytest.raises(ValueError, match="caption"):
        as_caption_list(["", "  "])


def test_limit_keeps_first_samples() -> None:
    assert len(make_dataset([["a"], ["b"], ["c"]], limit=2)) == 2
    assert len(make_dataset([["a"], ["b"]], limit=10)) == 2  # limit larger than data is fine


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit"):
        make_dataset([["a"]], limit=0)


def test_loader_receives_config_values() -> None:
    loader = make_loader([["a"]])
    config = DataConfig(dataset_id="some/dataset", cache_dir=None)
    FlickrDataset(config, FakeTokenizer(), split="test", dataset_loader=loader)
    assert loader.calls == [{"dataset_id": "some/dataset", "split": "test", "cache_dir": None}]


def test_numpy_images_are_converted_to_rgb() -> None:
    float_image = np.ones((8, 8), dtype=np.float32)  # grayscale in [0, 1]
    image = to_rgb_image(float_image)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert to_rgb_image(np.zeros((8, 8, 3), dtype=np.uint8)).mode == "RGB"


def test_transforms_keep_output_shape() -> None:
    image = Image.new("RGB", (50, 20), "gray")
    for train in (False, True):
        assert build_image_transform(16, train=train)(image).shape == (3, 16, 16)


def test_collator_pads_and_masks_labels() -> None:
    dataset = make_dataset([["ab"], ["abcde"]])
    batch = CaptionCollator(pad_token_id=PAD)([dataset[0], dataset[1]])

    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["input_ids"].shape == (2, 7)  # longest: BOS + 5 chars + EOS
    assert batch["labels"].shape == (2, 7)
    assert batch["index"].tolist() == [0, 1]

    # Short sequence: 4 real tokens, then 3 padding positions.
    assert batch["input_ids"][0, 4:].tolist() == [PAD] * 3
    assert batch["labels"][0, 4:].tolist() == [-100] * 3
    assert torch.equal(batch["labels"][0, :4], batch["input_ids"][0, :4])
    # Longest sequence has no padding, so labels equal inputs (no shift).
    assert torch.equal(batch["labels"][1], batch["input_ids"][1])
