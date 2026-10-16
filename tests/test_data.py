"""Offline tests for the data pipeline (no network, no real tokenizer)."""

from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from bia_captioner.config import DataConfig, data_config_for
from bia_captioner.data import (
    CaptionCollator,
    CaptionDataset,
    SyntheticShapes,
    as_caption_list,
    build_dataset,
    build_image_transform,
    holdout_indices,
    render_shapes,
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
) -> CaptionDataset:
    config = DataConfig(image_size=32, max_tokens=max_tokens)
    return CaptionDataset(
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
    assert as_caption_list(["", "  "]) == []
    with pytest.raises(ValueError, match="caption"):
        make_dataset([["", "  "]])


def test_captions_spread_over_several_columns() -> None:
    rows = [{"image": Image.new("RGB", (8, 8)), "c0": "a dog", "c1": "a puppy"}]
    config = DataConfig(image_size=16, caption_columns=("c0", "c1"))
    dataset = CaptionDataset(
        config, FakeTokenizer(), "train", dataset_loader=lambda *a, **k: FakeSplit(rows)
    )
    assert dataset.references(0) == ["a dog", "a puppy"]


def test_one_caption_per_row_is_grouped_by_image_id() -> None:
    # COCO style: 5 rows per image. Each image must appear once, with all its captions.
    colors = {1: "red", 2: "blue"}
    rows = [
        {"image": Image.new("RGB", (8, 8), colors[i]), "caption": f"cap {i}.{k}", "cocoid": i}
        for i in (1, 2)
        for k in range(3)
    ]
    rows.append({"image": Image.new("RGB", (8, 8), "red"), "caption": "late", "cocoid": 1})
    config = DataConfig(image_size=16, caption_columns=("caption",), group_column="cocoid")
    loader = lambda *a, **k: FakeSplit(rows)
    dataset = CaptionDataset(config, FakeTokenizer(), "train", dataset_loader=loader)
    assert len(dataset) == 2
    assert dataset.references(0) == ["cap 1.0", "cap 1.1", "cap 1.2", "late"]
    assert dataset.image(1).getpixel((0, 0)) == (0, 0, 255)  # image of the right group
    assert len(CaptionDataset(config, FakeTokenizer(), "train", 1, dataset_loader=loader)) == 1


def test_synthetic_captions_describe_the_picture() -> None:
    import random

    for seed in range(50):
        image, caption = render_shapes(random.Random(seed), canvas=64)
        words = caption.split()
        colors = [w for w in words if w in ("red", "green", "blue", "yellow")]
        assert words[0] == "a" and words[-1] in ("circle", "square", "triangle", "right")
        assert ("on the left" in caption) == (len(colors) == 2)
        # Every named color is actually drawn.
        pixels = set(map(tuple, np.asarray(image).reshape(-1, 3).tolist()))
        palette = {"red": (220, 40, 40), "green": (40, 160, 60), "blue": (40, 80, 220),
                   "yellow": (230, 190, 20)}  # fmt: skip
        assert all(palette[c] in pixels for c in colors)


def test_synthetic_dataset_is_deterministic_and_splits_differ() -> None:
    config = data_config_for("shapes", image_size=32)
    train = build_dataset(config, FakeTokenizer(), "train", limit=20)
    again = SyntheticShapes(config, FakeTokenizer(), "train", limit=20)
    val = SyntheticShapes(config, FakeTokenizer(), "validation", limit=20)
    assert isinstance(train, SyntheticShapes) and len(train) == 20
    assert torch.equal(train[3]["pixel_values"], again[3]["pixel_values"])
    assert train.captions == again.captions
    assert train.captions != val.captions
    assert train[0]["pixel_values"].shape == (3, 32, 32)


def test_limit_keeps_first_samples() -> None:
    assert len(make_dataset([["a"], ["b"], ["c"]], limit=2)) == 2
    assert len(make_dataset([["a"], ["b"]], limit=10)) == 2  # limit larger than data is fine


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit"):
        make_dataset([["a"]], limit=0)


def test_loader_receives_config_values() -> None:
    loader = make_loader([["a"]])
    config = DataConfig(dataset_id="some/dataset", cache_dir=None)
    CaptionDataset(config, FakeTokenizer(), split="test", dataset_loader=loader)
    assert loader.calls == [{"dataset_id": "some/dataset", "split": "test", "cache_dir": None}]


def test_numpy_images_are_converted_to_rgb() -> None:
    float_image = np.ones((8, 8), dtype=np.float32)  # grayscale in [0, 1]
    image = to_rgb_image(float_image)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert to_rgb_image(np.zeros((8, 8, 3), dtype=np.uint8)).mode == "RGB"


def test_transforms_keep_output_shape_and_use_the_given_normalisation() -> None:
    image = Image.new("RGB", (50, 20), (255, 255, 255))
    for train in (False, True):
        assert build_image_transform(16, train=train)(image).shape == (3, 16, 16)
    white = build_image_transform(4, False, mean=(0.5,) * 3, std=(0.5,) * 3)(image)
    assert torch.allclose(white, torch.ones_like(white))  # (1 - 0.5) / 0.5


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


def test_caption_separator_splits_joined_captions() -> None:
    # CUB birds store ~10 captions in one newline-joined string.
    assert as_caption_list("a red bird.\nit has a short beak\n", "\n") == [
        "a red bird.",
        "it has a short beak",
    ]
    assert as_caption_list("a red bird.\nit has a short beak") == [
        "a red bird. it has a short beak"
    ]


def test_holdout_splits_are_disjoint_complete_and_fixed() -> None:
    parts = {s: holdout_indices(1000, (0.05, 0.1), s) for s in ("train", "validation", "test")}
    assert [len(parts[s]) for s in ("train", "validation", "test")] == [850, 50, 100]
    assert sorted(parts["train"] + parts["validation"] + parts["test"]) == list(range(1000))
    assert parts == {s: holdout_indices(1000, (0.05, 0.1), s) for s in parts}  # same every time
    # Shuffled before splitting: datasets sorted by class still get every class in each split.
    assert max(parts["validation"]) - min(parts["validation"]) > 500
    with pytest.raises(ValueError, match="splits are"):
        holdout_indices(10, (0.1, 0.1), "val")


def test_holdout_dataset_reads_one_split_and_cuts_it() -> None:
    rows = [{"image": Image.new("RGB", (8, 8)), "text": f"flower {i}"} for i in range(40)]
    calls = []

    def loader(dataset_id: str, split: str, cache_dir: Any) -> FakeSplit:
        calls.append(split)
        return FakeSplit(rows)

    config = DataConfig(image_size=16, caption_columns=("text",), holdout=(0.1, 0.25))
    sizes = {
        split: len(CaptionDataset(config, FakeTokenizer(), split, dataset_loader=loader))
        for split in ("train", "validation", "test")
    }
    assert sizes == {"train": 26, "validation": 4, "test": 10}
    assert set(calls) == {"train"}  # only the one real split is ever loaded


def test_flowers_and_birds_presets() -> None:
    flowers, birds = data_config_for("flowers"), data_config_for("birds")
    assert flowers.holdout is not None and flowers.caption_columns == ("text",)
    assert birds.caption_separator == "\n" and birds.holdout is not None
