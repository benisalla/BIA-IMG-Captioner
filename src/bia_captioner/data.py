"""Datasets, image transforms and batch collation.

Two kinds of data, same interface (one item per image):

- `CaptionDataset`: any Hugging Face image-caption dataset (Flickr8k, Flickr30k,
  COCO, ...). Captions can be one list column, several string columns, or one
  caption per row grouped by an image id.
- `SyntheticShapes`: images of colored shapes generated on the fly, whose
  caption is fully determined by the picture. No download; the fastest way to
  show that the model really reads the image.

Training gives each image a *random* one of its captions, so every epoch goes
through each image once (encoder cost) while all captions get used. Validation
always uses the first caption, so the loss is comparable over time; BLEU
compares with all of them (see `references`).

The datasets return tensors of *different* lengths; the collator pads each
batch only up to its longest caption (dynamic padding), which saves compute.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import PreTrainedTokenizerBase

from bia_captioner.config import IMAGENET_MEAN, IMAGENET_STD, SYNTHETIC, DataConfig
from bia_captioner.tokenizer import IGNORE_INDEX, clean_caption, encode_caption


def build_image_transform(
    image_size: int,
    train: bool,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> v2.Compose:
    """PIL image -> normalised float tensor [3, H, W].

    `train=True` adds a mild random crop and colour jitter. No horizontal flip:
    it would make captions with "left" / "right" wrong.
    """
    if train:
        steps: list[Any] = [
            v2.RandomResizedCrop(image_size, scale=(0.75, 1.0), antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        steps = [v2.Resize((image_size, image_size), antialias=True)]
    steps += [
        v2.ToImage(),  # PIL -> uint8 tensor [3, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
        v2.Normalize(mean, std),  # what the encoder was trained with
    ]
    return v2.Compose(steps)


def to_rgb_image(image: Any) -> Image.Image:
    """Accept a PIL image or a numpy array and return an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        # Float images are assumed to be in [0, 1]; PIL needs uint8 in [0, 255].
        array = (np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def as_caption_list(value: Any) -> list[str]:
    """A caption cell may hold one string or a list of strings; always return a list."""
    captions = [value] if isinstance(value, str) else list(value)
    return [clean_caption(c) for c in captions if c and c.strip()]


class _CaptionedImages(Dataset[dict[str, torch.Tensor]]):
    """Shared logic: subclasses provide `__len__`, `image(i)` and `self.captions`."""

    captions: list[list[str]]

    def __init__(
        self, config: DataConfig, tokenizer: PreTrainedTokenizerBase, augment: bool
    ) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = config.max_tokens
        self.train = augment  # also picks a random caption per visit
        self.transform = build_image_transform(
            config.image_size, augment, config.image_mean, config.image_std
        )

    def image(self, index: int) -> Image.Image:
        raise NotImplementedError

    def references(self, index: int) -> list[str]:
        """All human captions of one image (the references for BLEU)."""
        return self.captions[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        captions = self.captions[index]
        caption = random.choice(captions) if self.train else captions[0]
        ids = encode_caption(self.tokenizer, caption, self.max_tokens)
        return {
            "pixel_values": self.transform(self.image(index)),
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),  # to look up references
        }


class CaptionDataset(_CaptionedImages):
    """Image / caption pairs from a Hugging Face dataset, one item per image."""

    def __init__(
        self,
        config: DataConfig,
        tokenizer: PreTrainedTokenizerBase,
        split: str,
        limit: int | None = None,
        train: bool = False,
        dataset_loader: Callable[..., Any] = load_dataset,
    ) -> None:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        super().__init__(config, tokenizer, augment=train)
        self.image_column = config.image_column

        # The loader is injectable so tests can run without the network.
        cache_dir = str(config.cache_dir) if config.cache_dir else None
        self.samples = dataset_loader(config.dataset_id, split=split, cache_dir=cache_dir)
        if limit is not None and config.group_column is None:
            # Keep only the first `limit` images (handy for quick overfit runs).
            self.samples = self.samples.select(range(min(limit, len(self.samples))))

        # Read the caption columns once: cheap (no image decoding) and needed for BLEU.
        columns = [list(self.samples[name]) for name in config.caption_columns]
        per_row = [[c for cell in cells for c in as_caption_list(cell)] for cells in zip(*columns)]
        if config.group_column is None:
            self.rows = list(range(len(per_row)))
            self.captions = per_row
        else:
            # One caption per row: gather the rows of each image (first row gives the image).
            groups: dict[Any, list[int]] = {}
            for row, key in enumerate(self.samples[config.group_column]):
                groups.setdefault(key, []).append(row)
            self.rows = [rows[0] for rows in groups.values()]
            self.captions = [[c for r in rows for c in per_row[r]] for rows in groups.values()]
            if limit is not None:
                self.rows, self.captions = self.rows[:limit], self.captions[:limit]
        if any(not captions for captions in self.captions):
            raise ValueError("found an image without any caption")

    def __len__(self) -> int:
        return len(self.rows)

    def image(self, index: int) -> Image.Image:
        return to_rgb_image(self.samples[self.rows[index]][self.image_column])


# ---------------------------------------------------------------------------
# Synthetic shapes
# ---------------------------------------------------------------------------

SHAPE_COLORS = {
    "red": (220, 40, 40),
    "green": (40, 160, 60),
    "blue": (40, 80, 220),
    "yellow": (230, 190, 20),
}
SHAPE_KINDS = ("circle", "square", "triangle")
SHAPE_SIZES = {"small": 0.45, "big": 0.85}  # fraction of the object's cell
SYNTHETIC_SPLITS = {"train": (0, 8000), "validation": (1, 500), "test": (2, 500)}  # (seed, size)


def render_shapes(rng: random.Random, canvas: int = 128) -> tuple[Image.Image, str]:
    """Draw one or two random shapes and return (image, caption).

    "a big red circle" or
    "a small blue square on the left and a big yellow triangle on the right".
    """
    count = rng.choice((1, 2))
    image = Image.new("RGB", (canvas, canvas), "white")
    draw = ImageDraw.Draw(image)
    cell = canvas if count == 1 else canvas // 2
    phrases = []
    for slot in range(count):
        size_name, color, kind = (
            rng.choice(list(SHAPE_SIZES)),
            rng.choice(list(SHAPE_COLORS)),
            rng.choice(SHAPE_KINDS),
        )
        side = SHAPE_SIZES[size_name] * cell * 0.8
        # Random position inside the object's cell (left / right half, or the whole image).
        x0 = slot * cell + rng.uniform(0, cell - side)
        y0 = rng.uniform(canvas / 2 - cell / 2, canvas / 2 + cell / 2 - side)
        box = (x0, y0, x0 + side, y0 + side)
        fill = SHAPE_COLORS[color]
        if kind == "circle":
            draw.ellipse(box, fill=fill)
        elif kind == "square":
            draw.rectangle(box, fill=fill)
        else:
            draw.polygon([(x0 + side / 2, y0), (x0, y0 + side), (x0 + side, y0 + side)], fill=fill)
        phrases.append(f"a {size_name} {color} {kind}")
    if count == 1:
        return image, phrases[0]
    return image, f"{phrases[0]} on the left and {phrases[1]} on the right"


class SyntheticShapes(_CaptionedImages):
    """Generated colored-shape images; train / validation / test use different seeds.

    The caption is fully determined by the image, so a model that ignores the
    image cannot do better than guessing, while a working model reaches ~100%.
    """

    def __init__(
        self,
        config: DataConfig,
        tokenizer: PreTrainedTokenizerBase,
        split: str,
        limit: int | None = None,
        train: bool = False,
    ) -> None:
        if split not in SYNTHETIC_SPLITS:
            raise ValueError(f"synthetic splits are {list(SYNTHETIC_SPLITS)}")
        # No augmentation: a random crop could cut a shape off or move it across halves.
        super().__init__(config, tokenizer, augment=False)
        self.seed, size = SYNTHETIC_SPLITS[split]
        self.size = size if limit is None else min(limit, size)
        self.captions = [[render_shapes(self._rng(i))[1]] for i in range(self.size)]

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed * 1_000_003 + index)  # same image for the same index

    def __len__(self) -> int:
        return self.size

    def image(self, index: int) -> Image.Image:
        return render_shapes(self._rng(index))[0]


def build_dataset(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    limit: int | None = None,
    train: bool = False,
    dataset_loader: Callable[..., Any] | None = None,
) -> Dataset:
    """The dataset named by config.dataset_id ("synthetic" or a Hugging Face id)."""
    if config.dataset_id == SYNTHETIC:
        return SyntheticShapes(config, tokenizer, split, limit=limit, train=train)
    extra = {} if dataset_loader is None else {"dataset_loader": dataset_loader}
    return CaptionDataset(config, tokenizer, split, limit=limit, train=train, **extra)


@dataclass
class CaptionCollator:
    """Stack samples into a batch, padding captions to the longest one.

    Labels are a copy of input_ids with padding set to -100 so the loss
    ignores it. They are *not* shifted here: the model does the shift itself.
    """

    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([sample["pixel_values"] for sample in batch])

        sequences = [sample["input_ids"] for sample in batch]
        lengths = torch.tensor([len(seq) for seq in sequences])
        # Right padding: real tokens first, pad tokens after.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.pad_token_id
        )

        # Mark padding by position, not by value, so a real token is never masked.
        is_padding = torch.arange(input_ids.size(1)) >= lengths[:, None]
        labels = input_ids.masked_fill(is_padding, IGNORE_INDEX)

        out = {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}
        if "index" in batch[0]:
            out["index"] = torch.stack([sample["index"] for sample in batch])
        return out


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "CaptionCollator",
    "CaptionDataset",
    "SyntheticShapes",
    "as_caption_list",
    "build_dataset",
    "build_image_transform",
    "render_shapes",
    "to_rgb_image",
]
