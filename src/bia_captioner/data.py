"""Dataset, image transforms and batch collation for Flickr30k.

One sample is a photo plus one of its (five) human-written captions:

- training: a *random* caption each time the image is seen, so every epoch
  goes through each image once (encoder cost) but all captions get used;
- validation: always the first caption, so the loss is comparable over time.
  BLEU is computed against all five (see `references`).

The dataset returns tensors of *different* lengths; the collator pads each
batch only up to its longest caption (dynamic padding), which saves compute.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import PreTrainedTokenizerBase

from bia_captioner.config import DataConfig
from bia_captioner.tokenizer import IGNORE_INDEX, clean_caption, encode_caption

# ImageNet statistics: the pretrained image encoders expect inputs normalised this way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(image_size: int, train: bool) -> v2.Compose:
    """PIL image -> normalised float tensor [3, H, W].

    Training uses a mild random crop and colour jitter. No horizontal flip:
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
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
    captions = [clean_caption(c) for c in captions if c and c.strip()]
    if not captions:
        raise ValueError("image without any caption")
    return captions


class FlickrDataset(Dataset[dict[str, torch.Tensor]]):
    """Image / caption pairs (one item per image), returned as unpadded tensors."""

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

        self.tokenizer = tokenizer
        self.max_tokens = config.max_tokens
        self.image_column = config.image_column
        self.train = train
        self.transform = build_image_transform(config.image_size, train=train)

        # The loader is injectable so tests can run without the network.
        cache_dir = str(config.cache_dir) if config.cache_dir else None
        self.samples = dataset_loader(config.dataset_id, split=split, cache_dir=cache_dir)
        if limit is not None:
            # Keep only the first `limit` images (handy for quick overfit runs).
            self.samples = self.samples.select(range(min(limit, len(self.samples))))
        # Read the caption column once: cheap (no image decoding) and needed for BLEU.
        self.captions = [as_caption_list(v) for v in self.samples[config.caption_column]]

    def __len__(self) -> int:
        return len(self.samples)

    def references(self, index: int) -> list[str]:
        """All human captions of one image (the references for BLEU)."""
        return self.captions[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = to_rgb_image(self.samples[index][self.image_column])
        captions = self.captions[index]
        caption = random.choice(captions) if self.train else captions[0]
        ids = encode_caption(self.tokenizer, caption, self.max_tokens)
        return {
            "pixel_values": self.transform(image),
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),  # to look up references
        }


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
    "FlickrDataset",
    "as_caption_list",
    "build_image_transform",
    "to_rgb_image",
]
