"""All hyperparameters live here, grouped by what they control.

Configs are frozen dataclasses: they cannot be changed by accident after
creation, and they are saved inside every checkpoint so a run can be reproduced.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Image encoders (see model.py). Every one except "bia-vit" can start from
# pretrained weights or from scratch (ModelConfig.pretrained).
# ---------------------------------------------------------------------------
BIA_VIT = "bia-vit"  # the project's own Vision Transformer, always from scratch
TORCHVISION_VITS = ("vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14")
TORCHVISION_RESNETS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
# Any Hugging Face vision backbone: "hf:<model id>", for example
#   hf:google/vit-base-patch16-224-in21k   ViT-B/16 pretrained on ImageNet-21k (14M images)
#   hf:facebook/dinov2-base                DINOv2 ViT-B/14, self-supervised on 142M images
#   hf:openai/clip-vit-base-patch16        CLIP ViT-B/16 image tower, 400M image-text pairs
#   hf:microsoft/resnet-50                 ResNet-50 (ImageNet-1k)
HF_PREFIX = "hf:"


def is_valid_encoder(name: str) -> bool:
    return (
        name == BIA_VIT
        or name in TORCHVISION_VITS
        or name in TORCHVISION_RESNETS
        or (name.startswith(HF_PREFIX) and len(name) > len(HF_PREFIX))
    )


# ---------------------------------------------------------------------------
# Datasets. A preset fills DataConfig's dataset fields; see docs/datasets.md.
# ---------------------------------------------------------------------------
SYNTHETIC = "synthetic"  # generated on the fly, no download (see data.SyntheticShapes)
DATASET_PRESETS: dict[str, dict[str, Any]] = {
    # 8k images of colored shapes, captions fully determined by the image.
    # The quickest proof that the architecture learns from pixels (minutes on a CPU).
    "shapes": {"dataset_id": SYNTHETIC, "caption_columns": ("caption",), "group_column": None},
    # "The CIFAR-10 of captioning": 6 552 high-resolution flower photos, one short
    # caption each (written by BLIP, so simple and consistent), ~0.3 GB. Learns in
    # minutes on a GPU, and the pictures are nice to show. One split only:
    # validation / test images are a seeded random 5% / 10% of it.
    "flowers": {
        "dataset_id": "pranked03/flowers-blip-captions",
        "caption_columns": ("text",),
        "group_column": None,
        "holdout": (0.05, 0.10),
    },
    # CUB-200 birds: 5 994 photos of 200 species, ~10 human captions each about
    # colors and body parts ("this bird has a red crown and a short black bill"),
    # ~0.6 GB. One split only: validation / test are a seeded random 8% / 8%.
    "birds": {
        "dataset_id": "Multimodal-Fatima/CUB_train",
        "caption_columns": ("description",),
        "caption_separator": "\n",
        "group_column": None,
        "holdout": (0.08, 0.08),
    },
    # 8 091 everyday photos, 5 captions each, 6k/1k/1k split, ~0.3 GB.
    "flickr8k": {
        "dataset_id": "jxie/flickr8k",
        "caption_columns": tuple(f"caption_{i}" for i in range(5)),
        "group_column": None,
    },
    # 31 014 photos, 5 captions each, Karpathy split 29k/1k/1k, ~4.3 GB.
    "flickr30k": {
        "dataset_id": "AnyModal/flickr30k",
        "caption_columns": ("original_alt_text",),
        "group_column": None,
    },
    # MS-COCO Karpathy split: 113k/5k/5k images, one caption per row (grouped by
    # image id into ~5 captions per image), ~21 GB. The standard captioning benchmark.
    "coco": {
        "dataset_id": "jxie/coco_captions",
        "caption_columns": ("caption",),
        "group_column": "cocoid",
    },
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the data comes from and how it is turned into tensors."""

    dataset_id: str = "AnyModal/flickr30k"  # Hugging Face dataset, or "synthetic"
    image_column: str = "image"
    # One column holding a list of captions, or several columns with one caption each.
    caption_columns: tuple[str, ...] = ("original_alt_text",)
    # A caption cell holding several captions joined by this string (e.g. "\n").
    caption_separator: str | None = None
    # Rows sharing this column's value are one image (datasets with one caption per row).
    group_column: str | None = None
    # For datasets with a single split: (validation, test) fractions carved out of
    # `train_split` with a fixed random order; the splits are then named
    # "train", "validation" and "test" as usual.
    holdout: tuple[float, float] | None = None
    tokenizer_id: str = "openai-community/gpt2"
    cache_dir: Path | None = None  # Hugging Face cache, None = library default
    train_split: str = "train"
    val_split: str = "validation"  # keep "test" untouched for the final evaluation
    image_size: int = 224
    # Normalisation the encoder expects (ImageNet for torchvision; read from the
    # Hugging Face image processor for "hf:" encoders, see train.py).
    image_mean: tuple[float, float, float] = IMAGENET_MEAN
    image_std: tuple[float, float, float] = IMAGENET_STD
    # Includes BOS and EOS. Flickr30k captions average 16 GPT-2 tokens; 64 fits 99.96%.
    max_tokens: int = 64

    def __post_init__(self) -> None:
        # Checkpoints store tuples as lists; turn them back into tuples.
        for name in ("caption_columns", "image_mean", "image_std", "holdout"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.caption_columns:
            raise ValueError("caption_columns must name at least one column")
        if self.holdout is not None and not (
            len(self.holdout) == 2
            and all(0 < f < 1 for f in self.holdout)
            and sum(self.holdout) < 1
        ):
            raise ValueError("holdout must be two fractions in (0, 1) summing to less than 1")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.max_tokens < 2:
            raise ValueError("max_tokens must be at least 2 (room for BOS and EOS)")


def data_config_for(dataset: str, **overrides: Any) -> DataConfig:
    """DataConfig for a preset name (see DATASET_PRESETS)."""
    if dataset not in DATASET_PRESETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(DATASET_PRESETS)}")
    return replace(DataConfig(), **{**DATASET_PRESETS[dataset], **overrides})


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape of the encoder-decoder network."""

    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    encoder: str = "vit_b_16"  # see the encoder list at the top of this file
    pretrained: bool = True  # start the encoder from pretrained weights (not for bia-vit)
    d_model: int = 512
    attention_heads: int = 8
    decoder_layers: int = 6
    ffn_mult: int = 4  # feed-forward hidden size = ffn_mult * d_model
    max_tokens: int = 64
    dropout: float = 0.1
    # Only used by the from-scratch "bia-vit" encoder (its width is d_model).
    patch_size: int = 16
    encoder_layers: int = 6

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not is_valid_encoder(self.encoder):
            raise ValueError(
                f"unknown encoder {self.encoder!r}: use {BIA_VIT!r}, one of "
                f"{TORCHVISION_VITS + TORCHVISION_RESNETS} or 'hf:<model id>'"
            )
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pad_token_id in (self.bos_token_id, self.eos_token_id):
            # If pad == eos, the loss ignores EOS and the model never learns to stop.
            raise ValueError("pad_token_id must differ from bos/eos token ids")

    @property
    def encoder_is_pretrained(self) -> bool:
        """True when the encoder starts from pretrained weights (bia-vit never does)."""
        return self.pretrained and self.encoder != BIA_VIT


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation and bookkeeping settings for one training run."""

    output_dir: Path = Path("checkpoints/run")
    seed: int = 42
    epochs: int = 15
    batch_size: int = 64
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    lr: float = 3e-4  # decoder learning rate
    # Encoder learning rate. None = 1e-5 for a pretrained encoder (it is already
    # good, a large LR would destroy it) and = lr for an encoder trained from scratch.
    encoder_lr: float | None = None
    freeze_encoder: bool = False  # True = train only the decoder (much faster)
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    augment: bool = True  # random crop + colour jitter on training images
    precision: str = "auto"  # "auto" | "fp32" | "bf16" | "fp16"
    num_workers: int = 4
    train_limit: int | None = None  # use only N images (tiny overfit runs)
    val_limit: int | None = None
    log_every: int = 50  # optimizer steps between log lines
    eval_every: int = 500  # optimizer steps between validations
    bleu_samples: int = 500  # val images captioned (greedy) for BLEU at each validation, 0 = off
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    resume: Path | None = None  # checkpoint to continue from

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16 or fp16")
        if self.encoder_lr is not None and self.encoder_lr < 0:
            raise ValueError("encoder_lr must be >= 0")


def resolve_encoder_lr(train_config: TrainConfig, model_config: ModelConfig) -> float:
    """The encoder LR to use: explicit value, or a default that depends on its init."""
    if train_config.encoder_lr is not None:
        return train_config.encoder_lr
    return 1e-5 if model_config.encoder_is_pretrained else train_config.lr


def config_to_dict(config: Any) -> dict[str, Any]:
    """Turn a config into plain types so it can be stored in a checkpoint."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}


__all__ = [
    "BIA_VIT",
    "DATASET_PRESETS",
    "HF_PREFIX",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SYNTHETIC",
    "TORCHVISION_RESNETS",
    "TORCHVISION_VITS",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "config_to_dict",
    "data_config_for",
    "is_valid_encoder",
    "resolve_encoder_lr",
]
