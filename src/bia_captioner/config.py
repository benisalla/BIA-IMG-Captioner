"""All hyperparameters live here, grouped by what they control.

Configs are frozen dataclasses: they cannot be changed by accident after
creation, and they are saved inside every checkpoint so a run can be reproduced.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Image encoders the model can use (see model.py):
#   "vit-b16"  : torchvision ViT-B/16 pretrained on ImageNet (default, best quality)
#   "resnet50" : torchvision ResNet-50 pretrained on ImageNet (lighter, faster)
#   "bia-vit"  : the hand-written BIA Vision Transformer, trained from scratch
ENCODERS = ("vit-b16", "resnet50", "bia-vit")
PRETRAINED_ENCODERS = ("vit-b16", "resnet50")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the data comes from and how it is turned into tensors."""

    # Flickr30k with the standard Karpathy splits: 29 000 / 1 014 / 1 000 images,
    # 5 human captions per image. Plain parquet, downloaded on first use (~4.3 GB).
    dataset_id: str = "AnyModal/flickr30k"
    image_column: str = "image"
    caption_column: str = "original_alt_text"  # list of 5 captions per image
    tokenizer_id: str = "openai-community/gpt2"
    cache_dir: Path | None = None  # Hugging Face cache, None = library default
    train_split: str = "train"
    val_split: str = "validation"  # keep "test" untouched for the final evaluation
    image_size: int = 224
    # Includes BOS and EOS. Flickr30k captions are short: 64 GPT-2 tokens fit ~99.9%.
    max_tokens: int = 64

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.max_tokens < 2:
            raise ValueError("max_tokens must be at least 2 (room for BOS and EOS)")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape of the encoder-decoder network."""

    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    encoder: str = "vit-b16"
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
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pad_token_id in (self.bos_token_id, self.eos_token_id):
            # If pad == eos, the loss ignores EOS and the model never learns to stop.
            raise ValueError("pad_token_id must differ from bos/eos token ids")


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
    # good, a large LR would destroy it) and = lr for the from-scratch BIA-ViT.
    encoder_lr: float | None = None
    freeze_encoder: bool = False  # True = train only the decoder (much faster)
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
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


def resolve_encoder_lr(train_config: TrainConfig, encoder: str) -> float:
    """The encoder LR to use: explicit value, or a sensible default for the encoder type."""
    if train_config.encoder_lr is not None:
        return train_config.encoder_lr
    return 1e-5 if encoder in PRETRAINED_ENCODERS else train_config.lr


def config_to_dict(config: Any) -> dict[str, Any]:
    """Turn a config into plain types so it can be stored in a checkpoint."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}


__all__ = [
    "ENCODERS",
    "PRETRAINED_ENCODERS",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "config_to_dict",
    "resolve_encoder_lr",
]
