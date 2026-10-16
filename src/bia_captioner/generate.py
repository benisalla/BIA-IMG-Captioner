"""Use a trained model: image in, caption out.

    python -m bia_captioner.generate --checkpoint checkpoints/run/best.pt --image photo.jpg

Everything needed to rebuild the network (its configs) is stored inside the
checkpoint, and the tokenizer is saved next to it, so no flags have to match.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from bia_captioner.config import DataConfig, ModelConfig
from bia_captioner.data import build_image_transform
from bia_captioner.model import ImageCaptioner
from bia_captioner.train import resolve_device

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: Path, device: torch.device | str = "cpu"
) -> tuple[ImageCaptioner, DataConfig]:
    """Rebuild the model from a checkpoint and put it in eval mode."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = ModelConfig(**checkpoint["model_config"])
    data_values = checkpoint["data_config"]
    data_config = DataConfig(**{**data_values, "cache_dir": None})  # cache path is machine-specific

    # pretrained_encoder=False: the trained weights come from the checkpoint anyway.
    model = ImageCaptioner(model_config, data_config.image_size, pretrained_encoder=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()  # eval: no dropout, BatchNorm uses its running stats
    return model, data_config


def load_tokenizer(checkpoint_path: Path) -> PreTrainedTokenizerBase:
    """The tokenizer that train.py saved next to the checkpoint."""
    return AutoTokenizer.from_pretrained(Path(checkpoint_path).parent / "tokenizer")


@torch.inference_mode()
def caption_image(
    model: ImageCaptioner,
    tokenizer: PreTrainedTokenizerBase,
    image: Image.Image,
    data_config: DataConfig,
    beam_size: int = 3,
    length_penalty: float = 0.6,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    """Caption one image.

    beam_size > 1 uses beam search (best quality, deterministic). With
    beam_size 1, temperature 0 is greedy and temperature > 0 samples
    (optionally restricted with top_k / top_p).
    """
    device = next(model.parameters()).device
    transform = build_image_transform(
        data_config.image_size, False, data_config.image_mean, data_config.image_std
    )
    pixel_values = transform(image.convert("RGB"))
    pixel_values = pixel_values[None].to(device)
    if beam_size > 1:
        tokens, _ = model.beam_search(
            pixel_values, beam_size=beam_size, length_penalty=length_penalty
        )[0]
    else:
        tokens = model.generate(pixel_values, temperature=temperature, top_k=top_k, top_p=top_p)[0]
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=True).strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Caption images with a trained BIA captioner.")
    p.add_argument("--checkpoint", type=Path, required=True, help="path to best.pt / last.pt")
    p.add_argument("--image", type=Path, required=True, nargs="+", help="one or more images")
    p.add_argument("--output", type=Path, default=None, help="save {image: caption} as JSON")
    p.add_argument("--beam-size", type=int, default=3, help="1 = greedy / sampling")
    p.add_argument("--length-penalty", type=float, default=0.6, help="beam search only")
    p.add_argument("--temperature", type=float, default=0.0, help="beam-size 1 only; 0 = greedy")
    p.add_argument("--top-k", type=int, default=None, help="sample only from the k best tokens")
    p.add_argument("--top-p", type=float, default=None, help="nucleus sampling threshold")
    p.add_argument("--device", default="auto")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (returns None: console scripts pass the return value to sys.exit)."""
    run(argv)


def run(argv: Sequence[str] | None = None) -> dict[str, str]:
    """Parse the flags, caption every image and return {image path: caption}."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, data_config = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.checkpoint)

    captions = {}
    for image_path in args.image:
        caption = caption_image(
            model, tokenizer, Image.open(image_path), data_config,
            beam_size=args.beam_size, length_penalty=args.length_penalty,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )  # fmt: skip
        captions[str(image_path)] = caption
        print(f"{image_path}: {caption}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(captions, indent=2) + "\n")
        logger.info("wrote %s", args.output)
    return captions


if __name__ == "__main__":
    main()
