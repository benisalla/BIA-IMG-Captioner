"""Score a trained model on a held-out split with corpus BLEU-1..4.

    python -m bia_captioner.evaluate --checkpoint checkpoints/run/best.pt --split test

Each image is captioned once (beam search by default) and compared with all
of its human captions. Optionally writes every prediction to a JSON file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import textwrap
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from bia_captioner.data import CaptionCollator, build_dataset
from bia_captioner.generate import load_model, load_tokenizer
from bia_captioner.metrics import corpus_bleu, exact_match
from bia_captioner.train import autocast, resolve_device, resolve_precision

logger = logging.getLogger(__name__)


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    tokenizer,
    device: torch.device,
    beam_size: int = 3,
    length_penalty: float = 0.6,
    batch_size: int = 32,
    num_workers: int = 2,
    amp_dtype: torch.dtype | None = None,
    shuffle_images: bool = False,
) -> list[str]:
    """Caption every image of the dataset, in order.

    shuffle_images=True is a sanity check: caption i is generated from another,
    randomly chosen image of the split (a fixed random permutation), so scores
    must collapse if the model really reads the image and stay the same if it
    only learned the language.
    """
    if shuffle_images:
        dataset = Subset(dataset, shuffled_order(len(dataset)))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=CaptionCollator(pad_token_id=tokenizer.pad_token_id),
    )  # fmt: skip
    captions: list[str] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        with autocast(device, amp_dtype):
            if beam_size > 1:  # beam search works on one image at a time
                sequences = [
                    model.beam_search(image[None], beam_size, length_penalty=length_penalty)[0][0]
                    for image in pixel_values
                ]
            else:  # greedy, the whole batch at once
                sequences = list(model.generate(pixel_values))
        captions += [
            tokenizer.decode(s.tolist(), skip_special_tokens=True).strip() for s in sequences
        ]
        logger.info("captioned %d / %d images", len(captions), len(dataset))
    return captions


def shuffled_order(n: int, seed: int = 0) -> list[int]:
    """For each image i, another image to take its pixels from (fixed, random).

    The images are put in a random cycle and each one takes the next one's
    place, so no image is ever paired with itself.
    """
    cycle = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    source = list(range(n))
    for k, image in enumerate(cycle):
        source[image] = cycle[(k + 1) % n]
    return source


def spread_indices(n: int, count: int) -> list[int]:
    """`count` indices spread evenly over range(n) (datasets are often sorted by class)."""
    count = min(count, n)
    return sorted({round(i * (n - 1) / max(count - 1, 1)) for i in range(count)})


def save_gallery(
    images: Sequence[Image.Image],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    path: Path,
    columns: int = 4,
    tile: int = 320,
) -> None:
    """Grid of images with the model's caption (and one human caption) under each.

    Made to be shared: a README, a slide or a post.
    """
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    chars = int(tile / 8.6)  # characters per line at these font sizes
    text_height = 84
    rows = -(-len(images) // columns)
    sheet = Image.new("RGB", (columns * tile, rows * (tile + text_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, prediction, refs) in enumerate(zip(images, predictions, references)):
        x, y = (i % columns) * tile, (i // columns) * (tile + text_height)
        side = min(image.size)  # centre square crop, like a thumbnail
        left, top = (image.width - side) // 2, (image.height - side) // 2
        square = image.convert("RGB").crop((left, top, left + side, top + side))
        sheet.paste(square.resize((tile - 8, tile - 8), Image.LANCZOS), (x + 4, y + 4))
        lines = textwrap.wrap(prediction, chars)[:3]
        draw.multiline_text((x + 8, y + tile), "\n".join(lines), fill="black", font=font)
        human = textwrap.shorten("human: " + refs[0], width=int(chars * 1.2), placeholder="...")
        draw.text((x + 8, y + tile + 18 * len(lines) + 6), human, fill=(110, 110, 110), font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BLEU evaluation of a trained BIA captioner.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="test", help="dataset split to score (default: test)")
    p.add_argument("--limit", type=int, default=None, help="score only the first N images")
    p.add_argument("--beam-size", type=int, default=3, help="1 = greedy")
    p.add_argument("--length-penalty", type=float, default=0.6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None, help="write scores + predictions (JSON)")
    p.add_argument("--gallery", type=Path, default=None, help="save a captioned image grid (PNG)")
    p.add_argument("--gallery-size", type=int, default=12, help="images in the gallery")
    p.add_argument(
        "--shuffle-images", action="store_true",
        help="sanity check: caption each image from another image's pixels (scores must drop)",
    )  # fmt: skip
    p.add_argument("--device", default="auto")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (returns None: console scripts pass the return value to sys.exit)."""
    run(argv)


def run(argv: Sequence[str] | None = None, dataset_loader=None) -> dict[str, float]:
    """Caption the split, print BLEU and return the scores."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpx2"):  # hide Hugging Face HTTP logs
        logging.getLogger(name).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    amp_dtype, _ = resolve_precision("auto", device)
    model, data_config = load_model(args.checkpoint, device)
    data_config = dataclasses.replace(data_config, cache_dir=args.cache_dir)
    tokenizer = load_tokenizer(args.checkpoint)

    dataset = build_dataset(
        data_config, tokenizer, args.split, limit=args.limit, dataset_loader=dataset_loader
    )
    predictions = predict(
        model, dataset, tokenizer, device, args.beam_size, args.length_penalty,
        args.batch_size, args.num_workers, amp_dtype, args.shuffle_images,
    )  # fmt: skip
    references = [dataset.references(i) for i in range(len(dataset))]
    scores = corpus_bleu(predictions, references)
    scores["exact_match"] = exact_match(predictions, references)

    shuffled = " | IMAGES SHUFFLED (sanity check)" if args.shuffle_images else ""
    print(f"{args.split}: {len(dataset)} images | beam {args.beam_size}{shuffled}")
    for name, value in scores.items():
        print(f"  {name.upper():11} {100 * value:6.2f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"prediction": p, "references": r} for p, r in zip(predictions, references)]
        args.output.write_text(json.dumps({"scores": scores, "predictions": rows}, indent=2))
        logger.info("wrote %s", args.output)
    if args.gallery is not None:
        picks = spread_indices(len(dataset), args.gallery_size)
        save_gallery(
            [dataset.image(i) for i in picks],
            [predictions[i] for i in picks],
            [references[i] for i in picks],
            args.gallery,
        )
        logger.info("wrote %s", args.gallery)
    return scores


if __name__ == "__main__":
    main()
