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
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bia_captioner.data import CaptionCollator, FlickrDataset
from bia_captioner.generate import load_model, load_tokenizer
from bia_captioner.metrics import corpus_bleu
from bia_captioner.train import autocast, resolve_device, resolve_precision

logger = logging.getLogger(__name__)


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    dataset: FlickrDataset,
    tokenizer,
    device: torch.device,
    beam_size: int = 3,
    length_penalty: float = 0.6,
    batch_size: int = 32,
    num_workers: int = 2,
    amp_dtype: torch.dtype | None = None,
) -> list[str]:
    """Caption every image of the dataset, in order."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=CaptionCollator(pad_token_id=tokenizer.pad_token_id),
    )  # fmt: skip
    captions: list[str] = []
    for batch_index, batch in enumerate(loader):
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

    extra = {} if dataset_loader is None else {"dataset_loader": dataset_loader}
    dataset = FlickrDataset(data_config, tokenizer, args.split, limit=args.limit, **extra)
    predictions = predict(
        model, dataset, tokenizer, device, args.beam_size, args.length_penalty,
        args.batch_size, args.num_workers, amp_dtype,
    )  # fmt: skip
    references = [dataset.references(i) for i in range(len(dataset))]
    scores = corpus_bleu(predictions, references)

    print(f"{args.split}: {len(dataset)} images | beam {args.beam_size}")
    for name, value in scores.items():
        print(f"  {name.upper():6} {100 * value:6.2f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"prediction": p, "references": r} for p, r in zip(predictions, references)]
        args.output.write_text(json.dumps({"scores": scores, "predictions": rows}, indent=2))
        logger.info("wrote %s", args.output)
    return scores


if __name__ == "__main__":
    main()
