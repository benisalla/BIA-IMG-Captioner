"""Tests for generate.py and evaluate.py: checkpoint -> model -> captions. CPU, no network."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from test_data import FakeSplit
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from bia_captioner import evaluate
from bia_captioner.config import DataConfig, ModelConfig, config_to_dict
from bia_captioner.generate import load_model, load_tokenizer, run
from bia_captioner.model import ImageCaptioner

IMAGE_SIZE = 32
WORDS = ["<pad>", "<bos>", "<eos>", "a", "dog", "runs", "on", "grass"]


def make_tokenizer() -> PreTrainedTokenizerFast:
    """A tiny word-level tokenizer that can be saved/loaded like GPT-2's."""
    vocab = {word: i for i, word in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="a"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>"
    )


def write_checkpoint(run_dir: Path) -> tuple[Path, ImageCaptioner]:
    """Save a checkpoint laid out exactly like train.py does (configs + tokenizer dir)."""
    torch.manual_seed(0)
    model_config = ModelConfig(
        vocab_size=len(WORDS), pad_token_id=0, bos_token_id=1, eos_token_id=2,
        encoder="bia-vit", d_model=32, attention_heads=2, decoder_layers=1,
        encoder_layers=1, patch_size=8, max_tokens=12, dropout=0.0,
    )  # fmt: skip
    model = ImageCaptioner(model_config, image_size=IMAGE_SIZE, pretrained_encoder=False)
    model.eval()
    data_config = DataConfig(image_size=IMAGE_SIZE, max_tokens=12)
    run_dir.mkdir(parents=True)
    path = run_dir / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": config_to_dict(model_config),
            "data_config": config_to_dict(data_config),
        },
        path,
    )
    make_tokenizer().save_pretrained(run_dir / "tokenizer")
    return path, model


def test_load_model_restores_the_same_weights(tmp_path: Path) -> None:
    path, original = write_checkpoint(tmp_path / "run")
    loaded, data_config = load_model(path)
    assert data_config.image_size == IMAGE_SIZE
    assert not loaded.training  # ready for inference
    pixels = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.testing.assert_close(loaded.generate(pixels), original.generate(pixels))


@pytest.mark.parametrize(
    "decoding",
    [
        ["--beam-size", "3"],
        ["--beam-size", "1"],
        ["--beam-size", "1", "--temperature", "0.8", "--top-p", "0.9"],
    ],
)
def test_cli_captions_every_image(tmp_path: Path, decoding: list[str]) -> None:
    path, _ = write_checkpoint(tmp_path / "run")
    image_paths = []
    for name in ("one.png", "two.jpg"):
        image_paths.append(tmp_path / name)
        Image.new("RGB", (100, 80), "white").save(image_paths[-1])  # any size: it is resized

    output = tmp_path / "out" / "captions.json"
    captions = run(
        ["--checkpoint", str(path), "--image", *map(str, image_paths),
         "--output", str(output), "--device", "cpu", *decoding]
    )  # fmt: skip

    assert list(captions) == [str(p) for p in image_paths]
    assert json.loads(output.read_text()) == captions
    for caption in captions.values():
        # Special tokens are stripped from the decoded text.
        assert "<bos>" not in caption and "<pad>" not in caption and "<eos>" not in caption
    assert len(load_tokenizer(path)) == len(WORDS)


def test_evaluate_cli_scores_a_split(tmp_path: Path) -> None:
    path, _ = write_checkpoint(tmp_path / "run")
    rows = [
        {"image": Image.new("RGB", (40, 30), color), "original_alt_text": ["a dog runs", "a dog"]}
        for color in ("white", "black", "red")
    ]

    def loader(dataset_id: str, split: str, cache_dir: Any) -> FakeSplit:
        assert split == "test"
        return FakeSplit(rows)

    output = tmp_path / "scores.json"
    scores = evaluate.run(
        ["--checkpoint", str(path), "--device", "cpu", "--num-workers", "0",
         "--batch-size", "2", "--output", str(output)],
        dataset_loader=loader,
    )  # fmt: skip
    assert set(scores) == {"bleu1", "bleu2", "bleu3", "bleu4", "exact_match"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    saved = json.loads(output.read_text())
    assert len(saved["predictions"]) == 3
    assert saved["predictions"][0]["references"] == ["a dog runs", "a dog"]


def test_evaluate_with_shuffled_images_runs(tmp_path: Path) -> None:
    path, _ = write_checkpoint(tmp_path / "run")
    rows = [{"image": Image.new("RGB", (40, 30), c), "original_alt_text": ["a dog"]}
            for c in ("white", "black")]  # fmt: skip
    scores = evaluate.run(
        ["--checkpoint", str(path), "--device", "cpu", "--num-workers", "0",
         "--beam-size", "1", "--shuffle-images"],
        dataset_loader=lambda *a, **k: FakeSplit(rows),
    )  # fmt: skip
    assert 0.0 <= scores["exact_match"] <= 1.0


def test_evaluate_writes_a_gallery(tmp_path: Path) -> None:
    path, _ = write_checkpoint(tmp_path / "run")
    rows = [{"image": Image.new("RGB", (60, 40), c), "original_alt_text": ["a dog runs"]}
            for c in ("white", "black", "red")]  # fmt: skip
    gallery = tmp_path / "gallery.png"
    evaluate.run(
        ["--checkpoint", str(path), "--device", "cpu", "--num-workers", "0", "--beam-size", "1",
         "--gallery", str(gallery), "--gallery-size", "2"],
        dataset_loader=lambda *a, **k: FakeSplit(rows),
    )  # fmt: skip
    image = Image.open(gallery)
    assert image.size[0] == 4 * 320 and image.size[1] > 320  # one row of 4 tiles + text


def test_evaluate_writes_an_examples_table(tmp_path: Path) -> None:
    path, _ = write_checkpoint(tmp_path / "run")
    rows = [{"image": Image.new("RGB", (60, 40), c), "original_alt_text": ["a dog | runs"]}
            for c in ("white", "black", "red")]  # fmt: skip
    examples = tmp_path / "examples"
    evaluate.run(
        ["--checkpoint", str(path), "--device", "cpu", "--num-workers", "0", "--beam-size", "1",
         "--examples", str(examples), "--examples-size", "2"],
        dataset_loader=lambda *a, **k: FakeSplit(rows),
    )  # fmt: skip
    table = (examples / "README.md").read_text().splitlines()
    assert table[0].startswith("# ")
    body = [line for line in table if line.startswith("| <img")]
    assert len(body) == 2  # first and last image of the split
    assert '<img src="00.jpg"' in body[0] and '<img src="01.jpg"' in body[1]
    assert "a dog \\| runs" in body[0]  # a "|" in a caption must not break the table
    assert body[0].endswith("| no |") or body[0].endswith("| yes |")
    assert Image.open(examples / "00.jpg").size == (160, 160)


def test_save_examples_marks_exact_matches(tmp_path: Path) -> None:
    from bia_captioner.evaluate import save_examples

    images = [Image.new("RGB", (32, 32), "red"), Image.new("RGB", (32, 32), "blue")]
    path = save_examples(
        images, ["a red square", "a red square"], [["a red square"], ["a blue square"]], tmp_path
    )
    rows = [line for line in path.read_text().splitlines() if line.startswith("| <img")]
    assert rows[0].endswith("| yes |") and rows[1].endswith("| no |")


def test_shuffled_order_never_keeps_an_image_and_is_fixed() -> None:
    from bia_captioner.evaluate import shuffled_order

    order = shuffled_order(50)
    assert sorted(order) == list(range(50))  # a permutation
    assert all(order[i] != i for i in range(50))  # nobody keeps its own image
    assert order == shuffled_order(50)
    assert max(abs(order[i] - i) for i in range(50)) > 10  # far apart, not neighbours


def test_gallery_picks_are_spread_over_the_split() -> None:
    from bia_captioner.evaluate import spread_indices

    assert spread_indices(655, 4) == [0, 218, 436, 654]
    assert spread_indices(3, 12) == [0, 1, 2]
