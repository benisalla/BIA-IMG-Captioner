"""End to end through the real command-line entry points, offline and on CPU:

    cap-train (shapes data, BIA-ViT) -> cap-generate -> cap-evaluate (+ gallery)

Only the GPT-2 download is replaced (by a word-level tokenizer, see conftest.py).
"""

import json
import random
from pathlib import Path

import pytest
import torch
from PIL import Image

from bia_captioner import evaluate, generate
from bia_captioner import train as train_module
from bia_captioner.data import render_shapes

TINY = [
    "--dataset", "shapes", "--encoder", "bia-vit", "--image-size", "32", "--patch-size", "8",
    "--d-model", "64", "--heads", "2", "--layers", "1", "--encoder-layers", "1",
    "--max-tokens", "24", "--num-workers", "0", "--device", "cpu",
]  # fmt: skip


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny model once for every test in this file; returns its output dir."""
    from conftest import SHAPE_WORDS, make_word_tokenizer
    from transformers import AutoTokenizer

    out = tmp_path_factory.mktemp("cli") / "run"
    patch = pytest.MonkeyPatch()
    patch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **k: make_word_tokenizer(SHAPE_WORDS, False)
    )
    try:
        # Memorise 8 training images (no regularisation, 100 steps), validate on 8 others.
        train_module.main([
            *TINY, "--epochs", "100", "--batch-size", "8", "--train-limit", "8",
            "--val-limit", "8", "--lr", "3e-3", "--warmup-steps", "2", "--dropout", "0",
            "--label-smoothing", "0", "--log-every", "25", "--eval-every", "50",
            "--bleu-samples", "8", "--output-dir", str(out),
        ])  # fmt: skip
    finally:
        patch.undo()
    return out


def test_cap_train_writes_a_complete_run(trained_run: Path) -> None:
    assert {p.name for p in trained_run.iterdir()} >= {"best.pt", "last.pt", "tokenizer"}
    checkpoint = torch.load(trained_run / "last.pt", weights_only=True)
    assert checkpoint["step"] == 100  # 8 images / batch 8 x 100 epochs
    assert checkpoint["data_config"]["dataset_id"] == "synthetic"
    assert checkpoint["model_config"]["encoder"] == "bia-vit"
    assert checkpoint["train_config"]["augment"] is True  # never applied to shapes, by design
    # Random guessing would be ln(19) ~ 2.9 for this vocabulary.
    assert checkpoint["best_val_loss"] < 2.0


def test_cap_generate_reproduces_a_memorised_caption(trained_run: Path, tmp_path: Path) -> None:
    # Training image 3 of the shapes "train" split (seed 0), saved as a PNG file. The final
    # weights (last.pt) have memorised it; best.pt is chosen on other images.
    image, caption = render_shapes(random.Random(0 * 1_000_003 + 3))
    path = tmp_path / "shape.png"
    image.save(path)
    for flags in (["--beam-size", "2"], ["--beam-size", "1"]):
        captions = generate.run(
            ["--checkpoint", str(trained_run / "last.pt"), "--image", str(path),
             "--device", "cpu", *flags]
        )  # fmt: skip
        assert captions[str(path)] == caption


def test_cap_evaluate_scores_the_test_split_and_draws_a_gallery(
    trained_run: Path, tmp_path: Path
) -> None:
    gallery, output = tmp_path / "gallery.png", tmp_path / "scores.json"
    scores = evaluate.run(
        ["--checkpoint", str(trained_run / "best.pt"), "--split", "test", "--limit", "8",
         "--beam-size", "1", "--num-workers", "0", "--device", "cpu",
         "--output", str(output), "--gallery", str(gallery), "--gallery-size", "4"]
    )  # fmt: skip
    assert set(scores) == {"bleu1", "bleu2", "bleu3", "bleu4", "exact_match"}
    assert len(json.loads(output.read_text())["predictions"]) == 8
    assert Image.open(gallery).size[0] == 4 * 320


def test_overfit_flag_memorises(tmp_path: Path, offline_tokenizer: None) -> None:
    out = tmp_path / "overfit"
    train_module.main([
        *TINY, "--overfit", "4", "--epochs", "60", "--lr", "3e-3", "--warmup-steps", "2",
        "--log-every", "20", "--eval-every", "60", "--bleu-samples", "4", "--output-dir", str(out),
    ])  # fmt: skip
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    assert checkpoint["best_val_loss"] < 0.3  # validation = the same 4 images, memorised
