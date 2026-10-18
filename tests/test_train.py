"""Tests for image_captioner.train. CPU only, no network: tiny fake data and model."""

import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from image_captioner.config import DataConfig, ModelConfig, TrainConfig
from image_captioner.train import (
    LAST_CHECKPOINT,
    DistInfo,
    build_dataloaders,
    build_optimizer,
    build_parser,
    caption_bleu,
    configs_from_args,
    count_correct,
    evaluate,
    format_lrs,
    get_dist_info,
    local_main_first,
    lr_lambda,
    model_sizes_from_args,
    resolve_device,
    resolve_precision,
    setup_distributed,
    train,
)

PAD, BOS, EOS, VOCAB = 0, 1, 2, 16
IMAGE_SIZE = 16


# --- fakes that follow the image_captioner.data / image_captioner.model interfaces --------


class FakeDataset(Dataset):
    """N random images, each with its own short token sequence [BOS, ..., EOS]."""

    def __init__(self, n: int = 8) -> None:
        g = torch.Generator().manual_seed(0)
        self.items = []
        for i in range(n):
            length = 3 + i % 4  # different lengths, so padding is exercised
            body = torch.randint(3, VOCAB, (length,), generator=g)
            self.items.append(
                {
                    "pixel_values": torch.randn(3, IMAGE_SIZE, IMAGE_SIZE, generator=g),
                    "input_ids": torch.cat([torch.tensor([BOS]), body, torch.tensor([EOS])]),
                    "index": torch.tensor(i),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]

    def references(self, index: int) -> list[str]:
        """Same text the FakeTokenizer decodes the true caption to."""
        ids = self.items[index]["input_ids"].tolist()
        return [FakeTokenizer().decode(ids, skip_special_tokens=True)]


@dataclass
class FakeCollator:
    """Same contract as image_captioner.data.CaptionCollator."""

    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        ids = nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=self.pad_token_id
        )
        labels = ids.clone()
        labels[ids == self.pad_token_id] = -100
        pixels = torch.stack([b["pixel_values"] for b in batch])
        index = torch.stack([b["index"] for b in batch])
        return {"pixel_values": pixels, "input_ids": ids, "labels": labels, "index": index}


class FakeTokenizer:
    pad_token_id, bos_token_id, eos_token_id = PAD, BOS, EOS

    def __len__(self) -> int:
        return VOCAB

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        special = {PAD, BOS, EOS} if skip_special_tokens else set()
        return " ".join(f"w{i}" for i in ids if i not in special)

    def save_pretrained(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")


class TinyModel(nn.Module):
    """Image vector + previous-token GRU -> next-token logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 32)
        self.embed = nn.Embedding(config.vocab_size, 32)
        self.rnn = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, config.vocab_size)

    def forward(self, pixel_values, input_ids, labels=None, label_smoothing=0.0):
        image = self.encoder(pixel_values.flatten(1))
        hidden, _ = self.rnn(self.embed(input_ids), image.unsqueeze(0))
        logits = self.head(hidden)
        loss = None
        if labels is not None:  # predict token t+1 from tokens <= t
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )
        return logits, loss

    def generate(self, pixel_values, max_new_tokens=None, temperature=0.0, top_k=None):
        ids = torch.full((pixel_values.size(0), 1), BOS, dtype=torch.long)
        for _ in range(max_new_tokens or self.config.max_tokens - 1):
            logits, _ = self(pixel_values, ids)
            ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return ids

    def parameter_groups(self, weight_decay, lr, encoder_lr):
        encoder = list(self.encoder.parameters())
        encoder_ids = {id(p) for p in encoder}
        rest = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {"name": "encoder", "params": encoder, "lr": encoder_lr, "weight_decay": weight_decay},
            {"name": "decoder", "params": rest, "lr": lr, "weight_decay": weight_decay},
        ]


def model_config(**overrides) -> ModelConfig:
    values = {
        "vocab_size": VOCAB, "pad_token_id": PAD, "bos_token_id": BOS, "eos_token_id": EOS,
        "encoder": "bia-vit", "d_model": 32, "attention_heads": 4, "decoder_layers": 1,
        "encoder_layers": 1, "patch_size": 8, "max_tokens": 10, "dropout": 0.0,
    }  # fmt: skip
    return ModelConfig(**{**values, **overrides})


def train_config(tmp_path: Path, **overrides) -> TrainConfig:
    """Small, fast CPU settings; tests override what they need."""
    defaults = {
        "output_dir": tmp_path / "run",
        "epochs": 20,
        "batch_size": 4,
        "lr": 1e-2,
        "encoder_lr": 1e-2,
        "weight_decay": 0.0,
        "warmup_steps": 2,
        "label_smoothing": 0.0,
        "num_workers": 0,
        "log_every": 1,
        "eval_every": 1000,
        "bleu_samples": 0,
        "device": "cpu",
        "precision": "fp32",
    }
    return TrainConfig(**{**defaults, **overrides})


def run(config: TrainConfig, model: nn.Module | None = None, callback=None):
    torch.manual_seed(0)
    model = model or TinyModel(model_config())
    dataset = FakeDataset()
    train_loader, val_loader = build_dataloaders(
        dataset, dataset, FakeCollator(PAD), config, torch.device("cpu")
    )
    state = train(
        model, train_loader, val_loader, FakeTokenizer(),
        DataConfig(image_size=IMAGE_SIZE), model_config(), config, callback=callback,
    )  # fmt: skip
    return state, model, val_loader


# --- tests ------------------------------------------------------------------------


def test_lr_lambda_warms_up_linearly() -> None:
    assert lr_lambda(0, warmup_steps=10, total_steps=100) == pytest.approx(0.1)
    assert lr_lambda(4, warmup_steps=10, total_steps=100) == pytest.approx(0.5)
    assert lr_lambda(9, warmup_steps=10, total_steps=100) == pytest.approx(1.0)


def test_lr_lambda_cosine_decays_to_zero() -> None:
    assert lr_lambda(10, warmup_steps=10, total_steps=110) == pytest.approx(1.0)
    assert lr_lambda(60, warmup_steps=10, total_steps=110) == pytest.approx(0.5)
    assert lr_lambda(110, warmup_steps=10, total_steps=110) == pytest.approx(0.0)
    assert lr_lambda(500, warmup_steps=10, total_steps=110) == pytest.approx(0.0)  # clamped


def test_lr_lambda_without_warmup() -> None:
    assert lr_lambda(0, warmup_steps=0, total_steps=10) == pytest.approx(1.0)


def test_resolve_precision_on_cpu() -> None:
    cpu = torch.device("cpu")
    assert resolve_precision("auto", cpu) == (None, False)
    assert resolve_precision("fp32", cpu) == (None, False)
    assert resolve_precision("bf16", cpu) == (torch.bfloat16, False)
    with pytest.raises(ValueError, match="CUDA"):
        resolve_precision("fp16", cpu)


def test_resolve_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_single_process_helpers_are_no_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    info = setup_distributed("cpu")
    assert info == DistInfo() == get_dist_info()
    assert info.is_main and not info.enabled
    with local_main_first(info):  # no barrier without a process group
        pass
    assert DistInfo(rank=1, world_size=2).enabled and not DistInfo(rank=1, world_size=2).is_main


def test_format_lrs_names_each_group() -> None:
    optimizer = torch.optim.SGD(
        [{"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-3, "name": "decoder"},
         {"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-5, "name": "encoder"}],
    )  # fmt: skip
    assert format_lrs(optimizer) == "decoder 1.0e-03 encoder 1.0e-05"


def test_overfit_flag_reuses_train_split_and_disables_regularisation() -> None:
    args = build_parser().parse_args(["--overfit", "4", "--batch-size", "32"])
    data_config, config = configs_from_args(args)
    assert data_config.val_split == data_config.train_split
    assert (config.train_limit, config.val_limit, config.batch_size) == (4, 4, 4)
    assert config.label_smoothing == 0.0 and not config.augment
    assert args.dropout == 0.0


def test_cli_model_flags_reach_the_model_config() -> None:
    args = build_parser().parse_args(["--encoder", "bia-vit", "--d-model", "256", "--heads", "4"])
    sizes = model_sizes_from_args(args)
    assert sizes["encoder"] == "bia-vit"
    assert (sizes["d_model"], sizes["attention_heads"]) == (256, 4)
    ModelConfig(vocab_size=10, pad_token_id=0, bos_token_id=1, eos_token_id=2, **sizes)


def test_dataset_preset_and_overrides_reach_the_data_config() -> None:
    args = build_parser().parse_args(["--dataset", "coco"])
    data_config, _ = configs_from_args(args)
    assert (data_config.dataset_id, data_config.group_column) == ("jxie/coco_captions", "cocoid")

    args = build_parser().parse_args(
        ["--dataset", "flickr8k", "--caption-columns", "a", "b", "--image-size", "128"]
    )
    data_config, _ = configs_from_args(args)
    assert data_config.dataset_id == "jxie/flickr8k"
    assert (data_config.caption_columns, data_config.image_size) == (("a", "b"), 128)


def test_holdout_and_separator_flags() -> None:
    args = build_parser().parse_args(
        ["--dataset-id", "u/d", "--caption-separator", "|", "--holdout", "0.1", "0.2"]
    )
    data_config, _ = configs_from_args(args)
    assert (data_config.caption_separator, data_config.holdout) == ("|", (0.1, 0.2))


def test_from_scratch_flag_disables_pretrained_weights() -> None:
    assert model_sizes_from_args(build_parser().parse_args([]))["pretrained"] is True
    args = build_parser().parse_args(["--encoder", "resnet50", "--from-scratch"])
    assert model_sizes_from_args(args)["pretrained"] is False


def test_optimizer_lr_depends_on_how_the_encoder_starts() -> None:
    model = TinyModel(model_config())
    for pretrained, encoder_lr in ((True, 1e-5), (False, 3e-4)):
        config = model_config(encoder="vit_b_16", pretrained=pretrained)
        optimizer = build_optimizer(model, TrainConfig(lr=3e-4), torch.device("cpu"), config)
        lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
        assert lrs == {"encoder": encoder_lr, "decoder": 3e-4}


def test_same_seed_gives_the_same_run(tmp_path: Path) -> None:
    first, _, _ = run(train_config(tmp_path / "a", epochs=3))
    second, _, _ = run(train_config(tmp_path / "b", epochs=3))
    assert first.history == second.history


def test_train_reduces_loss_and_writes_checkpoints(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    state, model, val_loader = run(config)

    assert state.step == 20 * 2  # 8 samples / batch 4 = 2 steps per epoch
    first_loss, last_loss = state.history[0][1], state.history[-1][1]
    assert last_loss < first_loss * 0.5
    assert math.isfinite(state.best_val_loss)
    assert evaluate(model, val_loader, torch.device("cpu"))["loss"] < first_loss
    assert model.training  # evaluate() restored train mode

    run_dir = config.output_dir
    assert (run_dir / LAST_CHECKPOINT).exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "tokenizer" / "tokenizer.json").exists()
    checkpoint = torch.load(run_dir / LAST_CHECKPOINT, weights_only=True)
    assert checkpoint["step"] == 40
    assert checkpoint["model_config"]["vocab_size"] == VOCAB
    assert checkpoint["train_config"]["output_dir"] == str(run_dir)


def test_resume_continues_step_count(tmp_path: Path) -> None:
    first, _, _ = run(train_config(tmp_path, epochs=2))
    assert first.step == 4

    resumed_config = train_config(tmp_path, epochs=3, resume=tmp_path / "run" / LAST_CHECKPOINT)
    resumed, _, _ = run(resumed_config)
    assert resumed.step == 6  # only the new epoch was trained
    assert [step for step, _ in resumed.history] == [5, 6]
    assert resumed.best_val_loss <= first.best_val_loss


def test_gradient_accumulation_counts_optimizer_steps(tmp_path: Path) -> None:
    state, _, _ = run(train_config(tmp_path, epochs=1, batch_size=2, grad_accum_steps=2))
    assert state.step == 2  # 4 micro-batches / 2 per step


def test_gradient_accumulation_matches_one_big_batch(tmp_path: Path) -> None:
    # The legacy trainer zeroed the gradients every micro-batch, so only the last one
    # counted. Here 2 x 4 accumulated must give the same update as 1 x 8.
    # (Same-length sequences, so the mean of the two means equals the overall mean.)
    def weights_after(**overrides) -> dict:
        torch.manual_seed(0)
        model = TinyModel(model_config())
        dataset = FakeDataset(n=8)
        for item in dataset.items:
            item["input_ids"] = item["input_ids"][:5]
        config = train_config(tmp_path / str(overrides), epochs=1, warmup_steps=0, **overrides)
        loader, _ = build_dataloaders(
            dataset, dataset, FakeCollator(PAD), config, torch.device("cpu")
        )
        train(
            model, loader, loader, FakeTokenizer(),
            DataConfig(image_size=IMAGE_SIZE), model_config(), config,
        )  # fmt: skip
        return {k: v.clone() for k, v in model.state_dict().items()}

    big = weights_after(batch_size=8, grad_accum_steps=1)
    accumulated = weights_after(batch_size=4, grad_accum_steps=2)
    for name in big:
        torch.testing.assert_close(big[name], accumulated[name], msg=f"{name} differs")


@pytest.mark.parametrize("freeze", [False, True])
def test_train_runs_with_real_model_and_collator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, freeze: bool
) -> None:
    """Same loop with the real ImageCaptioner (BIA-ViT encoder) and CaptionCollator."""
    from image_captioner import train as train_module
    from image_captioner.data import CaptionCollator
    from image_captioner.model import ImageCaptioner

    saved: list[str] = []
    monkeypatch.setattr(
        train_module, "save_checkpoint", lambda path, *a, **k: saved.append(path.name)
    )

    torch.manual_seed(0)
    config = train_config(tmp_path, epochs=1, eval_every=1, freeze_encoder=freeze, bleu_samples=4)
    model = ImageCaptioner(model_config(), image_size=IMAGE_SIZE, pretrained_encoder=False)
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    dataset = FakeDataset()
    train_loader, val_loader = build_dataloaders(
        dataset, dataset, CaptionCollator(PAD), config, torch.device("cpu")
    )
    events: list[dict] = []
    state = train(
        model, train_loader, val_loader, FakeTokenizer(),
        DataConfig(image_size=IMAGE_SIZE), model_config(), config, callback=events.append,
    )  # fmt: skip
    assert state.step == 2
    assert math.isfinite(state.best_val_loss)
    assert LAST_CHECKPOINT in saved
    assert "bleu4" in next(e for e in events if e["kind"] == "val")

    encoder_after = model.encoder.state_dict()
    unchanged = all(torch.equal(encoder_before[k], encoder_after[k]) for k in encoder_before)
    assert unchanged == freeze


def test_count_correct_skips_first_label_and_padding() -> None:
    # Position t predicts label t+1. Vocab of 3; the argmax at each position is 2, 1, 0, 0.
    logits = torch.tensor([[[0, 0, 9], [0, 9, 0], [9, 0, 0], [9, 0, 0]]], dtype=torch.float)
    labels = torch.tensor([[BOS, 2, 0, -100]])  # targets: 2 (right), 0 (wrong: guessed 1), pad
    assert count_correct(logits, labels).item() == 1


def test_caption_bleu_is_perfect_for_a_memorised_set(tmp_path: Path) -> None:
    config = train_config(tmp_path, epochs=60, batch_size=8)
    dataset = FakeDataset()
    _state, model, val_loader = run(config)
    scores = caption_bleu(model, val_loader, FakeTokenizer(), torch.device("cpu"))
    assert scores is not None and scores["bleu1"] > 0.9
    # Datasets without references are skipped, not crashed on.
    plain = torch.utils.data.DataLoader(list(dataset.items), batch_size=4)
    assert caption_bleu(model, plain, FakeTokenizer(), torch.device("cpu")) is None


def test_callback_receives_train_and_val_metrics(tmp_path: Path) -> None:
    events: list[dict] = []
    run(train_config(tmp_path, epochs=10, eval_every=10, bleu_samples=8), callback=events.append)

    train_events = [e for e in events if e["kind"] == "train"]
    val_events = [e for e in events if e["kind"] == "val"]
    assert len(train_events) == 20  # log_every=1, 2 steps per epoch
    assert len(val_events) == 2  # step 10 and step 20
    for event in train_events + val_events:
        assert 0.0 <= event["accuracy"] <= 1.0
        assert math.isfinite(event["loss"])
    assert {"lr", "grad_norm", "tokens_per_sec", "images_per_sec", "total_steps"} <= set(
        train_events[0]
    )
    assert {"perplexity", "bleu4", "sample_prediction", "sample_target"} <= set(val_events[0])
    # The model learns: accuracy goes up, loss goes down.
    assert train_events[-1]["accuracy"] > train_events[0]["accuracy"]
    assert val_events[-1]["loss"] < val_events[0]["loss"]
