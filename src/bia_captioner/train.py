"""Train the image captioning model.

The file is split into small pieces you can read top to bottom:

1. setup helpers   - seeding, device, mixed precision, multi-GPU (DDP) setup
2. data            - DataLoaders
3. optimisation    - AdamW + warmup/cosine learning-rate schedule
4. evaluation      - validation loss, perplexity, accuracy, BLEU, one decoded sample
5. checkpoints     - save / load everything needed to resume
6. train()         - the training loop (takes ready-made objects, easy to test)
7. CLI             - main() builds everything from command-line flags

Run it with:  python -m bia_captioner.train --help

Multi-GPU / multi-node: launch the same command with torchrun, e.g.
    torchrun --nproc_per_node=4 -m bia_captioner.train ...
torchrun starts one process per GPU and sets RANK / WORLD_SIZE / LOCAL_RANK;
the code below picks those up and wraps the model in DistributedDataParallel.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from bia_captioner.config import (
    BIA_VIT,
    DATASET_PRESETS,
    TORCHVISION_RESNETS,
    TORCHVISION_VITS,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_to_dict,
    data_config_for,
    resolve_encoder_lr,
)
from bia_captioner.metrics import corpus_bleu
from bia_captioner.tokenizer import IGNORE_INDEX, vocab_size

try:  # numpy is optional; we only need it to seed its RNG
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

logger = logging.getLogger(__name__)

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"


# ---------------------------------------------------------------------------
# 1. Setup helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed every random number generator we use."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
    # We do NOT set torch.backends.cudnn.deterministic = True:
    # it makes convolutions slower, and tiny run-to-run noise is fine here.


def resolve_device(name: str = "auto") -> torch.device:
    """Turn "auto" | "cpu" | "cuda" | "mps" into a torch.device."""
    # Under torchrun each process owns one GPU: cuda:LOCAL_RANK.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda", local_rank)
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """Pick the autocast dtype and whether we need a GradScaler.

    Returns (dtype, use_scaler). dtype None means plain fp32 (no autocast).
    - bf16 has the same range as fp32, so it needs no loss scaling.
    - fp16 has a small range; small gradients underflow to 0 without a GradScaler.
    """
    if precision == "fp32":
        return None, False
    if precision == "bf16":
        return torch.bfloat16, False
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training needs a CUDA device")
        return torch.float16, True
    if precision == "auto":
        if device.type != "cuda":
            return None, False  # CPU / MPS: keep it simple and exact
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True
    raise ValueError(f"unknown precision: {precision!r}")


@dataclass(frozen=True)
class DistInfo:
    """Who am I in a multi-process run? Defaults describe a single process."""

    rank: int = 0  # global id of this process (0 .. world_size-1)
    world_size: int = 1  # total number of processes (= total GPUs)
    local_rank: int = 0  # id of this process on its own machine

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Only the main process logs, decodes samples and writes checkpoints."""
        return self.rank == 0


def setup_distributed(device_name: str = "auto") -> DistInfo:
    """Join the process group if we were launched by torchrun; no-op otherwise."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistInfo()
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = device_name in ("auto", "cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)  # this process only talks to its own GPU
    # NCCL is the fast backend for GPUs; gloo works on CPU (handy for testing).
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank)


def get_dist_info() -> DistInfo:
    """Current process-group state (single process if none was started)."""
    if not (dist.is_available() and dist.is_initialized()):
        return DistInfo()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank)


@contextmanager
def local_main_first(info: DistInfo) -> Iterator[None]:
    """Let local rank 0 run the block first (downloads, cache building).

    The other processes on the same machine wait, then reuse the warm cache
    instead of all downloading the same files at once.
    """
    if info.enabled and info.local_rank != 0:
        dist.barrier()  # wait for the local main process
    yield
    if info.enabled and info.local_rank == 0:
        dist.barrier()  # release the others


def autocast(device: torch.device, dtype: torch.dtype | None) -> AbstractContextManager:
    """Mixed-precision context; does nothing when dtype is None."""
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def build_model_config(tokenizer: Any, data_config: DataConfig, **sizes: Any) -> ModelConfig:
    """ModelConfig whose vocab and special ids come straight from the tokenizer.

    `sizes` are the remaining ModelConfig fields (encoder, d_model, ...).
    """
    return ModelConfig(
        vocab_size=vocab_size(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_tokens=data_config.max_tokens,
        **sizes,
    )


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own numpy/python seed.

    PyTorch already seeds torch in each worker; we copy that seed to the others
    so random augmentations (and the random caption choice) differ between workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    collate_fn: Any,
    config: TrainConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create the train and validation DataLoaders.

    With several processes, a DistributedSampler gives each one a different
    slice of the data, so batch_size is *per process* (per GPU).
    """
    info = get_dist_info()
    workers = config.num_workers
    common = {
        "batch_size": config.batch_size,
        "collate_fn": collate_fn,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",  # faster CPU -> GPU copies
        "persistent_workers": workers > 0,  # do not restart workers every epoch
        "worker_init_fn": seed_worker if workers > 0 else None,
    }
    generator = torch.Generator()
    # Same order on every run; + rank so augmentation noise differs per process.
    generator.manual_seed(config.seed + info.rank)

    if info.enabled:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, seed=config.seed, drop_last=True
        )
        # Note: to split evenly, the sampler may repeat a few val samples.
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_dataset, sampler=train_sampler, drop_last=True, generator=generator, **common
        )
        val_loader = DataLoader(val_dataset, sampler=val_sampler, drop_last=False, **common)
        return train_loader, val_loader

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,  # every step sees a full batch
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Send every tensor of the batch to the device."""
    # non_blocking overlaps the copy with compute (works with pinned memory)
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# 3. Optimisation
# ---------------------------------------------------------------------------


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to 0.

    `step` is the number of optimizer steps already taken (starts at 0).
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # never exactly 0, so step 0 still learns
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(
    model: nn.Module, config: TrainConfig, device: torch.device, model_config: ModelConfig
) -> AdamW:
    """AdamW with the model's own parameter groups (no decay on norms/biases, etc.)."""
    groups = model.parameter_groups(
        weight_decay=config.weight_decay,
        lr=config.lr,
        encoder_lr=resolve_encoder_lr(config, model_config),
    )
    # fused=True runs the whole update in one CUDA kernel (faster); CUDA only.
    return AdamW(groups, lr=config.lr, betas=(0.9, 0.98), fused=device.type == "cuda")


def format_lrs(optimizer: Optimizer) -> str:
    """Current learning rate of each named param group, e.g. "decoder 3.0e-04 encoder 1.0e-05"."""
    lrs = {group.get("name", "lr"): group["lr"] for group in optimizer.param_groups}
    return " ".join(f"{name} {lr:.1e}" for name, lr in sorted(lrs.items()))


def build_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Warmup + cosine schedule. Call scheduler.step() once per optimizer step."""
    return LambdaLR(optimizer, lambda step: lr_lambda(step, warmup_steps, total_steps))


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------


def count_target_tokens(labels: torch.Tensor) -> torch.Tensor:
    """Number of tokens the loss is averaged over.

    The model predicts token t+1 from tokens <= t, so the first label is never
    a target. Padding (-100) is never a target either.
    """
    return (labels[:, 1:] != IGNORE_INDEX).sum()


def count_correct(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """How many next tokens the model guessed right (top-1), ignoring padding.

    0.6 means 6 caption tokens out of 10 are predicted exactly, when the model
    sees the true caption so far.
    """
    predictions = logits[:, :-1].argmax(dim=-1)  # guess for token t+1 at position t
    targets = labels[:, 1:]
    return ((predictions == targets) & (targets != IGNORE_INDEX)).sum()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Token-weighted mean loss, perplexity and token accuracy on a whole loader."""
    was_training = model.training
    model.eval()  # turns off dropout
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct_sum = torch.zeros((), device=device, dtype=torch.float64)
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast(device, amp_dtype):
            logits, loss = model(batch["pixel_values"], batch["input_ids"], labels=batch["labels"])
        # Each batch has a different number of real tokens, so weight by it.
        n_tokens = count_target_tokens(batch["labels"])
        loss_sum += loss.double() * n_tokens
        token_sum += n_tokens
        correct_sum += count_correct(logits, batch["labels"])
    model.train(was_training)
    if dist.is_initialized():
        # Every process saw a different slice: add up the totals of all of them.
        for total in (loss_sum, token_sum, correct_sum):
            dist.all_reduce(total)
    tokens = token_sum.clamp(min=1)
    mean_loss = (loss_sum / tokens).item()  # one GPU sync at the end
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 50.0)),
        "accuracy": (correct_sum / tokens).item(),
    }


@torch.inference_mode()
def caption_bleu(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    max_images: int | None = None,
) -> dict[str, float] | None:
    """Greedy-caption up to `max_images` images and score them with corpus BLEU.

    Needs a dataset with `references(index)` and batches with an "index" key
    (bia_captioner.data datasets + CaptionCollator); returns None otherwise.
    """
    references_of = getattr(loader.dataset, "references", None)
    if references_of is None:
        return None
    was_training = model.training
    model.eval()
    hypotheses: list[str] = []
    references: list[list[str]] = []
    for batch in loader:
        if "index" not in batch or (max_images is not None and len(hypotheses) >= max_images):
            break
        with autocast(device, amp_dtype):
            tokens = model.generate(batch["pixel_values"].to(device), temperature=0.0)
        hypotheses += [tokenizer.decode(t, skip_special_tokens=True) for t in tokens.tolist()]
        references += [references_of(int(i)) for i in batch["index"]]
    model.train(was_training)
    if not hypotheses:
        return None
    n = len(hypotheses) if max_images is None else min(max_images, len(hypotheses))
    return corpus_bleu(hypotheses[:n], references[:n])


@torch.inference_mode()
def decode_sample(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> tuple[str, str]:
    """Greedy-decode the first validation example. Returns (prediction, target)."""
    was_training = model.training
    model.eval()
    batch = next(iter(loader))
    pixel_values = batch["pixel_values"][:1].to(device)
    with autocast(device, amp_dtype):
        generated = model.generate(pixel_values, temperature=0.0)
    model.train(was_training)
    prediction = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    target = tokenizer.decode(batch["input_ids"][0].tolist(), skip_special_tokens=True)
    return prediction, target


def run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    step: int,
    bleu_samples: int = 0,
) -> dict[str, Any]:
    """Evaluate, log the metrics, BLEU and one decoded sample, return all of them."""
    metrics: dict[str, Any] = evaluate(model, val_loader, device, amp_dtype)  # all processes
    if not get_dist_info().is_main:
        return metrics
    if bleu_samples > 0:
        bleu = caption_bleu(model, val_loader, tokenizer, device, amp_dtype, bleu_samples)
        if bleu is not None:
            metrics.update(bleu)
    bleu_text = f" | bleu4 {metrics['bleu4']:.4f}" if "bleu4" in metrics else ""
    logger.info(
        "val | step %d | loss %.4f | ppl %.2f | acc %.3f%s",
        step, metrics["loss"], metrics["perplexity"], metrics["accuracy"], bleu_text,
    )  # fmt: skip
    prediction, target = decode_sample(model, val_loader, tokenizer, device, amp_dtype)
    logger.info("sample target:     %s", target)
    logger.info("sample prediction: %s", prediction)
    metrics["sample_prediction"] = prediction
    metrics["sample_target"] = target
    return metrics


# ---------------------------------------------------------------------------
# 5. Checkpoints
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    """Counters that must survive a restart."""

    step: int = 0  # optimizer steps done
    epoch: int = 0  # epoch we are in
    best_val_loss: float = math.inf
    history: list[tuple[int, float]] = field(default_factory=list)  # (step, train loss)


def rng_state() -> dict[str, Any]:
    """Snapshot every RNG so a resumed run continues the same random stream."""
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if np is not None:
        name, keys, pos, has_gauss, cached = np.random.get_state()
        # store plain lists so torch.load(weights_only=True) can read them back
        state["numpy"] = (name, keys.tolist(), pos, has_gauss, cached)
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore what rng_state() saved."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if np is not None and "numpy" in state:
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    state: TrainState,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> None:
    """Save everything needed to resume. Written atomically."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
        "best_val_loss": state.best_val_loss,
        "data_config": config_to_dict(data_config),
        "model_config": config_to_dict(model_config),
        "train_config": config_to_dict(train_config),
        "rng_state": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    # os.replace is atomic: a crash mid-save never leaves a half-written file.
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> TrainState:
    """Load a checkpoint into the given objects and return the saved counters."""
    # weights_only=True refuses to run arbitrary pickled code (safe loading).
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint["scaler"]:  # empty dict when scaler disabled
        scaler.load_state_dict(checkpoint["scaler"])
    set_rng_state(checkpoint["rng_state"])
    return TrainState(
        step=checkpoint["step"],
        epoch=checkpoint["epoch"],
        best_val_loss=checkpoint["best_val_loss"],
    )


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------


def epoch_batches(loader: DataLoader, epoch: int, seed: int, skip: int) -> Iterator[dict]:
    """Batches of one epoch, in a shuffle order that depends only on (seed, epoch).

    Re-seeding per epoch means a resumed run sees the same order, so we can
    skip the batches that were already used before the restart.
    """
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)  # new shuffle each epoch, same on every process
    elif loader.generator is not None:
        loader.generator.manual_seed(seed + epoch)
    return islice(iter(loader), skip, None)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: Any,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainState:
    """Run the full training loop and return the final counters.

    Everything is passed in already built, so tests can use tiny fakes.
    `callback` (optional) receives every logged metric as a dict, on the main
    process only; the notebook uses it to draw live training curves:
        {"kind": "train", "step", "epoch", "total_steps", "loss", "accuracy",
         "lr": {group: lr}, "grad_norm", "tokens_per_sec"}
        {"kind": "val", "step", "epoch", "loss", "perplexity", "accuracy",
         "bleu1".."bleu4" (when computed), "sample_prediction", "sample_target"}
    """
    cfg = train_config
    info = get_dist_info()
    device = resolve_device(cfg.device)
    amp_dtype, use_scaler = resolve_precision(cfg.precision, device)
    model.to(device)
    if cfg.freeze_encoder:
        model.freeze_encoder(True)  # before building the optimizer: frozen params are left out
    model.train()
    raw_model = model  # the plain model: used for checkpoints, eval and generation

    # One optimizer step = grad_accum_steps micro-batches.
    accum = cfg.grad_accum_steps
    steps_per_epoch = len(train_loader) // accum
    if steps_per_epoch == 0:
        raise ValueError("train set too small for batch_size * grad_accum_steps")
    total_steps = cfg.epochs * steps_per_epoch

    optimizer = build_optimizer(raw_model, cfg, device, model_config)
    scheduler = build_scheduler(optimizer, cfg.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if info.is_main:
        tokenizer.save_pretrained(output_dir / "tokenizer")  # inference needs the same vocab

    state = TrainState()
    if cfg.resume is not None:
        state = load_checkpoint(Path(cfg.resume), raw_model, optimizer, scheduler, scaler)
        logger.info("resumed from %s at step %d", cfg.resume, state.step)

    if info.enabled:
        # DDP averages the gradients of all processes after each backward pass.
        # It also copies rank 0's weights to everyone at start, so all begin equal.
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    def checkpoint(path: Path) -> None:
        if not info.is_main:
            return  # one writer is enough: all processes hold the same weights
        save_checkpoint(
            path, raw_model, optimizer, scheduler, scaler, state,
            data_config, model_config, cfg,
        )  # fmt: skip

    def emit(event: dict[str, Any]) -> None:
        if callback is not None and info.is_main:
            callback(event)

    def validate_and_save() -> None:
        metrics = run_validation(
            raw_model, val_loader, tokenizer, device, amp_dtype, state.step, cfg.bleu_samples
        )
        emit({"kind": "val", "step": state.step, "epoch": state.epoch, **metrics})
        val_loss = metrics["loss"]
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            checkpoint(output_dir / BEST_CHECKPOINT)
        checkpoint(output_dir / LAST_CHECKPOINT)

    logger.info(
        "training: %d epochs x %d steps (effective batch %d = %d per process x %d accum x %d "
        "processes) on %s, amp=%s, encoder=%s%s",
        cfg.epochs, steps_per_epoch, cfg.batch_size * accum * info.world_size,
        cfg.batch_size, accum, info.world_size, device, amp_dtype,
        model_config.encoder, " (frozen)" if cfg.freeze_encoder else "",
    )  # fmt: skip

    # Running sums for the log line (kept on device: no GPU sync every step).
    window_loss = torch.zeros((), device=device)
    window_tokens = torch.zeros((), device=device)  # real tokens, for tok/s
    window_targets = torch.zeros((), device=device)  # predicted tokens, for accuracy
    window_correct = torch.zeros((), device=device)
    window_images = 0
    window_start = time.perf_counter()

    for epoch in range(state.step // steps_per_epoch, cfg.epochs):
        state.epoch = epoch
        done_in_epoch = state.step - epoch * steps_per_epoch  # > 0 only right after resume
        batches = epoch_batches(train_loader, epoch, cfg.seed, skip=done_in_epoch * accum)
        micro_batches = (steps_per_epoch - done_in_epoch) * accum  # drop the leftover tail

        for micro_step, batch in enumerate(islice(batches, micro_batches)):
            batch = move_batch(batch, device)
            is_update_step = (micro_step + 1) % accum == 0
            # With DDP, skip the gradient all-reduce until the last micro-batch:
            # syncing once per optimizer step is enough and much cheaper.
            sync = model.no_sync() if info.enabled and not is_update_step else nullcontext()
            with sync:
                with autocast(device, amp_dtype):
                    logits, loss = model(
                        batch["pixel_values"],
                        batch["input_ids"],
                        labels=batch["labels"],
                        label_smoothing=cfg.label_smoothing,
                    )
                # Divide so the summed gradients equal the gradient of the mean loss.
                scaler.scale(loss / accum).backward()
            window_loss += loss.detach() / accum
            window_tokens += (batch["labels"] != IGNORE_INDEX).sum()
            window_targets += count_target_tokens(batch["labels"])
            window_correct += count_correct(logits.detach(), batch["labels"])
            window_images += batch["pixel_values"].shape[0]

            if not is_update_step:
                continue  # keep accumulating gradients

            # --- one optimizer step ---
            scaler.unscale_(optimizer)  # back to real gradient values before clipping
            max_norm = cfg.grad_clip if cfg.grad_clip > 0 else math.inf
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)  # skips the step if fp16 gradients overflowed
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # frees memory, slightly faster
            scheduler.step()
            state.step += 1

            if state.step % cfg.log_every == 0:
                elapsed = time.perf_counter() - window_start
                mean_loss = window_loss.item() / cfg.log_every
                accuracy = (window_correct / window_targets.clamp(min=1)).item()
                # loss/accuracy are this process's; speeds are estimated for all processes
                tokens_per_sec = window_tokens.item() * info.world_size / elapsed
                images_per_sec = window_images * info.world_size / elapsed
                state.history.append((state.step, mean_loss))
                logger.info(
                    "train | epoch %d | step %d/%d | loss %.4f | acc %.3f | lr %s | grad %.2f "
                    "| %.0f img/s",
                    epoch, state.step, total_steps, mean_loss, accuracy, format_lrs(optimizer),
                    grad_norm.item(), images_per_sec,
                )  # fmt: skip
                emit({
                    "kind": "train", "step": state.step, "epoch": epoch,
                    "total_steps": total_steps, "loss": mean_loss, "accuracy": accuracy,
                    "lr": {g.get("name", "lr"): g["lr"] for g in optimizer.param_groups},
                    "grad_norm": grad_norm.item(), "tokens_per_sec": tokens_per_sec,
                    "images_per_sec": images_per_sec,
                })  # fmt: skip
                for total in (window_loss, window_tokens, window_targets, window_correct):
                    total.zero_()
                window_images = 0
                window_start = time.perf_counter()

            # Validate every eval_every steps, and once more at the very end.
            if state.step % cfg.eval_every == 0 or state.step == total_steps:
                validate_and_save()
                window_start = time.perf_counter()  # don't count eval time in the speed

    state.epoch = cfg.epochs
    checkpoint(output_dir / LAST_CHECKPOINT)
    logger.info("done | best val loss %.4f", state.best_val_loss)
    return state


# ---------------------------------------------------------------------------
# 7. Command line
# ---------------------------------------------------------------------------


ENCODER_HELP = (
    f"image encoder: {BIA_VIT!r} (own ViT, from scratch), a torchvision ViT "
    f"{TORCHVISION_VITS}, a torchvision ResNet {TORCHVISION_RESNETS}, or any Hugging Face "
    "vision model as 'hf:<model id>' (e.g. hf:google/vit-base-patch16-224-in21k)"
)


def build_parser() -> argparse.ArgumentParser:
    """All CLI flags. Defaults come from the config dataclasses."""
    d, t = DataConfig(), TrainConfig()
    m = ModelConfig(vocab_size=1, pad_token_id=0, bos_token_id=1, eos_token_id=1)
    p = argparse.ArgumentParser(description="Train the BIA image captioning model.")

    data = p.add_argument_group("data")
    data.add_argument(
        "--dataset", choices=list(DATASET_PRESETS), default="flickr30k",
        help="dataset preset (see docs/datasets.md); the flags below override it",
    )  # fmt: skip
    data.add_argument("--dataset-id", default=None, help="Hugging Face dataset id")
    data.add_argument("--image-column", default=None)
    data.add_argument("--caption-columns", nargs="+", default=None)
    data.add_argument("--group-column", default=None, help="rows sharing it are one image")
    data.add_argument("--tokenizer-id", default=d.tokenizer_id)
    data.add_argument("--cache-dir", type=Path, default=d.cache_dir)
    data.add_argument("--train-split", default=d.train_split)
    data.add_argument("--val-split", default=d.val_split)
    data.add_argument("--image-size", type=int, default=d.image_size)
    data.add_argument("--max-tokens", type=int, default=d.max_tokens)

    model = p.add_argument_group("model")
    model.add_argument("--encoder", default=m.encoder, help=ENCODER_HELP)
    model.add_argument(
        "--from-scratch", action="store_true",
        help="random-init the encoder instead of loading pretrained weights",
    )  # fmt: skip
    model.add_argument("--d-model", type=int, default=m.d_model)
    model.add_argument("--heads", type=int, default=m.attention_heads)
    model.add_argument("--layers", type=int, default=m.decoder_layers, help="decoder layers")
    model.add_argument("--ffn-mult", type=int, default=m.ffn_mult)
    model.add_argument("--dropout", type=float, default=m.dropout)
    model.add_argument("--patch-size", type=int, default=m.patch_size, help="bia-vit only")
    model.add_argument("--encoder-layers", type=int, default=m.encoder_layers, help="bia-vit only")

    opt = p.add_argument_group("training")
    opt.add_argument("--epochs", type=int, default=t.epochs)
    opt.add_argument("--batch-size", type=int, default=t.batch_size)
    opt.add_argument("--grad-accum", type=int, default=t.grad_accum_steps)
    opt.add_argument("--lr", type=float, default=t.lr)
    opt.add_argument(
        "--encoder-lr", type=float, default=t.encoder_lr,
        help="default: 1e-5 for a pretrained encoder, = --lr for one trained from scratch",
    )  # fmt: skip
    opt.add_argument("--freeze-encoder", action="store_true", help="train only the decoder")
    opt.add_argument("--weight-decay", type=float, default=t.weight_decay)
    opt.add_argument("--warmup-steps", type=int, default=t.warmup_steps)
    opt.add_argument("--grad-clip", type=float, default=t.grad_clip)
    opt.add_argument("--label-smoothing", type=float, default=t.label_smoothing)
    opt.add_argument("--no-augment", action="store_true", help="no random crop / colour jitter")
    opt.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default=t.precision)
    opt.add_argument("--num-workers", type=int, default=t.num_workers)
    opt.add_argument("--train-limit", type=int, default=t.train_limit)
    opt.add_argument("--val-limit", type=int, default=t.val_limit)
    opt.add_argument("--log-every", type=int, default=t.log_every)
    opt.add_argument("--eval-every", type=int, default=t.eval_every)
    opt.add_argument("--bleu-samples", type=int, default=t.bleu_samples)
    opt.add_argument("--device", default=t.device)
    opt.add_argument("--output-dir", type=Path, default=t.output_dir)
    opt.add_argument("--resume", type=Path, default=t.resume)
    opt.add_argument("--seed", type=int, default=t.seed)
    opt.add_argument(
        "--overfit",
        type=int,
        metavar="N",
        default=None,
        help="sanity check: train and validate on the same N train images, no regularisation",
    )
    return p


def configs_from_args(args: argparse.Namespace) -> tuple[DataConfig, TrainConfig]:
    """Turn parsed flags into DataConfig and TrainConfig."""
    overrides = {
        "dataset_id": args.dataset_id,
        "image_column": args.image_column,
        "caption_columns": tuple(args.caption_columns) if args.caption_columns else None,
        "group_column": args.group_column,
    }
    data_config = data_config_for(
        args.dataset,
        **{k: v for k, v in overrides.items() if v is not None},
        tokenizer_id=args.tokenizer_id,
        cache_dir=args.cache_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        image_size=args.image_size,
        max_tokens=args.max_tokens,
    )
    train_config = TrainConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        freeze_encoder=args.freeze_encoder,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        augment=not args.no_augment,
        precision=args.precision,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        log_every=args.log_every,
        eval_every=args.eval_every,
        bleu_samples=args.bleu_samples,
        device=args.device,
        resume=args.resume,
    )
    if args.overfit is not None:
        # "Can the model memorise a tiny set?" If not, something is broken.
        n = args.overfit
        data_config = dataclasses.replace(data_config, val_split=data_config.train_split)
        train_config = dataclasses.replace(
            train_config,
            train_limit=n,
            val_limit=n,
            batch_size=min(train_config.batch_size, n),  # drop_last would leave 0 batches
            label_smoothing=0.0,  # smoothing keeps the loss above 0 even when memorised
            augment=False,  # memorise exactly these images
            bleu_samples=min(train_config.bleu_samples, n),
        )
        args.dropout = 0.0
    return data_config, train_config


def model_sizes_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """The ModelConfig fields that come from the CLI."""
    return {
        "encoder": args.encoder,
        "pretrained": not args.from_scratch,
        "d_model": args.d_model,
        "attention_heads": args.heads,
        "decoder_layers": args.layers,
        "ffn_mult": args.ffn_mult,
        "dropout": args.dropout,
        "patch_size": args.patch_size,
        "encoder_layers": args.encoder_layers,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: set up (optional) multi-process training, then train.

    Returns None on purpose: console scripts pass the return value to sys.exit,
    and anything else would be printed and turned into exit code 1.
    """
    args = build_parser().parse_args(argv)
    data_config, train_config = configs_from_args(args)
    info = setup_distributed(train_config.device)  # no-op unless launched by torchrun

    # Only the main process prints INFO lines; the others only warnings.
    logging.basicConfig(
        level=logging.INFO if info.is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2"):  # hide Hugging Face HTTP logs
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        _build_and_train(args, data_config, train_config, info)
    finally:
        if info.enabled:
            dist.destroy_process_group()


def _build_and_train(
    args: argparse.Namespace, data_config: DataConfig, train_config: TrainConfig, info: DistInfo
) -> TrainState:
    """Build tokenizer, data and model from the CLI flags, then train."""
    # Heavy imports here so `import bia_captioner.train` stays light for tests.
    from bia_captioner.data import CaptionCollator, build_dataset
    from bia_captioner.model import ImageCaptioner, encoder_normalization
    from bia_captioner.tokenizer import build_tokenizer

    seed_everything(train_config.seed)  # before building the model: same init weights
    with local_main_first(info):  # one download per machine, the rest reuse the cache
        tokenizer = build_tokenizer(data_config.tokenizer_id, data_config.cache_dir)
        model_config = build_model_config(tokenizer, data_config, **model_sizes_from_args(args))
        # Images must be normalised the way the encoder was trained (saved in checkpoints).
        mean, std = encoder_normalization(model_config)
        data_config = dataclasses.replace(data_config, image_mean=mean, image_std=std)
        train_dataset = build_dataset(
            data_config, tokenizer, data_config.train_split,
            limit=train_config.train_limit, train=train_config.augment,
        )  # fmt: skip
        val_dataset = build_dataset(
            data_config, tokenizer, data_config.val_split, limit=train_config.val_limit
        )
        model = ImageCaptioner(model_config, image_size=data_config.image_size)

    collator = CaptionCollator(pad_token_id=tokenizer.pad_token_id)
    device = resolve_device(train_config.device)
    train_loader, val_loader = build_dataloaders(
        train_dataset, val_dataset, collator, train_config, device
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    logger.info(
        "model: %.1fM parameters (encoder %s, %s, %.1fM) | data %s | train %d images | "
        "val %d images",
        n_params / 1e6, model_config.encoder,
        "pretrained" if model_config.encoder_is_pretrained else "from scratch",
        n_encoder / 1e6, data_config.dataset_id, len(train_dataset), len(val_dataset),
    )  # fmt: skip

    return train(
        model, train_loader, val_loader, tokenizer, data_config, model_config, train_config
    )


if __name__ == "__main__":
    main()
