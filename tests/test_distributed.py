"""Multi-process (DDP) training test on CPU.

Starts 2 processes with the gloo backend, exactly like torchrun would on 2 GPUs,
and checks they stay in sync: same step count and identical final weights.
"""

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_train import (  # reuse the tiny fakes from the single-process tests
    IMAGE_SIZE,
    PAD,
    FakeCollator,
    FakeDataset,
    FakeTokenizer,
    TinyModel,
    model_config,
    train_config,
)

from image_captioner.config import DataConfig
from image_captioner.train import LAST_CHECKPOINT, build_dataloaders, get_dist_info, train

WORLD_SIZE = 2


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank: int, port: int, tmp_path: Path) -> None:
    """What torchrun would run in each process."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE),
    )  # fmt: skip
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        assert get_dist_info().world_size == WORLD_SIZE
        torch.manual_seed(rank)  # different init per rank: DDP must broadcast rank 0's
        model = TinyModel(model_config())
        config = train_config(tmp_path, epochs=5, grad_accum_steps=2, batch_size=2)
        dataset = FakeDataset(n=16)
        train_loader, val_loader = build_dataloaders(
            dataset, dataset, FakeCollator(PAD), config, torch.device("cpu")
        )
        # 16 samples / 2 processes / batch 2 = 4 batches per process.
        assert len(train_loader) == 4
        state = train(
            model, train_loader, val_loader, FakeTokenizer(),
            DataConfig(image_size=IMAGE_SIZE), model_config(), config,
        )  # fmt: skip
        torch.save({"step": state.step, "model": model.state_dict()}, tmp_path / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_process_training_stays_in_sync(tmp_path: Path) -> None:
    mp.spawn(worker, args=(free_port(), tmp_path), nprocs=WORLD_SIZE, join=True)

    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=True)
    # 4 batches / 2 accumulation = 2 optimizer steps per epoch, 5 epochs.
    assert rank0["step"] == rank1["step"] == 10
    for name, weight in rank0["model"].items():
        torch.testing.assert_close(weight, rank1["model"][name], msg=f"{name} differs")
    # Only rank 0 writes checkpoints.
    assert (tmp_path / "run" / LAST_CHECKPOINT).exists()
