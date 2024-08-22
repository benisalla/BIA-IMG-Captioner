import gc
import time
from img_cap_engine.model.BIAImgCaptioner import BIAImgCaptioner
import torch
import torch.nn as nn
import torch.optim
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import math
from typing import Optional, Dict, Tuple, Union


def get_lr(step: int, n_emb: int, warmup_steps: int) -> float:
    """
    Compute the learning rate based on the current step, the embedding size, and the number of warmup steps.

    Parameters:
    -----------
    step : int
        The current training step.
    n_emb : int
        The dimensionality of the embeddings.
    warmup_steps : int
        The number of warmup steps.
    """

    sf = 2.0 * math.pow(n_emb, -0.5)  # Scale based on n_emb
    dp = math.pow(step, -0.5)  # LR during decay
    wp = step * math.pow(warmup_steps, -1.5)  # LR during warmup
    lr = sf * min(dp, wp)  # Final LR

    return lr


def save_checkpoint(
    epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, file_path: str
):
    """
    Save the model and optimizer state to a file.

    Parameters:
    -----------
    epoch : int
        The current epoch number.
    model : nn.Module
        The model to save.
    optimizer : torch.optim.Optimizer
        The optimizer to save.
    file_path : str
        The path to the file where the checkpoint will be saved.
    """

    optimizer_params = {
        "lr": optimizer.param_groups[0]["lr"],
        "betas": optimizer.param_groups[0]["betas"],
        "eps": optimizer.param_groups[0]["eps"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
    }

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_params": optimizer_params,
        "init_args": model.get_init_args(),
    }

    try:
        torch.save(state, file_path)
        print(f"Checkpoint saved successfully at {file_path}")
    except Exception as e:
        print(f"Failed to save checkpoint at {file_path}: {e}")


def load_checkpoint(
    file_path: str, device: str = "cpu"
) -> Tuple[nn.Module, torch.optim.Optimizer, int]:
    """
    Load a checkpoint from a file.

    Parameters:
    -----------
    file_path : str
        The path to the checkpoint file.
    device : str, optional
        The device to load the model onto (default is 'cpu').
    """
    checkpoint = torch.load(file_path, map_location=device)

    model = BIAImgCaptioner(**checkpoint["init_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=checkpoint["optimizer_params"]["lr"],
        betas=checkpoint["optimizer_params"]["betas"],
        eps=checkpoint["optimizer_params"]["eps"],
        weight_decay=checkpoint["optimizer_params"]["weight_decay"],
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    return model, optimizer, epoch


def get_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the accuracy of predictions.

    Parameters:
    -----------
    logits : torch.Tensor
        The predicted logits of shape (batch_size, sequence_length, num_classes).
    targets : torch.Tensor
        The ground truth targets of shape (batch_size, sequence_length).
    """

    targets = targets[:, 1:]
    logits = logits[:, :-1, :]
    predicted = logits.argmax(-1)
    correct_predictions = (predicted == targets).float()
    accuracy = correct_predictions.mean()
    return accuracy.item()


def update_lr(optimizer: torch.optim.Optimizer, new_lr: float):
    """
    Update the learning rate for the optimizer.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate is to be updated.
    new_lr : float
        The new learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


class AverageMeter:
    """
    A class for tracking the average value of a metric over time.

    Attributes:
    -----------
    value : float
        The most recent value.
    avg : float
        The average value.
    sum : float
        The sum of all values.
    count : int
        The number of values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def wrap_text(text: str, line_length: int) -> str:
    """
    Wrap text to a specified line length.

    Parameters:
    -----------
    text : str
        The text to wrap.
    line_length : int
        The maximum length of each line.
    """

    words = text.split(" ")
    wrapped_text = ""
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= line_length:
            current_line += word + " "
        else:
            wrapped_text += current_line.strip() + "\n"
            current_line = word + " "

    wrapped_text += current_line.strip()
    return wrapped_text


def imshow(
    img: torch.Tensor,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    line_length: int = 30,
):
    """
    Display an image with an optional title.

    Parameters:
    -----------
    img : torch.Tensor
        The image to display.
    title : str, optional
        The title of the image (default is None).
    ax : plt.Axes, optional
        The axes on which to plot the image (default is None).
    line_length : int, optional
        The maximum length of the title line (default is 30).
    """

    if ax is None:
        _, ax = plt.subplots()
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    if title is not None:
        wrapped_title = wrap_text(title, line_length)
        ax.set_title(wrapped_title, fontdict={"fontsize": 6, "color": "black"})
    ax.axis("off")


def show_batch_images(images: torch.Tensor, captions: torch.Tensor, tokenizer) -> None:
    """
    Display a batch of images with their corresponding captions.

    Parameters:
    -----------
    images : torch.Tensor
        The batch of images to display.
    captions : torch.Tensor
        The captions corresponding to the images.
    tokenizer
        The tokenizer used to decode the captions.
    """

    batch_size = images.size(0)
    n_rows = int(math.sqrt(batch_size))
    n_cols = math.ceil(batch_size / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(batch_size):
        ax = axes[i]
        title = tokenizer.decode(captions[i], skip_special_tokens=True)
        imshow(images[i], title=title, ax=ax)

    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def live_plot_dual(
    data_dict: Dict[str, Union[list, np.ndarray]],
    figsize: Tuple[int, int] = (12, 10),
    title: str = "",
) -> None:
    """
    Plot training and validation loss and accuracy in real-time.

    Parameters:
    -----------
    data_dict : dict
        A dictionary containing lists or arrays of training and validation losses and accuracies.
    figsize : Tuple[int, int], optional
        The size of the figure (default is (12, 10)).
    title : str, optional
        The title of the plot (default is an empty string).
    """

    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plotting Losses
    if "train_loss" in data_dict and data_dict["train_loss"]:
        ax1.plot(data_dict["train_loss"], "r-", label="Train Loss")
        ax1.annotate(
            f"{data_dict['train_loss'][-1]:.4f}",
            xy=(len(data_dict["train_loss"]) - 1, data_dict["train_loss"][-1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

    if "val_loss" in data_dict and data_dict["val_loss"]:
        ax1.plot(data_dict["val_loss"], "b-", label="Val Loss")
        ax1.annotate(
            f"{data_dict['val_loss'][-1]:.4f}",
            xy=(len(data_dict["val_loss"]) - 1, data_dict["val_loss"][-1]),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
        )

    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plotting Accuracy
    if "train_acc" in data_dict and data_dict["train_acc"]:
        ax2.plot(data_dict["train_acc"], "r-", label="Train Accuracy")
        ax2.annotate(
            f"{data_dict['train_acc'][-1]:.2f}%",
            xy=(len(data_dict["train_acc"]) - 1, data_dict["train_acc"][-1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

    if "val_acc" in data_dict and data_dict["val_acc"]:
        ax2.plot(data_dict["val_acc"], "b-", label="Val Accuracy")
        ax2.annotate(
            f"{data_dict['val_acc'][-1]:.2f}%",
            xy=(len(data_dict["val_acc"]) - 1, data_dict["val_acc"][-1]),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
        )

    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def validator(
    val_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Validate the model on a validation set.

    Parameters:
    -----------
    val_loader : torch.utils.data.DataLoader
        The data loader for the validation set.
    model : nn.Module
        The model to evaluate.
    criterion : nn.Module
        The loss function to use.
    device : str
        The device to perform computation on (e.g., "cpu" or "cuda").
    """

    model.eval()

    with torch.no_grad():
        losses = AverageMeter()
        t_acc = 0
        step = 0

        for img, idx, cap_len in val_loader:
            img, idx, cap_len = img.to(device), idx.to(device), cap_len.to(device)

            preds = model(img=img, idx=idx)
            loss = criterion(x=preds, y=idx[:, 1:], lens=cap_len - 1)
            losses.update(loss.item(), (cap_len - 1).sum().item())

            b_acc = get_accuracy(logits=preds, targets=idx)
            t_acc += b_acc
            step += 1

        avg_acc = t_acc / step
        val_loss = losses.avg
        return val_loss, avg_acc * 100


def trainer(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    device: str,
    batch_in_step: int,
    epochs: int,
    warmup_steps: int,
    n_emb: int,
    max_iters: Optional[int] = None,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float, Optional[float]]:
    """
    Train the model for one epoch.

    Parameters:
    -----------
    train_loader : torch.utils.data.DataLoader
        The data loader for the training set.
    model : nn.Module
        The model to train.
    criterion : nn.Module
        The loss function to use.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    epoch : int
        The current epoch number.
    step : int
        The current step number.
    device : str
        The device to perform computation on (e.g., "cpu" or "cuda").
    batch_in_step : int
        The number of steps per batch.
    epochs : int
        The total number of epochs.
    warmup_steps : int
        The number of warmup steps for learning rate scheduling.
    n_emb : int
        The dimensionality of the embeddings.
    max_iters : int, optional
        The maximum number of iterations (default is None, meaning no limit).
    grad_clip : float, optional
        The gradient clipping value (default is None, meaning no clipping).
    """

    model.train()
    data_time = AverageMeter()
    step_time = AverageMeter()
    losses = AverageMeter()
    t_acc = 0
    acc_step = 0
    curr_lr = None

    start_data_time = time.time()
    start_step_time = time.time()

    for i, (img, idx, cap_len) in enumerate(train_loader):
        if max_iters is not None and i >= max_iters:
            break

        img, idx, cap_len = img.to(device), idx.to(device), cap_len.to(device)

        data_time.update(time.time() - start_data_time)

        preds = model(img=img, idx=idx)
        loss = criterion(x=preds, y=idx[:, 1:], lens=cap_len - 1)
        losses.update(loss.item(), idx.size(0))

        t_acc += get_accuracy(logits=preds, targets=idx)
        acc_step += 1

        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if (i + 1) % batch_in_step == 0:
            optimizer.step()
            step += 1
            curr_lr = get_lr(step=step, n_emb=n_emb, warmup_steps=warmup_steps)
            update_lr(optimizer, new_lr=curr_lr)

            step_time.update(time.time() - start_step_time)

            start_step_time = time.time()

        start_data_time = time.time()

    avg_acc = t_acc / acc_step
    train_loss = losses.avg
    return train_loss, avg_acc * 100, curr_lr
