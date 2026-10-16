# Training guide

All training goes through one command, `cap-train` (same as
`python -m bia_captioner.train`). It runs on CPU, one GPU, several GPUs, or
several machines: only the way you **launch** it changes.

- [1. Install](#1-install)
- [2. Choose an encoder](#2-choose-an-encoder)
- [3. First: the overfit sanity check](#3-first-the-overfit-sanity-check)
- [4. One GPU](#4-one-gpu)
- [5. Several GPUs or machines](#5-several-gpus-or-machines)
- [6. Resume a run](#6-resume-a-run)
- [7. What gets saved](#7-what-gets-saved)
- [8. Reading the logs](#8-reading-the-logs)
- [9. All flags](#9-all-flags)
- [10. Tips](#10-tips)
- [11. How long does training take?](#11-how-long-does-training-take)

---

> **No GPU?** Use the self-contained notebook
> [`notebooks/train_colab_kaggle.ipynb`](../notebooks/train_colab_kaggle.ipynb):
> upload it to Kaggle or Colab, pick a GPU, *Run all*. Same code, same
> defaults, plus live training graphs.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install PyTorch **first**, for your hardware (see
<https://pytorch.org/get-started/locally/>):

```bash
# NVIDIA GPU (example: CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only (small download, enough for tests and the overfit check)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then the project itself:

```bash
pip install -e '.[dev]'
pytest            # should end with "N passed"
```

Flickr30k (~4.3 GB), the GPT-2 tokenizer and the pretrained encoder weights
are downloaded on the first run. Use `--cache-dir /big/disk/hf_cache` to put
the dataset elsewhere.

## 2. Choose an encoder

| `--encoder` | What it is | Typical command |
|---|---|---|
| `vit-b16` (default) | ViT-B/16 pretrained on ImageNet, fine-tuned with a small LR | `cap-train` |
| `resnet50` | ResNet-50 pretrained on ImageNet | `cap-train --encoder resnet50` |
| `bia-vit` | the BIA Vision Transformer, trained from scratch | `cap-train --encoder bia-vit --epochs 40` |

The encoder learning rate follows the encoder: **1e-5** for the pretrained
ones (they are already good; a large LR would wipe out what they learned on
ImageNet) and **the decoder LR** for BIA-ViT (it starts from random weights).
Override with `--encoder-lr`.

`--freeze-encoder` trains only the decoder. It is several times faster and uses
much less memory; quality is a bit lower. A good recipe on a small GPU is to
train frozen first, then fine-tune everything:

```bash
cap-train --freeze-encoder --epochs 8 --output-dir checkpoints/frozen
cap-train --epochs 15 --output-dir checkpoints/run    # full fine-tuning
```

(The second command starts from ImageNet again; it does not load the frozen
run. Use it when you have the GPU time.)

**About BIA-ViT.** A Vision Transformer trained from scratch on 29k photos will
caption noticeably worse than a pretrained encoder: ViTs have no built-in idea
of locality and learn it only from very large datasets. It is kept as a
learning / research option. If you use it, train longer (`--epochs 40` or
more) and keep the regularisation (dropout, label smoothing, augmentation).

## 3. First: the overfit sanity check

Before spending GPU hours, check the model can **memorise a few images**. If it
cannot, something is broken and a long run would be wasted.

```bash
cap-train --overfit 8 --epochs 60 --batch-size 8 --lr 1e-3 --warmup-steps 5 \
          --eval-every 20 --log-every 5 --output-dir checkpoints/overfit
```

`--overfit 8` trains *and* validates on the same 8 training images, with no
dropout, no label smoothing and no augmentation. Expected: the loss falls from
~10.8 (random guessing over 50k tokens) towards 0, BLEU goes to ~1, and the
logged `sample prediction` becomes the `sample target`. See
[testing.md](testing.md) for a real result.

## 4. One GPU

```bash
cap-train --output-dir checkpoints/run
```

The defaults are meant for one GPU: ViT-B/16, 15 epochs, batch 64.

- `--device auto` (default) picks CUDA when available.
- `--precision auto` (default) uses **bf16** on GPUs that support it (A100,
  H100, L4, RTX 30xx/40xx) and **fp16 + gradient scaling** on older ones (T4, V100).
- Out of memory? Lower `--batch-size` and raise `--grad-accum` by the same
  factor: the effective batch, and so the training dynamics, stay the same.

```bash
# same effective batch (64), half the memory
cap-train --batch-size 32 --grad-accum 2 --output-dir checkpoints/run
```

## 5. Several GPUs or machines

Launch the **same command** with `torchrun`. It starts one process per GPU;
the code detects this and uses `DistributedDataParallel` (DDP).

```bash
# all 4 GPUs of this machine
torchrun --standalone --nproc_per_node=4 -m bia_captioner.train --output-dir checkpoints/run

# 2 machines x 8 GPUs: run this on every machine (MASTER = address of machine 0)
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
    --rdzv_endpoint=MASTER:29500 --rdzv_id=cap-run-1 \
    -m bia_captioner.train --output-dir /shared/checkpoints/run
```

- each process gets its own GPU and its own slice of the data;
- **batch size is per GPU**: effective batch = `batch-size × grad-accum ×
  number of GPUs` (printed on the first log line). When you multiply it by k,
  multiplying `--lr` by about √k is a common starting point;
- with `--grad-accum`, gradients are only synchronised on the last
  micro-batch (`no_sync`), which saves communication;
- only process 0 logs, computes BLEU and writes checkpoints; the validation
  loss is summed over all processes;
- on each machine, local rank 0 downloads the dataset first and the others
  wait (`local_main_first`). Point `--cache-dir` at a shared disk to download
  only once.

The DDP path is covered by an automated 2-process test
(`tests/test_distributed.py`, gloo backend on CPU).

## 6. Resume a run

```bash
cap-train --resume checkpoints/run/last.pt --output-dir checkpoints/run
```

A checkpoint holds the model, optimizer, LR scheduler, gradient scaler, step
counter and random-number state, so training continues exactly where it
stopped (it even skips the batches already seen in the interrupted epoch).
**Pass the same flags as the original run**: they are read from the command
line; the configs saved in the checkpoint are a record only.

## 7. What gets saved

```
checkpoints/run/
  last.pt        latest state (written after every validation and at the end)
  best.pt        state with the lowest validation loss so far
  tokenizer/     the exact tokenizer used (with the added <|pad|> token)
```

Files are written to a temporary file first and then renamed, so a crash
during saving never leaves a broken checkpoint. `best.pt` is what you use for
[inference](inference.md). With ViT-B/16, each file is ~1.6 GB (weights +
AdamW state).

## 8. Reading the logs

```
train | epoch 3 | step 1400/6795 | loss 2.9120 | acc 0.452 | lr decoder 2.7e-04 encoder 9.1e-06 | grad 0.61 | 142 img/s
val | step 1500 | loss 2.6011 | ppl 13.48 | acc 0.471 | bleu4 0.2104
sample target:     A man in a red shirt is climbing a rock wall.
sample prediction: A man is climbing a rock wall.
```

- `loss`: cross-entropy per caption token, with label smoothing during
  training (so it never reaches 0). Starts near 10.8 (= ln of the vocabulary).
- `acc`: next-token accuracy given the true caption so far.
- `lr`: decoder and encoder learning rates (warmup, then cosine down to 0).
- `grad`: gradient norm before clipping (clipped to `--grad-clip`, default 1).
- `ppl`: perplexity = exp(validation loss).
- `bleu4`: corpus BLEU-4 of greedy captions on `--bleu-samples` validation
  images, against all 5 references. Use `cap-evaluate` (beam search, test
  split) for the number you report.
- `sample prediction`: the quickest way to see what the model has learned.

The numbers in this example are illustrative.

## 9. All flags

`cap-train --help` prints them all. Defaults come from
`src/bia_captioner/config.py`.

| Flag | Default | Meaning |
|------|---------|---------|
| **data** | | |
| `--dataset-id` | `AnyModal/flickr30k` | Hugging Face dataset |
| `--image-column` / `--caption-column` | `image` / `original_alt_text` | dataset columns (a caption cell may be one string or a list) |
| `--tokenizer-id` | `openai-community/gpt2` | base BPE vocabulary |
| `--cache-dir` | HF default | where downloads are stored |
| `--train-split` / `--val-split` | `train` / `validation` | dataset splits |
| `--image-size` | 224 | images are resized to this square (pretrained ViT-B/16 needs 224) |
| `--max-tokens` | 64 | max caption length incl. BOS/EOS |
| **model** | | |
| `--encoder` | `vit-b16` | `vit-b16`, `resnet50` or `bia-vit` |
| `--d-model` | 512 | decoder width (and BIA-ViT width) |
| `--heads` | 8 | attention heads (`d-model` must divide by it) |
| `--layers` | 6 | decoder layers |
| `--ffn-mult` | 4 | feed-forward size = ffn-mult × d-model |
| `--dropout` | 0.1 | dropout in the decoder (and BIA-ViT) |
| `--patch-size` | 16 | BIA-ViT patch size |
| `--encoder-layers` | 6 | BIA-ViT layers |
| `--no-pretrained-encoder` | off | random-init `vit-b16` / `resnet50` |
| **training** | | |
| `--epochs` | 15 | passes over the training images |
| `--batch-size` | 64 | images per step **per process/GPU** |
| `--grad-accum` | 1 | micro-batches per optimizer step |
| `--lr` | 3e-4 | peak decoder learning rate |
| `--encoder-lr` | auto | 1e-5 for pretrained encoders, `--lr` for BIA-ViT |
| `--freeze-encoder` | off | train only the decoder |
| `--weight-decay` | 0.05 | AdamW decay (not on biases, norms, embeddings) |
| `--warmup-steps` | 500 | linear LR warmup, in optimizer steps |
| `--grad-clip` | 1.0 | max gradient norm (0 = off) |
| `--label-smoothing` | 0.1 | label smoothing for the training loss |
| `--precision` | `auto` | `auto`, `fp32`, `bf16`, `fp16` |
| `--num-workers` | 4 | data-loading processes per GPU |
| `--train-limit` / `--val-limit` | all | use only the first N images |
| `--log-every` | 50 | optimizer steps between log lines |
| `--eval-every` | 500 | optimizer steps between validations (+ one at the end) |
| `--bleu-samples` | 500 | validation images captioned for BLEU (0 = off) |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--output-dir` | `checkpoints/run` | where checkpoints go |
| `--resume` | none | checkpoint to continue from |
| `--seed` | 42 | random seed |
| `--overfit N` | off | sanity check: train + validate on the same N images |

## 10. Tips

- **Always run the overfit check** after changing the model or data code.
- **Watch the decoded sample and BLEU**, not only the loss.
- **Model selection** uses the validation loss; validation BLEU usually peaks
  around the same time. Report BLEU on the *test* split with `cap-evaluate`.
- **Data loading too slow?** (GPU utilisation below ~90%) raise
  `--num-workers`: decoding JPEGs is the main CPU cost.
- **Reproducibility**: same seed + same hardware gives nearly the same run.
  cuDNN is left in its fast, slightly non-deterministic mode on purpose.

## 11. How long does training take?

One epoch = 29 000 images = 453 optimizer steps at batch 64.

| Hardware | `vit-b16` fine-tuned | `vit-b16 --freeze-encoder` | `resnet50` |
|---|---|---|---|
| A100 / H100 | ~1–2 min / epoch | < 1 min / epoch | ~1 min / epoch |
| RTX 3090 / 4090 | ~3–4 min / epoch | ~1–2 min / epoch | ~2 min / epoch |
| T4 (Kaggle / Colab) | ~6–8 min / epoch (`--batch-size 32 --grad-accum 2`) | ~2–3 min / epoch | ~4 min / epoch |
| Laptop CPU | hours per epoch | ~1 h / epoch | hours per epoch |

> These are **estimates** from the encoder's compute per image (ViT-B/16 is
> ~17.6 GFLOPs forward at 224 px; the decoder is small next to it), assuming
> ~30% of the GPU's peak. They have not been timed on these GPUs from this
> repo yet. Read `img/s` from your own log: minutes per epoch ≈ 29 000 /
> img/s / 60.

The legacy code needed one encoder pass per *caption* (5 per image) and
loaded images in the main process; the current pipeline needs one pass per
image and loads data in parallel, which is where most of the speed-up comes
from.
