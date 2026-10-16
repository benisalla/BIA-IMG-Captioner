# Testing guide

Three levels of checks, from seconds to minutes:

1. **Unit tests** (`pytest`): every module, CPU only, no network.
2. **Overfit sanity check**: real Flickr30k images, can the model memorise 8 of them?
3. **Launch checks**: the real CLIs (`cap-train`, `cap-generate`, `cap-evaluate`,
   `torchrun`) and the notebook.

## 1. Unit tests

```bash
pytest                       # everything (~1 min on a laptop CPU)
pytest tests/test_model.py   # one file
pytest -k beam -v            # tests whose name contains "beam"
ruff check src tests         # lint
ruff format --check src tests
```

No GPU and no internet needed: tests use tiny fake datasets, a tiny tokenizer,
small models and randomly initialised encoders.

| File | What it proves |
|------|----------------|
| `test_config.py` | invalid settings are rejected early (pad token = EOS, unknown encoder, bad precision); the encoder LR defaults to 1e-5 for pretrained encoders and to the decoder LR for BIA-ViT |
| `test_data.py` | images become normalised `[3,H,W]` tensors; captions become `<bos> … <eos>`; truncation drops EOS; validation always uses the first caption while training samples all of them; references are cleaned; the collator pads with -100 labels and does not shift them |
| `test_metrics.py` | BLEU: perfect match = 1, no overlap = 0, n-gram clipping, brevity penalty, a hand-computed BLEU-4 (the implementation also matches NLTK's `corpus_bleu`) |
| `test_model.py` | output shapes for **all three encoders**; the old 225 px / 16 px setting is rejected; initial loss ≈ ln(vocab); loss ignores padding; **causality**; padding cannot change real tokens; the image reaches the logits; `generate()` (greedy, top-k, top-p) starts with BOS, never emits BOS/pad, stops at EOS; **beam search finds a sequence greedy misses**, equals greedy with beam 1; top-p keeps the smallest nucleus; weight-decay groups (incl. ViT position / class tokens); a **frozen encoder** gets no gradient and stays in eval mode; the model overfits one batch and regenerates both captions |
| `test_train.py` | warmup + cosine LR; precision/device selection; CLI flags reach the configs; a training run lowers the loss and writes `last.pt`/`best.pt`/`tokenizer/`; **resume** continues the step count; **gradient accumulation gives exactly the same weights as one big batch** (the legacy bug); the real model + collator train end to end, with and without a frozen encoder; BLEU is ~1 on a memorised set; the callback receives train and validation metrics incl. BLEU |
| `test_distributed.py` | **DDP**: 2 processes (gloo backend on CPU) start from *different* random weights, train with gradient accumulation and end with **identical** weights and step counts; only rank 0 writes checkpoints |
| `test_generate.py` | a checkpoint reloads into exactly the same model; `cap-generate` captions every image with beam search, greedy and sampling and writes JSON without special tokens; `cap-evaluate` scores a split and saves predictions with their references |

## 2. Overfit sanity check (real data)

The single most useful test of a training pipeline: if the model cannot
memorise a handful of images, something is broken (labels shifted twice,
masks wrong, encoder disconnected, LR far off, ...).

```bash
cap-train --overfit 8 --encoder vit-b16 --epochs 60 --batch-size 8 --lr 1e-3 \
          --warmup-steps 5 --eval-every 20 --log-every 5 --output-dir checkpoints/overfit
```

Result on a laptop CPU (no GPU), real Flickr30k images, pretrained ViT-B/16
fine-tuned (137.3M parameters), ~16 minutes:

| step | train loss | val loss | val token acc | val BLEU-4 |
|-----:|-----------:|---------:|--------------:|-----------:|
| 5    | 9.25       |          |               |            |
| 20   | 1.81       | 1.20     | 0.796         | 0.27       |
| 40   | 0.044      | 0.028    | 1.000         | **1.00**   |
| 60   | 0.013      | **0.012** (perplexity 1.01) | 1.000 | 1.00 |

```
sample target:     A group of people stand in the back of a truck filled with cotton.
sample prediction: A group of people stand in the back of a truck filled with cotton.
```

Reloading `best.pt` in a fresh process, `cap-generate` (beam size 3)
reproduced the captions of the training images word for word, and
`cap-evaluate` on those 8 images gave BLEU-1..4 = 100.

What this does **not** tell you: how well the model captions *new* photos.
That needs a full GPU run and the test-split BLEU (`cap-evaluate`).

## 3. Launch checks

```bash
# training CLI, exit code must be 0 (the small BIA-ViT is fast on CPU)
cap-train --overfit 8 --encoder bia-vit --epochs 3 --batch-size 8 \
          --d-model 128 --heads 4 --layers 2 --encoder-layers 2 --output-dir /tmp/cap-check ; echo $?

# 2 processes on one machine (CPU uses the gloo backend, GPUs use NCCL)
torchrun --standalone --nproc_per_node=2 -m bia_captioner.train \
    --overfit 16 --encoder bia-vit --batch-size 4 --grad-accum 2 \
    --d-model 128 --heads 4 --layers 2 --encoder-layers 2 --epochs 6 --eval-every 3 \
    --output-dir /tmp/cap-ddp

# resume it for 2 more epochs: the log must say "resumed ... at step 6"
torchrun --standalone --nproc_per_node=2 -m bia_captioner.train \
    --overfit 16 --encoder bia-vit --batch-size 4 --grad-accum 2 \
    --d-model 128 --heads 4 --layers 2 --encoder-layers 2 --epochs 8 --eval-every 3 \
    --output-dir /tmp/cap-ddp --resume /tmp/cap-ddp/last.pt

# inference and evaluation
cap-generate --checkpoint checkpoints/overfit/best.pt --image photo.jpg
cap-evaluate --checkpoint checkpoints/overfit/best.pt --split train --limit 8
```

What to check in the output:

- the first line prints the **effective batch** and the encoder, e.g.
  `effective batch 16 = 4 per process x 2 accum x 2 processes ... encoder=bia-vit`;
- only **one** process prints INFO lines (rank 0);
- the loss goes down; the run ends with `done | best val loss ...`;
- `last.pt`, `best.pt` and `tokenizer/` exist in the output directory.

**The notebook** was executed end to end with `nbclient` in `SMOKE_TEST`
mode (ResNet-50, 64 images, small decoder) on real Flickr30k images: every
cell ran, including training with the live dashboard, beam-search captions,
test BLEU and the final zip.

## Not tested yet

- Real GPUs (CUDA, bf16/fp16, NCCL, fused AdamW) and a real multi-machine
  cluster. The code paths are standard PyTorch, but do a short run (or the
  notebook with `SMOKE_TEST = True`) on the GPU before a long job.
- Caption quality on unseen photos after a full training run.
