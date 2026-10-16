# Testing guide

Four levels of checks, from seconds to an hour:

1. **Unit tests and sanity checks** (`pytest`): every module, CPU only, no network.
2. **Synthetic shapes**: does the architecture learn to read images? (laptop CPU)
3. **Overfit check on real photos**: can the model memorise 8 Flickr30k images?
4. **Launch checks**: the real CLIs (`cap-train`, `cap-generate`, `cap-evaluate`,
   `torchrun`), the notebook and the web app.

## 1. Unit tests and sanity checks

```bash
pytest                       # everything (~1 min on a laptop CPU)
pytest tests/test_model.py   # one file
pytest -k beam -v            # tests whose name contains "beam"
ruff check src tests         # lint
ruff format --check src tests
```

No GPU and no internet needed: tests use tiny fake datasets, a tiny tokenizer,
small models and randomly initialised encoders.

### The neural-network sanity checks

The usual checklist for "is this network wired correctly?", each one an
automated test:

| Check | Why it matters | Test |
|---|---|---|
| Initial loss ≈ ln(vocabulary) | a random model must be a uniform guesser; much higher means bad init / scaling | `test_initial_loss_is_close_to_random_guessing` |
| Overfit one batch | if it cannot memorise 2 captions, the loss, masks or optimiser are broken | `test_can_overfit_one_batch` |
| Every trainable parameter gets a gradient (each encoder family) | dead layers waste compute and crash DDP | `test_every_trainable_parameter_receives_a_gradient` |
| No look-ahead | future tokens must not change past predictions (causal mask) | `test_causal_future_tokens_do_not_change_past_logits` |
| Padding is invisible | pads must not change real tokens' outputs or the loss | `test_padding_does_not_change_real_token_logits`, `test_loss_is_finite_and_ignores_minus_100` |
| Samples are independent | a reshape mixing batch and time axes, or BatchNorm in train mode, leaks across samples | `test_samples_in_a_batch_do_not_influence_each_other` |
| The image reaches the output | otherwise it is a language model, not a captioner | `test_the_image_changes_the_prediction`; on data: `cap-evaluate --shuffle-images` |
| Eval mode is deterministic | dropout must be off at inference | `test_eval_mode_is_deterministic_and_train_mode_uses_dropout` |
| Mixed precision is stable | bf16 loss finite and close to fp32 | `test_bf16_autocast_gives_finite_loss_close_to_fp32` |
| Gradient accumulation = one big batch | the legacy bug: same weights after 2×4 accumulated as after 1×8 | `test_gradient_accumulation_matches_one_big_batch` |
| Same seed = same run | needed to compare experiments | `test_same_seed_gives_the_same_run` |
| LR schedule | warmup and cosine values | `test_lr_lambda_*` |
| Resume | a restarted run continues the step count and data order | `test_resume_continues_step_count` |
| Multi-process | 2 DDP processes end with identical weights | `test_two_process_training_stays_in_sync` |

### What each file proves

| File | What it proves |
|------|----------------|
| `test_config.py` | invalid settings are rejected early (pad = EOS, unknown encoder, bad precision); every encoder name and dataset preset is valid; BIA-ViT is never "pretrained"; the encoder LR is 1e-5 when pretrained and the decoder LR when from scratch |
| `test_data.py` | images become normalised `[3,H,W]` tensors (with the encoder's own mean / std); captions become `<bos> … <eos>`; truncation drops EOS; validation uses the first caption, training samples all of them; captions from one list column, several columns, or **one per row grouped by image id (COCO)**; the synthetic shapes are deterministic, differ between splits, and every color named in a caption is in the picture; the collator pads with -100 labels and does not shift them |
| `test_metrics.py` | BLEU: perfect = 1, no overlap = 0, clipping, brevity penalty, a hand-computed BLEU-4 (also matches NLTK); exact match ignores case / punctuation |
| `test_model.py` | the sanity checks above; shapes for **each encoder family** (BIA-ViT, torchvision ViT, torchvision ResNet, Hugging Face ViT); **`pretrained` decides which weights are loaded** (and a checkpoint load never downloads); pretrained torchvision ViTs reject a wrong image size; the Hugging Face encoder interpolates positions, finds and freezes its unused pooler; greedy / top-k / top-p generation; **beam search finds a sequence greedy misses**, equals greedy with beam 1; weight-decay groups; a **frozen encoder** gets no gradient and stays in eval mode |
| `test_train.py` | LR schedule, precision / device selection, CLI flags and dataset presets reach the configs, `--from-scratch`, encoder LR defaults; a training run lowers the loss and writes `last.pt` / `best.pt` / `tokenizer/`; resume; gradient accumulation; reproducibility; the real model + collator train end to end, with and without a frozen encoder; BLEU ~1 on a memorised set; the callback receives train / validation metrics |
| `test_distributed.py` | **DDP**: 2 processes (gloo on CPU) start from *different* weights, train with gradient accumulation and end **identical**; only rank 0 writes checkpoints |
| `test_generate.py` | a checkpoint reloads into exactly the same model; `cap-generate` with beam search, greedy and sampling; `cap-evaluate` writes scores and predictions, and runs with `--shuffle-images` |

## 2. Synthetic shapes: does the architecture read images?

Synthetic images of colored shapes whose caption is fully determined by the
picture (see [datasets.md](datasets.md#1-shapes-the-simplest-proof-the-architecture-works)).
The question: does the model caption **unseen** images from their pixels, or
only produce plausible-sounding captions?

```bash
cap-train --dataset shapes --encoder bia-vit --image-size 64 --patch-size 8 \
          --d-model 128 --heads 4 --layers 2 --encoder-layers 3 \
          --epochs 8 --lr 1e-3 --warmup-steps 100 --eval-every 250 --bleu-samples 200 \
          --output-dir checkpoints/shapes
cap-evaluate --checkpoint checkpoints/shapes/best.pt --split test --beam-size 1
cap-evaluate --checkpoint checkpoints/shapes/best.pt --split test --beam-size 1 --shuffle-images
```

Real run on a laptop CPU: **BIA-ViT trained from scratch** (7.6M parameters
in total, 0.6M in the encoder), stopped after 250 optimizer steps (2 of the 8
planned epochs, ~35 minutes; the CPU was shared with other jobs):

| step | train loss | train token acc |
|-----:|-----------:|----------------:|
| 25   | 9.97       | 0.31 |
| 100  | 2.50       | 0.77 |
| 175  | 1.66       | 0.87 |

(Training loss includes label smoothing 0.1, so it cannot reach 0.)
Validation at step 250: loss 0.27, token accuracy 0.913, BLEU-4 0.67.

Test split (500 images never seen in training), greedy decoding:

| | BLEU-1 | BLEU-4 | exact caption |
|---|---:|---:|---:|
| real images | 84.1 | **67.3** | **43.2%** |
| images shuffled (`--shuffle-images`, the control) | 46.1 | 25.0 | 0.6% |

The control is what proves the architecture works: given the *wrong* image the
same model drops from 43% exact captions to 0.6%, so its captions come from
the pixels, not from a memorised language pattern. The remaining BLEU of the
control is the language prior (every caption starts "a small/big ... on the
left and ...").

After 2 epochs colors, sizes, counts and left / right are mostly right; most
errors are the shape word (circle / square / triangle), the hardest attribute
at 64 px with 8 px patches:

```
target:     a small green square on the left and a small blue circle on the right
prediction: a small green triangle on the left and a small blue circle on the right
```

The full 8-epoch schedule (or a GPU) should push exact match much higher; that
run was not completed here.

## 3. Overfit sanity check (real photos)

The single most useful test of a training pipeline: if the model cannot
memorise a handful of images, something is broken (labels shifted twice,
masks wrong, encoder disconnected, LR far off, ...).

```bash
cap-train --overfit 8 --encoder vit_b_16 --epochs 60 --batch-size 8 --lr 1e-3 \
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

## 4. Launch checks

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
