# Legacy code (kept for reference)

`img_cap_engine/` is the first version of this project, unchanged. The
maintained code is in [`src/bia_captioner`](../src/bia_captioner). This page
lists what was wrong with the legacy version and where each problem is fixed,
because together these problems explain why it trained badly.

## Why training did not work

The biggest problems, in order of impact:

1. **The ViT encoder was trained from scratch on 29k images.** Vision
   Transformers have no built-in image prior: they need millions of images
   (ImageNet-21k, JFT) to learn useful features. On Flickr30k alone the
   encoder learns little, so the decoder falls back to generic captions
   ("a man in a blue shirt ..."). **Fix:** `--encoder vit-b16` / `resnet50`
   use ImageNet-pretrained encoders with a small learning rate; the BIA-ViT is
   still available as `--encoder bia-vit`.
2. **Gradient accumulation threw away gradients.** `trainer()` called
   `optimizer.zero_grad()` before *every* backward pass but `optimizer.step()`
   only every `batch_in_step` (25) batches, so each update used the gradient of
   a single micro-batch and 24 out of 25 batches were wasted. **Fix:** zero
   once per optimizer step and divide the loss by the number of micro-batches
   (`train.py`, tested by `test_gradient_accumulation_matches_one_big_batch`).
3. **Almost no optimizer steps happened.** `max_iters = 20` stopped each epoch
   after 20 batches while an update needed 25, so with that config the
   optimizer *never* stepped. And with the Noam schedule's 8 000 warmup steps
   at one step per 25 batches (~200 steps per epoch), the learning rate would
   still be warming up after 40 epochs. **Fix:** full epochs, 500 warmup steps
   then cosine decay, separate encoder / decoder learning rates.
4. **The model could not be built.** `ResMHAtten.__init__` did
   `self.ln = ResMHAtten(n_emb)` (itself, with missing arguments) instead of a
   LayerNorm. **Fix:** `model.MultiHeadAttention` + `nn.LayerNorm`.
5. **Data loading was single-process and sequential.** The custom
   `DataLoader` class opened and decoded each JPEG in the training loop with no
   worker processes, so the GPU was idle most of the time, and validation used
   batches of one image. Each caption was also its own sample, so every image
   went through the encoder 5 times per epoch. **Fix:** a `torch.utils.data`
   `Dataset` with worker processes, pinned memory and dynamic padding; one
   item per image with a random caption each time, so each epoch encodes each
   image once.

## All issues found

### Model

| Legacy | Problem | Fix |
|---|---|---|
| `ResMHAtten`: `self.ln = ResMHAtten(n_emb)` | recursive construction: crashes | real LayerNorm in each block |
| `ResMHAtten`: `is_self = torch.equal(kv, q)` | guesses self- vs cross-attention by comparing whole tensors on every call: slow, and wrong if they are ever equal | explicit `context=` / `causal=` arguments |
| `ResMHAtten`: cross-attention `kv` not normalised | the encoder output went into cross-attention with an uncontrolled scale | encoder memory → projection + position → `LayerNorm` |
| `Encoder` | pre-norm blocks with no final LayerNorm: the output scale grows with depth | final `LayerNorm` in `BIAViT` |
| `Encoder`: init of `c_proj.weight` | no parameter has that name, so the scaled residual init never ran | scaled init on `out_proj` / `fc2` |
| `BIAImgCaptioner.init_weights` | Xavier init overwrote the N(0, 0.02) init of the decoder, including the tied embedding | one consistent GPT-2 style init |
| `Decoder`: `pte(idx) * sqrt(n_emb)` with tied output weights | scaling ×22.6 of an embedding shared with the output layer | no scaling (GPT-2 style), pad row initialised to zero |
| `BIAConv2DEmbedding.cls_token` | created, initialised, never used | removed |
| `im_size = 225`, `p_size = 16` | 225 is not a multiple of 16: the last pixel row/column is silently dropped | 224; `BIAViT` rejects sizes that are not a multiple of the patch |
| hand-written `bmm` attention | slow, materialises the full attention matrix | `F.scaled_dot_product_attention` (Flash / memory-efficient kernels) |
| `n_block = 10` for *both* encoder and decoder, `max_seq_len = 1024` | a large model for 29k images; 1024 positions for captions that average 16 tokens | separate `encoder_layers` / `decoder_layers`, `max_tokens = 64` (fits 99.96% of captions) |
| `d_rate = 0.0` | no regularisation on a small dataset: overfits | dropout 0.1, label smoothing 0.1, weight decay 0.05, mild augmentation |

### Generation (`describe`)

| Legacy | Problem | Fix |
|---|---|---|
| `scores = flogits / max(temperature + 1.0, ...)` | divides by T+1, so temperature 0 actually means 1 | `generate(temperature=0)` is true greedy |
| top-k / top-p filtering inside beam search | mixes sampling filters into a deterministic search and corrupts the beam scores | separate `beam_search()` (log-probs + GNMT length penalty) and `generate()` (greedy / top-k / top-p sampling) |
| beam search could pick `<PAD>` / `<SOS>` | invalid tokens in captions | special tokens are masked during decoding |
| `self = self.to(device)` with a string device | fragile | device read from the parameters |

### Training and evaluation

| Legacy | Problem | Fix |
|---|---|---|
| `zero_grad()` every batch | see "Why training did not work" #2 | correct accumulation |
| `clip_grad_norm_` on each micro-batch | clipped partial gradients | clipped once per optimizer step, after unscaling |
| loss not divided by accumulation steps | update size depends on `batch_in_step` | `loss / accum` |
| `get_accuracy` | counted padding positions as correct/incorrect | token accuracy over real tokens only |
| no mixed precision | 2-3× slower on modern GPUs | bf16 / fp16 autocast + GradScaler |
| Adam with `weight_decay` (L2) on every parameter | decays LayerNorms, biases, embeddings | AdamW, decay only on weight matrices |
| `train.py` / `finetune.py` | empty files | `cap-train` CLI, resume, DDP / multi-GPU |
| no caption metric | loss alone says little about caption quality | corpus BLEU-1..4 during validation and with `cap-evaluate` |
| checkpoints with `torch.load` (unsafe pickle) | can execute code from a file | `weights_only=True`, atomic writes |

### Data

| Legacy | Problem | Fix |
|---|---|---|
| `<SOS>`, `<EOS>`, `<PAD>` added as new tokens to GPT-2 | 3 untrained embeddings; `"<SOS>" + cap` relies on the BPE splitting correctly | GPT-2's own `<\|endoftext\|>` for BOS/EOS, one dedicated `<\|pad\|>` |
| every caption tokenised twice (once for lengths, once per batch) | wasted CPU | tokenised once per sample |
| Kaggle-only paths (`/kaggle/input/...`, pre-made `train.json`) | not reproducible elsewhere | Hugging Face dataset with the standard Karpathy splits |

### App

`app/main.py` was copied from the translation project: it imported
`translation_engine`, called `model.translate()` and loaded translation
checkpoints, so it could not run. **Fix:** `app/streamlit_app.py`.
