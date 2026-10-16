"""Build notebooks/train_colab_kaggle.ipynb: a self-contained training notebook.

Every module of src/bia_captioner is embedded verbatim (as %%writefile cells),
so the notebook runs the exact package code and needs no GitHub access.
Regenerate after changing src/bia_captioner:
    python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODULES = [
    "__init__",
    "config",
    "tokenizer",
    "data",
    "model",
    "metrics",
    "train",
    "generate",
    "evaluate",
]

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md("""
# BIA Image Captioner: training notebook (Kaggle & Colab)

Train the image-captioning model with a **free Kaggle or Colab GPU**, on
synthetic shapes, Flickr8k, Flickr30k or COCO. The notebook is **self-contained**: section 2 writes the
`bia_captioner` package (identical to the repo's `src/`), and training goes
through the same `train()` function as the `cap-train` command.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4 x2* (or P100) | *Runtime → Change runtime type → T4 GPU* |
| Internet | *Settings → Internet → On* (to download data) | on by default |
| Session limit | 12 h per session, 30 h GPU / week | ~12 h, can disconnect when idle |
| Checkpoints kept in | `/kaggle/working` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

The first run downloads the dataset (none for shapes, ~4.3 GB for Flickr30k) and the
pretrained encoder weights.

**Sections:** 1. Install · 2. The package · 3. Settings · 4. Data ·
5. Model · 6. Training with a live dashboard · 7. Results · 8. Download
""")

md("""
## 1. Install

Kaggle and Colab already have PyTorch, torchvision, transformers and
matplotlib. This only makes sure `datasets` is recent enough.
""")
code("""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets>=3.6"])
print("ready")
""")

# ---------------------------------------------------------------------------
md("""
## 2. The package

The cells below write `src/bia_captioner/*.py`, copied from the repo at build
time. Read them in this order: `config` → `tokenizer` → `data` → `model` →
`metrics` → `train`.
""")
code("""
import os

os.makedirs("src/bia_captioner", exist_ok=True)
""")
for module in MODULES:
    source = (ROOT / "src" / "bia_captioner" / f"{module}.py").read_text()
    code(f"%%writefile src/bia_captioner/{module}.py\n{source}")
code("""
import importlib
import sys

sys.path.insert(0, "src")
for name in [m for m in sys.modules if m.startswith("bia_captioner")]:
    del sys.modules[name]  # pick up the files just written if this cell is re-run
import bia_captioner.train as T

importlib.reload(T)
print("bia_captioner ready")
""")

# ---------------------------------------------------------------------------
md("""
## 3. Settings

**Dataset** (`DATASET`, see `docs/datasets.md`):

| `DATASET` | Images | What for |
|---|---|---|
| `"shapes"` | 8 000 generated shapes, no download | prove the architecture learns from pixels (minutes) |
| `"flowers"` | 6 552 flower photos, 1 simple caption each, ~0.3 GB | the "CIFAR-10 of captioning": good results fast, nice to show |
| `"birds"` | 5 994 bird photos, ~10 human captions each, ~0.6 GB | fine-grained human descriptions |
| `"flickr8k"` | 8 091 photos, ~0.3 GB | first real-photo run, fast |
| `"flickr30k"` | 31 014 photos, ~4.3 GB | the default real benchmark |
| `"coco"` | 123 287 photos, ~21 GB | the standard benchmark; Kaggle disk is tight, Colab Pro / own GPU |

**Encoder** (`ENCODER`) and **how it starts** (`PRETRAINED`):

| `ENCODER` | Family | `PRETRAINED = True` loads |
|---|---|---|
| `"bia-vit"` | the project's own ViT | (always from scratch) |
| `"vit_b_16"`, `"vit_b_32"`, `"vit_l_16"` | torchvision ViT | ImageNet-1k weights |
| `"resnet18"` … `"resnet152"` | torchvision ResNet | ImageNet-1k weights |
| `"hf:google/vit-base-patch16-224-in21k"` | Hugging Face ViT | ImageNet-21k (14M images) |
| `"hf:facebook/dinov2-base"` | Hugging Face DINOv2 | self-supervised, 142M images |
| `"hf:openai/clip-vit-base-patch16"` | CLIP image tower | 400M image-text pairs |

`PRETRAINED = False` starts any of them from random weights. Pretrained
encoders get a small learning rate (1e-5), encoders trained from scratch the
decoder's (3e-4). `FREEZE_ENCODER = True` trains only the decoder: much faster,
slightly lower quality. Batch size and precision adapt to the GPU.

`SMOKE_TEST = True` runs a tiny version (few images, small model) to check the
whole notebook in a few minutes. Keep it `False` for real training.
""")
code("""
import dataclasses
from pathlib import Path

import torch

from bia_captioner.config import TrainConfig, data_config_for

SMOKE_TEST = False          # True = tiny quick check of the whole notebook
DATASET = "flowers"         # "shapes" | "flowers" | "birds" | "flickr8k" | "flickr30k" | "coco"
ENCODER = "vit_b_16"        # "bia-vit" | "vit_b_16" | "resnet50" | "hf:<model id>" ...
PRETRAINED = True           # False = train the encoder from scratch
FREEZE_ENCODER = False      # True = train only the decoder
USE_GOOGLE_DRIVE = False    # Colab only: keep checkpoints on Drive (survive disconnects)
RESUME = True               # continue from last.pt in OUTPUT_DIR if it exists

RUN_NAME = f"{DATASET}-{ENCODER.replace('hf:', '').replace('/', '-')}"
RUN_NAME += "-pretrained" if PRETRAINED and ENCODER != "bia-vit" else "-scratch"

# --- where checkpoints go ------------------------------------------------------
ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ
if ON_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = Path(f"/content/drive/MyDrive/bia-captioner/{RUN_NAME}")
elif ON_KAGGLE:
    OUTPUT_DIR = Path(f"/kaggle/working/checkpoints/{RUN_NAME}")
elif ON_COLAB:
    OUTPUT_DIR = Path(f"/content/checkpoints/{RUN_NAME}")
else:
    OUTPUT_DIR = Path(f"checkpoints/{RUN_NAME}")

# --- device, precision, batch size -----------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_BATCH = TrainConfig().batch_size  # 64: the effective batch we keep
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    gpu_gb = props.total_memory / 1e9
    precision = "bf16" if props.major >= 8 else "fp16"  # native bf16 needs compute capability 8+
    batch_size = 64 if gpu_gb >= 20 or FREEZE_ENCODER else 32
    print(f"GPU: {props.name} ({gpu_gb:.0f} GB)")
else:
    precision, batch_size = "fp32", 16
    print("No GPU: enable one in the notebook settings (fine only for SMOKE_TEST / shapes).")

data_config = data_config_for(DATASET)
train_config = TrainConfig(
    output_dir=OUTPUT_DIR,
    batch_size=batch_size,
    grad_accum_steps=max(1, TARGET_BATCH // batch_size),
    freeze_encoder=FREEZE_ENCODER,
    precision=precision,
    num_workers=min(4, os.cpu_count() or 1),
    device=device.type,
    eval_every=500,
    log_every=25,
)
model_size = {"encoder": ENCODER, "pretrained": PRETRAINED}
if not PRETRAINED or ENCODER == "bia-vit":
    train_config = dataclasses.replace(train_config, epochs=40)  # from scratch: longer

if DATASET == "shapes":
    # Small images and a small model are enough for shapes (bia-vit: 8px patches).
    data_config = dataclasses.replace(data_config, image_size=64 if not PRETRAINED else 224)
    model_size.update(d_model=128, attention_heads=4, decoder_layers=2, encoder_layers=3,
                      patch_size=8)
    train_config = dataclasses.replace(train_config, epochs=8, lr=1e-3, warmup_steps=100,
                                       eval_every=250, bleu_samples=200)

if SMOKE_TEST:
    model_size.update(d_model=128, attention_heads=4, decoder_layers=2, encoder_layers=2,
                      dropout=0.0)
    OUTPUT_DIR = OUTPUT_DIR.parent / "smoke-test"  # never mixed with a real run
    train_config = dataclasses.replace(
        train_config, output_dir=OUTPUT_DIR, epochs=4, batch_size=8, grad_accum_steps=1,
        train_limit=64, val_limit=16, warmup_steps=2, log_every=2, eval_every=8,
        bleu_samples=16, lr=1e-3, num_workers=0,
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if RESUME and (OUTPUT_DIR / T.LAST_CHECKPOINT).exists():
    train_config = dataclasses.replace(train_config, resume=OUTPUT_DIR / T.LAST_CHECKPOINT)

amp_dtype, _ = T.resolve_precision(train_config.precision, device)
for name, cfg in [("DataConfig", data_config), ("TrainConfig", train_config)]:
    print(f"\\n{name}:")
    for key, value in dataclasses.asdict(cfg).items():
        print(f"  {key:18} {value}")
print(f"\\nmodel: {model_size}")
""")

# ---------------------------------------------------------------------------
md("""
## 4. Data

One item per image. Training sees every image once per epoch with a random one
of its captions; validation always uses the first caption, and BLEU compares
with all of them. Images are normalised the way the chosen encoder expects.
""")
code("""
import logging

import matplotlib.pyplot as plt

from bia_captioner.data import CaptionCollator, build_dataset
from bia_captioner.model import encoder_normalization
from bia_captioner.tokenizer import build_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    force=True)
for name in ("httpx", "httpx2"):  # hide Hugging Face HTTP logs
    logging.getLogger(name).setLevel(logging.WARNING)
T.seed_everything(train_config.seed)

tokenizer = build_tokenizer(data_config.tokenizer_id, data_config.cache_dir)
model_config = T.build_model_config(tokenizer, data_config, **model_size)
mean, std = encoder_normalization(model_config)
data_config = dataclasses.replace(data_config, image_mean=mean, image_std=std)

train_dataset = build_dataset(data_config, tokenizer, data_config.train_split,
                              limit=train_config.train_limit, train=train_config.augment)
val_dataset = build_dataset(data_config, tokenizer, data_config.val_split,
                            limit=train_config.val_limit)
print(f"train: {len(train_dataset):,} images | validation: {len(val_dataset):,} images")


def show_image(pixel_values, ax, title=""):
    \"\"\"Undo the normalisation and draw the image.\"\"\"
    m = torch.tensor(data_config.image_mean).view(3, 1, 1)
    s = torch.tensor(data_config.image_std).view(3, 1, 1)
    ax.imshow((pixel_values.cpu() * s + m).clamp(0, 1).permute(1, 2, 0))
    ax.set_title(title, fontsize=8, wrap=True)
    ax.axis("off")


fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i, ax in enumerate(axes):
    item = val_dataset[i]
    show_image(item["pixel_values"], ax, val_dataset.references(i)[0])
plt.show()
""")
code("""
collator = CaptionCollator(pad_token_id=tokenizer.pad_token_id)
train_loader, val_loader = T.build_dataloaders(train_dataset, val_dataset, collator,
                                               train_config, device)
print(f"{len(train_loader):,} train batches per epoch")
""")

# ---------------------------------------------------------------------------
md("## 5. Model")
code("""
import math

from bia_captioner.model import ImageCaptioner

model = ImageCaptioner(model_config, image_size=data_config.image_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_encoder = sum(p.numel() for p in model.encoder.parameters())
start = "pretrained" if model_config.encoder_is_pretrained else "from scratch"
print(f"parameters: {n_params / 1e6:.1f}M (encoder {ENCODER}, {start}, {n_encoder / 1e6:.1f}M)")
print(f"image tokens per image: {model.num_image_tokens}")

# Sanity check: an untrained decoder should be close to random guessing, ln(vocab).
batch = next(iter(val_loader))
with torch.no_grad(), T.autocast(device, amp_dtype):
    small = T.move_batch({k: v[:4] for k, v in batch.items()}, device)
    _, loss = model(small["pixel_values"], small["input_ids"], labels=small["labels"])
print(f"initial loss {loss.item():.2f} (random guessing = {math.log(model_config.vocab_size):.2f})")
""")

# ---------------------------------------------------------------------------
md("""
## 6. Training with a live dashboard

Same loop as `cap-train`: AdamW (smaller LR for a pretrained encoder), warmup +
cosine schedule, mixed precision, gradient accumulation and clipping, label
smoothing. Every `eval_every` steps it validates (loss, accuracy, BLEU on 500
validation images, one sample caption) and saves `last.pt`, plus `best.pt`
when the validation loss improves.

**Interrupted, or hit the session limit?** Run the notebook again with
`RESUME = True`: it continues from `last.pt` (same step, same data order).
""")
code("""
import json

from IPython.display import clear_output

METRICS_FILE = OUTPUT_DIR / "metrics.json"
history = {"train": [], "val": []}
if train_config.resume is not None and METRICS_FILE.exists():
    history = json.loads(METRICS_FILE.read_text())


def draw_dashboard():
    train, val = history["train"], history["val"]
    clear_output(wait=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    (ax_loss, ax_acc, ax_bleu), (ax_lr, ax_grad, ax_speed) = axes

    def series(rows, key):
        rows = [r for r in rows if key in r]
        return [r["step"] for r in rows], [r[key] for r in rows]

    ax_loss.plot(*series(train, "loss"), label="train", alpha=0.8)
    ax_loss.plot(*series(val, "loss"), "o-", label="validation")
    ax_loss.set_title("loss (lower is better)")
    ax_acc.plot(*series(train, "accuracy"), label="train", alpha=0.8)
    ax_acc.plot(*series(val, "accuracy"), "o-", label="validation")
    ax_acc.set_title("next-token accuracy")
    for n in (1, 4):
        ax_bleu.plot(*series(val, f"bleu{n}"), "o-", label=f"BLEU-{n}")
    ax_bleu.set_title("validation BLEU (greedy)")
    for group in ("decoder", "encoder"):
        ax_lr.plot(*series(train, f"lr_{group}"), label=group)
    ax_lr.set_title("learning rate")
    ax_lr.set_yscale("log")
    ax_grad.plot(*series(train, "grad_norm"), color="tab:red", alpha=0.8)
    ax_grad.axhline(train_config.grad_clip, ls="--", color="gray", label="clip")
    ax_grad.set_title("gradient norm (before clipping)")
    ax_speed.plot(*series(train, "images_per_sec"), color="tab:green")
    ax_speed.set_title("speed (images / second)")
    for ax in axes.flat:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    plt.show()
    if train:
        last = train[-1]
        print(f"step {last['step']:,}/{last['total_steps']:,} | epoch {last['epoch'] + 1} | "
              f"train loss {last['loss']:.4f} | {last['images_per_sec']:.0f} img/s")
    if val:
        last = val[-1]
        print(f"validation: loss {last['loss']:.4f} | BLEU-4 {100 * last.get('bleu4', 0):.2f}")
        print(f"  true:      {last['sample_target']}")
        print(f"  generated: {last['sample_prediction']}")


def on_metrics(event):
    row = {k: v for k, v in event.items() if k not in ("kind", "lr")}
    for group, lr in event.get("lr", {}).items():
        row[f"lr_{group}"] = lr
    history[event["kind"]].append(row)
    METRICS_FILE.write_text(json.dumps(history))
    draw_dashboard()


state = T.train(model, train_loader, val_loader, tokenizer, data_config, model_config,
                train_config, callback=on_metrics)
print(f"Training finished. Best validation loss {state.best_val_loss:.4f}")
""")

# ---------------------------------------------------------------------------
md("""
## 7. Results

### 7.1 Captions for validation images (best checkpoint, beam search)
""")
code("""
from bia_captioner.generate import load_model

best_model, _ = load_model(OUTPUT_DIR / T.BEST_CHECKPOINT, device)
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
for i, ax in enumerate(axes.flat):
    i = i % len(val_dataset)
    item = val_dataset[i]
    with torch.inference_mode(), T.autocast(device, amp_dtype):
        tokens, _ = best_model.beam_search(item["pixel_values"][None].to(device), beam_size=3)[0]
    caption = tokenizer.decode(tokens.tolist(), skip_special_tokens=True).strip()
    show_image(item["pixel_values"], ax, f"{caption}\\n(true: {val_dataset.references(i)[0]})")
plt.tight_layout()
plt.show()
""")

md("""
### 7.2 BLEU on the test split, a shareable gallery and an examples table

The test split was never used for training or model selection. Beam size 3.
`gallery.png` shows test photos with the model's captions (and the reference
caption in grey): ready for a README or a post. `examples/README.md` is a
table of real inputs and outputs (thumbnail, model caption, reference caption,
exact match), ready to commit to GitHub. Both use test images spread evenly
over the split, not hand-picked ones.
""")
code("""
from bia_captioner import evaluate

test_args = ["--checkpoint", str(OUTPUT_DIR / T.BEST_CHECKPOINT), "--split", "test",
             "--beam-size", "3", "--output", str(OUTPUT_DIR / "test_predictions.json"),
             "--gallery", str(OUTPUT_DIR / "gallery.png"),
             "--examples", str(OUTPUT_DIR / "examples")]
if SMOKE_TEST:
    test_args += ["--limit", "16"]
test_scores = evaluate.run(test_args)

from IPython.display import Image as ShowImage, display

display(ShowImage(filename=str(OUTPUT_DIR / "gallery.png")))  # ready to share
print((OUTPUT_DIR / "examples" / "README.md").read_text())
""")

# ---------------------------------------------------------------------------
md("""
## 8. Download the trained model

Zips `best.pt`, the tokenizer, the metrics, the test predictions, the
gallery and the examples table (`last.pt` is left out: it is only needed to resume). Use it with the repo:

```bash
unzip bia-captioner-run.zip -d checkpoints/run
cap-generate --checkpoint checkpoints/run/best.pt --image photo.jpg
```
""")
code("""
import shutil
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "bundle"
    bundle.mkdir()
    for name in [T.BEST_CHECKPOINT, "metrics.json", "test_predictions.json", "gallery.png"]:
        if (OUTPUT_DIR / name).exists():
            shutil.copy(OUTPUT_DIR / name, bundle / name)
    shutil.copytree(OUTPUT_DIR / "tokenizer", bundle / "tokenizer")
    if (OUTPUT_DIR / "examples").exists():
        shutil.copytree(OUTPUT_DIR / "examples", bundle / "examples")
    archive = shutil.make_archive(str(OUTPUT_DIR.parent / "bia-captioner-run"), "zip", bundle)
print("created", archive)

if ON_COLAB:
    from google.colab import files

    files.download(archive)
elif ON_KAGGLE:
    print("Kaggle: click 'Save Version', then download it from the notebook's Output tab.")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.validate(nb)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
