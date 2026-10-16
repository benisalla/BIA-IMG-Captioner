# Datasets: from a 2-minute proof to real-world photos

Pick the dataset with `cap-train --dataset <name>` (or `DATASET = "<name>"` in
the notebook). Climb the ladder in this order: each step answers a different
question.

| Step | `--dataset` | Images | Captions | Download | Question it answers |
|---|---|---|---|---|---|
| 0 | `--overfit 8` (any dataset) | 8 | as the dataset | as the dataset | is the training loop wired correctly? |
| 1 | `shapes` | 8 000 / 500 / 500, generated | 1, determined by the image | **none** | does the model really read the image? |
| 2 | **`flowers`** | 5 569 / 328 / 655 photos | 1 short, consistent | 0.3 GB | **the "CIFAR-10 of captioning"**: real, beautiful photos, good results fast |
| 3 | `birds` | 5 034 / 480 / 480 photos | ~10 human-written | 0.6 GB | the "CIFAR-100": fine-grained, human language, still simple |
| 4 | `flickr8k` | 6 000 / 1 000 / 1 000 photos | 5 human | 0.3 GB | open-domain everyday photos, small |
| 5 | `flickr30k` (default) | 29 000 / 1 014 / 1 000 photos | 5 human | 4.3 GB | a standard benchmark |
| 6 | `coco` | 113 287 / 5 000 / 5 000 photos | ~5 human | 21 GB | the reference benchmark; compare with papers |
| 7 | external (below) | 3M – 12M+ web images | 1 alt-text | 0.5 – 2 TB | real-world, open-domain robustness |

The idea is the same as CIFAR-10 for image classification: if the model learns
steps 1–3 well, the architecture and training code are right, and going to
steps 4–7 is a matter of more data, a bigger model and more compute (more
decoder layers, a larger pretrained encoder, more epochs, more GPUs).

## 1. `shapes`: the simplest proof the architecture works

Images of one or two colored shapes on a white background, generated on the
fly (no download, deterministic, different seeds for train / validation /
test):

```
"a big red circle"
"a small blue square on the left and a big yellow triangle on the right"
```

Why it is the right first experiment:

- **The caption is fully determined by the image.** A decoder that ignores
  the image can only guess colors, shapes, sizes and positions. Compare the
  test scores with `cap-evaluate --shuffle-images` (each caption generated
  from another image): a model that reads images collapses there, a language
  model does not. A plain loss curve on real photos cannot tell these apart
  early on, because the language model alone already lowers the loss.
- **It tests what captioning needs**: object recognition (shape), attributes
  (color, size), counting (one or two objects) and spatial relations
  (left / right).
- **It is cheap**: a 7.6M-parameter model with the BIA-ViT trained from
  scratch learns it on a laptop CPU: after 2 epochs, 43% exact captions on
  unseen test images vs 0.6% with shuffled images (see
  [testing.md](testing.md#2-synthetic-shapes-does-the-architecture-read-images)).
  Any encoder works, pretrained or not.

```bash
cap-train --dataset shapes --encoder bia-vit --image-size 64 --patch-size 8 \
          --d-model 128 --heads 4 --layers 2 --encoder-layers 3 \
          --epochs 8 --lr 1e-3 --warmup-steps 100 --eval-every 250 --bleu-samples 200 \
          --output-dir checkpoints/shapes
cap-evaluate --checkpoint checkpoints/shapes/best.pt --split test --beam-size 1
cap-evaluate --checkpoint checkpoints/shapes/best.pt --split test --beam-size 1 --shuffle-images
```

## 2. `flowers`: the "CIFAR-10 of captioning"

[pranked03/flowers-blip-captions](https://huggingface.co/datasets/pranked03/flowers-blip-captions):
the 102-category Oxford Flowers photos (6 552 here, ~500 px, sharp and
colorful), each with one short caption written by the BLIP captioning model:

```
"a purple passion flower with white flowers and green leaves"
"a blue passion flower with a yellow flower in the middle of the flower"
```

Why it is the right "does it work on real photos?" dataset:

- **Real, high-quality images** that look good in a README or a post.
- **Simple, consistent language**: one object type, a small vocabulary
  (colors, petal shapes, flower names), one caption style. The model reaches
  good, clearly image-specific captions after a few epochs.
- **Small**: 0.3 GB, ~5.6k training images. With a frozen pretrained encoder
  it even trains on a laptop CPU (see [testing.md](testing.md)); on a free
  Kaggle / Colab GPU it should take minutes (estimate).

The dataset has a single split; `--dataset flowers` holds out a fixed random
5% for validation and 10% for test (the rows are sorted by flower species, so
the random cut keeps every species in every split).

```bash
cap-train --dataset flowers --encoder vit_b_16 --epochs 10 --output-dir checkpoints/flowers   # GPU
cap-train --dataset flowers --encoder resnet50 --freeze-encoder --d-model 256 --heads 4 \
          --layers 3 --epochs 4 --batch-size 32 --lr 5e-4 --output-dir checkpoints/flowers  # CPU
cap-evaluate --checkpoint checkpoints/flowers/best.pt --split test --gallery flowers.png
```

`--gallery` saves a grid of test photos with the model's captions under them,
ready to share.

Measured on a laptop CPU with a frozen ResNet-50 and 4 epochs: test BLEU-4
47.7, with the species named correctly in most captions (see the gallery in
[testing.md](testing.md#3-flowers-real-photos-quick-results)).

## 3. `birds`: the next step (CUB-200)

[Multimodal-Fatima/CUB_train](https://huggingface.co/datasets/Multimodal-Fatima/CUB_train):
5 994 photos of 200 bird species from CUB-200-2011, each with ~10
**human-written** captions (Reed et al., 2016) about colors and body parts:

```
"this is a brown bird with a white face and a long downward pointing beak."
"aquatic large bird with long hooked bill, white face, and brown body."
```

Harder than flowers (fine-grained details, ten different phrasings per image,
human typos) but still a narrow domain with a small vocabulary. BLEU is
computed against all ten captions. Held-out validation / test: 8% / 8%.

## 4. `flickr8k`: small, open-domain photos

[jxie/flickr8k](https://huggingface.co/datasets/jxie/flickr8k): 8 091
everyday photos (people, dogs, sports), 5 captions each, the usual
6k / 1k / 1k split. Small enough to train a pretrained encoder in minutes on
one GPU, so it is the place to tune settings before a long run.

## 5. `flickr30k`: a standard benchmark

[AnyModal/flickr30k](https://huggingface.co/datasets/AnyModal/flickr30k):
31 014 photos, 5 captions each, Karpathy split (29 000 / 1 014 / 1 000). The
defaults of this repo are tuned for it.

## 6. `coco`: the reference benchmark

[jxie/coco_captions](https://huggingface.co/datasets/jxie/coco_captions):
MS-COCO 2014 with the Karpathy split (113 287 / 5 000 / 5 000 images, ~5
captions each, 80 object categories in everyday scenes). Most captioning
papers report COCO Karpathy-test BLEU-4 / CIDEr, so this is where results are
comparable. The dataset stores one caption per row; `group_column = "cocoid"`
gathers the rows of each image, so an epoch still visits every image once.

Needs ~21 GB of disk (Kaggle's working disk is limited; use Colab with Google
Drive, or your own GPU machine). One epoch is ~3.9× a Flickr30k epoch.

## 7. Large, real-world data (beyond the presets)

For open-domain images (products, screenshots, art, outdoor scenes, text in
images) the curated datasets above are too narrow. These need their own
download pipeline (they are distributed as URLs or WebDataset shards) and
plenty of disk, so they are recommendations rather than presets:

| Dataset | Size | What it adds | Get it |
|---|---|---|---|
| [Conceptual Captions 3M](https://ai.google.com/research/ConceptualCaptions/) | 3.3M images | web alt-text, very diverse | URLs in `google-research-datasets/conceptual_captions`; download with [img2dataset](https://github.com/rom1504/img2dataset), or the shards in `pixparse/cc3m-wds` |
| [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m) | 12M images | more, noisier alt-text | img2dataset |
| [SBU Captions](https://huggingface.co/datasets/vicenteor/sbu_captions) | 1M images | user-written Flickr captions | URLs, img2dataset |
| [TextCaps](https://textvqa.org/textcaps/) | 28k images | captions that must read text in the image | download page |
| [VizWiz-Captions](https://vizwiz.org/tasks-and-datasets/image-captioning/) | 39k images | photos taken by blind people: blur, odd framing | download page |
| [nocaps](https://nocaps.org/) | 15k images (val + test) | **evaluation only**: objects never seen in COCO | download page |

A common recipe is to **pre-train on CC3M** (lots of diverse but noisy
captions), then **fine-tune on COCO** (clean, human-written) and **evaluate on
nocaps** for novel objects. To use one of these, write a small
`torch.utils.data.Dataset` with the same interface as `CaptionDataset`
(`__getitem__` returning `pixel_values` / `input_ids` / `index`, and
`references(index)`), or convert the data to a Hugging Face dataset and pass
`--dataset-id`, `--image-column`, `--caption-columns` and `--group-column`.

## Using any Hugging Face dataset

No preset needed if the dataset has an image column and caption column(s):

```bash
# one column holding a list of captions
cap-train --dataset-id user/my-data --image-column image --caption-columns captions
# several columns with one caption each
cap-train --dataset-id user/my-data --caption-columns caption_0 caption_1 caption_2
# one caption per row, several rows per image
cap-train --dataset-id user/my-data --caption-columns caption --group-column image_id
# several captions joined in one string
cap-train --dataset-id user/my-data --caption-columns description --caption-separator $'\n'
# only one split on the Hub: hold out 5% for validation and 10% for test
cap-train --dataset-id user/my-data --caption-columns text --holdout 0.05 0.10
```

Without `--holdout`, the dataset needs `train` and `validation` splits (or
pass `--train-split` / `--val-split`), and a `test` split for `cap-evaluate`.
