# Datasets: from a 2-minute proof to real-world photos

Pick the dataset with `cap-train --dataset <name>` (or `DATASET = "<name>"` in
the notebook). Climb the ladder in this order: each step answers a different
question.

| Step | `--dataset` | Images | Captions | Download | Question it answers |
|---|---|---|---|---|---|
| 0 | `--overfit 8` (any dataset) | 8 | as the dataset | as the dataset | is the training loop wired correctly? |
| 1 | `shapes` | 8 000 / 500 / 500, generated | 1, determined by the image | **none** | does the model really read the image? |
| 2 | `flickr8k` | 6 000 / 1 000 / 1 000 photos | 5 human | 0.3 GB | does it describe real photos? (fast iterations) |
| 3 | `flickr30k` (default) | 29 000 / 1 014 / 1 000 photos | 5 human | 4.3 GB | how good is it on a standard benchmark? |
| 4 | `coco` | 113 287 / 5 000 / 5 000 photos | 5 human | 21 GB | the reference benchmark; compare with papers |
| 5 | external (below) | 3M – 12M+ web images | 1 alt-text | 0.5 – 2 TB | robustness on real-world, open-domain images |

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

## 2. `flickr8k`: small real photos

[jxie/flickr8k](https://huggingface.co/datasets/jxie/flickr8k): 8 091
everyday photos (people, dogs, sports), 5 captions each, the usual
6k / 1k / 1k split. Small enough to train a pretrained encoder in minutes on
one GPU, so it is the place to tune settings before a long run.

## 3. `flickr30k`: the default benchmark

[AnyModal/flickr30k](https://huggingface.co/datasets/AnyModal/flickr30k):
31 014 photos, 5 captions each, Karpathy split (29 000 / 1 014 / 1 000). The
defaults of this repo are tuned for it.

## 4. `coco`: the reference benchmark

[jxie/coco_captions](https://huggingface.co/datasets/jxie/coco_captions):
MS-COCO 2014 with the Karpathy split (113 287 / 5 000 / 5 000 images, ~5
captions each, 80 object categories in everyday scenes). Most captioning
papers report COCO Karpathy-test BLEU-4 / CIDEr, so this is where results are
comparable. The dataset stores one caption per row; `group_column = "cocoid"`
gathers the rows of each image, so an epoch still visits every image once.

Needs ~21 GB of disk (Kaggle's working disk is limited; use Colab with Google
Drive, or your own GPU machine). One epoch is ~3.9× a Flickr30k epoch.

## 5. Large, real-world data (beyond the presets)

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
```

The dataset needs `train` and `validation` splits (or pass `--train-split` /
`--val-split`), and a `test` split for `cap-evaluate`.
