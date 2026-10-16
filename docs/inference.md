# Using a trained model

You need a run directory produced by `cap-train` (or the notebook):

```
checkpoints/run/
  best.pt        model weights + the configs needed to rebuild it
  tokenizer/     the tokenizer used during training
```

No other flags are needed: the encoder, model size, image size, image
normalisation, dataset and vocabulary are all read from the checkpoint.

## Command line

```bash
# print the caption of each image
cap-generate --checkpoint checkpoints/run/best.pt --image photo1.jpg photo2.jpg

# also save them as {image: caption} JSON
cap-generate --checkpoint checkpoints/run/best.pt --image photos/*.jpg --output captions.json

# sampling instead of beam search (different caption on each call)
cap-generate --checkpoint checkpoints/run/best.pt --image photo.jpg \
             --beam-size 1 --temperature 0.8 --top-p 0.9
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkpoint` | required | `best.pt` or `last.pt` |
| `--image` | required | one or more image files (any size; resized automatically) |
| `--output` | none | write `{image: caption}` as JSON |
| `--beam-size` | 3 | beam search width; 1 = greedy or sampling |
| `--length-penalty` | 0.6 | beam search: >0 favours longer captions, 0 = raw log-probability |
| `--temperature` | 0 | with `--beam-size 1`: 0 = greedy, >0 = sample (higher is more random) |
| `--top-k` | none | sample only among the k most likely tokens |
| `--top-p` | none | sample only among the most likely tokens whose probabilities sum to p |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |

**Beam search (default)** gives the best captions. **Greedy** is faster and
usually close. **Sampling** gives more varied, sometimes less accurate captions.

## Evaluate on the test split

```bash
cap-evaluate --checkpoint checkpoints/run/best.pt --split test --output test_predictions.json
```

Captions every image of the checkpoint's dataset split (Flickr30k test: 1 000
images; `shapes` test: 500), beam size 3, and prints corpus BLEU-1..4 against
all human captions of each image. `--output` saves the scores and
every prediction next to its references. `--limit N` scores only the first N
images, `--beam-size 1` is greedy and much faster.

Two ways to *show* the results, both using images spread evenly over the
split (not hand-picked):

```bash
cap-evaluate --checkpoint checkpoints/run/best.pt --split test \
             --gallery gallery.png --gallery-size 12 \
             --examples examples/ --examples-size 10
```

- `--gallery` saves one image grid with the model's caption (black) and a
  reference caption (grey) under each photo: for a slide or a post.
- `--examples` writes `examples/README.md`, a table with a thumbnail, the
  model's caption, the reference caption and whether they match exactly, plus
  the thumbnails: GitHub renders it as a page. The tables in
  [docs/examples](examples) were made this way.

## Web demo

```bash
pip install -e '.[app]'
streamlit run app/streamlit_app.py -- --checkpoint checkpoints/run/best.pt
```

Upload a photo; the sidebar switches between beam search, greedy and sampling.
With beam search, every beam and its score are shown.

## From Python

```python
from pathlib import Path
from PIL import Image
from bia_captioner.generate import caption_image, load_model, load_tokenizer

checkpoint = Path("checkpoints/run/best.pt")
model, data_config = load_model(checkpoint, device="cuda")   # already in eval mode
tokenizer = load_tokenizer(checkpoint)

print(caption_image(model, tokenizer, Image.open("photo.jpg"), data_config))
```

For many images at once, batch them yourself and call `model.generate`
(greedy / sampling, batched) directly; `model.beam_search` takes one image at
a time. Both return token ids starting with BOS; decode them with
`tokenizer.decode(ids, skip_special_tokens=True)`.

Images should look like the training data: everyday scenes with people and
animals for Flickr / COCO, colored shapes for `shapes`. Very different images
(drawings, screenshots, medical images) are out of distribution and give poor
captions; [datasets.md](datasets.md) lists larger, more varied datasets.
