# Using a trained model

You need a run directory produced by `cap-train` (or the notebook):

```
checkpoints/run/
  best.pt        model weights + the configs needed to rebuild it
  tokenizer/     the tokenizer used during training
```

No other flags are needed: the encoder type, model size, image size and
vocabulary are all read from the checkpoint.

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

Captions the 1 000 test images (beam size 3) and prints corpus BLEU-1..4
against the 5 human captions of each image. `--output` saves the scores and
every prediction next to its references. `--limit N` scores only the first N
images, `--beam-size 1` is greedy and much faster.

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

print(caption_image(model, tokenizer, Image.open("photo.jpg"), data_config.image_size))
```

For many images at once, batch them yourself and call `model.generate`
(greedy / sampling, batched) directly; `model.beam_search` takes one image at
a time. Both return token ids starting with BOS; decode them with
`tokenizer.decode(ids, skip_special_tokens=True)`.

Photos should look like Flickr30k (everyday scenes with people and animals).
Very different images (drawings, screenshots, medical images) are out of
distribution and give poor captions.
