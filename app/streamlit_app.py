"""Streamlit demo: upload a photo, get its caption.

pip install -e '.[app]'
streamlit run app/streamlit_app.py -- --checkpoint checkpoints/run/best.pt
"""

import argparse
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from bia_captioner.data import build_image_transform
from bia_captioner.generate import load_model, load_tokenizer
from bia_captioner.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/run/best.pt"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args(sys.argv[1:])


@st.cache_resource(show_spinner="Loading the model...")
def load(checkpoint: str, device_name: str):
    device = resolve_device(device_name)
    model, data_config = load_model(Path(checkpoint), device)
    return model, data_config, load_tokenizer(Path(checkpoint)), device


@torch.inference_mode()
def captions_for(image: Image.Image, settings: dict) -> list[tuple[str, float | None]]:
    """[(caption, score)] best first; score is None for greedy / sampling."""
    model, data_config, tokenizer, device = load(settings["checkpoint"], settings["device"])
    transform = build_image_transform(
        data_config.image_size, False, data_config.image_mean, data_config.image_std
    )
    pixels = transform(image.convert("RGB"))
    pixels = pixels[None].to(device)
    if settings["mode"] == "Beam search":
        beams = model.beam_search(
            pixels, beam_size=settings["beam_size"], length_penalty=settings["length_penalty"]
        )
        return [(tokenizer.decode(t.tolist(), skip_special_tokens=True), s) for t, s in beams]
    temperature = 0.0 if settings["mode"] == "Greedy" else settings["temperature"]
    tokens = model.generate(
        pixels, temperature=temperature, top_k=settings["top_k"], top_p=settings["top_p"]
    )
    return [(tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True), None)]


def sidebar(args: argparse.Namespace) -> dict:
    st.sidebar.header("Model")
    settings = {
        "checkpoint": st.sidebar.text_input("Checkpoint", str(args.checkpoint)),
        "device": args.device,
    }
    st.sidebar.header("Decoding")
    settings["mode"] = st.sidebar.radio("Strategy", ["Beam search", "Greedy", "Sampling"])
    settings["beam_size"] = st.sidebar.slider("Beam size", 1, 10, 3)
    settings["length_penalty"] = st.sidebar.slider("Length penalty", 0.0, 2.0, 0.6, 0.1)
    settings["temperature"] = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, 0.05)
    top_k = st.sidebar.slider("Top-k (0 = off)", 0, 200, 50)
    settings["top_k"] = top_k or None
    settings["top_p"] = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95, 0.01)
    return settings


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="BIA Image Captioner", page_icon="🖼️", layout="centered")
    st.title("BIA Image Captioner")
    st.caption("Image encoder + Transformer decoder trained on Flickr30k.")
    settings = sidebar(args)

    if not Path(settings["checkpoint"]).exists():
        st.error(f"Checkpoint not found: {settings['checkpoint']}. Train one with `cap-train`.")
        return

    upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        return
    image = Image.open(upload)
    st.image(image, use_container_width=True)
    with st.spinner("Writing a caption..."):
        results = captions_for(image, settings)

    best, _ = results[0]
    st.subheader(best.strip().capitalize() or "(empty caption)")
    if len(results) > 1:
        with st.expander("All beams"):
            for caption, score in results:
                st.write(f"`{score:.3f}`  {caption.strip()}")


if __name__ == "__main__":
    main()
