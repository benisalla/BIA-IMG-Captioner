"""Small CPU tests for the image -> caption model (no network, tiny sizes)."""

import pytest
import torch

from bia_captioner.config import ModelConfig
from bia_captioner.model import IGNORE_INDEX, ImageCaptioner, _sample

PAD, BOS, EOS = 0, 1, 2
IMAGE_SIZE = 64
# Image tokens at 64px: bia-vit (patch 16) 4x4 = 16, resnet50 2x2 = 4, vit-b16 4x4 + class = 17.
TOKENS = {"bia-vit": 16, "resnet50": 4, "vit-b16": 17}


def make_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 50,
        "pad_token_id": PAD,
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        "encoder": "bia-vit",
        "d_model": 32,
        "attention_heads": 2,
        "decoder_layers": 1,
        "encoder_layers": 1,
        "max_tokens": 12,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)  # type: ignore[arg-type]


def make_model(**overrides: object) -> ImageCaptioner:
    torch.manual_seed(0)
    model = ImageCaptioner(make_config(**overrides), IMAGE_SIZE, pretrained_encoder=False)
    return model.eval()


def images(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.mark.parametrize("encoder", ["bia-vit", "resnet50", "vit-b16"])
def test_shapes_for_every_encoder(encoder: str) -> None:
    model = make_model(encoder=encoder)
    pixels, ids = images(), torch.randint(3, 50, (2, 7))
    assert model.encode(pixels).shape == (2, TOKENS[encoder], 32)
    logits, loss = model(pixels, ids)
    assert logits.shape == (2, 7, 50)
    assert loss is None


def test_wrong_image_size_is_rejected() -> None:
    model = make_model(encoder="resnet50")
    with pytest.raises(ValueError, match="image tokens"):
        model.encode(torch.randn(1, 3, 128, 128))


def test_bia_vit_needs_whole_patches() -> None:
    with pytest.raises(ValueError, match="multiple of patch_size"):
        ImageCaptioner(make_config(), image_size=225)  # the old 225px / 16px setting


def test_initial_loss_is_close_to_random_guessing() -> None:
    model = make_model()
    ids = torch.randint(3, 50, (4, 10))
    _, loss = model(images(4), ids, ids.clone())
    assert loss is not None
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.3


def test_loss_is_finite_and_ignores_minus_100() -> None:
    model = make_model()
    pixels = images()
    ids = torch.tensor([[BOS, 5, 6, EOS, PAD, PAD], [BOS, 7, 8, 9, 10, EOS]])
    labels = ids.masked_fill(ids.eq(PAD), IGNORE_INDEX)
    _, loss = model(pixels, ids, labels)
    assert loss is not None and torch.isfinite(loss)

    # What sits at ignored positions (here: real tokens instead of pad) must not matter.
    other_ids = ids.clone()
    other_ids[0, 4:] = torch.tensor([7, 8])
    _, same = model(pixels, other_ids, labels)
    assert torch.allclose(loss, same)

    # Label smoothing changes the loss but keeps it finite.
    _, smooth = model(pixels, ids, labels, label_smoothing=0.1)
    assert smooth is not None and torch.isfinite(smooth) and not torch.allclose(loss, smooth)


def test_causal_future_tokens_do_not_change_past_logits() -> None:
    model = make_model()
    pixels = images()
    ids = torch.randint(3, 50, (2, 8))
    changed = ids.clone()
    changed[:, 5:] = torch.randint(3, 50, (2, 3))
    logits_a, _ = model(pixels, ids)
    logits_b, _ = model(pixels, changed)
    assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-5)
    assert not torch.allclose(logits_a[:, 5:], logits_b[:, 5:])


def test_padding_does_not_change_real_token_logits() -> None:
    model = make_model()
    pixels = images(1)
    short = torch.tensor([[BOS, 5, 6, EOS]])
    padded = torch.tensor([[BOS, 5, 6, EOS, PAD, PAD, PAD]])
    logits_short, _ = model(pixels, short)
    logits_padded, _ = model(pixels, padded)
    assert torch.allclose(logits_short, logits_padded[:, :4], atol=1e-5)
    assert torch.isfinite(logits_padded).all()


def test_the_image_changes_the_prediction() -> None:
    # The legacy model never normalised the cross-attention inputs; check the
    # image actually reaches the logits.
    model = make_model()
    ids = torch.randint(3, 50, (1, 5))
    logits_a, _ = model(images(1), ids)
    logits_b, _ = model(images(1), ids)
    assert not torch.allclose(logits_a, logits_b)


def test_generate_starts_with_bos_and_respects_max_tokens() -> None:
    model = make_model()
    out = model.generate(images(3))
    assert out.dtype == torch.long
    assert out.shape[0] == 3 and 1 < out.shape[1] <= model.config.max_tokens
    assert out[:, 0].eq(BOS).all()
    assert not out[:, 1:].eq(BOS).any()  # BOS is never generated

    # Asking for more tokens than allowed is capped at max_tokens.
    assert model.generate(images(1), max_new_tokens=1000).shape[1] <= model.config.max_tokens
    assert model.generate(images(1), max_new_tokens=3).shape[1] == 4

    # Sampling paths also work.
    for kwargs in ({"top_k": 5}, {"top_p": 0.9}, {"top_k": 10, "top_p": 0.5}):
        sampled = model.generate(images(2), temperature=1.0, **kwargs)
        assert sampled[:, 0].eq(BOS).all()


def test_generate_stops_at_eos_and_pads_finished_rows() -> None:
    model = make_model()
    # Make the final hidden state all ones and only the EOS row of the head
    # non-zero: every row picks EOS first, so generation stops after one step.
    with torch.no_grad():
        model.final_norm.weight.zero_()
        model.final_norm.bias.fill_(1.0)
        model.output.weight.zero_()  # tied: also zeroes the token embedding
        model.output.weight[EOS] = 1.0
    out = model.generate(images(2))
    assert out.tolist() == [[BOS, EOS], [BOS, EOS]]


def test_generate_fills_pad_after_eos() -> None:
    model = make_model()
    # Row 0 sees EOS as its first greedy pick, row 1 only later -> row 0 must be padded.
    calls = {"n": 0}

    def fake_decode(memory: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(ids.shape[0], ids.shape[1], model.config.vocab_size)
        logits[0, :, EOS] = 1.0
        logits[1, :, 7] = 1.0
        if calls["n"] >= 2:
            logits[1, :, EOS] = 2.0
        calls["n"] += 1
        return logits

    model.decode = fake_decode  # type: ignore[method-assign]
    out = model.generate(images(2))
    assert out.tolist() == [[BOS, EOS, PAD, PAD], [BOS, 7, 7, EOS]]


def fixed_next_token_model(table: dict[tuple[int, ...], dict[int, float]]) -> ImageCaptioner:
    """A model whose next-token probabilities depend only on the prefix (via `table`)."""
    model = make_model()

    def fake_decode(memory: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((ids.shape[0], ids.shape[1], model.config.vocab_size), -30.0)
        for row, prefix in enumerate(ids.tolist()):
            for token, prob in table.get(tuple(prefix), {EOS: 1.0}).items():
                logits[row, -1, token] = torch.log(torch.tensor(prob))
        return logits

    model.decode = fake_decode  # type: ignore[method-assign]
    return model


def test_beam_search_finds_the_better_sequence_greedy_misses() -> None:
    # Greedy takes 5 (p=0.6) and then is stuck with 0.3 -> total 0.18.
    # Beam search also keeps 6 (p=0.4), whose continuation is certain -> total 0.4.
    table = {
        (BOS,): {5: 0.6, 6: 0.4},
        (BOS, 5): {7: 0.3, 8: 0.3, 9: 0.4 - 1e-6},
        (BOS, 6): {7: 1.0},
    }
    model = fixed_next_token_model(table)
    greedy = model.generate(images(1))[0].tolist()
    assert greedy[:2] == [BOS, 5]

    results = model.beam_search(images(1), beam_size=2, length_penalty=0.0)
    best, score = results[0]
    assert best.tolist() == [BOS, 6, 7, EOS]
    assert score == pytest.approx(torch.log(torch.tensor(0.4)).item(), abs=1e-4)
    assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)


def test_beam_search_with_beam_one_matches_greedy() -> None:
    model = make_model()
    pixels = images(1)
    greedy = model.generate(pixels)[0]
    (beam, _), *_ = model.beam_search(pixels, beam_size=1)
    # Greedy pads finished rows; strip that before comparing.
    assert beam.tolist() == greedy[greedy != PAD].tolist()


def test_beam_search_respects_max_tokens_and_needs_one_image() -> None:
    model = make_model()
    for tokens, _ in model.beam_search(images(1), beam_size=3):
        assert tokens[0] == BOS and len(tokens) <= model.config.max_tokens
    with pytest.raises(ValueError, match="one image"):
        model.beam_search(images(2))


def test_top_p_keeps_the_smallest_nucleus() -> None:
    torch.manual_seed(0)
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    picks = {_sample(logits, 1.0, None, 0.7).item() for _ in range(200)}
    assert picks == {0, 1}  # 0.5 + 0.3 >= 0.7: tokens 2 and 3 are never sampled


def test_parameter_groups_cover_every_parameter_once() -> None:
    model = make_model(encoder="resnet50")
    groups = model.parameter_groups(weight_decay=0.1, lr=1e-3, encoder_lr=1e-4)
    seen = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters()}
    assert len(seen) == len(set(seen)), "a parameter is in two groups"
    assert set(seen) == expected

    group_of = {id(p): g for g in groups for p in g["params"]}
    # Tied embedding / output weight: no decay, decoder lr.
    tied = group_of[id(model.token_embedding.weight)]
    assert tied["weight_decay"] == 0.0 and tied["lr"] == 1e-3
    assert group_of[id(model.image_position)]["weight_decay"] == 0.0
    assert group_of[id(model.image_projection.weight)]["weight_decay"] == 0.1
    assert group_of[id(model.image_projection.bias)]["weight_decay"] == 0.0
    conv = model.encoder.body[0].weight
    assert group_of[id(conv)]["lr"] == 1e-4 and group_of[id(conv)]["weight_decay"] == 0.1

    # Works with a real optimizer.
    torch.optim.AdamW(groups)


def test_vit_position_and_class_tokens_get_no_weight_decay() -> None:
    model = make_model(encoder="vit-b16")
    groups = model.parameter_groups(weight_decay=0.1, lr=1e-3, encoder_lr=1e-4)
    group_of = {id(p): g for g in groups for p in g["params"]}
    assert group_of[id(model.encoder.vit.class_token)]["weight_decay"] == 0.0
    assert group_of[id(model.encoder.vit.encoder.pos_embedding)]["weight_decay"] == 0.0


def test_frozen_encoder_gets_no_gradient_and_stays_in_eval() -> None:
    model = make_model(encoder="resnet50")
    model.freeze_encoder(True)
    model.train()
    assert not model.encoder.training  # BatchNorm keeps its running statistics
    assert model.blocks.training

    groups = model.parameter_groups(0.1, lr=1e-3, encoder_lr=1e-4)
    assert {g["name"] for g in groups} == {"decoder"}  # frozen params are left out

    ids = torch.randint(3, 50, (2, 6))
    _, loss = model(images(), ids, ids.clone())
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.image_projection.weight.grad is not None

    model.freeze_encoder(False)
    assert model.encoder.training and all(p.requires_grad for p in model.encoder.parameters())


def test_weight_tying_and_zero_pad_row() -> None:
    model = make_model()
    assert model.output.weight is model.token_embedding.weight
    assert model.token_embedding.weight[PAD].abs().sum() == 0


def test_can_overfit_one_batch() -> None:
    # bia-vit only: training ResNet-50 for 60 steps on a CPU takes over a minute.
    model = make_model()
    model.train()
    pixels = images()
    ids = torch.tensor([[BOS, 5, 6, 7, EOS, PAD], [BOS, 9, 8, 7, 6, EOS]])
    labels = ids.masked_fill(ids.eq(PAD), IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameter_groups(0.0, lr=3e-3, encoder_lr=3e-3))

    _, first = model(pixels, ids, labels)
    for _ in range(60):
        _, loss = model(pixels, ids, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert first is not None
    assert loss.item() < 0.2 * first.item()

    # And it reproduces both captions from the images alone.
    model.eval()
    out = model.generate(pixels)
    assert out[0, :5].tolist() == [BOS, 5, 6, 7, EOS]
    assert out[1].tolist() == [BOS, 9, 8, 7, 6, EOS]
