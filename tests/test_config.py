from pathlib import Path

import pytest

from bia_captioner.config import (
    DATASET_PRESETS,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_to_dict,
    data_config_for,
    resolve_encoder_lr,
)


def make_model_config(**overrides: object) -> ModelConfig:
    values = {"vocab_size": 100, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 1}
    values.update(overrides)
    return ModelConfig(**values)  # type: ignore[arg-type]


def test_default_data_config_is_valid() -> None:
    config = DataConfig()
    assert config.image_size == 224
    assert config.max_tokens == 64


def test_model_dimension_must_match_attention_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        make_model_config(d_model=10, attention_heads=3)


def test_pad_token_must_differ_from_eos() -> None:
    # pad == eos would make the loss ignore EOS, so the model never learns to stop.
    with pytest.raises(ValueError, match="pad_token_id"):
        make_model_config(pad_token_id=1)


@pytest.mark.parametrize(
    "encoder",
    ["bia-vit", "vit_b_16", "vit_h_14", "resnet18", "resnet152", "hf:facebook/dinov2-base"],
)
def test_known_encoders_are_accepted(encoder: str) -> None:
    assert make_model_config(encoder=encoder).encoder == encoder


@pytest.mark.parametrize("encoder", ["vgg", "vit-b16", "hf:", "resnet"])
def test_unknown_encoder_is_rejected(encoder: str) -> None:
    with pytest.raises(ValueError, match="encoder"):
        make_model_config(encoder=encoder)


def test_bia_vit_is_never_pretrained() -> None:
    assert make_model_config(encoder="resnet50").encoder_is_pretrained
    assert not make_model_config(encoder="resnet50", pretrained=False).encoder_is_pretrained
    assert not make_model_config(encoder="bia-vit", pretrained=True).encoder_is_pretrained


def test_every_dataset_preset_builds_a_valid_config() -> None:
    for name in DATASET_PRESETS:
        assert data_config_for(name).dataset_id == DATASET_PRESETS[name]["dataset_id"]
    assert data_config_for("coco").group_column == "cocoid"
    assert len(data_config_for("flickr8k").caption_columns) == 5
    assert data_config_for("flickr8k", image_size=64).image_size == 64  # overrides win
    with pytest.raises(ValueError, match="unknown dataset"):
        data_config_for("imagenet")


def test_lists_from_a_checkpoint_become_tuples() -> None:
    config = DataConfig(caption_columns=["a", "b"], image_mean=[0.5] * 3, image_std=[0.5] * 3)
    assert config.caption_columns == ("a", "b") and config.image_mean == (0.5, 0.5, 0.5)


def test_train_config_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp8")


def test_encoder_lr_defaults_depend_on_how_the_encoder_starts() -> None:
    config = TrainConfig(lr=3e-4)
    pretrained = make_model_config(encoder="vit_b_16")
    scratch = make_model_config(encoder="resnet50", pretrained=False)
    assert resolve_encoder_lr(config, pretrained) == 1e-5  # pretrained: small LR
    assert resolve_encoder_lr(config, scratch) == 3e-4  # from scratch: same as decoder
    assert resolve_encoder_lr(config, make_model_config(encoder="bia-vit")) == 3e-4
    assert resolve_encoder_lr(TrainConfig(encoder_lr=5e-5), pretrained) == 5e-5  # explicit wins


def test_config_to_dict_stringifies_paths() -> None:
    assert config_to_dict(TrainConfig(output_dir=Path("a/b")))["output_dir"] == "a/b"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"caption_columns": ()}, "caption_columns"),
        ({"holdout": (0.5, 0.6)}, "holdout"),
        ({"holdout": (0.0, 0.1)}, "holdout"),
        ({"image_size": 0}, "image_size"),
        ({"max_tokens": 1}, "max_tokens"),
    ],
)
def test_invalid_data_config_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DataConfig(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vocab_size": 0}, "vocab_size"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_model_config_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_model_config(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"grad_accum_steps": 0}, "grad_accum_steps"),
        ({"epochs": 0}, "epochs"),
        ({"encoder_lr": -1.0}, "encoder_lr"),
    ],
)
def test_invalid_train_config_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainConfig(**overrides)


def test_configs_are_frozen() -> None:
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        TrainConfig().lr = 1.0  # type: ignore[misc]
