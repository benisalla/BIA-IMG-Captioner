from pathlib import Path

import pytest

from bia_captioner.config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_to_dict,
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


def test_unknown_encoder_is_rejected() -> None:
    with pytest.raises(ValueError, match="encoder"):
        make_model_config(encoder="vgg")


def test_train_config_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp8")


def test_encoder_lr_defaults_depend_on_the_encoder() -> None:
    config = TrainConfig(lr=3e-4)
    assert resolve_encoder_lr(config, "vit-b16") == 1e-5  # pretrained: small LR
    assert resolve_encoder_lr(config, "resnet50") == 1e-5
    assert resolve_encoder_lr(config, "bia-vit") == 3e-4  # from scratch: same as decoder
    assert resolve_encoder_lr(TrainConfig(encoder_lr=5e-5), "vit-b16") == 5e-5  # explicit wins


def test_config_to_dict_stringifies_paths() -> None:
    assert config_to_dict(TrainConfig(output_dir=Path("a/b")))["output_dir"] == "a/b"
