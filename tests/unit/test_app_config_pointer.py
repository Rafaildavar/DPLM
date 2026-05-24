# -*- coding: utf-8 -*-

from app.services.app_config import AppConfig, ConfigStore


def test_pointer_smoothing_is_loaded_as_float():
    config = AppConfig.from_dict(
        {"recognition": {"pointer_smoothing": "0.72"}}
    )

    assert config.recognition.pointer_smoothing == 0.72


def test_pointer_smoothing_validation_range():
    config = AppConfig.from_dict(
        {"recognition": {"pointer_smoothing": "1.50"}}
    )

    result = ConfigStore().validate(config)

    assert not result.ok
    assert any("pointer_smoothing" in error for error in result.errors)
