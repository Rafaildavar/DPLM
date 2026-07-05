# -*- coding: utf-8 -*-

from pathlib import Path

from cv.gesture_features import FEATURE_STATIC_MEAN
from scripts import ml_smoke


def test_optional_legacy_pickle_error_is_skipped_in_default_smoke(tmp_path, monkeypatch):
    profile = _write_profile(tmp_path, required=False)
    monkeypatch.setattr(ml_smoke.joblib, "load", _raise_bitgenerator_error)

    result = ml_smoke._run_profile(
        tmp_path,
        profile,
        allow_optional_skips=True,
    )

    assert result.status == "skipped"
    assert "legacy optional artifact" in result.error


def test_explicit_legacy_pickle_error_still_fails(tmp_path, monkeypatch):
    profile = _write_profile(tmp_path, required=False)
    monkeypatch.setattr(ml_smoke.joblib, "load", _raise_bitgenerator_error)

    result = ml_smoke._run_profile(
        tmp_path,
        profile,
        allow_optional_skips=False,
    )

    assert result.status == "failed"
    assert "BitGenerator" in result.error


def _write_profile(tmp_path: Path, *, required: bool) -> ml_smoke.SmokeProfile:
    (tmp_path / "legacy.pkl").write_bytes(b"placeholder")
    (tmp_path / "classes.json").write_text('["a", "b"]', encoding="utf-8")
    (tmp_path / "feature_dim.txt").write_text("42", encoding="utf-8")
    (tmp_path / "feature_mode.txt").write_text(FEATURE_STATIC_MEAN, encoding="utf-8")
    return ml_smoke.SmokeProfile(
        name="legacy_sequence_mlp",
        model="legacy.pkl",
        classes="classes.json",
        feature_dim="feature_dim.txt",
        feature_mode="feature_mode.txt",
        required=required,
    )


def _raise_bitgenerator_error(_path):
    raise ValueError(
        "<class 'numpy.random._mt19937.MT19937'> "
        "is not a known BitGenerator module."
    )
