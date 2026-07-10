# -*- coding: utf-8 -*-
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.database import Base, Command, Gesture, GestureSample, RecognitionModel, Settings
from app.services.binding_settings import KEY_CONFIDENCE_THRESHOLD, KEY_COOLDOWN_MS
from scripts.seed_database import ACTIVE_MODEL_NAME, seed_session


def _make_demo_files(tmp_path):
    models_dir = tmp_path / "models"
    gestures_dir = tmp_path / "gestures"
    models_dir.mkdir()
    gestures_dir.mkdir()

    (models_dir / "classes.json").write_text(
        json.dumps(["new", "new2"], ensure_ascii=False),
        encoding="utf-8",
    )
    (models_dir / "feature_dim.txt").write_text("42", encoding="utf-8")
    (models_dir / "knn.pkl").write_bytes(b"demo")

    for label, count in (("New", 2), ("new2", 1), ("zoom", 1)):
        label_dir = gestures_dir / label
        label_dir.mkdir()
        for idx in range(count):
            (label_dir / f"sample_{idx:04d}.npy").write_bytes(b"sample")

    return models_dir, gestures_dir


def test_seed_session_creates_demo_model_gestures_and_bindings(tmp_path):
    models_dir, gestures_dir = _make_demo_files(tmp_path)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        stale = Gesture(label="zoom", samples_path="data/gestures/zoom", model_class_id=0)
        session.add(stale)
        session.commit()

        summary = seed_session(session, models_dir=models_dir, gestures_dir=gestures_dir)

        assert summary["classes"] == ["new", "new2"]
        assert summary["gestures"]["created"] == 2
        assert summary["gestures"]["stale"] == 1
        assert summary["gestures"]["samples"] == 3

        new = session.query(Gesture).filter_by(label="new").one()
        new2 = session.query(Gesture).filter_by(label="new2").one()
        zoom = session.query(Gesture).filter_by(label="zoom").one()
        assert new.model_class_id == 0
        assert new2.model_class_id == 1
        assert "New" in str(new.samples_path)
        assert zoom.model_class_id is None

        assert session.query(GestureSample).count() == 3

        telegram = session.query(Command).filter_by(name="Open telegram").one()
        browser = session.query(Command).filter_by(name="open browser S").one()
        assert telegram.gesture_id == new.id
        assert browser.gesture_id == new2.id
        assert json.loads(telegram.action_spec)["app"] == "Telegram"
        assert json.loads(browser.action_spec)["app"] == "Safari"

        model = session.query(RecognitionModel).filter_by(name=ACTIVE_MODEL_NAME).one()
        assert model.is_active is True
        assert model.feature_dim == 42
        assert model.n_classes == 2
        assert model.n_samples == 3

        assert (
            session.query(Settings)
            .filter_by(key=KEY_CONFIDENCE_THRESHOLD)
            .one()
            .value
            == "0.6500"
        )
        assert session.query(Settings).filter_by(key=KEY_COOLDOWN_MS).one().value == "1500"

        second = seed_session(session, models_dir=models_dir, gestures_dir=gestures_dir)
        assert second["gestures"]["created"] == 0
        assert second["gestures"]["samples"] == 0
        assert session.query(Gesture).filter_by(label="new").count() == 1
        assert session.query(Command).filter_by(name="Open telegram").count() == 1
        assert session.query(RecognitionModel).filter_by(name=ACTIVE_MODEL_NAME).count() == 1
    finally:
        session.close()
