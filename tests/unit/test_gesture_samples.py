# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.database import Base, Gesture, GestureSample, RecognitionModel
from app.services.gesture_samples import (
    assign_gesture_model_class_ids,
    load_dataset_from_db,
    record_gesture_sample,
    register_recognition_model,
    sync_gesture_dataset_to_db,
)


def test_record_gesture_sample_upserts_metadata(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        sample_path = tmp_path / "sample_0000.npy"
        np.save(sample_path, np.zeros((30, 21, 2), dtype=np.float32))

        row = record_gesture_sample(
            session,
            label="pinch",
            sample_index=0,
            features_path=sample_path,
            frames=30,
            hand_count=1,
            source="test",
        )

        assert session.query(Gesture).filter_by(label="pinch").one().id == row.gesture_id
        assert session.query(GestureSample).count() == 1
        assert row.frames == 30
        assert row.hand_count == 1
    finally:
        session.close()


def test_load_dataset_from_db(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        data_root = tmp_path / "gestures"
        for label in ("a", "b"):
            label_dir = data_root / label
            label_dir.mkdir(parents=True)
            for idx in range(2):
                sample_path = label_dir / f"sample_{idx:04d}.npy"
                np.save(sample_path, np.zeros((30, 21, 2), dtype=np.float32))
                record_gesture_sample(
                    session,
                    label=label,
                    sample_index=idx,
                    features_path=sample_path,
                    frames=30,
                    hand_count=1,
                    source="test",
                )

        x, y, classes = load_dataset_from_db(session)
        assert x.shape == (4, 42)
        assert y.tolist() == [0, 0, 1, 1]
        assert classes == ["a", "b"]
    finally:
        session.close()


def test_assign_gesture_model_class_ids(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        data_root = tmp_path / "gestures"
        for label in ("a", "b"):
            label_dir = data_root / label
            label_dir.mkdir(parents=True)
            for idx in range(2):
                np.save(label_dir / f"sample_{idx:04d}.npy", np.zeros((30, 21, 2), dtype=np.float32))

        synced = sync_gesture_dataset_to_db(session, data_root=data_root, classes=["a", "b"])
        updated = assign_gesture_model_class_ids(session, ["a", "b"])
        register_recognition_model(
            session,
            name="test-knn",
            algorithm="knn",
            model_path=Path("models/knn.pkl"),
            classes_path=Path("models/classes.json"),
            feature_dim=42,
            n_classes=2,
            n_samples=synced,
        )

        assert synced == 4
        assert updated == 2
        assert session.query(Gesture).count() == 2
        assert session.query(GestureSample).count() == 4
        model = session.query(RecognitionModel).one()
        assert model.name == "test-knn"
        assert model.is_active is True
        assert model.n_samples == 4
    finally:
        session.close()
