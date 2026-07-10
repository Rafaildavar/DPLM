# -*- coding: utf-8 -*-

import base64
import json

import numpy as np

from app.flet_app.views.gestures import (
    ANIMATED_PREVIEW_ENV,
    DatasetGesturesView,
    GesturesView,
)


class _Controller:
    def get_db_gestures(self):
        return []

    def list_recorded_gestures(self):
        return []

    def sync_dataset_to_db(self):
        return {"created": 0, "updated": 0, "total": 0, "samples": 0}


def _visible_text(control):
    parts = []
    seen = set()

    def visit(obj):
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(obj, (str, int, float, bool)):
            parts.append(str(obj))
            return
        for attr in ("value", "label", "text", "tooltip", "key"):
            value = getattr(obj, attr, None)
            if isinstance(value, (str, int, float, bool)):
                parts.append(str(value))
        visit(getattr(obj, "content", None))
        for attr in ("controls", "options"):
            for child in getattr(obj, attr, []) or []:
                visit(child)

    visit(control)
    return "\n".join(parts)


def _dynamic_sample(path):
    seq = np.zeros((12, 44), dtype=np.float32)
    base = np.asarray(
        [[0.38 + (idx % 4) * 0.04, 0.72 - (idx // 4) * 0.05] for idx in range(21)],
        dtype=np.float32,
    )
    for frame_idx in range(seq.shape[0]):
        pose = base.copy()
        pose[:, 0] += frame_idx * 0.01
        seq[frame_idx, :42] = pose.reshape(-1)
        seq[frame_idx, 42:44] = [0.25 + frame_idx * 0.035, 0.55]
    np.save(path, seq)


def test_dynamic_preview_is_static_png_by_default(monkeypatch, tmp_path):
    sample_path = tmp_path / "sample_0000.npy"
    _dynamic_sample(sample_path)
    view = GesturesView(object(), _Controller())
    monkeypatch.delenv(ANIMATED_PREVIEW_ENV, raising=False)
    monkeypatch.setattr(
        view,
        "_sample_gif_base64",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("gif used")),
    )

    preview = view._gesture_preview_src(
        {"samplePreviewPath": str(sample_path), "gestureType": "dynamic"}
    )

    assert preview is not None
    assert base64.b64decode(preview).startswith(b"\x89PNG")


def test_dynamic_preview_gif_requires_opt_in(monkeypatch, tmp_path):
    sample_path = tmp_path / "sample_0000.npy"
    _dynamic_sample(sample_path)
    view = GesturesView(object(), _Controller())
    monkeypatch.setenv(ANIMATED_PREVIEW_ENV, "1")
    monkeypatch.setattr(view, "_sample_gif_base64", lambda *_args, **_kwargs: "gif")

    preview = view._gesture_preview_src(
        {"samplePreviewPath": str(sample_path), "gestureType": "dynamic"}
    )

    assert preview == "gif"


def test_static_xyz_sample_type_is_not_misclassified_as_dynamic(tmp_path):
    sample_path = tmp_path / "sample_0000.npy"
    np.save(sample_path, np.zeros((30, 63), dtype=np.float32))
    sample_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "raw_feature_dim": 63,
                "include_global_motion": False,
                "include_landmark_z": True,
                "sample_feature_format": "landmark_xyz",
            }
        ),
        encoding="utf-8",
    )
    view = GesturesView(object(), _Controller())

    gesture_type = view._gesture_type(
        {"label": "custom_static", "samplePreviewPath": str(sample_path)}
    )

    assert gesture_type == "static"


def test_dynamic_xyz_wrist_sample_type_stays_dynamic(tmp_path):
    sample_path = tmp_path / "sample_0000.npy"
    np.save(sample_path, np.zeros((72, 65), dtype=np.float32))
    sample_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "raw_feature_dim": 65,
                "include_global_motion": True,
                "include_landmark_z": True,
                "sample_feature_format": "landmark_xyz_wrist_xy",
            }
        ),
        encoding="utf-8",
    )
    view = GesturesView(object(), _Controller())

    gesture_type = view._gesture_type(
        {"label": "custom_dynamic", "samplePreviewPath": str(sample_path)}
    )

    assert gesture_type == "dynamic"


def test_gesture_library_copy_is_user_recorded_not_import():
    view = GesturesView(object(), _Controller())
    text = _visible_text(view.build())

    assert "стандартные и свои жесты" in text
    assert "Данные жестов" not in text
    assert "Импорт" not in text
    assert "после записи на вкладке" in text


def test_dataset_gestures_view_owns_dataset_manager_copy():
    view = DatasetGesturesView(object(), _Controller())
    text = _visible_text(view.build())

    assert "Данные жестов" in text
    assert "классы, записи и пользовательские сэмплы" in text
    assert "Gesture Library" not in text


def test_dataset_manager_hides_system_classes_and_marks_standard_gestures():
    view = DatasetGesturesView(object(), _Controller())
    view._dataset_rows = [
        {
            "label": "open_hand",
            "samples": 12,
            "realSamples": 12,
            "augmentedSamples": 0,
            "userRecordedSamples": 0,
            "canDelete": False,
            "systemClass": False,
        },
        {
            "label": "my_wave",
            "samples": 5,
            "realSamples": 5,
            "augmentedSamples": 5,
            "userRecordedSamples": 5,
            "canDelete": True,
            "systemClass": False,
        },
        {
            "label": "no_gesture_static",
            "samples": 20,
            "realSamples": 20,
            "augmentedSamples": 0,
            "userRecordedSamples": 0,
            "canDelete": False,
            "systemClass": True,
        },
    ]

    view._render_dataset_manager()
    text = _visible_text(view._dataset_column)

    assert "open_hand" in text
    assert "стандартный" in text
    assert "my_wave" in text
    assert "мой" in text
    assert "no_gesture_static" not in text
    assert view._dataset_standard.value == "1"
    assert view._dataset_custom.value == "1"
