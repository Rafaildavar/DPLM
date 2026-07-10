import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.gesture_online_infer import GestureOnlineInfer
from cv.dynamic_completion import (
    build_dynamic_completion_evidence,
    build_dynamic_completion_profiles,
    completion_progress_prefix,
    evaluate_completion_profile,
)
from cv.dynamic_motion import canonical_dynamic_sequence
from cv.gesture_features import global_trajectory_xy


def _sequence(
    *,
    kind: str,
    variant: int,
    frames: int = 48,
) -> np.ndarray:
    sequence = np.zeros((frames, 65), dtype=np.float32)
    progress = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    points = np.zeros((frames, 21, 3), dtype=np.float32)
    base_x = np.linspace(-0.45, 0.45, 21, dtype=np.float32)
    base_y = np.linspace(0.0, -0.9, 21, dtype=np.float32)
    points[:, :, 0] = base_x[None, :]
    points[:, :, 1] = base_y[None, :]

    jitter = (variant - 3.5) * 0.002
    if kind == "translation":
        sequence[:, -2] = 0.82 - progress * (0.48 + jitter)
        sequence[:, -1] = 0.50 + 0.01 * np.sin(progress * np.pi)
        points[:, 8, 0] += progress * (0.04 + jitter)
    else:
        sequence[:, -2] = 0.50 + (0.12 + abs(jitter)) * np.sin(
            progress * np.pi * 2.0
        )
        sequence[:, -1] = 0.55
        for tip in (4, 8, 12, 16, 20):
            direction = -1.0 if tip in (4, 8) else 1.0
            points[:, tip, 0] += direction * progress * (0.35 + jitter)
            points[:, tip, 1] -= progress * (0.15 + jitter)

    sequence[:, :63] = points.reshape(frames, 63)
    return sequence


def _write_class(root: Path, label: str, kind: str) -> list[np.ndarray]:
    label_dir = root / label
    label_dir.mkdir(parents=True)
    sequences: list[np.ndarray] = []
    for index in range(8):
        sequence = _sequence(kind=kind, variant=index)
        path = label_dir / f"sample_{index:04d}.npy"
        np.save(path, sequence)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"projected_hand_scale_median": 0.50}),
            encoding="utf-8",
        )
        sequences.append(sequence)
    return sequences


def test_raw_completion_evidence_survives_amplitude_normalization() -> None:
    full = _sequence(kind="translation", variant=0)
    partial = full[:12]

    full_evidence = build_dynamic_completion_evidence(full, motion_scale=0.50)
    partial_evidence = build_dynamic_completion_evidence(
        partial,
        motion_scale=0.50,
    )
    full_canonical = canonical_dynamic_sequence(full, target_frames=72)
    partial_canonical = canonical_dynamic_sequence(partial, target_frames=72)

    assert full_evidence["relative_displacement"] > 0.80
    assert partial_evidence["relative_displacement"] < 0.30
    assert np.isclose(
        np.linalg.norm(
            global_trajectory_xy(full_canonical)[-1]
            - global_trajectory_xy(full_canonical)[0]
        ),
        0.50,
        atol=0.02,
    )
    assert np.isclose(
        np.linalg.norm(
            global_trajectory_xy(partial_canonical)[-1]
            - global_trajectory_xy(partial_canonical)[0]
        ),
        0.50,
        atol=0.02,
    )


def test_profiles_are_trained_for_arbitrary_user_labels(tmp_path: Path) -> None:
    data_root = tmp_path / "gestures"
    translation = _write_class(data_root, "user_alpha", "translation")
    shape = _write_class(data_root, "my_custom_shape", "shape")

    payload = build_dynamic_completion_profiles(
        data_root,
        ["user_alpha", "my_custom_shape", "partial_swipe"],
        target_dim=65,
        target_frames=72,
    )

    assert set(payload["classes"]) == {"user_alpha", "my_custom_shape"}
    assert payload["classes"]["user_alpha"]["validation"]["grouped_cv"] is True
    assert payload["classes"]["user_alpha"]["validation"]["source_groups"] == 16
    assert payload["classes"]["user_alpha"]["negative_sample_counts"] == {
        "cross_class_full": 8,
        "cross_class_prefix": 32,
        "own_prefix": 32,
    }
    assert payload["classes"]["my_custom_shape"]["validation"][
        "complete_recall"
    ] >= 0.95

    for label, sequences in (
        ("user_alpha", translation),
        ("my_custom_shape", shape),
    ):
        profile = payload["classes"][label]
        full_results = []
        partial_results = []
        for sequence in sequences:
            full_results.append(
                evaluate_completion_profile(
                    profile,
                    build_dynamic_completion_evidence(
                        sequence,
                        motion_scale=0.50,
                        target_frames=72,
                    ),
                )
            )
            prefix = completion_progress_prefix(
                sequence,
                0.40,
                motion_scale=0.50,
            )
            partial_results.append(
                evaluate_completion_profile(
                    profile,
                    build_dynamic_completion_evidence(
                        prefix,
                        motion_scale=0.50,
                        target_frames=72,
                    ),
                )
            )
        assert sum(result["accepted"] for result in full_results) >= 7
        assert sum(result["accepted"] for result in partial_results) <= 1

    alpha_profile = payload["classes"]["user_alpha"]
    foreign_prefix_results = [
        evaluate_completion_profile(
            alpha_profile,
            build_dynamic_completion_evidence(
                completion_progress_prefix(
                    sequence,
                    0.70,
                    motion_scale=0.50,
                ),
                motion_scale=0.50,
                target_frames=72,
            ),
        )
        for sequence in shape
    ]
    assert sum(result["accepted"] for result in foreign_prefix_results) <= 1


def test_missing_profile_is_backward_compatible() -> None:
    result = evaluate_completion_profile(None, {})

    assert result["enabled"] is False
    assert result["accepted"] is True
    assert result["reason"] == "profile_missing"


def test_online_infer_suppresses_incomplete_candidate(monkeypatch) -> None:
    infer = GestureOnlineInfer.__new__(GestureOnlineInfer)
    infer._window = deque(maxlen=72)
    infer._classes = ["custom_motion"]
    infer._clf = object()
    infer._last_dynamic_decision = {"source": "model_prediction"}
    infer._pending_dynamic_prediction = None
    infer._pending_dynamic_repeats = 0
    infer._pending_dynamic_motion_scale = 0.0
    infer._hand_lost_grace_frames = 2
    infer._hand_lost_streak = 0
    rejected: dict[str, object] = {}
    infer._dynamic_segmenter = SimpleNamespace(
        reject_completed_candidate=lambda sequence, **kwargs: rejected.update(
            sequence=sequence,
            **kwargs,
        )
    )

    monkeypatch.setattr(infer, "_dynamic_motion_gate", lambda: (True, {}))
    monkeypatch.setattr(
        infer,
        "_build_model_feature",
        lambda: np.zeros(4, dtype=np.float32),
    )
    monkeypatch.setattr(
        infer,
        "_dynamic_prediction",
        lambda *_args, **_kwargs: ("custom_motion", 0.99),
    )
    monkeypatch.setattr(infer, "_registered_dynamic_label", lambda label: label)
    monkeypatch.setattr(infer, "_intent_feature_payload", lambda *_args: [])
    monkeypatch.setattr(infer, "_with_hand_tracking", lambda payload: payload)
    monkeypatch.setattr(
        infer,
        "_dynamic_completion_assessment",
        lambda _label, _evidence: {
            "enabled": True,
            "accepted": False,
            "score": 0.22,
            "threshold": 0.80,
            "reason": "incomplete_gesture",
        },
    )
    update = SimpleNamespace(
        phase="completed",
        frames=18,
        end_reason="still",
        completed_sequence=np.zeros((72, 65), dtype=np.float32),
        completion_evidence={
            "relative_displacement": 0.18,
            "motion_scale": 0.50,
        },
        raw_sequence=np.zeros((18, 65), dtype=np.float32),
    )

    out = infer._classify_completed_dynamic_update(
        update,
        "[]",
        motion_scale=0.0,
    )

    assert out["label"] == ""
    assert out["confidence"] == 0.0
    assert out["dynamic_decision"]["source"] == "completion_rejected"
    assert out["dynamic_decision"]["completion_original_source"] == (
        "model_prediction"
    )
    assert np.asarray(rejected["sequence"]).shape == (18, 65)
    assert rejected["motion_scale"] == 0.50
    assert out["temporal"]["completion"]["accepted"] is False


def test_completion_timeout_clears_stale_raw_intent(monkeypatch) -> None:
    infer = GestureOnlineInfer.__new__(GestureOnlineInfer)
    infer._window = deque([np.ones(65, dtype=np.float32)], maxlen=72)
    infer._intent_window = deque(
        [np.ones(65, dtype=np.float32)],
        maxlen=88,
    )
    infer._last_dynamic_decision = {"source": "completion_rejected"}
    infer._hand_lost_grace_frames = 2
    infer._hand_lost_streak = 0
    infer._dynamic_segmenter = SimpleNamespace(
        update=lambda *_args, **_kwargs: SimpleNamespace(
            phase="idle",
            frames=0,
            end_reason="completion_timeout",
            completed_sequence=None,
            completion_evidence=None,
        )
    )
    monkeypatch.setattr(infer, "_intent_feature_payload", lambda *_args: [])
    monkeypatch.setattr(infer, "_with_hand_tracking", lambda payload: payload)

    out = infer._process_segmented_dynamic_frame(
        np.zeros(65, dtype=np.float32),
        "[]",
        motion_scale=0.50,
    )

    assert out["temporal"]["end_reason"] == "completion_timeout"
    assert list(infer._window) == []
    assert list(infer._intent_window) == []
    assert infer._last_dynamic_decision == {}
