import numpy as np

from cv.dynamic_motion import (
    DynamicMotionSegmenter,
    canonical_dynamic_sequence,
    resample_sequence,
)
from cv.gesture_features import trajectory_features


def _frame(x: float, y: float) -> np.ndarray:
    frame = np.zeros(44, dtype=np.float32)
    frame[-2:] = [x, y]
    return frame


def test_resample_sequence_preserves_endpoints():
    sequence = np.stack([_frame(0.8, 0.5), _frame(0.3, 0.5)])

    result = resample_sequence(sequence, target_frames=36)

    assert result.shape == (36, 44)
    assert np.allclose(result[0], sequence[0])
    assert np.allclose(result[-1], sequence[-1])


def test_canonical_dynamic_sequence_trims_static_edges():
    points = [0.8] * 5 + list(np.linspace(0.8, 0.3, 12)) + [0.3] * 7
    sequence = np.stack([_frame(x, 0.5) for x in points])

    result = canonical_dynamic_sequence(sequence, target_frames=36)
    motion = trajectory_features(result, target_dim=44)

    assert result.shape == (36, 44)
    assert motion[0] < -0.45
    assert abs(float(motion[1])) < 1e-6


def test_segmenter_emits_one_completed_swipe_after_motion_stops():
    segmenter = DynamicMotionSegmenter(cooldown_frames=4)
    updates = []
    points = [0.8] * 6 + list(np.linspace(0.8, 0.3, 14)) + [0.3] * 6
    for x in points:
        updates.append(segmenter.update(_frame(x, 0.5)))

    completed = [item for item in updates if item.completed_sequence is not None]

    assert len(completed) == 1
    assert completed[0].phase == "completed"
    assert completed[0].completed_sequence.shape == (36, 44)
    motion = trajectory_features(completed[0].completed_sequence, target_dim=44)
    assert motion[0] < -0.4


def test_segmenter_handles_different_gesture_speeds():
    for motion_frames in (8, 14, 28):
        segmenter = DynamicMotionSegmenter()
        points = [0.8] * 6 + list(np.linspace(0.8, 0.3, motion_frames)) + [0.3] * 6

        completed = [
            update.completed_sequence
            for x in points
            if (update := segmenter.update(_frame(x, 0.5))).completed_sequence
            is not None
        ]

        assert len(completed) == 1
        assert completed[0].shape == (36, 44)


def test_segmenter_does_not_emit_for_stationary_hand():
    segmenter = DynamicMotionSegmenter()

    updates = [segmenter.update(_frame(0.5, 0.5)) for _ in range(80)]

    assert all(item.completed_sequence is None for item in updates)
    assert updates[-1].phase == "idle"


def test_segmenter_cooldown_ignores_immediate_return_motion():
    segmenter = DynamicMotionSegmenter(cooldown_frames=8)
    outward = [0.8] * 6 + list(np.linspace(0.8, 0.3, 14)) + [0.3] * 6
    outward_updates = [segmenter.update(_frame(x, 0.5)) for x in outward]
    return_updates = [
        segmenter.update(_frame(x, 0.5)) for x in np.linspace(0.3, 0.8, 8)
    ]

    assert sum(item.completed_sequence is not None for item in outward_updates) == 1
    assert all(item.completed_sequence is None for item in return_updates)
    assert sum(item.phase == "cooldown" for item in return_updates) >= 4
