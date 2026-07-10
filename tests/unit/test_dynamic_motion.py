import numpy as np

from cv.dynamic_motion import (
    DynamicMotionSegmenter,
    canonical_dynamic_sequence,
    create_runtime_dynamic_segmenter,
    normalize_global_trajectory,
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


def test_global_trajectory_is_position_and_amplitude_invariant():
    base = np.stack([_frame(x, 0.5) for x in np.linspace(0.8, 0.3, 20)])
    transformed = base.copy()
    transformed[:, -2:] = (
        np.asarray([0.2, 0.75], dtype=np.float32) + (base[:, -2:] - base[0, -2:]) * 0.2
    )

    base_normalized = canonical_dynamic_sequence(base, target_frames=36)
    transformed_normalized = canonical_dynamic_sequence(
        transformed,
        target_frames=36,
    )

    assert np.allclose(base_normalized, transformed_normalized, atol=1e-5)
    assert np.allclose(
        normalize_global_trajectory(base_normalized),
        base_normalized,
        atol=1e-6,
    )


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
    assert completed[0].completion_evidence is not None
    assert completed[0].completion_evidence["raw_displacement"] > 0.45
    motion = trajectory_features(completed[0].completed_sequence, target_dim=44)
    assert motion[0] < -0.4


def test_segmenter_emits_swipe_after_short_velocity_drop_grace():
    segmenter = DynamicMotionSegmenter(cooldown_frames=0)
    points = [
        0.80,
        0.78,
        0.74,
        0.68,
        0.61,
        0.54,
        0.47,
        0.40,
        0.34,
        0.31,
        0.30,
        0.30,
        0.30,
        0.30,
    ]

    completed = [
        update
        for x in points
        if (update := segmenter.update(_frame(x, 0.5))).completed_sequence
        is not None
    ]

    assert len(completed) == 1
    assert completed[0].end_reason == "velocity_drop"
    motion = trajectory_features(completed[0].completed_sequence, target_dim=44)
    assert motion[0] < -0.4


def test_segmenter_does_not_split_compound_gesture_on_corner_pause():
    segmenter = DynamicMotionSegmenter(cooldown_frames=0)
    points = (
        [(0.50, 0.82), (0.50, 0.76), (0.50, 0.68), (0.50, 0.60)]
        + [(0.50, 0.54), (0.50, 0.53)]
        + [(0.44, 0.53), (0.38, 0.53), (0.32, 0.53), (0.30, 0.53)]
        + [(0.30, 0.53), (0.30, 0.53), (0.30, 0.53), (0.30, 0.53)]
    )

    updates = [segmenter.update(_frame(x, y)) for x, y in points]
    completion_indices = [
        index for index, update in enumerate(updates) if update.completed_sequence is not None
    ]

    assert completion_indices == [len(points) - 1]
    completed = updates[completion_indices[0]].completed_sequence
    assert completed is not None
    motion = trajectory_features(completed, target_dim=44)
    assert motion[0] < -0.20
    assert motion[1] < -0.20


def test_segmenter_emits_swipe_when_hand_leaves_after_motion():
    segmenter = DynamicMotionSegmenter(cooldown_frames=0)
    points = [0.80, 0.76, 0.70, 0.64, 0.58, 0.52, 0.46, 0.40, 0.34, 0.30]

    updates = [segmenter.update(_frame(x, 0.5)) for x in points]
    hand_lost = segmenter.finish_due_to_hand_lost()

    assert all(update.completed_sequence is None for update in updates)
    assert hand_lost.completed_sequence is not None
    assert hand_lost.end_reason == "hand_lost"
    motion = trajectory_features(hand_lost.completed_sequence, target_dim=44)
    assert motion[0] < -0.4


def test_online_segmenter_emits_fast_horizontal_swipe_before_hand_leaves():
    from app.gesture_online_infer import GestureOnlineInfer

    segmenter = GestureOnlineInfer._create_dynamic_segmenter(target_frames=36)
    points = [0.80, 0.72, 0.64, 0.56, 0.48]

    updates = [
        segmenter.update(_frame(x, 0.5), motion_scale=0.50)
        for x in points
    ]
    hand_lost = segmenter.finish_due_to_hand_lost()

    assert all(update.completed_sequence is None for update in updates)
    assert hand_lost.completed_sequence is not None
    assert hand_lost.end_reason == "hand_lost"
    motion = trajectory_features(hand_lost.completed_sequence, target_dim=44)
    assert motion[0] < -0.4


def test_runtime_segmenter_waits_five_frames_before_tentative_completion():
    segmenter = create_runtime_dynamic_segmenter(target_frames=72)

    assert segmenter.completion_grace_frames == 5


def test_rejected_prefix_waits_and_then_resumes_same_gesture():
    segmenter = create_runtime_dynamic_segmenter(target_frames=72)
    first_leg = [0.80, 0.72, 0.64, 0.56, 0.48] + [0.48] * 8
    first_updates = [
        segmenter.update(_frame(x, 0.5), motion_scale=0.50)
        for x in first_leg
    ]
    rejected = next(
        update for update in first_updates if update.completed_sequence is not None
    )
    assert rejected.raw_sequence is not None

    segmenter.reject_completed_candidate(
        rejected.raw_sequence,
        motion_scale=0.50,
    )
    waiting = [
        segmenter.update(_frame(0.48, 0.5), motion_scale=0.50)
        for _ in range(6)
    ]

    assert all(update.phase == "awaiting_continuation" for update in waiting)
    assert all(update.completed_sequence is None for update in waiting)

    second_leg = [0.40, 0.32, 0.24, 0.20] + [0.20] * 8
    resumed = [
        segmenter.update(_frame(x, 0.5), motion_scale=0.50)
        for x in second_leg
    ]
    completed = [
        update for update in resumed if update.completed_sequence is not None
    ]

    assert len(completed) == 1
    assert completed[0].raw_sequence is not None
    motion = trajectory_features(completed[0].raw_sequence, target_dim=44)
    assert motion[0] < -0.55


def test_confirmed_returning_motion_can_finish_near_its_start():
    segmenter = create_runtime_dynamic_segmenter(target_frames=72)
    points = [
        0.50,
        0.54,
        0.61,
        0.69,
        0.75,
        0.69,
        0.61,
        0.54,
        0.50,
        *([0.50] * 9),
    ]

    updates = [
        segmenter.update(_frame(x, 0.5), motion_scale=0.50)
        for x in points
    ]
    completed = [
        update for update in updates if update.completed_sequence is not None
    ]

    assert len(completed) == 1
    assert completed[0].end_reason in {"still", "velocity_drop"}
    assert completed[0].frames < segmenter.max_active_frames


def test_rejected_prefix_times_out_without_emitting_a_command():
    segmenter = create_runtime_dynamic_segmenter(target_frames=72)
    first_leg = [0.80, 0.72, 0.64, 0.56, 0.48] + [0.48] * 8
    rejected = next(
        update
        for x in first_leg
        if (
            update := segmenter.update(_frame(x, 0.5), motion_scale=0.50)
        ).completed_sequence
        is not None
    )
    assert rejected.raw_sequence is not None
    segmenter.reject_completed_candidate(
        rejected.raw_sequence,
        motion_scale=0.50,
    )

    waiting = [
        segmenter.update(_frame(0.48, 0.5), motion_scale=0.50)
        for _ in range(segmenter.continuation_wait_frames)
    ]

    assert all(update.completed_sequence is None for update in waiting)
    assert waiting[-1].end_reason == "completion_timeout"


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


def test_segmenter_uses_hand_scale_for_far_camera_motion():
    segmenter = DynamicMotionSegmenter()
    points = [0.8] * 6 + list(np.linspace(0.8, 0.72, 14)) + [0.72] * 6

    completed = [
        update.completed_sequence
        for x in points
        if (
            update := segmenter.update(
                _frame(x, 0.5),
                motion_scale=0.08,
            )
        ).completed_sequence
        is not None
    ]

    assert len(completed) == 1
    motion = trajectory_features(completed[0], target_dim=44)
    assert np.isclose(motion[0], -0.5, atol=0.01)


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
