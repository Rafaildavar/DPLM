from app.services.live_gesture_state import (
    PHASE_CONFIRMED,
    PHASE_IDLE,
    PHASE_PENDING,
    PHASE_REJECTED,
    LiveGestureState,
)


def test_live_gesture_state_confirms_after_required_frames():
    state = LiveGestureState()

    confirmed, confidence, snapshot = state.observe(
        "swipe_down",
        0.8,
        required_frames=3,
        route="dynamic",
    )

    assert confirmed is False
    assert confidence == 0.8
    assert snapshot.phase == PHASE_PENDING
    assert snapshot.progress == 1 / 3
    assert snapshot.as_dict()["displayText"] == "swipe_down 1/3"

    state.observe("swipe_down", 0.7, required_frames=3, route="dynamic")
    confirmed, confidence, snapshot = state.observe(
        "swipe_down",
        0.9,
        required_frames=3,
        route="dynamic",
    )

    assert confirmed is True
    assert confidence == (0.8 + 0.7 + 0.9) / 3
    assert snapshot.phase == PHASE_CONFIRMED
    assert snapshot.progress == 1.0
    assert snapshot.route == "dynamic"


def test_live_gesture_state_switches_candidate_and_resets_on_empty_label():
    state = LiveGestureState()

    state.observe("new", 0.8, required_frames=4)
    confirmed, confidence, snapshot = state.observe("new2", 0.6, required_frames=4)

    assert confirmed is False
    assert confidence == 0.6
    assert snapshot.label == "new2"
    assert snapshot.frames == 1
    assert state.pending_label == "new2"

    confirmed, confidence, snapshot = state.observe("", 0.0, required_frames=4)

    assert confirmed is False
    assert confidence == 0.0
    assert snapshot.phase == PHASE_IDLE
    assert state.pending_frames == 0


def test_live_gesture_state_marks_rejection_for_ui():
    state = LiveGestureState()
    state.observe("random_motion", 0.95, required_frames=1)

    snapshot = state.mark_rejected(
        "random_motion",
        0.95,
        reason="negative_label",
        route="dynamic",
    )

    assert snapshot.phase == PHASE_REJECTED
    assert snapshot.reason == "negative_label"
    assert snapshot.as_dict()["displayText"] == "random_motion rejected"
