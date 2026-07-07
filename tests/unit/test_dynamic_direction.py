from cv.dynamic_direction import classify_swipe_direction


def test_classify_swipe_direction_prefers_completed_motion_over_pose() -> None:
    decision = classify_swipe_direction(
        {
            "dx": -0.50,
            "dy": 0.03,
            "path_length": 0.55,
            "displacement": 0.50,
        },
        ["swipe_down", "swipe_left", "swipe_up"],
    )

    assert decision.accepted is True
    assert decision.label == "swipe_left"
    assert decision.axis == "horizontal"
    assert decision.direction == "left"
    assert decision.confidence >= 0.80


def test_classify_swipe_direction_returns_trained_compact_label() -> None:
    decision = classify_swipe_direction(
        {
            "dx": -0.50,
            "dy": 0.03,
            "path_length": 0.55,
            "displacement": 0.50,
        },
        ["SwipeLeft"],
    )

    assert decision.accepted is True
    assert decision.label == "SwipeLeft"


def test_classify_swipe_direction_rejects_ambiguous_diagonal_motion() -> None:
    decision = classify_swipe_direction(
        {
            "dx": -0.32,
            "dy": -0.30,
            "path_length": 0.48,
            "displacement": 0.44,
        },
        ["swipe_left", "swipe_up"],
    )

    assert decision.accepted is False
    assert decision.reason == "ambiguous_axis"


def test_classify_swipe_direction_requires_available_runtime_class() -> None:
    decision = classify_swipe_direction(
        {
            "dx": 0.0,
            "dy": -0.50,
            "path_length": 0.50,
            "displacement": 0.50,
        },
        ["swipe_left"],
    )

    assert decision.accepted is False
    assert decision.reason == "missing_direction_label"
    assert decision.direction == "up"
