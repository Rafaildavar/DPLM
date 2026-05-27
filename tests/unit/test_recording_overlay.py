import numpy as np

from cv.recording_overlay import draw_recording_overlay


def test_draw_recording_overlay_preserves_frame_and_adds_panels() -> None:
    frame = np.full((720, 1280, 3), 170, dtype=np.uint8)

    rendered = draw_recording_overlay(
        frame.copy(),
        label="Свайп влево",
        saved=3,
        target_samples=20,
        recording=True,
        frame_count=14,
        sequence_length=30,
        hands_count=1,
    )

    assert rendered.shape == frame.shape
    assert not np.array_equal(rendered[:116], frame[:116])
    assert not np.array_equal(rendered[-126:], frame[-126:])


def test_draw_recording_overlay_shows_countdown_state() -> None:
    frame = np.full((720, 1280, 3), 170, dtype=np.uint8)

    rendered = draw_recording_overlay(
        frame.copy(),
        label="Клик",
        saved=0,
        target_samples=10,
        recording=False,
        frame_count=0,
        sequence_length=30,
        hands_count=1,
        countdown_seconds=3,
    )

    assert not np.array_equal(rendered[270:450, 560:720], frame[270:450, 560:720])
