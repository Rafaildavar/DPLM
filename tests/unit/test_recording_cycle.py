from cv.recording_cycle import RecordingCycle


def test_recording_cycle_counts_down_and_stops_at_exact_length() -> None:
    cycle = RecordingCycle(sequence_length=3, countdown_seconds=3.0)

    cycle.start(10.0)
    assert cycle.countdown_value(10.2) == 3
    assert cycle.capture("too early", 12.9) == (False, False)
    assert cycle.frames == []

    assert cycle.capture("one", 13.0) == (True, False)
    assert cycle.capture("two", 13.1) == (False, False)
    assert cycle.capture("three", 13.2) == (False, True)
    assert cycle.ready
    assert not cycle.recording

    cycle.capture("extra", 13.3)
    assert cycle.frames == ["one", "two", "three"]


def test_recording_cycle_restarts_and_clears_previous_take() -> None:
    cycle = RecordingCycle(sequence_length=2, countdown_seconds=0.0)
    cycle.start(0.0)
    cycle.capture("old", 0.0)

    cycle.start(1.0)
    assert cycle.frames == []
    cycle.capture("new-1", 1.0)
    cycle.capture("new-2", 1.1)

    assert cycle.ready
    assert cycle.frames == ["new-1", "new-2"]
