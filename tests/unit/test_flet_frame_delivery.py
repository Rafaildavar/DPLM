import threading

import app.flet_app.views.home as home_view
from app.flet_app.views.home import HomeView
from app.flet_app.views.training import TrainingView


class _QueuedPage:
    def __init__(self):
        self.calls = []
        self.update_calls = 0

    def run_thread(self, func, *args):
        self.calls.append((func, args))

    def update(self):
        self.update_calls += 1


class _Controller:
    latest_jpeg_bytes = b"jpeg"


class _Image:
    def __init__(self):
        self.src = ""
        self.visible = False
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


class _Text:
    def __init__(self, value=""):
        self.value = value
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


class _GestureController:
    is_recognizing = True
    live_recognition_active = True


def _frame_view(view_type):
    view = object.__new__(view_type)
    view._page = _QueuedPage()
    view._controller = _Controller()
    view._visible = True
    view._frame_update_lock = threading.Lock()
    view._frame_update_pending = False
    view._camera_image = _Image()
    return view


def _gesture_view():
    view = object.__new__(HomeView)
    view._page = _QueuedPage()
    view._controller = _GestureController()
    view._gesture_text = _Text("—")
    view._gesture_clear_lock = threading.Lock()
    view._gesture_clear_token = 0
    return view


def test_home_coalesces_frames_while_ui_update_is_pending():
    view = _frame_view(HomeView)

    view._on_frame()
    view._on_frame()

    assert len(view._page.calls) == 1
    callback, args = view._page.calls.pop()
    callback(*args)
    assert view._frame_update_pending is False
    assert view._camera_image.visible is True


def test_hidden_training_view_does_not_enqueue_camera_frames():
    view = _frame_view(TrainingView)
    view._visible = False

    view._on_frame()

    assert view._page.calls == []


def test_home_holds_last_gesture_prediction_before_clearing(monkeypatch):
    timers = []

    class FakeTimer:
        daemon = False

        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

    monkeypatch.setattr(home_view.threading, "Timer", FakeTimer)
    view = _gesture_view()

    view._apply_gesture("swipe_down")
    view._apply_gesture("")

    assert view._gesture_text.value == "swipe_down"
    assert timers[0].delay == home_view.LIVE_PREDICTION_HOLD_SECONDS

    timers[0].callback()
    callback, args = view._page.calls.pop()
    callback(*args)

    assert view._gesture_text.value == "—"
