import threading

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


def _frame_view(view_type):
    view = object.__new__(view_type)
    view._page = _QueuedPage()
    view._controller = _Controller()
    view._visible = True
    view._frame_update_lock = threading.Lock()
    view._frame_update_pending = False
    view._camera_image = _Image()
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
