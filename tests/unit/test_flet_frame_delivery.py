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
        self.color = None

    def update(self):
        self.update_calls += 1


class _Control:
    def __init__(self):
        self.controls = []
        self.icon = None
        self.tooltip = ""
        self.disabled = False
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


class _GestureController:
    is_recognizing = True
    live_recognition_active = True


class _InspectorController:
    def __init__(self, history):
        self.history = list(history)
        self.cleared = False
        self.exported = []

    def get_live_gesture_inspector_history(self):
        return [dict(row) for row in self.history]

    def clear_live_gesture_inspector_history(self):
        self.history = []
        self.cleared = True

    def export_live_gesture_inspector_history(self, fmt):
        self.exported.append(fmt)

        class _Path:
            name = f"live_gesture_inspector.{fmt}"

        return _Path()


class _RecognitionLabelsController:
    def list_recognition_labels(self):
        return ["FreshGesture"]


class _ToggleController:
    def __init__(self):
        self.toggle_calls = 0

    def toggle_recognition(self):
        self.toggle_calls += 1


class _Event:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _ReleaseController:
    def __init__(self):
        self.recognition_model_mode = "auto"
        self.model_variant = "production"
        self.dynamic_model_profile = "dynamic_landmark_lstm_backbone"
        self.static_rejection_method = "open_set_policy"
        self.two_hands_mode = False
        self.auto_execute = True
        self.gesture_mode = True
        self.show_landmark_overlay = True
        self.pointer_mode = False
        self.status = "Idle"
        self.is_camera_active = False
        self.is_recognizing = False
        self.live_recognition_active = False
        for name in (
            "camera_frame_updated",
            "gesture_detected",
            "command_executed",
            "recognition_event_recorded",
            "confidence_changed",
            "gesture_state_changed",
            "status_changed",
            "recognizing_changed",
            "camera_active_changed",
            "two_hands_changed",
            "gesture_mode_changed",
            "pointer_mode_changed",
            "pointer_state_changed",
            "landmark_overlay_changed",
            "recognition_model_mode_changed",
            "dynamic_model_profile_changed",
            "model_variant_changed",
            "static_rejection_method_changed",
            "live_evaluation_changed",
        ):
            setattr(self, name, _Event())

    def list_recognition_labels(self):
        return ["SwipeLeft", "diagonal", "zoom"]

    def list_model_variants(self):
        return [{"key": "production", "label": "production"}]

    def list_dynamic_model_profiles(self):
        return [
            {
                "key": "dynamic_landmark_lstm_backbone",
                "label": "dynamic_landmark_lstm_backbone",
                "quick_selectable": True,
            }
        ]


def _visible_text(control):
    parts = []
    seen = set()

    def visit(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        for attr in ("value", "label", "text", "tooltip", "key"):
            value = getattr(obj, attr, None)
            if isinstance(value, (str, int, float, bool)):
                parts.append(str(value))
        visit(getattr(obj, "content", None))
        for child in getattr(obj, "controls", []) or []:
            visit(child)

    visit(control)
    return "\n".join(parts)


def _contains_control(root, target):
    seen = set()

    def visit(obj):
        if obj is None or id(obj) in seen:
            return False
        if obj is target:
            return True
        seen.add(id(obj))
        if visit(getattr(obj, "content", None)):
            return True
        return any(visit(child) for child in getattr(obj, "controls", []) or [])

    return visit(root)


def test_home_release_layout_hides_developer_controls() -> None:
    controller = _ReleaseController()
    view = HomeView(_QueuedPage(), controller)

    root = view.build()
    text = _visible_text(root)

    assert "Управление жестами" in text
    assert "Главная" in text
    assert "Распознавание выключено" in text
    assert "Последнее действие" in text
    assert "Недавние действия" in text
    assert "Quick Settings" not in text
    assert "Recognition Inspector" not in text
    assert "Live Evaluation" not in text
    assert not _contains_control(root, view._model_mode_dd)
    assert not _contains_control(root, view._dynamic_profile_dd)


def test_home_release_state_follows_recognition_and_camera() -> None:
    controller = _ReleaseController()
    view = HomeView(_QueuedPage(), controller)
    view.build()
    controller.live_recognition_active = True
    controller.is_recognizing = True
    controller.is_camera_active = True

    view._apply_button_state()

    assert view._release_header_status.value == "Активно"
    assert view._release_state_title.value == "Распознавание включено"
    assert view._release_state_caption.value == "Ожидаю жест"
    assert view._camera_state_text.value == "Камера включена"
    assert view._camera_empty_state.visible is False
    assert view._toggle_btn.content.value == "Остановить"


def _frame_view(view_type):
    view = object.__new__(view_type)
    view._page = _QueuedPage()
    view._controller = _Controller()
    view._visible = True
    view._frame_update_lock = threading.Lock()
    view._frame_update_pending = False
    view._camera_image = _Image()
    return view


def test_home_expected_label_options_do_not_add_stale_swipe_fallbacks():
    view = object.__new__(HomeView)
    view._controller = _RecognitionLabelsController()

    labels = view._recognition_label_options()

    assert labels == ["no_command", "FreshGesture"]
    assert "swipe_down" not in labels
    assert "swipe_left" not in labels


def test_home_toggle_runs_recognition_change_off_ui_handler():
    view = object.__new__(HomeView)
    view._page = _QueuedPage()
    view._controller = _ToggleController()

    view._on_toggle(None)

    assert view._controller.toggle_calls == 0
    assert len(view._page.calls) == 1
    callback, args = view._page.calls.pop()
    callback(*args)
    assert view._controller.toggle_calls == 1


def _gesture_view():
    view = object.__new__(HomeView)
    view._page = _QueuedPage()
    view._controller = _GestureController()
    view._gesture_text = _Text("—")
    view._gesture_clear_lock = threading.Lock()
    view._gesture_clear_token = 0
    return view


def _inspector_row(sequence, phase="pending"):
    return {
        "sequence": sequence,
        "phase": phase,
        "label": f"gesture_{sequence}",
        "confidence": 0.5,
        "progress": 0.5,
        "frames": 1,
        "requiredFrames": 2,
        "mode": "auto",
        "route": "dynamic",
        "model": "sequence_mlp",
        "staticReject": "open_set_policy",
        "reason": "demo",
    }


def _inspector_view(history):
    view = object.__new__(HomeView)
    view._controller = _InspectorController(history)
    view._recognition_inspector_list = _Control()
    view._recognition_inspector_status_text = _Text("live")
    view._recognition_inspector_pause_btn = _Control()
    view._recognition_inspector_step_btn = _Control()
    view._recognition_inspector_clear_btn = _Control()
    view._recognition_inspector_export_jsonl_btn = _Control()
    view._recognition_inspector_export_csv_btn = _Control()
    view._recognition_inspector_paused = False
    view._recognition_inspector_snapshot = []
    view._recognition_inspector_last_seen_sequence = 0
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


def test_home_recognition_inspector_can_pause_step_and_clear():
    view = _inspector_view([_inspector_row(2, "confirmed"), _inspector_row(1)])

    view._refresh_recognition_inspector()
    view._on_recognition_inspector_pause_toggle(None)
    view._controller.history.insert(0, _inspector_row(3, "rejected"))
    view._refresh_recognition_inspector()

    assert view._recognition_inspector_paused is True
    assert view._recognition_inspector_snapshot[0]["sequence"] == 2
    assert view._recognition_inspector_step_btn.disabled is False

    view._on_recognition_inspector_step(None)

    assert view._recognition_inspector_snapshot[0]["sequence"] == 3
    assert view._recognition_inspector_last_seen_sequence == 3

    view._on_recognition_inspector_clear(None)

    assert view._controller.cleared is True
    assert view._recognition_inspector_snapshot == []
