# -*- coding: utf-8 -*-
"""Управление курсором по ключевым точкам MediaPipe Hands."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

try:
    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None  # type: ignore[assignment]
    PYAUTOGUI_AVAILABLE = False

ACCESSIBILITY_HINT = (
    "Pointer: включите «Универсальный доступ» для Python/Terminal в "
    "Системные настройки → Конфиденциальность и безопасность → Универсальный доступ"
)

INDEX_FINGER_TIP = 8
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
THUMB_TIP = 4
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_MCP = 9
PINKY_FINGER_MCP = 17
Point = tuple[float, float]


def macos_accessibility_trusted() -> bool:
    """Проверить, разрешён ли процессу Python управление компьютером (macOS)."""
    if sys.platform != "darwin":
        return True
    try:
        import ctypes
        import ctypes.util

        lib_path = ctypes.util.find_library("ApplicationServices")
        if not lib_path:
            return True
        lib = ctypes.cdll.LoadLibrary(lib_path)
        lib.AXIsProcessTrusted.restype = bool  # type: ignore[attr-defined]
        return bool(lib.AXIsProcessTrusted())
    except Exception:
        return True


def macos_open_accessibility_settings() -> None:
    """Открыть раздел «Универсальный доступ» в системных настройках."""
    if sys.platform != "darwin":
        return
    urls = (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?"
        "Privacy_Accessibility",
    )
    for url in urls:
        try:
            subprocess.run(["open", url], check=False, timeout=3)
            return
        except Exception:
            continue


@dataclass
class PointerUpdateResult:
    ok: bool
    moved: bool = False
    clicked: bool = False
    drag_started: bool = False
    dragging: bool = False
    drag_ended: bool = False
    error: str = ""


@dataclass
class PointerAction:
    x: int
    y: int
    click: bool = False
    mouse_down: bool = False
    mouse_up: bool = False
    down_x: Optional[int] = None
    down_y: Optional[int] = None


def parse_landmarks_json(landmarks_json: str) -> list[dict[str, Any]]:
    """Разобрать JSON рук (новый и legacy-формат)."""
    if not landmarks_json or landmarks_json == "[]":
        return []
    try:
        raw = json.loads(landmarks_json)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []

    hands: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            pts = item.get("landmarks") or item.get("points")
            if not isinstance(pts, list):
                continue
            hands.append(
                {
                    "landmarks": pts,
                    "handedness": str(item.get("handedness") or ""),
                }
            )
        elif isinstance(item, list):
            hands.append({"landmarks": item, "handedness": ""})
    return hands


def _pick_pointer_hand(hands: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not hands:
        return None
    for hand in hands:
        if (hand.get("handedness") or "").strip().lower() == "right":
            return hand
    return hands[0]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_degrees(a: Point, b: Point, c: Point) -> float:
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    ab_len = math.hypot(*ab)
    cb_len = math.hypot(*cb)
    if ab_len <= 1e-6 or cb_len <= 1e-6:
        return 180.0
    cosine = (ab[0] * cb[0] + ab[1] * cb[1]) / (ab_len * cb_len)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _landmark_point(landmarks: list[Any], index: int) -> Optional[Point]:
    try:
        point = landmarks[index]
        return float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None


class PointerControlService:
    """Указательный палец двигает курсор; thumb+middle pinch даёт click/drag."""

    def __init__(
        self,
        *,
        smoothing: float = 0.28,
        edge_margin: float = 0.08,
        move_deadzone_px: float = 4.0,
        click_debounce_s: float = 0.42,
        drag_start_px: float = 18.0,
        drag_gain: float = 1.35,
        enable_index_bend_fallback: bool = False,
    ) -> None:
        self.smoothing = max(0.05, min(0.95, float(smoothing)))
        self.edge_margin = max(0.0, min(0.4, float(edge_margin)))
        self.move_deadzone_px = max(0.0, float(move_deadzone_px))
        self.click_debounce_s = max(0.1, float(click_debounce_s))
        self.drag_start_px = max(4.0, float(drag_start_px))
        self.drag_gain = max(0.5, min(3.0, float(drag_gain)))
        self.enable_index_bend_fallback = bool(enable_index_bend_fallback)
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._accessibility_warned = False
        self._missing_frames = 0
        self._index_folded = False
        self._pinch_active = False
        self._trigger_active = False
        self._button_down = False
        self._pending_click = False
        self._dragging = False
        self._fold_anchor_x: Optional[float] = None
        self._fold_anchor_y: Optional[float] = None
        self._fold_reference_x: Optional[float] = None
        self._fold_reference_y: Optional[float] = None
        self._last_click_ts = 0.0

    def reset(self) -> None:
        self._smooth_x = None
        self._smooth_y = None
        self._missing_frames = 0
        self._index_folded = False
        self._pinch_active = False
        self._trigger_active = False
        self._button_down = False
        self._pending_click = False
        self._dragging = False
        self._fold_anchor_x = None
        self._fold_anchor_y = None
        self._fold_reference_x = None
        self._fold_reference_y = None

    def compute(self, landmarks_json: str) -> tuple[PointerUpdateResult, Optional[PointerAction]]:
        if not PYAUTOGUI_AVAILABLE:
            return (
                PointerUpdateResult(
                    ok=False,
                    error="pyautogui не установлен (pip install pyautogui)",
                ),
                None,
            )

        hand = _pick_pointer_hand(parse_landmarks_json(landmarks_json))
        if hand is None:
            self._missing_frames += 1
            if self._missing_frames >= 4:
                release = self._release_button_action()
                self.reset()
                if release is not None:
                    return (
                        PointerUpdateResult(ok=True, drag_ended=True),
                        release,
                    )
            return PointerUpdateResult(ok=True, moved=False), None
        self._missing_frames = 0

        landmarks = hand.get("landmarks") or []
        if not isinstance(landmarks, list) or len(landmarks) <= INDEX_FINGER_TIP:
            return PointerUpdateResult(ok=True, moved=False), None

        index_tip = _landmark_point(landmarks, INDEX_FINGER_TIP)
        if index_tip is None:
            return PointerUpdateResult(ok=True, moved=False), None
        ix, iy = index_tip
        index_folded = self._is_index_folded(landmarks)
        pinch_active = self._is_thumb_middle_pinched(landmarks)
        trigger_active = pinch_active or (
            self.enable_index_bend_fallback and index_folded
        )

        screen_w, screen_h = pyautogui.size()
        pointer_x, pointer_y = self._landmark_to_screen(ix, iy, screen_w, screen_h)
        target_x, target_y = pointer_x, pointer_y

        previous_x = self._smooth_x
        previous_y = self._smooth_y
        was_trigger_active = self._trigger_active
        just_pressed = trigger_active and not was_trigger_active
        just_released = not trigger_active and was_trigger_active

        if (
            trigger_active
            and self._fold_anchor_x is not None
            and self._fold_anchor_y is not None
            and self._fold_reference_x is not None
            and self._fold_reference_y is not None
        ):
            target_x = (
                self._fold_anchor_x
                + (pointer_x - self._fold_reference_x) * self.drag_gain
            )
            target_y = (
                self._fold_anchor_y
                + (pointer_y - self._fold_reference_y) * self.drag_gain
            )

        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x = target_x
            self._smooth_y = target_y
            if just_pressed:
                self._button_down = True
                self._pending_click = True
                self._dragging = False
                self._fold_anchor_x = self._smooth_x
                self._fold_anchor_y = self._smooth_y
                self._fold_reference_x = pointer_x
                self._fold_reference_y = pointer_y
        elif just_pressed:
            self._button_down = True
            self._pending_click = True
            self._dragging = False
            self._fold_anchor_x = self._smooth_x
            self._fold_anchor_y = self._smooth_y
            self._fold_reference_x = pointer_x
            self._fold_reference_y = pointer_y
            target_x = self._smooth_x
            target_y = self._smooth_y
        elif just_released:
            target_x = self._smooth_x
            target_y = self._smooth_y
        else:
            distance_px = math.hypot(target_x - self._smooth_x, target_y - self._smooth_y)
            alpha = self._drag_alpha(distance_px) if self._dragging else self._adaptive_alpha(
                distance_px, screen_w, screen_h
            )
            self._smooth_x = self._smooth_x * (1.0 - alpha) + target_x * alpha
            self._smooth_y = self._smooth_y * (1.0 - alpha) + target_y * alpha
            self._limit_step(previous_x, previous_y, screen_w, screen_h)

        clicked = False
        mouse_down = just_pressed
        mouse_up = False
        drag_started = just_pressed
        drag_ended = False

        if trigger_active and self._button_down and self._pending_click and not self._dragging:
            if self._fold_drag_distance() >= self.drag_start_px:
                self._dragging = True
                self._pending_click = False

        if just_released:
            mouse_up = self._button_down
            if self._dragging:
                drag_ended = True
            elif self._pending_click:
                clicked = self._click_requested()
            self._button_down = False
            self._dragging = False
            self._pending_click = False
            self._fold_anchor_x = None
            self._fold_anchor_y = None
            self._fold_reference_x = None
            self._fold_reference_y = None

        self._index_folded = index_folded
        self._pinch_active = pinch_active
        self._trigger_active = trigger_active
        moved = True
        if previous_x is not None and previous_y is not None:
            deadzone = 1.0 if self._button_down else self.move_deadzone_px
            moved = (
                math.hypot(self._smooth_x - previous_x, self._smooth_y - previous_y)
                >= deadzone
            )

        if not moved and not clicked and not mouse_down and not mouse_up:
            return PointerUpdateResult(ok=True, moved=False), None

        anchor_x = None
        anchor_y = None
        if drag_started:
            anchor_x = self._screen_x(self._fold_anchor_x, screen_w)
            anchor_y = self._screen_y(self._fold_anchor_y, screen_h)

        action = PointerAction(
            x=self._screen_x(self._smooth_x, screen_w),
            y=self._screen_y(self._smooth_y, screen_h),
            click=False,
            mouse_down=mouse_down,
            mouse_up=mouse_up,
            down_x=anchor_x,
            down_y=anchor_y,
        )
        return (
            PointerUpdateResult(
                ok=True,
                moved=moved,
                clicked=clicked,
                drag_started=drag_started,
                dragging=self._dragging,
                drag_ended=drag_ended,
            ),
            action,
        )

    def _adaptive_alpha(self, distance_px: float, screen_w: int, screen_h: int) -> float:
        base = self.smoothing
        screen_diag = math.hypot(float(screen_w), float(screen_h))
        if distance_px <= 6.0:
            return max(0.05, base * 0.18)
        if distance_px <= 24.0:
            return max(0.10, base * 0.36)
        if distance_px <= 90.0:
            return max(0.20, base * 0.70)
        if distance_px >= screen_diag * 0.12:
            return min(0.92, base + 0.34)
        return min(0.82, max(0.28, base + 0.10))

    def _landmark_to_screen(
        self,
        x: float,
        y: float,
        screen_w: int,
        screen_h: int,
    ) -> tuple[float, float]:
        margin = self.edge_margin
        span = max(1e-6, 1.0 - 2.0 * margin)
        nx = max(margin, min(1.0 - margin, x))
        ny = max(margin, min(1.0 - margin, y))
        return (
            (nx - margin) / span * float(screen_w),
            (ny - margin) / span * float(screen_h),
        )

    def _drag_alpha(self, distance_px: float) -> float:
        if distance_px <= 24.0:
            return 0.34
        if distance_px <= 120.0:
            return 0.48
        return 0.66

    def _limit_step(
        self,
        previous_x: Optional[float],
        previous_y: Optional[float],
        screen_w: int,
        screen_h: int,
    ) -> None:
        if previous_x is None or previous_y is None:
            return
        if self._smooth_x is None or self._smooth_y is None:
            return
        dx = self._smooth_x - previous_x
        dy = self._smooth_y - previous_y
        distance = math.hypot(dx, dy)
        screen_diag = math.hypot(float(screen_w), float(screen_h))
        max_step = max(80.0, min(420.0, screen_diag * (0.12 + self.smoothing * 0.20)))
        if distance <= max_step or distance <= 1e-6:
            return
        scale = max_step / distance
        self._smooth_x = previous_x + dx * scale
        self._smooth_y = previous_y + dy * scale

    def _fold_drag_distance(self) -> float:
        if None in (
            self._smooth_x,
            self._smooth_y,
            self._fold_anchor_x,
            self._fold_anchor_y,
        ):
            return 0.0
        assert self._smooth_x is not None and self._smooth_y is not None
        assert self._fold_anchor_x is not None and self._fold_anchor_y is not None
        return math.hypot(
            self._smooth_x - self._fold_anchor_x,
            self._smooth_y - self._fold_anchor_y,
        )

    def _screen_x(self, value: Optional[float], screen_w: int) -> int:
        if value is None:
            value = 0.0
        return max(0, min(int(screen_w) - 1, int(round(value))))

    def _screen_y(self, value: Optional[float], screen_h: int) -> int:
        if value is None:
            value = 0.0
        return max(0, min(int(screen_h) - 1, int(round(value))))

    def _release_button_action(self) -> Optional[PointerAction]:
        if not self._button_down or self._smooth_x is None or self._smooth_y is None:
            return None
        screen_w, screen_h = pyautogui.size()
        return PointerAction(
            x=self._screen_x(self._smooth_x, screen_w),
            y=self._screen_y(self._smooth_y, screen_h),
            mouse_up=True,
        )

    def _is_index_folded(self, landmarks: list[Any]) -> bool:
        mcp = _landmark_point(landmarks, INDEX_FINGER_MCP)
        pip = _landmark_point(landmarks, INDEX_FINGER_PIP)
        dip = _landmark_point(landmarks, INDEX_FINGER_DIP)
        tip = _landmark_point(landmarks, INDEX_FINGER_TIP)
        if None in (mcp, pip, dip, tip):
            return False

        assert mcp is not None and pip is not None and dip is not None and tip is not None
        finger_len = _distance(mcp, pip) + _distance(pip, dip) + _distance(dip, tip)
        if finger_len <= 1e-6:
            return False

        extension_ratio = _distance(mcp, tip) / finger_len
        pip_angle = _angle_degrees(mcp, pip, dip)
        dip_angle = _angle_degrees(pip, dip, tip)
        bend_angle = min(pip_angle, dip_angle)

        if self._index_folded:
            return extension_ratio < 0.78 and bend_angle < 165.0
        return extension_ratio < 0.66 and bend_angle < 152.0

    def _is_thumb_middle_pinched(self, landmarks: list[Any]) -> bool:
        thumb_tip = _landmark_point(landmarks, THUMB_TIP)
        middle_tip = _landmark_point(landmarks, MIDDLE_FINGER_TIP)
        index_mcp = _landmark_point(landmarks, INDEX_FINGER_MCP)
        middle_mcp = _landmark_point(landmarks, MIDDLE_FINGER_MCP)
        pinky_mcp = _landmark_point(landmarks, PINKY_FINGER_MCP)
        wrist = _landmark_point(landmarks, 0)
        if None in (thumb_tip, middle_tip, index_mcp, middle_mcp, pinky_mcp, wrist):
            return False
        assert thumb_tip is not None and middle_tip is not None
        assert index_mcp is not None and middle_mcp is not None
        assert pinky_mcp is not None and wrist is not None

        palm_span = max(
            _distance(index_mcp, pinky_mcp),
            _distance(wrist, middle_mcp),
            1e-6,
        )
        ratio = _distance(thumb_tip, middle_tip) / palm_span
        return ratio < (0.72 if self._pinch_active else 0.48)

    def _click_requested(self) -> bool:
        now = time.monotonic()
        clicked = (now - self._last_click_ts) >= self.click_debounce_s
        if clicked:
            self._last_click_ts = now
        return clicked

    def apply(self, action: PointerAction) -> None:
        if not PYAUTOGUI_AVAILABLE:
            return
        if sys.platform == "darwin" and not macos_accessibility_trusted():
            if not self._accessibility_warned:
                self._accessibility_warned = True
                print(f"[!] {ACCESSIBILITY_HINT}", flush=True)
            return
        if action.mouse_down and action.down_x is not None and action.down_y is not None:
            pyautogui.moveTo(action.down_x, action.down_y, duration=0, _pause=False)
            pyautogui.mouseDown(_pause=False)
            if action.x != action.down_x or action.y != action.down_y:
                pyautogui.moveTo(action.x, action.y, duration=0, _pause=False)
        else:
            pyautogui.moveTo(action.x, action.y, duration=0, _pause=False)
        if action.click:
            pyautogui.click(_pause=False)
        if action.mouse_up:
            pyautogui.mouseUp(_pause=False)

    def update(self, landmarks_json: str) -> PointerUpdateResult:
        """Синхронный путь (тесты/CLI): compute + apply в одном вызове."""
        result, action = self.compute(landmarks_json)
        if result.ok and action is not None:
            try:
                self.apply(action)
            except Exception as exc:
                return PointerUpdateResult(ok=False, error=str(exc))
        return result
