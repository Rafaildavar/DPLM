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
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
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
    tab_switched: str = ""
    error: str = ""


@dataclass
class PointerAction:
    x: int
    y: int
    click: bool = False
    hotkey: tuple[str, ...] = ()


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
    """Кончик указательного двигает курсор; сгибание указательного кликает."""

    def __init__(
        self,
        *,
        smoothing: float = 0.28,
        edge_margin: float = 0.08,
        move_deadzone_px: float = 4.0,
        click_debounce_s: float = 0.42,
        tab_swipe_threshold: float = 0.10,
        tab_swipe_vertical_tolerance: float = 0.22,
        tab_swipe_cooldown_s: float = 0.65,
    ) -> None:
        self.smoothing = max(0.05, min(0.95, float(smoothing)))
        self.edge_margin = max(0.0, min(0.4, float(edge_margin)))
        self.move_deadzone_px = max(0.0, float(move_deadzone_px))
        self.click_debounce_s = max(0.1, float(click_debounce_s))
        self.tab_swipe_threshold = max(0.05, min(0.5, float(tab_swipe_threshold)))
        self.tab_swipe_vertical_tolerance = max(
            0.03,
            min(0.3, float(tab_swipe_vertical_tolerance)),
        )
        self.tab_swipe_cooldown_s = max(0.0, float(tab_swipe_cooldown_s))
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._accessibility_warned = False
        self._missing_frames = 0
        self._index_folded = False
        self._last_click_ts = 0.0
        self._two_finger_swipe_points: list[tuple[float, Point]] = []
        self._last_tab_swipe_ts = 0.0

    def reset(self) -> None:
        self._smooth_x = None
        self._smooth_y = None
        self._missing_frames = 0
        self._index_folded = False
        self._two_finger_swipe_points.clear()

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
                self.reset()
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

        screen_w, screen_h = pyautogui.size()
        margin = self.edge_margin
        span = max(1e-6, 1.0 - 2.0 * margin)
        nx = max(margin, min(1.0 - margin, ix))
        ny = max(margin, min(1.0 - margin, iy))
        target_x = (nx - margin) / span * float(screen_w)
        target_y = (ny - margin) / span * float(screen_h)

        previous_x = self._smooth_x
        previous_y = self._smooth_y
        freeze_for_click = index_folded and previous_x is not None and previous_y is not None

        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x = target_x
            self._smooth_y = target_y
        elif freeze_for_click:
            target_x = self._smooth_x
            target_y = self._smooth_y
        else:
            distance_px = math.hypot(target_x - self._smooth_x, target_y - self._smooth_y)
            alpha = self._adaptive_alpha(distance_px, screen_w, screen_h)
            self._smooth_x = self._smooth_x * (1.0 - alpha) + target_x * alpha
            self._smooth_y = self._smooth_y * (1.0 - alpha) + target_y * alpha
            self._limit_step(previous_x, previous_y, screen_w, screen_h)

        clicked = self._click_requested(index_folded)
        tab_swipe = self._tab_swipe_requested(landmarks, index_folded)
        moved = True
        if previous_x is not None and previous_y is not None:
            moved = (
                math.hypot(self._smooth_x - previous_x, self._smooth_y - previous_y)
                >= self.move_deadzone_px
            )

        if not moved and not clicked and not tab_swipe:
            return PointerUpdateResult(ok=True, moved=False), None

        action = PointerAction(
            x=max(0, min(int(screen_w) - 1, int(round(self._smooth_x)))),
            y=max(0, min(int(screen_h) - 1, int(round(self._smooth_y)))),
            click=clicked,
            hotkey=self._tab_swipe_hotkey(tab_swipe) if tab_swipe else (),
        )
        return (
            PointerUpdateResult(
                ok=True,
                moved=moved,
                clicked=clicked,
                tab_switched=tab_swipe,
            ),
            action,
        )

    def _adaptive_alpha(self, distance_px: float, screen_w: int, screen_h: int) -> float:
        base = self.smoothing
        if distance_px <= 10.0:
            return max(0.06, base * 0.22)
        if distance_px <= 36.0:
            return max(0.08, base * 0.35)
        if distance_px <= 120.0:
            return max(0.12, base * 0.55)
        screen_diag = math.hypot(float(screen_w), float(screen_h))
        if distance_px >= screen_diag * 0.20:
            return min(0.70, base + 0.18)
        return max(0.16, base * 0.75)

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
        max_step = max(40.0, min(180.0, screen_diag * 0.075))
        if distance <= max_step or distance <= 1e-6:
            return
        scale = max_step / distance
        self._smooth_x = previous_x + dx * scale
        self._smooth_y = previous_y + dy * scale

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

    def _click_requested(self, index_folded: bool) -> bool:
        now = time.monotonic()
        clicked = (
            index_folded
            and not self._index_folded
            and (now - self._last_click_ts) >= self.click_debounce_s
        )
        if clicked:
            self._last_click_ts = now
        self._index_folded = index_folded
        return clicked

    def _finger_extended(
        self,
        landmarks: list[Any],
        *,
        mcp_index: int,
        pip_index: int,
        dip_index: int,
        tip_index: int,
    ) -> bool:
        mcp = _landmark_point(landmarks, mcp_index)
        pip = _landmark_point(landmarks, pip_index)
        dip = _landmark_point(landmarks, dip_index)
        tip = _landmark_point(landmarks, tip_index)
        if None in (mcp, pip, dip, tip):
            return False

        assert mcp is not None and pip is not None and dip is not None and tip is not None
        finger_len = _distance(mcp, pip) + _distance(pip, dip) + _distance(dip, tip)
        if finger_len <= 1e-6:
            return False
        extension_ratio = _distance(mcp, tip) / finger_len
        pip_angle = _angle_degrees(mcp, pip, dip)
        dip_angle = _angle_degrees(pip, dip, tip)
        straight_enough = extension_ratio >= 0.58 and min(pip_angle, dip_angle) >= 132.0
        tip_ahead_of_knuckles = tip[1] <= pip[1] + 0.04 and tip[1] <= mcp[1] + 0.02
        return straight_enough or (extension_ratio >= 0.50 and tip_ahead_of_knuckles)

    def _two_finger_swipe_center(self, landmarks: list[Any]) -> Optional[Point]:
        index_tip = _landmark_point(landmarks, INDEX_FINGER_TIP)
        middle_tip = _landmark_point(landmarks, MIDDLE_FINGER_TIP)
        if index_tip is None or middle_tip is None:
            return None
        index_extended = self._finger_extended(
            landmarks,
            mcp_index=INDEX_FINGER_MCP,
            pip_index=INDEX_FINGER_PIP,
            dip_index=INDEX_FINGER_DIP,
            tip_index=INDEX_FINGER_TIP,
        )
        middle_extended = self._finger_extended(
            landmarks,
            mcp_index=MIDDLE_FINGER_MCP,
            pip_index=MIDDLE_FINGER_PIP,
            dip_index=MIDDLE_FINGER_DIP,
            tip_index=MIDDLE_FINGER_TIP,
        )
        if not index_extended or not middle_extended:
            return None
        if _distance(index_tip, middle_tip) > 0.30:
            return None
        return (index_tip[0] + middle_tip[0]) / 2.0, (index_tip[1] + middle_tip[1]) / 2.0

    def _tab_swipe_requested(self, landmarks: list[Any], index_folded: bool) -> str:
        if index_folded:
            self._two_finger_swipe_points.clear()
            return ""

        center = self._two_finger_swipe_center(landmarks)
        if center is None:
            self._two_finger_swipe_points.clear()
            return ""

        now = time.monotonic()
        self._two_finger_swipe_points.append((now, center))
        self._two_finger_swipe_points = [
            point for point in self._two_finger_swipe_points if now - point[0] <= 0.70
        ][-10:]
        if len(self._two_finger_swipe_points) < 2:
            return ""

        start = self._two_finger_swipe_points[0][1]
        dx = center[0] - start[0]
        dy = center[1] - start[1]
        if abs(dx) < self.tab_swipe_threshold:
            return ""
        if abs(dy) > self.tab_swipe_vertical_tolerance:
            self._two_finger_swipe_points = [(now, center)]
            return ""
        if abs(dx) < abs(dy) * 1.2:
            return ""
        if (now - self._last_tab_swipe_ts) < self.tab_swipe_cooldown_s:
            self._two_finger_swipe_points = [(now, center)]
            return ""

        self._last_tab_swipe_ts = now
        self._two_finger_swipe_points.clear()
        return "left" if dx < 0 else "right"

    def _tab_swipe_hotkey(self, direction: str) -> tuple[str, ...]:
        if direction == "left":
            return ("ctrl", "tab")
        return ("ctrl", "shift", "tab")

    def apply(self, action: PointerAction) -> None:
        if not PYAUTOGUI_AVAILABLE:
            return
        if sys.platform == "darwin" and not macos_accessibility_trusted():
            if not self._accessibility_warned:
                self._accessibility_warned = True
                print(f"[!] {ACCESSIBILITY_HINT}", flush=True)
            return
        pyautogui.moveTo(action.x, action.y, duration=0, _pause=False)
        if action.click:
            pyautogui.click(_pause=False)
        if action.hotkey:
            print(f"[i] Pointer: tab swipe hotkey {'+'.join(action.hotkey)}", flush=True)
            pyautogui.hotkey(*action.hotkey)

    def update(self, landmarks_json: str) -> PointerUpdateResult:
        """Синхронный путь (тесты/CLI): compute + apply в одном вызове."""
        result, action = self.compute(landmarks_json)
        if result.ok and action is not None:
            try:
                self.apply(action)
            except Exception as exc:
                return PointerUpdateResult(ok=False, error=str(exc))
        return result
