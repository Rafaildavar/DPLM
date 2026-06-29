"""
Экран «Жесты» — список жестов из БД (карточки с привязанной командой).

Аналог QML ``GestureListScreen.qml``. Источник данных — таблица ``gestures``
через ``AppController.get_db_gestures()``. Здесь показываем только активные
и обученные жесты: static-классы из ``model_class_id`` и dynamic-классы из
``models/dynamic_classes.json``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE_HIGH,
    COLOR_WARNING,
    surface_card,
)
from app.services.gesture_taxonomy import (
    GESTURE_TYPE_DYNAMIC,
    GESTURE_TYPE_NEGATIVE,
    GESTURE_TYPE_QUASI_STATIC,
    GESTURE_TYPE_STATIC,
    load_gesture_taxonomy,
)


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class GesturesView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller
        self._rows: list[dict] = []
        self._selected_id: int | None = None
        self._filter = "all"
        self._command_help_open = False
        self._preview_cache: dict[str, str | None] = {}
        self._gesture_type_cache: dict[str, str] = {}
        try:
            self._taxonomy = load_gesture_taxonomy()
        except Exception:
            self._taxonomy = None

        self._list_column = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO)
        self._info_text = ft.Text("", size=12, color=COLOR_MUTED)
        self._summary_total = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=COLOR_ON_SURFACE)
        self._summary_dynamic = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=COLOR_ACCENT)
        self._summary_bound = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=COLOR_SUCCESS)
        self._summary_unbound = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=COLOR_WARNING)
        self._detail_body = ft.Column(
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
        self._empty_title = ft.Text(
            "Жестов пока нет",
            size=18,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._empty_hint = ft.Text(
            "Записанные классы появятся здесь после импорта.",
            size=12,
            color=COLOR_MUTED,
            text_align=ft.TextAlign.CENTER,
        )
        self._search_field = ft.TextField(
            label="Поиск",
            dense=True,
            height=46,
            border_color=COLOR_SURFACE_HIGH,
            focused_border_color=COLOR_ACCENT,
            on_change=self._on_filters_changed,
        )
        self._filter_dd = ft.Dropdown(
            label="Показать",
            value="all",
            dense=True,
            height=46,
            width=250,
            filled=True,
            fill_color="#171A1D",
            bgcolor="#171A1D",
            border_color=COLOR_SURFACE_HIGH,
            border_radius=8,
            focused_border_color=COLOR_ACCENT,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=6),
            leading_icon=ft.Icons.FILTER_ALT,
            trailing_icon=ft.Icons.KEYBOARD_ARROW_DOWN,
            color=COLOR_ON_SURFACE,
            text_style=ft.TextStyle(size=13, color=COLOR_ON_SURFACE),
            label_style=ft.TextStyle(size=11, color=COLOR_MUTED),
            menu_height=240,
            menu_width=260,
            options=[
                ft.DropdownOption(key="all", text="все"),
                ft.DropdownOption(key="bound", text="привязанные"),
                ft.DropdownOption(key="unbound", text="без команды"),
                ft.DropdownOption(key="dynamic", text="динамические"),
                ft.DropdownOption(key="static", text="статические"),
                ft.DropdownOption(key="one_hand", text="одна рука"),
                ft.DropdownOption(key="two_hands", text="две руки"),
            ],
            on_select=self._on_filters_changed,
        )
        self._empty_state = ft.Container(
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=12,
                controls=[
                    ft.Container(
                        width=54,
                        height=54,
                        border_radius=8,
                        bgcolor="#171A1D",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(ft.Icons.GESTURE, size=30, color=COLOR_ACCENT),
                    ),
                    self._empty_title,
                    self._empty_hint,
                ],
            ),
            padding=32,
            alignment=ft.Alignment.CENTER,
            visible=False,
        )

    def on_show(self) -> None:
        self._refresh()

    def on_hide(self) -> None:
        pass

    def _refresh(self) -> None:
        self._rows = self._controller.get_db_gestures()
        ids = {self._row_id(row) for row in self._rows}
        if self._selected_id not in ids:
            self._selected_id = self._row_id(self._rows[0]) if self._rows else None
        self._render()

    def _render(self) -> None:
        rows = self._filtered_rows()
        self._list_column.controls = [self._gesture_card(row) for row in rows]

        total = len(self._rows)
        bound = sum(1 for row in self._rows if self._bound_command(row))
        dynamic = sum(1 for row in self._rows if self._gesture_type(row) == GESTURE_TYPE_DYNAMIC)
        self._summary_total.value = str(total)
        self._summary_dynamic.value = str(dynamic)
        self._summary_bound.value = str(bound)
        self._summary_unbound.value = str(max(0, total - bound))

        has_rows = len(rows) > 0
        if total > 0 and not has_rows:
            self._empty_title.value = "Ничего не найдено"
            self._empty_hint.value = "Измени поиск или фильтр."
        else:
            self._empty_title.value = "Жестов пока нет"
            self._empty_hint.value = "Записанные классы появятся здесь после импорта."
        self._empty_state.visible = not has_rows
        self._list_column.visible = has_rows
        self._render_detail()
        try:
            self._page.update()
        except Exception:
            pass

    def _filtered_rows(self) -> list[dict]:
        query = str(self._search_field.value or "").strip().lower()
        mode = str(self._filter_dd.value or self._filter or "all")
        rows = list(self._rows)
        if query:
            rows = [
                row
                for row in rows
                if query in self._label(row).lower()
                or query in self._description(row).lower()
                or query in self._bound_command(row).lower()
            ]
        if mode == "bound":
            rows = [row for row in rows if self._bound_command(row)]
        elif mode == "unbound":
            rows = [row for row in rows if not self._bound_command(row)]
        elif mode == "dynamic":
            rows = [row for row in rows if self._gesture_type(row) == GESTURE_TYPE_DYNAMIC]
        elif mode == "static":
            rows = [
                row
                for row in rows
                if self._gesture_type(row)
                in {GESTURE_TYPE_STATIC, GESTURE_TYPE_QUASI_STATIC}
            ]
        elif mode == "one_hand":
            rows = [row for row in rows if not row.get("isTwoHands")]
        elif mode == "two_hands":
            rows = [row for row in rows if row.get("isTwoHands")]
        return rows

    def _on_filters_changed(self, _e) -> None:
        self._filter = str(self._filter_dd.value or "all")
        rows = self._filtered_rows()
        ids = {self._row_id(row) for row in rows}
        if self._selected_id not in ids:
            self._selected_id = self._row_id(rows[0]) if rows else None
        self._render()

    def _select_row(self, row: dict) -> None:
        row_id = self._row_id(row)
        if row_id != self._selected_id:
            self._command_help_open = False
        self._selected_id = row_id
        self._render()

    def _row_id(self, row: dict) -> int:
        try:
            return int(row.get("id") or 0)
        except (TypeError, ValueError):
            return 0

    def _label(self, row: dict) -> str:
        return str(row.get("label") or "").strip() or "gesture"

    def _description(self, row: dict) -> str:
        text = str(row.get("description") or "").strip()
        prefix = "Auto-imported from "
        if text.startswith(prefix):
            path = text[len(prefix):].strip()
            marker = "data/gestures/"
            if marker in path:
                path = marker + path.split(marker, 1)[1]
            return f"Импортировано из {path}"
        return text

    def _bound_command(self, row: dict) -> str:
        return str(row.get("boundCommandName") or "").strip()

    def _sample_count(self, row: dict) -> int | None:
        for key in ("samples", "sampleCount", "nSamples"):
            try:
                value = int(row.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value
        return None

    def _gesture_type(self, row: dict) -> str:
        raw = str(row.get("gestureType") or row.get("type") or "").strip().lower()
        if raw in {
            GESTURE_TYPE_STATIC,
            GESTURE_TYPE_QUASI_STATIC,
            GESTURE_TYPE_DYNAMIC,
            GESTURE_TYPE_NEGATIVE,
        }:
            return raw

        label = self._label(row)
        if self._taxonomy is not None:
            try:
                gesture_type = self._taxonomy.gesture_type_for_label(label)
                if gesture_type and gesture_type != GESTURE_TYPE_STATIC:
                    return gesture_type
            except Exception:
                pass

        sample_path = self._gesture_sample_path(row)
        if sample_path is None:
            return GESTURE_TYPE_STATIC
        cache_key = str(sample_path)
        cached = self._gesture_type_cache.get(cache_key)
        if cached:
            return cached

        gesture_type = GESTURE_TYPE_STATIC
        meta_path = sample_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                scope = str(meta.get("source_scope") or meta.get("gesture_type") or "").lower()
                if bool(meta.get("include_global_motion")) or scope == GESTURE_TYPE_DYNAMIC:
                    gesture_type = GESTURE_TYPE_DYNAMIC
                elif scope == GESTURE_TYPE_NEGATIVE:
                    gesture_type = GESTURE_TYPE_NEGATIVE
                elif int(meta.get("raw_feature_dim") or 0) >= 44:
                    gesture_type = GESTURE_TYPE_DYNAMIC
            except Exception:
                pass

        if gesture_type == GESTURE_TYPE_STATIC:
            try:
                import numpy as np

                arr = np.load(sample_path, mmap_mode="r", allow_pickle=False)
                shape = tuple(int(item) for item in arr.shape)
                if len(shape) >= 2 and shape[-1] >= 44:
                    gesture_type = GESTURE_TYPE_DYNAMIC
            except Exception:
                pass

        self._gesture_type_cache[cache_key] = gesture_type
        return gesture_type

    def _gesture_type_label(self, gesture_type: str) -> str:
        return {
            GESTURE_TYPE_DYNAMIC: "динамический",
            GESTURE_TYPE_QUASI_STATIC: "quasi-static",
            GESTURE_TYPE_NEGATIVE: "отсев",
            GESTURE_TYPE_STATIC: "статический",
        }.get(gesture_type, "статический")

    def _gesture_type_color(self, gesture_type: str) -> str:
        return {
            GESTURE_TYPE_DYNAMIC: COLOR_ACCENT,
            GESTURE_TYPE_QUASI_STATIC: COLOR_WARNING,
            GESTURE_TYPE_NEGATIVE: COLOR_DANGER,
            GESTURE_TYPE_STATIC: COLOR_MUTED,
        }.get(gesture_type, COLOR_MUTED)

    def _gesture_type_icon(self, gesture_type: str) -> str:
        return {
            GESTURE_TYPE_DYNAMIC: ft.Icons.AUTO_AWESOME_MOTION,
            GESTURE_TYPE_QUASI_STATIC: ft.Icons.TIMELINE,
            GESTURE_TYPE_NEGATIVE: ft.Icons.RADAR,
            GESTURE_TYPE_STATIC: ft.Icons.BACK_HAND,
        }.get(gesture_type, ft.Icons.BACK_HAND)

    def _action_spec(self, row: dict) -> dict:
        raw = row.get("boundCommandActionSpec") or row.get("actionSpec") or {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}
        return {}

    def _gesture_sample_path(self, row: dict) -> Path | None:
        raw = str(row.get("samplePreviewPath") or row.get("samplesPath") or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.is_file() and path.suffix == ".npy":
            return path
        if path.is_dir():
            for pattern in ("sample_*.npy", "aug_sample_*.npy", "*.npy"):
                matches = sorted(path.glob(pattern))
                if matches:
                    return matches[0]
        return None

    def _gesture_preview_src(self, row: dict) -> str | None:
        sample_path = self._gesture_sample_path(row)
        if sample_path is None:
            return None
        try:
            cache_key = f"{sample_path}:{sample_path.stat().st_mtime_ns}"
        except OSError:
            cache_key = str(sample_path)
        if cache_key in self._preview_cache:
            return self._preview_cache[cache_key]
        try:
            import numpy as np

            arr = np.load(sample_path, allow_pickle=False)
            seq = np.asarray(arr, dtype=float)
            if seq.ndim == 3:
                frame = seq[min(len(seq) // 2, len(seq) - 1)]
                if frame.shape[0] < 21 or frame.shape[1] < 2:
                    return None
                points = frame[:21, :2]
                trail = seq[:, 0, :2] if seq.shape[0] > 1 else None
            elif seq.ndim == 2 and seq.shape[0] > 0:
                flat_seq = seq.reshape(seq.shape[0], -1)
                idx = min(flat_seq.shape[0] // 2, flat_seq.shape[0] - 1)
                nonzero = [
                    i for i in range(flat_seq.shape[0])
                    if np.any(np.abs(flat_seq[i, :42]) > 1e-6)
                ]
                if nonzero:
                    idx = nonzero[len(nonzero) // 2]
                flat = flat_seq[idx]
                if flat.shape[0] < 42:
                    return None
                points = flat[:42].reshape(21, 2)
                trail = flat_seq[:, 42:44] if flat_seq.shape[1] >= 44 else None
            else:
                return None
            if not np.isfinite(points).all() or np.all(np.abs(points) < 1e-6):
                return None
            preview = self._sample_png_base64(
                points,
                trail=trail,
                dynamic=self._gesture_type(row) == GESTURE_TYPE_DYNAMIC,
            )
            self._preview_cache[cache_key] = preview
            return preview
        except Exception:
            self._preview_cache[cache_key] = None
            return None

    def _sample_png_base64(self, points, *, trail=None, dynamic: bool = False) -> str:
        import cv2
        import numpy as np

        pts = np.asarray(points, dtype=float)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        span_x = max(max_x - min_x, 1e-4)
        span_y = max(max_y - min_y, 1e-4)
        width, height = 560, 300
        pad = 42
        scale = min((width - pad * 2) / span_x, (height - pad * 2) / span_y)
        offset_x = (width - span_x * scale) / 2
        offset_y = (height - span_y * scale) / 2

        def map_point(point) -> tuple[float, float]:
            x = offset_x + (float(point[0]) - min_x) * scale
            y = offset_y + (float(point[1]) - min_y) * scale
            return x, y

        mapped = [map_point(point) for point in pts]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = (18, 20, 22)
        for x in range(48, width, 48):
            cv2.line(canvas, (x, 0), (x, height), (28, 31, 35), 1, cv2.LINE_AA)
        for y in range(48, height, 48):
            cv2.line(canvas, (0, y), (width, y), (28, 31, 35), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (1, 1), (width - 2, height - 2), (52, 57, 66), 2)

        if trail is not None:
            tr = np.asarray(trail, dtype=float)
            if tr.ndim == 2 and tr.shape[1] >= 2:
                tr = tr[np.isfinite(tr).all(axis=1)]
            else:
                tr = np.empty((0, 2), dtype=float)
            if len(tr) >= 2 and not np.all(np.abs(tr[:, :2]) < 1e-6):
                t_min = tr[:, :2].min(axis=0)
                t_max = tr[:, :2].max(axis=0)
                t_span = np.maximum(t_max - t_min, 1e-4)
                trail_points = []
                for item in tr[:, :2]:
                    x = width - 132 + ((item[0] - t_min[0]) / t_span[0]) * 84
                    y = 42 + ((item[1] - t_min[1]) / t_span[1]) * 68
                    trail_points.append((int(round(x)), int(round(y))))
                for i, (p1, p2) in enumerate(zip(trail_points, trail_points[1:])):
                    blend = i / max(1, len(trail_points) - 1)
                    color = (
                        int(78 + 80 * blend),
                        int(168 + 28 * blend),
                        int(216 - 70 * blend),
                    )
                    cv2.line(canvas, p1, p2, color, 5 if dynamic else 3, cv2.LINE_AA)
                cv2.circle(canvas, trail_points[-1], 6, (216, 199, 82), -1, cv2.LINE_AA)

        glow = canvas.copy()
        for a, b in HAND_CONNECTIONS:
            if a < len(mapped) and b < len(mapped):
                x1, y1 = mapped[a]
                x2, y2 = mapped[b]
                cv2.line(
                    glow,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (82, 199, 216),
                    11,
                    cv2.LINE_AA,
                )
        canvas = cv2.addWeighted(glow, 0.22, canvas, 0.78, 0)

        for a, b in HAND_CONNECTIONS:
            if a < len(mapped) and b < len(mapped):
                x1, y1 = mapped[a]
                x2, y2 = mapped[b]
                cv2.line(
                    canvas,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (82, 199, 216),
                    6,
                    cv2.LINE_AA,
                )

        for x, y in mapped:
            cv2.circle(
                canvas,
                (int(round(x)), int(round(y))),
                8,
                (18, 20, 22),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                (int(round(x)), int(round(y))),
                5,
                (231, 238, 240),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                (int(round(x)), int(round(y))),
                2,
                (216, 199, 82) if dynamic else (82, 199, 216),
                -1,
                cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:
            raise ValueError("preview encode failed")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _border(self, color: str, width: float = 1) -> ft.Border:
        side = ft.BorderSide(width, color)
        return ft.Border(top=side, right=side, bottom=side, left=side)

    def _chip(self, text: str, color: str, *, icon: str | None = None) -> ft.Container:
        controls: list[ft.Control] = []
        if icon is not None:
            controls.append(ft.Icon(icon, size=13, color=color))
        controls.append(ft.Text(text, size=11, color=color, no_wrap=True))
        return ft.Container(
            bgcolor="#171A1D",
            border=self._border(COLOR_SURFACE_HIGH),
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=9, vertical=5),
            content=ft.Row(
                spacing=6,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=controls,
            ),
        )

    def _gesture_preview(self, row: dict) -> ft.Container:
        src = self._gesture_preview_src(row)
        gesture_type = self._gesture_type(row)
        gesture_color = self._gesture_type_color(gesture_type)
        if src:
            media: ft.Control = ft.Image(
                src=src,
                fit=ft.BoxFit.CONTAIN,
                border_radius=8,
                gapless_playback=True,
                filter_quality=ft.FilterQuality.HIGH,
            )
        else:
            media = ft.Column(
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.BACK_HAND, size=42, color=COLOR_ACCENT),
                    ft.Text("Нет sample preview", size=12, color=COLOR_MUTED),
                ],
            )
        return ft.Container(
            height=124,
            bgcolor="#101316",
            border=self._border(COLOR_SURFACE_HIGH),
            border_radius=8,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=ft.Stack(
                expand=True,
                controls=[
                    media,
                    ft.Container(
                        left=12,
                        top=12,
                        bgcolor="#101316",
                        border_radius=8,
                        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                        content=ft.Row(
                            spacing=6,
                            tight=True,
                            controls=[
                                ft.Icon(ft.Icons.CENTER_FOCUS_STRONG, size=14, color=COLOR_ACCENT),
                                ft.Text("реальная запись", size=11, color=COLOR_ON_SURFACE),
                            ],
                        ),
                    ),
                    ft.Container(
                        right=12,
                        top=12,
                        bgcolor="#101316",
                        border_radius=8,
                        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                        content=ft.Row(
                            spacing=6,
                            tight=True,
                            controls=[
                                ft.Icon(self._gesture_type_icon(gesture_type), size=14, color=gesture_color),
                                ft.Text(self._gesture_type_label(gesture_type), size=11, color=gesture_color),
                            ],
                        ),
                    ),
                ],
            ),
        )

    def _status_chip(self, bound: bool) -> ft.Container:
        return self._chip(
            "привязано" if bound else "без команды",
            COLOR_SUCCESS if bound else COLOR_WARNING,
            icon=ft.Icons.LINK if bound else ft.Icons.LINK_OFF,
        )

    def _metric_tile(self, title: str, value: ft.Text, icon: str, color: str) -> ft.Container:
        return ft.Container(
            bgcolor="#171A1D",
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=12, vertical=9),
            content=ft.Row(
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        width=32,
                        height=32,
                        border_radius=8,
                        bgcolor="#101316",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(icon, size=17, color=color),
                    ),
                    ft.Column(
                        spacing=0,
                        controls=[
                            ft.Text(title, size=11, color=COLOR_MUTED),
                            value,
                        ],
                    ),
                ],
            ),
        )

    def _table_header(self) -> ft.Container:
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            border_radius=8,
            bgcolor="#101316",
            content=ft.Row(
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(expand=3, content=ft.Text("Жест", size=11, color=COLOR_MUTED)),
                    ft.Container(width=118, content=ft.Text("Тип", size=11, color=COLOR_MUTED)),
                    ft.Container(width=78, content=ft.Text("Samples", size=11, color=COLOR_MUTED)),
                    ft.Container(expand=2, content=ft.Text("Команда", size=11, color=COLOR_MUTED)),
                    ft.Container(width=112, content=ft.Text("Статус", size=11, color=COLOR_MUTED)),
                ],
            ),
        )

    def _mini_gesture_preview(self, row: dict) -> ft.Container:
        gesture_type = self._gesture_type(row)
        color = self._gesture_type_color(gesture_type)
        return ft.Container(
            width=56,
            height=42,
            border_radius=8,
            bgcolor="#101316",
            border=self._border(COLOR_SURFACE_HIGH),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            alignment=ft.Alignment.CENTER,
            content=ft.Icon(self._gesture_type_icon(gesture_type), color=color, size=20),
        )

    def _gesture_card(self, row: dict) -> ft.Container:
        label = self._label(row)
        bound = self._bound_command(row)
        selected = self._row_id(row) == self._selected_id
        samples = self._sample_count(row)
        hands = "две руки" if row.get("isTwoHands") else "одна рука"
        description = self._description(row)
        subtitle = description or hands
        gesture_type = self._gesture_type(row)
        type_color = self._gesture_type_color(gesture_type)

        return ft.Container(
            bgcolor="#171A1D" if not selected else "#162A2E",
            border=self._border(COLOR_ACCENT if selected else COLOR_SURFACE_HIGH, 1.2 if selected else 1),
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            on_click=lambda _e, item=row: self._select_row(item),
            content=ft.Row(
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        expand=3,
                        content=ft.Row(
                            spacing=10,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                self._mini_gesture_preview(row),
                                ft.Column(
                                    spacing=3,
                                    expand=True,
                                    controls=[
                                        ft.Text(
                                            label,
                                            size=14,
                                            weight=ft.FontWeight.W_600,
                                            color=COLOR_ON_SURFACE,
                                            no_wrap=True,
                                        ),
                                        ft.Text(subtitle, size=11, color=COLOR_MUTED, no_wrap=True),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    ft.Container(
                        width=118,
                        content=self._chip(
                            self._gesture_type_label(gesture_type),
                            type_color,
                            icon=self._gesture_type_icon(gesture_type),
                        ),
                    ),
                    ft.Container(
                        width=78,
                        content=ft.Text(
                            str(samples) if samples is not None else "trained",
                            size=13,
                            color=COLOR_ON_SURFACE,
                            no_wrap=True,
                        ),
                    ),
                    ft.Container(
                        expand=2,
                        content=ft.Text(
                            bound or "не назначена",
                            size=13,
                            color=COLOR_SUCCESS if bound else COLOR_MUTED,
                            no_wrap=True,
                        ),
                    ),
                    ft.Container(
                        width=112,
                        content=self._status_chip(bool(bound)),
                    ),
                ],
            ),
        )

    def _detail_line(self, icon: str, title: str, value: str, color: str) -> ft.Container:
        return ft.Container(
            bgcolor="#171A1D",
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=11, vertical=9),
            content=ft.Row(
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(icon, size=17, color=color),
                    ft.Column(
                        spacing=1,
                        expand=True,
                        controls=[
                            ft.Text(title, size=11, color=COLOR_MUTED),
                            ft.Text(value, size=13, color=COLOR_ON_SURFACE, no_wrap=True),
                        ],
                    ),
                ],
            ),
        )

    def _action_summary(self, row: dict) -> str:
        spec = self._action_spec(row)
        action = str(spec.get("action") or "").strip()
        script_path = str(row.get("boundCommandScriptPath") or "").strip()
        command_desc = str(row.get("boundCommandDescription") or "").strip()
        if command_desc:
            return command_desc
        if not self._bound_command(row):
            return "Команда пока не назначена."
        if action == "open_url":
            return f"Откроет сайт: {spec.get('url', '')}".strip()
        if action == "open_app":
            return f"Запустит приложение: {spec.get('app', '')}".strip()
        if action == "open_path":
            return f"Откроет файл или папку: {spec.get('path', '')}".strip()
        if action == "scroll":
            clicks = int(spec.get("clicks") or 0)
            direction = "вниз" if clicks < 0 else "вверх"
            return f"Прокрутит страницу {direction} на {abs(clicks)} щелчков."
        if action == "press":
            return f"Нажмёт клавишу: {spec.get('key', '')}".strip()
        if action == "key_combination":
            keys = ", ".join(str(k) for k in spec.get("keys", []))
            return f"Нажмёт сочетание клавиш: {keys}".strip()
        if action == "media_key":
            kind = str(spec.get("kind") or "")
            return f"Отправит media-key: {kind}".strip()
        if action == "notify":
            return f"Покажет уведомление: {spec.get('message', '')}".strip()
        if action == "wait":
            return f"Подождёт {spec.get('seconds', '')} сек."
        if action == "run_script":
            return f"Запустит скрипт: {spec.get('script_path', '')}".strip()
        if action == "sequence":
            steps = spec.get("steps") or []
            return f"Выполнит сценарий из {len(steps)} шагов."
        if action in {"volume_up", "volume_down", "mute_toggle", "screenshot", "lock_screen"}:
            labels = {
                "volume_up": "Увеличит громкость.",
                "volume_down": "Уменьшит громкость.",
                "mute_toggle": "Включит или выключит звук.",
                "screenshot": "Сделает снимок экрана.",
                "lock_screen": "Заблокирует экран.",
            }
            return labels[action]
        if script_path.startswith("open:"):
            return f"Откроет ресурс: {script_path.removeprefix('open:')}"
        if script_path:
            return f"Выполнит ресурс: {script_path}"
        return "Команда выполнится через системный executor."

    def _action_details(self, row: dict) -> list[ft.Control]:
        spec = self._action_spec(row)
        if not spec:
            script_path = str(row.get("boundCommandScriptPath") or "").strip()
            if not script_path:
                return []
            return [self._detail_line(ft.Icons.CODE, "Источник", script_path, COLOR_MUTED)]

        controls: list[ft.Control] = []
        action = str(spec.get("action") or "unknown")
        controls.append(self._detail_line(ft.Icons.ROUTE, "Тип действия", action, COLOR_ACCENT))
        platform = str(spec.get("platform") or row.get("boundCommandPlatform") or "all")
        controls.append(self._detail_line(ft.Icons.DESKTOP_MAC, "Платформа", platform, COLOR_MUTED))
        for key in ("url", "app", "path", "key", "kind", "message", "seconds", "script_path"):
            if key in spec and spec.get(key) not in (None, ""):
                controls.append(
                    self._detail_line(
                        ft.Icons.CHEVRON_RIGHT,
                        key,
                        str(spec.get(key)),
                        COLOR_MUTED,
                    )
                )
        if action == "key_combination" and spec.get("keys"):
            controls.append(
                self._detail_line(
                    ft.Icons.KEYBOARD,
                    "keys",
                    " + ".join(str(k) for k in spec.get("keys", [])),
                    COLOR_MUTED,
                )
            )
        if action == "sequence":
            steps = spec.get("steps") if isinstance(spec.get("steps"), list) else []
            for i, step in enumerate(steps[:4], start=1):
                if isinstance(step, dict):
                    controls.append(
                        self._detail_line(
                            ft.Icons.FORMAT_LIST_NUMBERED,
                            f"Шаг {i}",
                            self._short_action_text(step),
                            COLOR_MUTED,
                        )
                    )
        return controls

    def _short_action_text(self, spec: dict) -> str:
        action = str(spec.get("action") or "action")
        for key in ("url", "app", "path", "key", "kind", "message", "script_path"):
            if spec.get(key):
                return f"{action}: {spec.get(key)}"
        if action == "key_combination":
            return f"{action}: {' + '.join(str(k) for k in spec.get('keys', []))}"
        return action

    def _toggle_command_help(self) -> None:
        self._command_help_open = not self._command_help_open
        self._render_detail()
        try:
            self._detail_body.update()
        except Exception:
            try:
                self._page.update()
            except Exception:
                pass

    def _command_panel(self, row: dict) -> ft.Container:
        bound = self._bound_command(row)
        summary = self._action_summary(row)
        controls: list[ft.Control] = [
            ft.Row(
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.TERMINAL, size=18, color=COLOR_SUCCESS if bound else COLOR_WARNING),
                    ft.Column(
                        spacing=2,
                        expand=True,
                        controls=[
                            ft.Text("Команда", size=12, color=COLOR_MUTED),
                            ft.Text(bound or "Не назначена", size=16, weight=ft.FontWeight.W_600, color=COLOR_ON_SURFACE, no_wrap=True),
                        ],
                    ),
                    ft.Icon(
                        ft.Icons.EXPAND_LESS if self._command_help_open else ft.Icons.EXPAND_MORE,
                        size=18,
                        color=COLOR_MUTED,
                    ),
                ],
            ),
            ft.Text(summary, size=12, color=COLOR_MUTED),
        ]
        if self._command_help_open:
            controls.extend(self._action_details(row))

        return ft.Container(
            bgcolor="#171A1D",
            border_radius=8,
            padding=12,
            on_click=lambda _e: self._toggle_command_help(),
            content=ft.Column(spacing=10, controls=controls),
        )

    def _selected_row(self) -> dict | None:
        for row in self._rows:
            if self._row_id(row) == self._selected_id:
                return row
        return None

    def _render_detail(self) -> None:
        row = self._selected_row()
        if row is None:
            self._detail_body.controls = [
                ft.Container(
                    height=240,
                    alignment=ft.Alignment.CENTER,
                    content=ft.Column(
                        spacing=10,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.Icons.TOUCH_APP, size=38, color=COLOR_MUTED),
                            ft.Text("Выбери жест", size=15, color=COLOR_ON_SURFACE),
                        ],
                    ),
                )
            ]
            return

        label = self._label(row)
        bound = self._bound_command(row)
        hands = "две руки" if row.get("isTwoHands") else "одна рука"
        samples = self._sample_count(row)
        gesture_type = self._gesture_type(row)
        description = self._description(row) or "Описание не задано"
        self._detail_body.controls = [
            self._gesture_preview(row),
            ft.Column(
                spacing=4,
                controls=[
                    ft.Text(
                        label,
                        size=22,
                        weight=ft.FontWeight.BOLD,
                        color=COLOR_ON_SURFACE,
                        no_wrap=True,
                    ),
                    ft.Text(f"id #{self._row_id(row)}", size=12, color=COLOR_MUTED),
                ],
            ),
            self._command_panel(row),
            ft.Text(description, size=12, color=COLOR_MUTED),
            self._detail_line(
                self._gesture_type_icon(gesture_type),
                "Тип жеста",
                self._gesture_type_label(gesture_type),
                self._gesture_type_color(gesture_type),
            ),
            self._detail_line(ft.Icons.PAN_TOOL_ALT, "Режим", hands, COLOR_ACCENT),
            self._detail_line(
                ft.Icons.DATASET,
                "Данные",
                f"{samples} samples" if samples is not None else "Обучен в модели",
                COLOR_ACCENT,
            ),
            ft.Row(
                spacing=8,
                wrap=True,
                controls=[
                    self._status_chip(bool(bound)),
                    self._chip(
                        self._gesture_type_label(gesture_type),
                        self._gesture_type_color(gesture_type),
                        icon=self._gesture_type_icon(gesture_type),
                    ),
                    self._chip("active", COLOR_SUCCESS, icon=ft.Icons.CHECK_CIRCLE),
                ],
            )
        ]

    def _on_import_click(self, _e) -> None:
        summary = self._controller.sync_dataset_to_db()
        if summary["total"] == 0:
            self._info_text.value = (
                "В папке data/gestures/ нет ни одного класса с записанными "
                "примерами. Запиши хотя бы один жест на вкладке «Обучение»."
            )
            self._info_text.color = COLOR_MUTED
        else:
            self._info_text.value = (
                f"✓ Синхронизировано: добавлено {summary['created']}, "
                f"обновлено {summary['updated']}, классов: {summary['total']}, "
                f"сэмплов: {summary.get('samples', 0)}"
            )
            self._info_text.color = COLOR_SUCCESS
        try:
            self._info_text.update()
        except Exception:
            pass
        self._refresh()

    def build(self) -> ft.Control:
        import_btn = ft.FilledButton(
            content=ft.Text("Импорт", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.CLOUD_UPLOAD,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            ),
            on_click=self._on_import_click,
            tooltip="Создать строки в таблице gestures для всех папок с npy",
        )
        refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Обновить",
            on_click=lambda _e: self._refresh(),
        )

        header = surface_card(
            ft.Row(
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        expand=True,
                        content=ft.Row(
                            spacing=10,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Container(
                                    width=36,
                                    height=36,
                                    border_radius=8,
                                    bgcolor="#171A1D",
                                    alignment=ft.Alignment.CENTER,
                                    content=ft.Icon(ft.Icons.GESTURE, color=COLOR_ACCENT, size=20),
                                ),
                                ft.Column(
                                    spacing=1,
                                    expand=True,
                                    controls=[
                                        ft.Text(
                                            "Gesture Library",
                                            size=19,
                                            weight=ft.FontWeight.BOLD,
                                            color=COLOR_ON_SURFACE,
                                        ),
                                        ft.Text(
                                            "активные обученные жесты",
                                            size=12,
                                            color=COLOR_MUTED,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    ft.Row(
                        spacing=8,
                        controls=[
                            self._metric_tile("Всего", self._summary_total, ft.Icons.DATA_ARRAY, COLOR_ACCENT),
                            self._metric_tile("Dynamic", self._summary_dynamic, ft.Icons.AUTO_AWESOME_MOTION, COLOR_ACCENT),
                            self._metric_tile("Привязано", self._summary_bound, ft.Icons.LINK, COLOR_SUCCESS),
                            self._metric_tile("Без команды", self._summary_unbound, ft.Icons.LINK_OFF, COLOR_WARNING),
                        ],
                    ),
                    ft.Row(
                        spacing=8,
                        alignment=ft.MainAxisAlignment.END,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[import_btn, refresh_btn],
                    ),
                ],
            ),
            padding=14,
            radius=8,
        )
        tools = surface_card(
            ft.ResponsiveRow(
                spacing=10,
                run_spacing=10,
                controls=[
                    ft.Container(content=self._search_field, col={"xs": 12, "md": 8}),
                    ft.Container(content=self._filter_dd, col={"xs": 12, "md": 4}),
                ],
            ),
            padding=12,
            radius=8,
        )
        library = ft.ResponsiveRow(
            spacing=14,
            run_spacing=14,
            controls=[
                ft.Container(
                    col={"xs": 12, "lg": 8},
                    content=surface_card(
                        ft.Column(
                            spacing=10,
                            expand=True,
                            controls=[
                                self._table_header(),
                                ft.Container(
                                    content=ft.Stack(
                                        controls=[self._list_column, self._empty_state],
                                        expand=True,
                                    ),
                                    expand=True,
                                ),
                            ],
                        ),
                        padding=12,
                        radius=8,
                    ),
                ),
                ft.Container(
                    col={"xs": 12, "lg": 4},
                    content=surface_card(
                        ft.Column(
                            spacing=12,
                            expand=True,
                            controls=[
                                ft.Row(
                                    spacing=10,
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                    controls=[
                                        ft.Icon(ft.Icons.INFO_OUTLINE, size=18, color=COLOR_ACCENT),
                                        ft.Text(
                                            "Gesture Details",
                                            size=15,
                                            weight=ft.FontWeight.W_600,
                                            color=COLOR_ON_SURFACE,
                                        ),
                                    ],
                                ),
                                ft.Container(
                                    content=self._detail_body,
                                    expand=True,
                                ),
                            ],
                        ),
                        padding=14,
                        radius=8,
                    ),
                ),
            ],
        )
        return ft.Column(
            spacing=12,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                header,
                tools,
                self._info_text,
                ft.Container(content=library, expand=True),
            ],
        )
