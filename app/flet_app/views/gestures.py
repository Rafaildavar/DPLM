"""
Экран «Жесты»: пользовательские классы и записи из локального хранилища.

Источник данных — таблица ``gestures`` через
``AppController.get_db_gestures()``. Здесь показываются только классы,
записанные пользователем через приложение.
"""
from __future__ import annotations

import base64
import json
import os
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
from cv.gesture_features import hand_feature_blocks


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
ANIMATED_PREVIEW_ENV = "DPLM_GESTURE_ANIMATED_PREVIEWS"
PREVIEW_CACHE_LIMIT = 48


def _animated_previews_enabled() -> bool:
    value = str(os.environ.get(ANIMATED_PREVIEW_ENV, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


class GesturesView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller
        self._rows: list[dict] = []
        self._dataset_rows: list[dict] = []
        self._selected_id: int | None = None
        self._filter = "all"
        self._command_help_open = False
        self._preview_cache: dict[str, str | None] = {}
        self._gesture_type_cache: dict[str, str] = {}
        self._pending_delete_label = ""
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
        self._dataset_standard = ft.Text("0", size=18, weight=ft.FontWeight.BOLD, color=COLOR_ACCENT)
        self._dataset_custom = ft.Text("0", size=18, weight=ft.FontWeight.BOLD, color=COLOR_SUCCESS)
        self._dataset_samples = ft.Text("0", size=18, weight=ft.FontWeight.BOLD, color=COLOR_ON_SURFACE)
        self._dataset_augmented = ft.Text("0", size=18, weight=ft.FontWeight.BOLD, color=COLOR_WARNING)
        self._dataset_column = ft.Column(
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
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
            "Записанные классы появятся здесь после записи на вкладке «Обучение».",
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
            self._empty_hint.value = (
                "Записанные классы появятся здесь после записи на вкладке «Обучение»."
            )
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
            rows = []
        return rows

    def _visible_dataset_rows(self) -> list[dict]:
        return [row for row in self._dataset_rows if not bool(row.get("systemClass"))]

    def _dataset_role(self, row: dict) -> str:
        if bool(row.get("systemClass")):
            return "system"
        try:
            user_samples = int(row.get("userRecordedSamples") or 0)
        except (TypeError, ValueError):
            user_samples = 0
        if bool(row.get("canDelete")) or user_samples > 0:
            return "custom"
        return "standard"

    def _dataset_role_chip(self, role: str) -> ft.Container:
        if role == "custom":
            return self._chip("мой", COLOR_SUCCESS, icon=ft.Icons.EDIT)
        if role == "standard":
            return self._chip("стандартный", COLOR_ACCENT, icon=ft.Icons.STAR)
        return self._chip("служебный", COLOR_MUTED, icon=ft.Icons.LOCK_OUTLINE)

    def _render_dataset_manager(self) -> None:
        rows = self._visible_dataset_rows()
        standard = sum(1 for row in rows if self._dataset_role(row) == "standard")
        custom = sum(1 for row in rows if self._dataset_role(row) == "custom")
        samples = 0
        augmented = 0
        for row in rows:
            try:
                samples += int(row.get("realSamples") or row.get("samples") or 0)
            except (TypeError, ValueError):
                pass
            try:
                augmented += int(row.get("augmentedSamples") or 0)
            except (TypeError, ValueError):
                pass

        self._dataset_standard.value = str(standard)
        self._dataset_custom.value = str(custom)
        self._dataset_samples.value = str(samples)
        self._dataset_augmented.value = str(augmented)
        if rows:
            self._dataset_column.controls = [self._dataset_row_card(row) for row in rows]
        else:
            self._dataset_column.controls = [
                ft.Container(
                    bgcolor="#171A1D",
                    border_radius=8,
                    padding=14,
                    alignment=ft.Alignment.CENTER,
                    content=ft.Text(
                        "Пока нет записанных жестов.",
                        size=12,
                        color=COLOR_MUTED,
                    ),
                )
            ]

    def _dataset_row_card(self, row: dict) -> ft.Container:
        label = str(row.get("label") or "").strip() or "gesture"
        role = self._dataset_role(row)
        can_delete = bool(row.get("canDelete")) and role == "custom"
        pending_delete = self._pending_delete_label.lower() == label.lower() and can_delete
        try:
            real_samples = int(row.get("realSamples") or row.get("samples") or 0)
        except (TypeError, ValueError):
            real_samples = 0
        try:
            augmented_samples = int(row.get("augmentedSamples") or 0)
        except (TypeError, ValueError):
            augmented_samples = 0

        if pending_delete:
            actions: list[ft.Control] = [
                ft.IconButton(
                    icon=ft.Icons.DELETE_FOREVER,
                    icon_color=COLOR_DANGER,
                    tooltip=f"Удалить записи {label}",
                    on_click=lambda _e, value=label: self._delete_samples(value),
                ),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_color=COLOR_MUTED,
                    tooltip="Отмена",
                    on_click=lambda _e: self._cancel_delete_samples(),
                ),
            ]
        elif can_delete:
            actions = [
                ft.IconButton(
                    icon=ft.Icons.DELETE,
                    icon_color=COLOR_DANGER,
                    tooltip=f"Удалить мои записи {label}",
                    on_click=lambda _e, value=label: self._request_delete_samples(value),
                )
            ]
        else:
            actions = [
                ft.IconButton(
                    icon=ft.Icons.LOCK_OUTLINE,
                    icon_color=COLOR_MUTED,
                    tooltip="Стандартные записи нельзя удалить отсюда",
                    disabled=True,
                )
            ]

        return ft.Container(
            bgcolor="#171A1D",
            border=self._border(COLOR_SURFACE_HIGH),
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=12, vertical=9),
            content=ft.Row(
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.FOLDER, size=18, color=COLOR_ACCENT),
                    ft.Column(
                        spacing=2,
                        expand=True,
                        controls=[
                            ft.Text(
                                label,
                                size=13,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                                no_wrap=True,
                            ),
                            ft.Text(
                                f"{real_samples} записей · {augmented_samples} аугм.",
                                size=11,
                                color=COLOR_MUTED,
                                no_wrap=True,
                            ),
                        ],
                    ),
                    self._dataset_role_chip(role),
                    *actions,
                ],
            ),
        )

    def _request_delete_samples(self, label: str) -> None:
        self._pending_delete_label = label
        self._render_dataset_manager()
        try:
            self._page.update()
        except Exception:
            pass

    def _cancel_delete_samples(self) -> None:
        self._pending_delete_label = ""
        self._render_dataset_manager()
        try:
            self._page.update()
        except Exception:
            pass

    def _delete_samples(self, label: str) -> None:
        self._pending_delete_label = ""
        delete_recorded = getattr(self._controller, "delete_recorded_samples", None)
        if not callable(delete_recorded):
            self._info_text.value = "Удаление записей недоступно."
            self._info_text.color = COLOR_DANGER
            self._render()
            return
        summary = delete_recorded(label)
        if summary.get("ok"):
            self._info_text.value = f"Удалены записи «{label}». Переобучи модель после изменений."
            self._info_text.color = COLOR_SUCCESS
        else:
            self._info_text.value = str(summary.get("error") or "Не удалось удалить записи.")
            self._info_text.color = COLOR_DANGER
        self._refresh()

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
                explicit_types = getattr(self._taxonomy, "label_to_type", {})
                explicit_type = explicit_types.get(str(label or "").strip().lower())
                if explicit_type:
                    return explicit_type
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
                feature_format = str(meta.get("sample_feature_format") or "").lower()
                raw_feature_dim = int(meta.get("raw_feature_dim") or 0)
                if bool(meta.get("include_global_motion")) or scope == GESTURE_TYPE_DYNAMIC:
                    gesture_type = GESTURE_TYPE_DYNAMIC
                elif scope == GESTURE_TYPE_NEGATIVE:
                    gesture_type = GESTURE_TYPE_NEGATIVE
                elif scope in {GESTURE_TYPE_STATIC, GESTURE_TYPE_QUASI_STATIC}:
                    gesture_type = scope
                elif "wrist_xy" in feature_format:
                    gesture_type = GESTURE_TYPE_DYNAMIC
                elif self._looks_like_dynamic_feature_dim(raw_feature_dim):
                    gesture_type = GESTURE_TYPE_DYNAMIC
            except Exception:
                pass

        if gesture_type == GESTURE_TYPE_STATIC:
            try:
                import numpy as np

                arr = np.load(sample_path, mmap_mode="r", allow_pickle=False)
                shape = tuple(int(item) for item in arr.shape)
                if len(shape) >= 2 and self._looks_like_dynamic_feature_dim(shape[-1]):
                    gesture_type = GESTURE_TYPE_DYNAMIC
            except Exception:
                pass

        self._gesture_type_cache[cache_key] = gesture_type
        return gesture_type

    @staticmethod
    def _looks_like_dynamic_feature_dim(feature_dim: int) -> bool:
        # Legacy dynamic samples: 21 * xy + wrist_xy = 44 per hand.
        # New dynamic samples: 21 * xyz + wrist_xy = 65 per hand.
        # Static xyz samples are 63/126, so a generic ">= 44" check is wrong.
        return int(feature_dim or 0) in {44, 65, 88, 130}

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
        is_dynamic = self._gesture_type(row) == GESTURE_TYPE_DYNAMIC
        animated = bool(is_dynamic and _animated_previews_enabled())
        try:
            cache_key = (
                f"{sample_path}:{sample_path.stat().st_mtime_ns}:"
                f"{is_dynamic}:{animated}"
            )
        except OSError:
            cache_key = f"{sample_path}:{is_dynamic}:{animated}"
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
                frames = seq[:, :21, :2] if is_dynamic and seq.shape[0] > 1 else None
            elif seq.ndim == 2 and seq.shape[0] > 0:
                flat_seq = seq.reshape(seq.shape[0], -1)
                idx = min(flat_seq.shape[0] // 2, flat_seq.shape[0] - 1)
                nonzero = [
                    i for i in range(flat_seq.shape[0])
                    if np.any(np.abs(flat_seq[i, :42]) > 1e-6)
                ]
                if nonzero:
                    idx = nonzero[len(nonzero) // 2]
                blocks = hand_feature_blocks(flat_seq.shape[1])
                if is_dynamic and blocks:
                    block_candidates = []
                    for block in blocks:
                        pose_flat = flat_seq[:, block.start : block.pose_end]
                        if pose_flat.shape[1] < block.pose_dim:
                            continue
                        pose_frames_full = pose_flat.reshape(
                            flat_seq.shape[0],
                            21,
                            block.coords_per_point,
                        )
                        pose_frames = pose_frames_full[:, :, :2]
                        pose_signal = float(np.nanmean(np.abs(pose_flat)))
                        if block.global_start is not None:
                            wrist = flat_seq[:, block.global_start : block.global_start + 2]
                        else:
                            wrist = pose_frames[:, 0, :2]
                        wrist_finite = wrist[np.isfinite(wrist).all(axis=1)]
                        wrist_displacement = (
                            float(np.linalg.norm(wrist_finite[-1] - wrist_finite[0]))
                            if len(wrist_finite) >= 2
                            else 0.0
                        )
                        block_candidates.append(
                            (wrist_displacement + pose_signal, pose_frames, wrist)
                        )
                    if block_candidates:
                        _score, frames, trail = max(
                            block_candidates,
                            key=lambda item: item[0],
                        )
                        points = frames[idx]
                    else:
                        flat = flat_seq[idx]
                        if flat.shape[0] < 42:
                            return None
                        points = flat[:42].reshape(21, 2)
                        trail = None
                        frames = flat_seq[:, :42].reshape(flat_seq.shape[0], 21, 2)
                else:
                    flat = flat_seq[idx]
                    if flat.shape[0] < 42:
                        return None
                    points = flat[:42].reshape(21, 2)
                    trail = None
                    frames = (
                        flat_seq[:, :42].reshape(flat_seq.shape[0], 21, 2)
                        if is_dynamic and flat_seq.shape[0] > 1
                        else None
                    )
            else:
                return None
            if not np.isfinite(points).all() or np.all(np.abs(points) < 1e-6):
                return None
            if animated and frames is not None:
                preview = self._sample_gif_base64(frames, trail=trail)
            elif is_dynamic and frames is not None:
                preview = self._sample_dynamic_png_base64(frames, trail=trail)
            else:
                preview = self._sample_png_base64(points, trail=trail, dynamic=False)
            self._preview_cache[cache_key] = preview
            self._trim_preview_cache()
            return preview
        except Exception:
            self._preview_cache[cache_key] = None
            self._trim_preview_cache()
            return None

    def _trim_preview_cache(self) -> None:
        overflow = len(self._preview_cache) - PREVIEW_CACHE_LIMIT
        if overflow <= 0:
            return
        for key in list(self._preview_cache.keys())[:overflow]:
            self._preview_cache.pop(key, None)

    def _sample_dynamic_png_base64(self, frames, *, trail=None) -> str:
        import numpy as np

        seq = np.asarray(frames, dtype=float)
        valid = [
            i
            for i in range(seq.shape[0])
            if np.isfinite(seq[i]).all() and not np.all(np.abs(seq[i]) < 1e-6)
        ]
        if not valid:
            raise ValueError("dynamic preview has no valid frames")
        idx = valid[min(len(valid) // 2, len(valid) - 1)]
        layout = self._camera_motion_preview_layout(seq, trail, valid)
        progress = valid.index(idx) / max(1, len(valid) - 1)
        png_bytes = self._sample_camera_motion_png_bytes(
            seq[idx],
            all_frames=seq,
            layout=layout,
            frame_index=idx,
            progress=progress,
        )
        return base64.b64encode(png_bytes).decode("ascii")

    def _sample_gif_base64(self, frames, *, trail=None) -> str:
        import io

        import numpy as np
        from PIL import Image

        seq = np.asarray(frames, dtype=float)
        valid = [
            i for i in range(seq.shape[0])
            if np.isfinite(seq[i]).all() and not np.all(np.abs(seq[i]) < 1e-6)
        ]
        if not valid:
            raise ValueError("animated preview has no valid frames")
        wanted = np.linspace(0, len(valid) - 1, num=min(12, len(valid)), dtype=int)
        indices = [valid[int(item)] for item in wanted]

        layout = self._camera_motion_preview_layout(seq, trail, valid)
        images: list[Image.Image] = []
        for frame_no, idx in enumerate(indices):
            png_bytes = self._sample_camera_motion_png_bytes(
                seq[idx],
                all_frames=seq,
                layout=layout,
                frame_index=idx,
                progress=frame_no / max(1, len(indices) - 1),
            )
            images.append(Image.open(io.BytesIO(png_bytes)).convert("P", palette=Image.Palette.ADAPTIVE))

        out = io.BytesIO()
        images[0].save(
            out,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=115,
            loop=0,
            optimize=True,
            disposal=2,
        )
        return base64.b64encode(out.getvalue()).decode("ascii")

    def _preview_canvas(self, width: int, height: int):
        import cv2
        import numpy as np

        y, x = np.mgrid[0:height, 0:width]
        nx = (x - width / 2) / max(1.0, width / 2)
        ny = (y - height / 2) / max(1.0, height / 2)
        spotlight = np.clip(1.0 - (nx * nx + ny * ny), 0.0, 1.0)
        vertical = y / max(1.0, height - 1)

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :, 0] = np.clip(16 + spotlight * 14 + vertical * 5, 0, 255).astype(np.uint8)
        canvas[:, :, 1] = np.clip(18 + spotlight * 12 + vertical * 4, 0, 255).astype(np.uint8)
        canvas[:, :, 2] = np.clip(20 + spotlight * 8 + vertical * 3, 0, 255).astype(np.uint8)

        grid = canvas.copy()
        for x_pos in range(28, width, 28):
            color = (29, 33, 36) if x_pos % 56 else (35, 40, 43)
            cv2.line(grid, (x_pos, 0), (x_pos, height), color, 1, cv2.LINE_AA)
        for y_pos in range(28, height, 28):
            color = (29, 33, 36) if y_pos % 56 else (35, 40, 43)
            cv2.line(grid, (0, y_pos), (width, y_pos), color, 1, cv2.LINE_AA)
        canvas = cv2.addWeighted(grid, 0.46, canvas, 0.54, 0)

        frame = canvas.copy()
        cv2.rectangle(frame, (1, 1), (width - 2, height - 2), (54, 61, 70), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (8, 8), (width - 9, height - 9), (25, 29, 33), 1, cv2.LINE_AA)
        canvas = cv2.addWeighted(frame, 0.76, canvas, 0.24, 0)

        accent = (216, 199, 82)
        corner = 24
        for x0, y0, sx, sy in (
            (12, 12, 1, 1),
            (width - 12, 12, -1, 1),
            (12, height - 12, 1, -1),
            (width - 12, height - 12, -1, -1),
        ):
            cv2.line(canvas, (x0, y0), (x0 + sx * corner, y0), accent, 1, cv2.LINE_AA)
            cv2.line(canvas, (x0, y0), (x0, y0 + sy * corner), accent, 1, cv2.LINE_AA)

        return canvas

    def _camera_motion_preview_layout(self, seq, trail, valid_indices) -> dict:
        import numpy as np

        width, height = 560, 300
        frames = np.asarray(seq, dtype=float)
        valid_frames = frames[valid_indices]
        path = None
        origin_mode = "center"
        path_is_camera = False
        if trail is not None:
            trail_arr = np.asarray(trail, dtype=float)
            if trail_arr.ndim == 2 and trail_arr.shape[0] >= frames.shape[0] and trail_arr.shape[1] >= 2:
                trail_arr = trail_arr[: frames.shape[0], :2]
                finite = np.isfinite(trail_arr).all(axis=1)
                if finite.any() and not np.all(np.abs(trail_arr[finite]) < 1e-6):
                    path = trail_arr
                    origin_mode = "wrist"
                    finite_path = trail_arr[finite]
                    path_is_camera = bool(
                        finite_path[:, 0].min() >= -0.08
                        and finite_path[:, 1].min() >= -0.08
                        and finite_path[:, 0].max() <= 1.08
                        and finite_path[:, 1].max() <= 1.08
                    )

        origins = (
            valid_frames[:, :1, :]
            if origin_mode == "wrist"
            else valid_frames.mean(axis=1, keepdims=True)
        )
        local_points = (valid_frames - origins).reshape(-1, 2)
        local_points = local_points[np.isfinite(local_points).all(axis=1)]
        if len(local_points) == 0:
            local_points = np.zeros((1, 2), dtype=float)
        local_min = local_points.min(axis=0)
        local_max = local_points.max(axis=0)
        local_span = np.maximum(local_max - local_min, 1e-4)
        hand_scale = min((width * 0.32) / local_span[0], (height * 0.56) / local_span[1])

        if path is None:
            path = frames.mean(axis=1)

        path = np.asarray(path, dtype=float)
        finite_path = path[np.isfinite(path).all(axis=1)]
        if len(finite_path) < 2:
            center = np.array([[width / 2, (height - 20) / 2]], dtype=float)
            path_px = np.repeat(center, frames.shape[0], axis=0)
        else:
            content_min = np.array([22.0, 18.0], dtype=float)
            content_max = np.array([width - 22.0, height - 32.0], dtype=float)
            if path_is_camera:
                content_span = np.maximum(content_max - content_min, 1e-4)
                world_min = finite_path.min(axis=0) + local_min
                world_max = finite_path.max(axis=0) + local_max
                world_span = np.maximum(world_max - world_min, 1e-4)
                fit_scale = min(content_span[0] / world_span[0], content_span[1] / world_span[1]) * 0.9
                max_hand_scale = min((width * 0.46) / local_span[0], (height * 0.72) / local_span[1])
                path_scale = min(float(fit_scale), float(max_hand_scale))
                world_px = world_span * path_scale
                offset = content_min + (content_span - world_px) / 2 - world_min * path_scale
                path_px = path * path_scale + offset
                hand_scale = path_scale
            else:
                min_xy = finite_path.min(axis=0)
                max_xy = finite_path.max(axis=0)
                span_xy = np.maximum(max_xy - min_xy, 1e-4)
                path_w = min(width * 0.42, max(28.0, span_xy[0] * width * 0.9))
                path_h = min(height * 0.42, max(20.0, span_xy[1] * height * 0.9))
                path_scale = min(path_w / span_xy[0], path_h / span_xy[1])
                offset = np.array(
                    [
                        (width - span_xy[0] * path_scale) / 2 - min_xy[0] * path_scale,
                        (height - 30 - span_xy[1] * path_scale) / 2 - min_xy[1] * path_scale,
                    ],
                    dtype=float,
                )
                path_px = path * path_scale + offset

            if not np.isfinite(path_px).all():
                center = np.array([[width / 2, (height - 20) / 2]], dtype=float)
                path_px = np.repeat(center, frames.shape[0], axis=0)
            else:
                finite_px = path_px[np.isfinite(path_px).all(axis=1)]
                if len(finite_px):
                    content_min = np.array([22.0, 18.0], dtype=float)
                    content_max = np.array([width - 22.0, height - 32.0], dtype=float)
                    for _ in range(2):
                        ext_min = finite_px.min(axis=0) + local_min * hand_scale
                        ext_max = finite_px.max(axis=0) + local_max * hand_scale
                        ext_span = np.maximum(ext_max - ext_min, 1e-4)
                        content_span = np.maximum(content_max - content_min, 1e-4)
                        overflow = max(
                            float(ext_span[0] / content_span[0]),
                            float(ext_span[1] / content_span[1]),
                            1.0,
                        )
                        if overflow <= 1.0:
                            break
                        hand_scale /= overflow

                    ext_min = finite_px.min(axis=0) + local_min * hand_scale
                    ext_max = finite_px.max(axis=0) + local_max * hand_scale
                    shift = np.zeros(2, dtype=float)
                    for axis in (0, 1):
                        if ext_min[axis] < content_min[axis]:
                            shift[axis] += content_min[axis] - ext_min[axis]
                        if ext_max[axis] + shift[axis] > content_max[axis]:
                            shift[axis] += content_max[axis] - (ext_max[axis] + shift[axis])
                    path_px = path_px + shift

        return {
            "width": width,
            "height": height,
            "hand_scale": float(hand_scale),
            "path_px": path_px,
            "origin_mode": origin_mode,
        }

    def _sample_camera_motion_png_bytes(
        self,
        points,
        *,
        all_frames,
        layout: dict,
        frame_index: int,
        progress: float,
    ) -> bytes:
        import cv2
        import numpy as np

        width = int(layout["width"])
        height = int(layout["height"])
        hand_scale = float(layout["hand_scale"])
        path_px = np.asarray(layout["path_px"], dtype=float)
        idx = max(0, min(int(frame_index), len(path_px) - 1))

        canvas = self._preview_canvas(width, height)

        full_path = path_px[np.isfinite(path_px).all(axis=1)]
        if len(full_path) >= 2:
            path_shadow = canvas.copy()
            for p1, p2 in zip(full_path, full_path[1:]):
                cv2.line(
                    path_shadow,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])), int(round(p2[1]))),
                    (12, 15, 18),
                    8,
                    cv2.LINE_AA,
                )
            canvas = cv2.addWeighted(path_shadow, 0.22, canvas, 0.78, 0)
            for p1, p2 in zip(full_path, full_path[1:]):
                cv2.line(
                    canvas,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])), int(round(p2[1]))),
                    (52, 62, 64),
                    2,
                    cv2.LINE_AA,
                )

        visible_path = path_px[: idx + 1]
        if len(visible_path) >= 2:
            finite = visible_path[np.isfinite(visible_path).all(axis=1)]
            path_glow = canvas.copy()
            for p1, p2 in zip(finite, finite[1:]):
                cv2.line(
                    path_glow,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])), int(round(p2[1]))),
                    (216, 199, 82),
                    12,
                    cv2.LINE_AA,
                )
            canvas = cv2.addWeighted(path_glow, 0.18, canvas, 0.82, 0)
            for i, (p1, p2) in enumerate(zip(finite, finite[1:])):
                blend = i / max(1, len(finite) - 1)
                color = (
                    int(216 - 126 * blend),
                    int(199 + 6 * blend),
                    int(82 + 150 * blend),
                )
                cv2.line(
                    canvas,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])), int(round(p2[1]))),
                    color,
                    5,
                    cv2.LINE_AA,
                )
            cv2.circle(
                canvas,
                (int(round(finite[-1][0])), int(round(finite[-1][1]))),
                11,
                (216, 199, 82),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                (int(round(finite[-1][0])), int(round(finite[-1][1]))),
                5,
                (245, 231, 112),
                -1,
                cv2.LINE_AA,
            )

        def map_hand(frame_points, anchor_point) -> list[tuple[float, float]]:
            frame = np.asarray(frame_points, dtype=float)
            if str(layout.get("origin_mode") or "") == "wrist" and len(frame):
                center_point = frame[0]
            else:
                center_point = frame.mean(axis=0)
            return [
                (
                    float(anchor_point[0] + (point[0] - center_point[0]) * hand_scale),
                    float(anchor_point[1] + (point[1] - center_point[1]) * hand_scale),
                )
                for point in frame
            ]

        frames = np.asarray(all_frames, dtype=float)
        if idx > 0 and frames.ndim == 3:
            ghost_indices = np.linspace(0, idx, num=min(4, idx + 1), dtype=int)[:-1]
            for ghost_no, ghost_idx in enumerate(ghost_indices):
                if ghost_idx >= len(frames) or ghost_idx >= len(path_px):
                    continue
                ghost_mapped = map_hand(frames[int(ghost_idx)], path_px[int(ghost_idx)])
                overlay = canvas.copy()
                for a, b in HAND_CONNECTIONS:
                    if a < len(ghost_mapped) and b < len(ghost_mapped):
                        x1, y1 = ghost_mapped[a]
                        x2, y2 = ghost_mapped[b]
                        cv2.line(
                            overlay,
                            (int(round(x1)), int(round(y1))),
                            (int(round(x2)), int(round(y2))),
                            (74, 87, 86),
                            5,
                            cv2.LINE_AA,
                        )
                alpha = 0.09 + 0.04 * ghost_no
                canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)

        pts = np.asarray(points, dtype=float)
        anchor = path_px[idx]
        mapped = map_hand(pts, anchor)

        glow = canvas.copy()
        for a, b in HAND_CONNECTIONS:
            if a < len(mapped) and b < len(mapped):
                x1, y1 = mapped[a]
                x2, y2 = mapped[b]
                cv2.line(
                    glow,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (0, 215, 255),
                    14,
                    cv2.LINE_AA,
                )
        canvas = cv2.addWeighted(glow, 0.18, canvas, 0.82, 0)

        for a, b in HAND_CONNECTIONS:
            if a < len(mapped) and b < len(mapped):
                x1, y1 = mapped[a]
                x2, y2 = mapped[b]
                cv2.line(
                    canvas,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (18, 20, 22),
                    8,
                    cv2.LINE_AA,
                )

        for a, b in HAND_CONNECTIONS:
            if a < len(mapped) and b < len(mapped):
                x1, y1 = mapped[a]
                x2, y2 = mapped[b]
                cv2.line(
                    canvas,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (58, 211, 244),
                    5,
                    cv2.LINE_AA,
                )
                cv2.line(
                    canvas,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (112, 232, 255),
                    2,
                    cv2.LINE_AA,
                )

        for x, y in mapped:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 8, (18, 20, 22), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 5, (231, 238, 240), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (216, 199, 82), -1, cv2.LINE_AA)

        bar_w = max(12, int((width - 44) * max(0.0, min(1.0, float(progress)))))
        cv2.rectangle(canvas, (22, height - 18), (width - 22, height - 13), (35, 39, 43), -1)
        cv2.rectangle(canvas, (22, height - 18), (22 + bar_w, height - 13), (216, 199, 82), -1)

        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:
            raise ValueError("camera motion preview encode failed")
        return encoded.tobytes()

    def _sample_png_base64(self, points, *, trail=None, dynamic: bool = False) -> str:
        return base64.b64encode(
            self._sample_png_bytes(points, trail=trail, dynamic=dynamic)
        ).decode("ascii")

    def _sample_png_bytes(
        self,
        points,
        *,
        trail=None,
        dynamic: bool = False,
        bounds: tuple[float, float, float, float] | None = None,
        progress: float | None = None,
    ) -> bytes:
        import cv2
        import numpy as np

        pts = np.asarray(points, dtype=float)
        if bounds is None:
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)
        else:
            min_x, min_y, max_x, max_y = bounds
        span_x = max(max_x - min_x, 1e-4)
        span_y = max(max_y - min_y, 1e-4)
        width, height = 560, 300
        pad = 58
        scale = min((width - pad * 2) / span_x, (height - pad * 2) / span_y)
        offset_x = (width - span_x * scale) / 2
        offset_y = (height - span_y * scale) / 2

        def map_point(point) -> tuple[float, float]:
            x = offset_x + (float(point[0]) - min_x) * scale
            y = offset_y + (float(point[1]) - min_y) * scale
            return x, y

        mapped = [map_point(point) for point in pts]
        canvas = self._preview_canvas(width, height)
        if progress is not None:
            bar_w = max(12, int((width - 44) * max(0.0, min(1.0, float(progress)))))
            cv2.rectangle(canvas, (22, height - 18), (width - 22, height - 13), (35, 39, 43), -1)
            cv2.rectangle(canvas, (22, height - 18), (22 + bar_w, height - 13), (216, 199, 82), -1)

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
                        int(216 - 126 * blend),
                        int(199 + 6 * blend),
                        int(82 + 150 * blend),
                    )
                    cv2.line(canvas, p1, p2, color, 5 if dynamic else 3, cv2.LINE_AA)
                cv2.circle(canvas, trail_points[-1], 7, (216, 199, 82), -1, cv2.LINE_AA)
                cv2.circle(canvas, trail_points[-1], 3, (245, 231, 112), -1, cv2.LINE_AA)

        glow = canvas.copy()
        for a, b in HAND_CONNECTIONS:
            if a < len(mapped) and b < len(mapped):
                x1, y1 = mapped[a]
                x2, y2 = mapped[b]
                cv2.line(
                    glow,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (0, 215, 255),
                    14,
                    cv2.LINE_AA,
                )
        canvas = cv2.addWeighted(glow, 0.18, canvas, 0.82, 0)

        for a, b in HAND_CONNECTIONS:
            if a < len(mapped) and b < len(mapped):
                x1, y1 = mapped[a]
                x2, y2 = mapped[b]
                cv2.line(
                    canvas,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (18, 20, 22),
                    8,
                    cv2.LINE_AA,
                )

        for a, b in HAND_CONNECTIONS:
            if a < len(mapped) and b < len(mapped):
                x1, y1 = mapped[a]
                x2, y2 = mapped[b]
                cv2.line(
                    canvas,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (58, 211, 244),
                    5,
                    cv2.LINE_AA,
                )
                cv2.line(
                    canvas,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (112, 232, 255),
                    2,
                    cv2.LINE_AA,
                )

        for x, y in mapped:
            cv2.circle(
                canvas,
                (int(round(x)), int(round(y))),
                9,
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
                (216, 199, 82),
                -1,
                cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:
            raise ValueError("preview encode failed")
        return encoded.tobytes()

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
        real_badge = ft.Container(
            bgcolor="#101316",
            border=self._border(COLOR_SURFACE_HIGH),
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=9, vertical=5),
            content=ft.Row(
                spacing=6,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.CENTER_FOCUS_STRONG, size=13, color=COLOR_ACCENT),
                    ft.Text("запись", size=11, color=COLOR_ON_SURFACE, no_wrap=True),
                ],
            ),
        )
        type_badge = ft.Container(
            bgcolor="#101316",
            border=self._border(COLOR_SURFACE_HIGH),
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=9, vertical=5),
            content=ft.Row(
                spacing=6,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(self._gesture_type_icon(gesture_type), size=13, color=gesture_color),
                    ft.Text(self._gesture_type_label(gesture_type), size=11, color=gesture_color, no_wrap=True),
                ],
            ),
        )
        if src:
            media: ft.Control = ft.Image(
                src=src,
                fit=ft.BoxFit.CONTAIN,
                border_radius=8,
                gapless_playback=bool(
                    gesture_type == GESTURE_TYPE_DYNAMIC
                    and _animated_previews_enabled()
                ),
                filter_quality=ft.FilterQuality.MEDIUM,
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
            height=204,
            bgcolor="#0E1114",
            border=self._border(COLOR_SURFACE_HIGH),
            border_radius=8,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            padding=8,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Row(
                        spacing=8,
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            real_badge,
                            type_badge,
                        ],
                    ),
                    ft.Container(
                        expand=True,
                        bgcolor="#101316",
                        border_radius=8,
                        clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        content=ft.Stack(
                            expand=True,
                            controls=[media],
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
        hands = "одна рука"
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
        hands = "одна рука"
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

    def _build_dataset_manager_card(self) -> ft.Control:
        metrics = ft.ResponsiveRow(
            spacing=10,
            run_spacing=10,
            controls=[
                ft.Container(
                    content=self._metric_tile(
                        "Стандартные",
                        self._dataset_standard,
                        ft.Icons.STAR,
                        COLOR_ACCENT,
                    ),
                    col={"xs": 6, "md": 3},
                ),
                ft.Container(
                    content=self._metric_tile(
                        "Мои",
                        self._dataset_custom,
                        ft.Icons.EDIT,
                        COLOR_SUCCESS,
                    ),
                    col={"xs": 6, "md": 3},
                ),
                ft.Container(
                    content=self._metric_tile(
                        "Записи",
                        self._dataset_samples,
                        ft.Icons.DATA_ARRAY,
                        COLOR_ON_SURFACE,
                    ),
                    col={"xs": 6, "md": 3},
                ),
                ft.Container(
                    content=self._metric_tile(
                        "Аугментации",
                        self._dataset_augmented,
                        ft.Icons.AUTO_FIX_HIGH,
                        COLOR_WARNING,
                    ),
                    col={"xs": 6, "md": 3},
                ),
            ],
        )
        card = surface_card(
            ft.Column(
                spacing=12,
                expand=True,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    ft.Row(
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.Icons.DATASET, size=18, color=COLOR_ACCENT),
                            ft.Text(
                                "Данные жестов",
                                size=15,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                                expand=True,
                            ),
                        ],
                    ),
                    metrics,
                    ft.Container(
                        content=self._dataset_column,
                        expand=True,
                    ),
                ],
            ),
            padding=14,
            radius=8,
        )
        card.expand = True
        return card

    def _on_sync_click(self, _e) -> None:
        self._controller.sync_dataset_to_db()
        self._refresh()
        total = len(self._rows)
        samples = sum(int(row.get("sampleCount") or 0) for row in self._rows)
        if total == 0:
            self._info_text.value = (
                "Пока нет пользовательских записей. Запиши жест на вкладке «Обучение»."
            )
            self._info_text.color = COLOR_MUTED
        else:
            self._info_text.value = (
                f"✓ Обновлено: пользовательских жестов {total}, сэмплов {samples}"
            )
            self._info_text.color = COLOR_SUCCESS
        try:
            self._info_text.update()
        except Exception:
            pass

    def build(self) -> ft.Control:
        sync_btn = ft.FilledButton(
            content=ft.Text("Обновить", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SYNC,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            ),
            on_click=self._on_sync_click,
            tooltip="Обновить список записанных жестов",
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
                                            "стандартные и свои жесты",
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
                        controls=[sync_btn],
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


class DatasetGesturesView(GesturesView):
    """Separate screen for gesture dataset classes and user-recorded samples."""

    def _refresh(self) -> None:
        try:
            self._rows = self._controller.get_db_gestures()
        except Exception:
            self._rows = []
        list_recorded = getattr(self._controller, "list_recorded_gestures", None)
        if callable(list_recorded):
            try:
                self._dataset_rows = list_recorded()
            except Exception:
                self._dataset_rows = []
        else:
            self._dataset_rows = []
        self._render()

    def _render(self) -> None:
        self._render_dataset_manager()
        try:
            self._page.update()
        except Exception:
            pass

    def build(self) -> ft.Control:
        sync_btn = ft.FilledButton(
            content=ft.Text("Обновить", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SYNC,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            ),
            on_click=self._on_sync_click,
            tooltip="Обновить список записанных жестов",
        )
        header = surface_card(
            ft.Row(
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        width=40,
                        height=40,
                        border_radius=8,
                        bgcolor="#171A1D",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(ft.Icons.DATASET, color=COLOR_ACCENT, size=22),
                    ),
                    ft.Column(
                        spacing=2,
                        expand=True,
                        controls=[
                            ft.Text(
                                "Данные жестов",
                                size=20,
                                weight=ft.FontWeight.BOLD,
                                color=COLOR_ON_SURFACE,
                            ),
                            ft.Text(
                                "классы, записи и пользовательские сэмплы",
                                size=12,
                                color=COLOR_MUTED,
                            ),
                        ],
                    ),
                    sync_btn,
                ],
            ),
            padding=14,
            radius=8,
        )
        return ft.Column(
            spacing=12,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                header,
                ft.Container(
                    content=self._build_dataset_manager_card(),
                    expand=True,
                ),
                self._info_text,
            ],
        )
