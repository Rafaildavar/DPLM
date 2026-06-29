"""
Экран «Обучение» — UI-обёртка над CLI ``cv/record_gestures.py`` и
``cv/train_classifier.py``.

Обычный режим оставляет только параметры, которые нужны пользователю:
имя жеста, число сэмплов, режим двух рук и запуск обучения.

Режим разработчика сохраняет прежние технические поля: длину записи,
папку датасета, путь модели и количество соседей K.

После успешного обучения модель попадает в ``models/knn.pkl`` —
``GestureOnlineInfer`` подхватит её при следующем запуске встроенного
распознавания на Главной.
"""
from __future__ import annotations

import base64
import threading

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_SURFACE_HIGH,
    COLOR_WARNING,
    surface_card,
)


_MAX_LOG_LINES = 400
_DEFAULT_DATA_ROOT = "data/gestures"
_DEFAULT_MODEL_OUT = "models/knn.pkl"
_DEFAULT_RECORD_SAMPLES = 8
_DEFAULT_RECORD_FRAMES = 30
_DEFAULT_DYNAMIC_MODEL_OUT = "models/dynamic_knn.pkl"
_DYNAMIC_MODEL_OUT_BY_TYPE = {
    "knn": "models/dynamic_knn.pkl",
    "svm": "models/dynamic_svm.pkl",
    "extra_trees": "models/dynamic_extra_trees.pkl",
    "sequence_knn": "models/dynamic_sequence_knn.pkl",
    "sequence_mlp": "models/dynamic_sequence_mlp.pkl",
    "rf": "models/dynamic_rf.pkl",
    "logreg": "models/dynamic_logreg.pkl",
}
_DEFAULT_DYNAMIC_CLASSES_OUT = "models/dynamic_classes.json"
_DEFAULT_DYNAMIC_FEATURE_DIM_OUT = "models/dynamic_feature_dim.txt"
_DEFAULT_DYNAMIC_FEATURE_MODE_OUT = "models/dynamic_feature_mode.txt"
_DEFAULT_DYNAMIC_RECORD_SAMPLES = 10
_DEFAULT_DYNAMIC_RECORD_FRAMES = 36
_DEFAULT_NEGATIVE_SAMPLES_PER_LABEL = 20
_DEFAULT_NEGATIVE_SEED = 42
_STATIC_TRAINING_SCOPE = "static,quasi_static,negative"
_DYNAMIC_TRAINING_SCOPE = "dynamic,negative"
_DEFAULT_DYNAMIC_FEATURE_MODE = "dynamic_stats"
_DEFAULT_MODEL_TYPE = "knn"
_DEFAULT_K_NEIGHBORS = 5
_PLACEHOLDER_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _dynamic_model_out_for_type(model_type: str) -> str:
    clean = str(model_type or _DEFAULT_MODEL_TYPE).strip().lower()
    return _DYNAMIC_MODEL_OUT_BY_TYPE.get(clean, f"models/dynamic_{clean}.pkl")


def _known_dynamic_model_outputs() -> set[str]:
    return set(_DYNAMIC_MODEL_OUT_BY_TYPE.values())


def _dynamic_metadata_out_for_type(model_type: str) -> tuple[str, str, str]:
    clean = str(model_type or _DEFAULT_MODEL_TYPE).strip().lower()
    if clean == "sequence_knn":
        return (
            "models/dynamic_sequence_classes.json",
            "models/dynamic_sequence_feature_dim.txt",
            "models/dynamic_sequence_feature_mode.txt",
        )
    if clean == "sequence_mlp":
        return (
            "models/dynamic_sequence_mlp_classes.json",
            "models/dynamic_sequence_mlp_feature_dim.txt",
            "models/dynamic_sequence_mlp_feature_mode.txt",
        )
    return (
        _DEFAULT_DYNAMIC_CLASSES_OUT,
        _DEFAULT_DYNAMIC_FEATURE_DIM_OUT,
        _DEFAULT_DYNAMIC_FEATURE_MODE_OUT,
    )


class TrainingView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller
        self._visible = False
        self._frame_update_lock = threading.Lock()
        self._frame_update_pending = False

        # --- Поля обычного режима ----------------------------------------
        self._user_rec_label = ft.TextField(
            label="Имя жеста",
            hint_text="например: zoom",
            border_color=COLOR_SURFACE_HIGH,
        )
        self._user_rec_samples = ft.TextField(
            label="Реальных дублей",
            value=str(_DEFAULT_RECORD_SAMPLES),
            width=140,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._user_rec_two_hands = ft.Switch(
            label="Две руки", value=False, active_color=COLOR_ACCENT
        )
        self._user_rec_start_btn = ft.FilledButton(
            content=ft.Text("Записать примеры", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.FIBER_MANUAL_RECORD,
            style=ft.ButtonStyle(
                bgcolor=COLOR_DANGER,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=14),
            ),
            on_click=lambda e: self._on_record_start(e, mode="user"),
        )
        self._user_tr_start_btn = ft.FilledButton(
            content=ft.Text("Обучить модель", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.MODEL_TRAINING,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=14),
            ),
            on_click=lambda e: self._on_train_start(e, mode="user"),
        )

        # --- Поля «Запись примеров» --------------------------------------
        self._rec_label = ft.TextField(
            label="Имя жеста",
            hint_text="например: zoom",
            border_color=COLOR_SURFACE_HIGH,
        )
        self._rec_samples = ft.TextField(
            label="Реальных дублей",
            value=str(_DEFAULT_RECORD_SAMPLES),
            width=120,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._rec_frames = ft.TextField(
            label="Длина (кадров)",
            value=str(_DEFAULT_RECORD_FRAMES),
            width=160,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._rec_two_hands = ft.Switch(
            label="Две руки", value=False, active_color=COLOR_ACCENT
        )
        self._rec_start_btn = ft.FilledButton(
            content=ft.Text("Записать примеры", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.FIBER_MANUAL_RECORD,
            style=ft.ButtonStyle(
                bgcolor=COLOR_DANGER,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=14),
            ),
            on_click=lambda e: self._on_record_start(e, mode="developer"),
        )

        # --- Поля «Динамический жест» ------------------------------------
        self._dyn_rec_label = ft.TextField(
            label="Имя динамического жеста",
            hint_text="например: swipe_right",
            border_color=COLOR_SURFACE_HIGH,
        )
        self._dyn_rec_samples = ft.TextField(
            label="Реальных дублей",
            value=str(_DEFAULT_DYNAMIC_RECORD_SAMPLES),
            width=120,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._dyn_rec_frames = ft.TextField(
            label="Длина (кадров)",
            value=str(_DEFAULT_DYNAMIC_RECORD_FRAMES),
            width=160,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._dyn_rec_two_hands = ft.Switch(
            label="Две руки", value=False, active_color=COLOR_ACCENT
        )
        self._dyn_rec_start_btn = ft.FilledButton(
            content=ft.Text("Записать динамику", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.FIBER_MANUAL_RECORD,
            style=ft.ButtonStyle(
                bgcolor=COLOR_DANGER,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=14),
            ),
            on_click=lambda e: self._on_record_start(e, mode="dynamic"),
        )
        self._dyn_feature_mode = ft.Dropdown(
            label="Признаки",
            value=_DEFAULT_DYNAMIC_FEATURE_MODE,
            border_color=COLOR_SURFACE_HIGH,
            options=[
                ft.DropdownOption(key="dynamic_stats", text="dynamic_stats"),
                ft.DropdownOption(key="dynamic_sequence", text="dynamic_sequence"),
                ft.DropdownOption(key="hybrid_stats", text="hybrid_stats"),
                ft.DropdownOption(key="static_stats", text="static_stats"),
            ],
            editable=False,
        )
        self._dyn_model_type = ft.Dropdown(
            label="Модель",
            value=_DEFAULT_MODEL_TYPE,
            border_color=COLOR_SURFACE_HIGH,
            options=[
                ft.DropdownOption(key="knn", text="knn"),
                ft.DropdownOption(key="svm", text="svm"),
                ft.DropdownOption(key="extra_trees", text="extra_trees"),
                ft.DropdownOption(key="sequence_knn", text="sequence_knn"),
                ft.DropdownOption(key="sequence_mlp", text="sequence_mlp"),
                ft.DropdownOption(key="rf", text="rf"),
                ft.DropdownOption(key="logreg", text="logreg"),
            ],
            editable=False,
            on_select=self._on_dynamic_model_type_changed,
        )
        self._dyn_model_out = ft.TextField(
            label="Файл dynamic-модели",
            value=_DEFAULT_DYNAMIC_MODEL_OUT,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._dyn_tr_neighbors = ft.TextField(
            label="K",
            value=str(_DEFAULT_K_NEIGHBORS),
            width=100,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._dyn_tr_start_btn = ft.FilledButton(
            content=ft.Text("Обучить dynamic модель", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.MODEL_TRAINING,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=14),
            ),
            on_click=lambda e: self._on_train_start(e, mode="dynamic"),
        )

        # --- Поля «Негативные примеры» -----------------------------------
        self._neg_samples_per_label = ft.TextField(
            label="Сэмплов на negative-класс",
            value=str(_DEFAULT_NEGATIVE_SAMPLES_PER_LABEL),
            width=220,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._neg_seed = ft.TextField(
            label="Seed",
            value=str(_DEFAULT_NEGATIVE_SEED),
            width=120,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._neg_generate_btn = ft.FilledButton(
            content=ft.Text("Сгенерировать negative", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.MODEL_TRAINING,
            style=ft.ButtonStyle(
                bgcolor=COLOR_SURFACE_HIGH,
                color=COLOR_ON_SURFACE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=14),
            ),
            on_click=self._on_negative_generate,
        )

        # --- Поля «Обучение» ---------------------------------------------
        self._tr_data_root = ft.TextField(
            label="Папка датасета",
            value=_DEFAULT_DATA_ROOT,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._tr_out_path = ft.TextField(
            label="Выходной файл модели",
            value=_DEFAULT_MODEL_OUT,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._tr_neighbors = ft.TextField(
            label="K (соседи)",
            value=str(_DEFAULT_K_NEIGHBORS),
            width=120,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._tr_model_type = ft.Dropdown(
            label="Модель",
            value=_DEFAULT_MODEL_TYPE,
            border_color=COLOR_SURFACE_HIGH,
            options=[
                ft.DropdownOption(key="knn", text="knn"),
                ft.DropdownOption(key="svm", text="svm"),
                ft.DropdownOption(key="extra_trees", text="extra_trees"),
                ft.DropdownOption(key="rf", text="rf"),
                ft.DropdownOption(key="logreg", text="logreg"),
            ],
            editable=False,
        )
        self._tr_start_btn = ft.FilledButton(
            content=ft.Text("Обучить модель", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.MODEL_TRAINING,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=14),
            ),
            on_click=lambda e: self._on_train_start(e, mode="developer"),
        )

        self._cancel_btn = ft.OutlinedButton(
            content=ft.Text("Остановить"),
            icon=ft.Icons.STOP,
            on_click=self._on_cancel,
            disabled=True,
        )

        # --- Лог -----------------------------------------------------------
        self._log_text = ft.Text(
            "Готов к работе.\n",
            size=12,
            color=COLOR_ON_SURFACE,
            selectable=True,
            font_family="Menlo",
        )
        self._log_scroll = ft.Container(
            content=ft.Column(
                controls=[self._log_text],
                scroll=ft.ScrollMode.AUTO,
                spacing=0,
            ),
            bgcolor="#0e0e18",
            border_radius=10,
            padding=10,
            height=260,
            width=float("inf"),
        )

        # --- Список записанных классов ------------------------------------
        self._datasets_column = ft.Column(spacing=8)
        self._dataset_classes_value = ft.Text(
            "0",
            size=18,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._dataset_samples_value = ft.Text(
            "0",
            size=18,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._dataset_real_aug_value = ft.Text(
            "0 / 0",
            size=18,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_SURFACE,
        )
        self._dataset_balance_value = ft.Text(
            "empty",
            size=18,
            weight=ft.FontWeight.BOLD,
            color=COLOR_MUTED,
        )
        self._pending_delete_label = ""
        self._last_recording_state: dict = {"active": False}

        # --- Превью записи -------------------------------------------------
        self._camera_image = ft.Image(
            src=_PLACEHOLDER_DATA_URL,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
            expand=True,
            visible=False,
        )
        self._recording_badge = ft.Container(
            visible=False,
            bgcolor=COLOR_DANGER,
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=10, vertical=5),
            content=ft.Row(
                spacing=6,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.FIBER_MANUAL_RECORD, color=ft.Colors.WHITE, size=14),
                    ft.Text(
                        "REC",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.WHITE,
                    ),
                ],
            ),
        )
        self._recording_title = ft.Text(
            "Запись не запущена",
            size=14,
            weight=ft.FontWeight.W_600,
            color=COLOR_ON_SURFACE,
        )
        self._recording_detail = ft.Text(
            "После старта здесь появится камера и прогресс сохранения сэмплов.",
            size=12,
            color=COLOR_MUTED,
        )
        self._recording_progress = ft.ProgressBar(
            value=0.0,
            color=COLOR_ACCENT,
            bgcolor=COLOR_SURFACE_HIGH,
        )
        self._recording_progress_text = ft.Text("0%", size=12, color=COLOR_MUTED)

        frame_event = getattr(controller, "camera_frame_updated", None)
        if frame_event is not None:
            frame_event.connect(self._on_frame)
        camera_event = getattr(controller, "camera_active_changed", None)
        if camera_event is not None:
            camera_event.connect(self._on_camera_active)
        recording_event = getattr(controller, "sample_recording_changed", None)
        if recording_event is not None:
            recording_event.connect(self._on_recording_state)

    # ---- Жизненный цикл --------------------------------------------------

    def on_show(self) -> None:
        self._visible = True
        self._refresh_datasets()

    def on_hide(self) -> None:
        self._visible = False

    # ---- Хелперы ---------------------------------------------------------

    def _refresh_datasets(self) -> None:
        rows = self._controller.list_recorded_gestures()
        self._datasets_column.controls.clear()
        total_samples = 0
        total_real_samples = 0
        total_augmented_samples = 0
        class_counts: list[int] = []
        if not rows:
            self._datasets_column.controls.append(
                ft.Text(
                    "В папке data/gestures/ пока ничего нет — запишите первый жест выше.",
                    size=12,
                    color=COLOR_MUTED,
                )
            )
        else:
            for row in rows:
                label = str(row["label"])
                samples = int(row["samples"])
                real_samples = int(row.get("realSamples") or samples)
                augmented_samples = int(row.get("augmentedSamples") or 0)
                total_samples += samples
                total_real_samples += real_samples
                total_augmented_samples += augmented_samples
                class_counts.append(samples)
                pending_delete = self._pending_delete_label == label
                actions: list[ft.Control]
                if pending_delete:
                    actions = [
                        ft.Text("подтвердить", size=12, color=COLOR_DANGER),
                        ft.IconButton(
                            icon=ft.Icons.DELETE_FOREVER,
                            icon_color=COLOR_DANGER,
                            tooltip=f"Подтвердить удаление {label}",
                            on_click=lambda _e, value=label: self._delete_samples(value),
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            icon_color=COLOR_MUTED,
                            tooltip="Отмена",
                            on_click=lambda _e: self._cancel_delete_samples(),
                        ),
                    ]
                else:
                    actions = [
                        ft.IconButton(
                            icon=ft.Icons.DELETE,
                            icon_color=COLOR_DANGER,
                            tooltip=f"Удалить семплы {label}",
                            on_click=lambda _e, value=label: self._request_delete_samples(value),
                        )
                    ]
                self._datasets_column.controls.append(
                    ft.Container(
                        bgcolor="#1A1E22",
                        border_radius=8,
                        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                        content=ft.Row(
                            spacing=10,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Icon(ft.Icons.FOLDER, size=18, color=COLOR_ACCENT),
                                ft.Text(
                                    label,
                                    size=13,
                                    weight=ft.FontWeight.W_600,
                                    color=COLOR_ON_SURFACE,
                                    expand=True,
                                ),
                                ft.Text(
                                    (
                                        f"{samples} обучающих "
                                        f"({real_samples} real + {augmented_samples} aug)"
                                        if augmented_samples
                                        else f"{samples} сэмплов"
                                    ),
                                    size=12,
                                    color=COLOR_MUTED,
                                ),
                                *actions,
                            ],
                        ),
                    )
                )
        if class_counts:
            spread = max(class_counts) - min(class_counts)
            tolerance = max(3, int(max(class_counts) * 0.25))
            balance_label = "healthy" if spread <= tolerance else "uneven"
            balance_color = COLOR_SUCCESS if spread <= tolerance else COLOR_WARNING
        else:
            balance_label = "empty"
            balance_color = COLOR_MUTED
        self._dataset_classes_value.value = str(len(rows))
        self._dataset_samples_value.value = str(total_samples)
        self._dataset_real_aug_value.value = f"{total_real_samples} / {total_augmented_samples}"
        self._dataset_balance_value.value = balance_label
        self._dataset_balance_value.color = balance_color
        try:
            self._datasets_column.update()
            self._dataset_classes_value.update()
            self._dataset_samples_value.update()
            self._dataset_real_aug_value.update()
            self._dataset_balance_value.update()
        except Exception:
            pass

    def _request_delete_samples(self, label: str) -> None:
        self._pending_delete_label = label
        self._refresh_datasets()

    def _cancel_delete_samples(self) -> None:
        self._pending_delete_label = ""
        self._refresh_datasets()

    def _delete_samples(self, label: str) -> None:
        self._pending_delete_label = ""
        summary = self._controller.delete_recorded_samples(label)
        if summary.get("ok"):
            self._append_log(
                f"[✓] Удалены семплы «{label}»: файлов {summary['filesDeleted']}, "
                f"строк БД {summary['sampleRowsDeleted']}, "
                f"отвязано команд {summary['commandsUnbound']}"
            )
            self._append_log("[i] Переобучи модель, чтобы удалить этот класс из knn.pkl/classes.json")
        else:
            self._append_log(f"[!] Удаление «{label}»: {summary.get('error') or 'ошибка'}")
        self._refresh_datasets()

    def _append_log(self, line: str) -> None:
        # Этот метод вызывается из фонового потока — маршалируем в UI.
        self._page.run_thread(self._apply_log_line, line)

    def _apply_log_line(self, line: str) -> None:
        current = self._log_text.value or ""
        lines = current.split("\n")
        lines.append(line)
        if len(lines) > _MAX_LOG_LINES:
            lines = lines[-_MAX_LOG_LINES:]
        self._log_text.value = "\n".join(lines)
        try:
            self._log_text.update()
        except Exception:
            pass

    def _set_running(self, running: bool) -> None:
        self._page.run_thread(self._apply_running_state, running)

    def _apply_running_state(self, running: bool) -> None:
        for control in (
            self._user_rec_start_btn,
            self._user_tr_start_btn,
            self._rec_start_btn,
            self._tr_start_btn,
            self._dyn_rec_start_btn,
            self._dyn_tr_start_btn,
            self._neg_generate_btn,
        ):
            control.disabled = running
        self._cancel_btn.disabled = not running
        for control in (
            self._user_rec_start_btn,
            self._user_tr_start_btn,
            self._rec_start_btn,
            self._tr_start_btn,
            self._dyn_rec_start_btn,
            self._dyn_tr_start_btn,
            self._neg_generate_btn,
            self._cancel_btn,
        ):
            try:
                control.update()
            except Exception:
                pass

    def _on_frame(self) -> None:
        if not self._visible:
            return
        with self._frame_update_lock:
            if self._frame_update_pending:
                return
            self._frame_update_pending = True
        data = self._controller.latest_jpeg_bytes
        if not data:
            with self._frame_update_lock:
                self._frame_update_pending = False
            return
        b64 = base64.b64encode(data).decode("ascii")
        try:
            self._page.run_thread(
                self._apply_frame,
                f"data:image/jpeg;base64,{b64}",
            )
        except Exception:
            with self._frame_update_lock:
                self._frame_update_pending = False

    def _apply_frame(self, data_url: str) -> None:
        try:
            if not self._visible:
                return
            self._camera_image.src = data_url
            self._camera_image.visible = True
            self._camera_image.update()
        except Exception:
            try:
                self._page.update()
            except Exception:
                pass
        finally:
            with self._frame_update_lock:
                self._frame_update_pending = False

    def _on_camera_active(self, active: bool) -> None:
        self._page.run_thread(self._apply_camera_active, active)

    def _apply_camera_active(self, active: bool) -> None:
        if active:
            if not self._last_recording_state.get("active"):
                self._recording_title.value = "Камера активна"
                self._recording_detail.value = "Кадр готов к записи жеста."
        else:
            self._camera_image.src = _PLACEHOLDER_DATA_URL
            self._camera_image.visible = False
            self._recording_badge.visible = False
            self._recording_progress.value = 0.0
            self._recording_progress_text.value = "0%"
            self._recording_title.value = "Камера остановлена"
            self._recording_detail.value = "Нажми «Записать примеры», чтобы начать."
        self._update_recording_controls()

    def _on_recording_state(self, state: dict) -> None:
        self._page.run_thread(self._apply_recording_state, state)

    def _apply_recording_state(self, state: dict) -> None:
        self._last_recording_state = dict(state or {})
        active = bool(self._last_recording_state.get("active"))
        label = str(self._last_recording_state.get("label") or "—")
        progress = max(0.0, min(1.0, float(self._last_recording_state.get("progress") or 0.0)))
        saved = int(self._last_recording_state.get("saved") or 0)
        target = int(self._last_recording_state.get("target") or 0)
        sample_index = int(self._last_recording_state.get("sampleIndex") or 0)
        current_frames = int(self._last_recording_state.get("currentFrames") or 0)
        target_frames = int(self._last_recording_state.get("targetFrames") or 0)
        message = str(self._last_recording_state.get("message") or "")

        self._recording_badge.visible = active
        self._recording_progress.value = progress
        self._recording_progress.color = COLOR_DANGER if active else COLOR_SUCCESS
        self._recording_progress_text.value = f"{int(round(progress * 100))}%"
        if active:
            self._recording_title.value = f"Записывается жест «{label}»"
            self._recording_detail.value = (
                f"Сохранено {saved}/{target}; текущий сэмпл {sample_index}/{target}, "
                f"кадры {current_frames}/{target_frames}"
            )
            if message:
                self._recording_detail.value += f" · {message}"
        else:
            self._recording_title.value = "Запись завершена" if progress >= 1.0 else "Запись не запущена"
            self._recording_detail.value = message or "Нажми «Записать примеры», чтобы начать."
        self._update_recording_controls()

    def _update_recording_controls(self) -> None:
        for control in (
            self._camera_image,
            self._recording_badge,
            self._recording_title,
            self._recording_detail,
            self._recording_progress,
            self._recording_progress_text,
        ):
            try:
                control.update()
            except Exception:
                pass

    # ---- Действия --------------------------------------------------------

    def _parse_int(self, value: str, default: int) -> int:
        try:
            return int((value or "").strip())
        except (TypeError, ValueError):
            return default

    def _on_dynamic_model_type_changed(self, _e) -> None:
        model_type = str(self._dyn_model_type.value or "knn")
        current = str(self._dyn_model_out.value or "").strip()
        suggested = _dynamic_model_out_for_type(model_type)
        if not current or current in _known_dynamic_model_outputs():
            self._dyn_model_out.value = suggested
            try:
                self._dyn_model_out.update()
            except Exception:
                pass
        if model_type in {"sequence_knn", "sequence_mlp"}:
            self._dyn_feature_mode.value = "dynamic_sequence"
            try:
                self._dyn_feature_mode.update()
            except Exception:
                pass

    def _on_record_start(self, _e, *, mode: str = "developer") -> None:
        if mode == "user":
            label = (self._user_rec_label.value or "").strip()
            samples = max(
                1,
                self._parse_int(
                    self._user_rec_samples.value,
                    _DEFAULT_RECORD_SAMPLES,
                ),
            )
            frames = _DEFAULT_RECORD_FRAMES
            two_hands = bool(self._user_rec_two_hands.value)
        elif mode == "dynamic":
            label = (self._dyn_rec_label.value or "").strip()
            samples = max(
                1,
                self._parse_int(
                    self._dyn_rec_samples.value,
                    _DEFAULT_DYNAMIC_RECORD_SAMPLES,
                ),
            )
            frames = max(
                1,
                self._parse_int(
                    self._dyn_rec_frames.value,
                    _DEFAULT_DYNAMIC_RECORD_FRAMES,
                ),
            )
            two_hands = bool(self._dyn_rec_two_hands.value)
        else:
            label = (self._rec_label.value or "").strip()
            samples = max(
                1,
                self._parse_int(self._rec_samples.value, _DEFAULT_RECORD_SAMPLES),
            )
            frames = max(
                1,
                self._parse_int(self._rec_frames.value, _DEFAULT_RECORD_FRAMES),
            )
            two_hands = bool(self._rec_two_hands.value)

        if not label:
            self._append_log("[!] Укажи имя жеста")
            return

        self._append_log(
            f"[i] Запись «{label}»: {samples} реальных дублей, {frames} кадров"
            + (" (две руки)" if two_hands else "")
        )
        self._append_log(
            "Запись выполняется во встроенной камере: держи жест в кадре, "
            "сэмплы сохранятся автоматически."
        )

        ok = self._controller.start_recording(
            label=label,
            num_samples=samples,
            frames=frames,
            two_hands=two_hands,
            include_global_motion=(mode == "dynamic"),
            on_line=self._append_log,
            on_done=self._on_subprocess_done,
        )
        if not ok:
            self._append_log("[!] Не удалось запустить запись (возможно, уже идёт другая задача).")
            return
        self._set_running(True)

    def _on_negative_generate(self, _e) -> None:
        samples = max(
            1,
            self._parse_int(
                self._neg_samples_per_label.value,
                _DEFAULT_NEGATIVE_SAMPLES_PER_LABEL,
            ),
        )
        seed = self._parse_int(self._neg_seed.value, _DEFAULT_NEGATIVE_SEED)
        self._append_log(
            "[i] Генерация negative samples: пользователь записывает только "
            "настоящие dynamic-жесты, отрицательные траектории строятся автоматически."
        )
        self._append_log(f"[i] samples_per_label={samples}, seed={seed}")
        ok = self._controller.start_negative_generation(
            samples_per_label=samples,
            seed=seed,
            on_line=self._append_log,
            on_done=self._on_subprocess_done,
        )
        if not ok:
            self._append_log(
                "[!] Не удалось запустить генерацию negative (возможно, уже идёт другая задача)."
            )
            return
        self._set_running(True)

    def _on_train_start(self, _e, *, mode: str = "developer") -> None:
        if mode == "user":
            data_root = _DEFAULT_DATA_ROOT
            out_path = _DEFAULT_MODEL_OUT
            neighbors = _DEFAULT_K_NEIGHBORS
            feature_mode = "static_mean"
            expect_dim = 42
            model_type = _DEFAULT_MODEL_TYPE
            classes_out_path = ""
            feature_dim_out_path = ""
            feature_mode_out_path = ""
            training_scope = _STATIC_TRAINING_SCOPE
            self._append_log(
                "[i] Обучение KNN со стандартными static/negative параметрами проекта"
            )
        elif mode == "dynamic":
            data_root = _DEFAULT_DATA_ROOT
            neighbors = max(
                1,
                self._parse_int(self._dyn_tr_neighbors.value, _DEFAULT_K_NEIGHBORS),
            )
            feature_mode = (
                str(self._dyn_feature_mode.value or _DEFAULT_DYNAMIC_FEATURE_MODE)
                .strip()
                or _DEFAULT_DYNAMIC_FEATURE_MODE
            )
            expect_dim = None
            model_type = str(self._dyn_model_type.value or _DEFAULT_MODEL_TYPE).strip()
            out_path = (
                self._dyn_model_out.value
                or _dynamic_model_out_for_type(model_type)
            ).strip()
            (
                classes_out_path,
                feature_dim_out_path,
                feature_mode_out_path,
            ) = _dynamic_metadata_out_for_type(model_type)
            training_scope = _DYNAMIC_TRAINING_SCOPE
            self._append_log(
                f"[i] Обучение отдельной dynamic-модели: "
                f"model={model_type}, feature_mode={feature_mode}, scope={training_scope}"
            )
        else:
            data_root = (self._tr_data_root.value or _DEFAULT_DATA_ROOT).strip()
            out_path = (self._tr_out_path.value or _DEFAULT_MODEL_OUT).strip()
            neighbors = max(
                1,
                self._parse_int(self._tr_neighbors.value, _DEFAULT_K_NEIGHBORS),
            )
            feature_mode = "static_mean"
            expect_dim = 42
            model_type = str(self._tr_model_type.value or _DEFAULT_MODEL_TYPE).strip()
            classes_out_path = ""
            feature_dim_out_path = ""
            feature_mode_out_path = ""
            training_scope = _STATIC_TRAINING_SCOPE

        self._append_log(
            f"[i] Обучение: data={data_root}, out={out_path}, "
            f"model={model_type}, k={neighbors}, feature_mode={feature_mode}"
        )
        ok = self._controller.start_training(
            data_root=data_root,
            out_path=out_path,
            neighbors=neighbors,
            expect_dim=expect_dim,
            feature_mode=feature_mode,
            model_type=model_type,
            classes_out_path=classes_out_path,
            feature_dim_out_path=feature_dim_out_path,
            feature_mode_out_path=feature_mode_out_path,
            training_scope=training_scope,
            on_line=self._append_log,
            on_done=self._on_subprocess_done,
        )
        if not ok:
            self._append_log("[!] Не удалось запустить обучение (возможно, уже идёт другая задача).")
            return
        self._set_running(True)

    def _on_cancel(self, _e) -> None:
        self._controller.cancel_training()
        self._append_log("[i] Остановка процесса…")

    def _on_subprocess_done(self, code: int) -> None:
        status = "✓ успешно" if code == 0 else f"⚠ код выхода {code}"
        self._append_log(f"[i] Процесс завершён ({status})")
        self._set_running(False)
        self._page.run_thread(self._refresh_datasets)

    # ---- Сборка дерева ---------------------------------------------------

    def _section_title(self, text: str) -> ft.Text:
        return ft.Text(
            text,
            size=14,
            weight=ft.FontWeight.W_600,
            color=COLOR_ON_SURFACE,
        )

    def _panel_title(
        self,
        icon: str,
        title: str,
        *,
        color: str = COLOR_ACCENT,
        trailing: ft.Control | None = None,
    ) -> ft.Row:
        controls: list[ft.Control] = [
            ft.Container(
                width=30,
                height=30,
                border_radius=8,
                bgcolor="#1A1E22",
                alignment=ft.Alignment.CENTER,
                content=ft.Icon(icon, size=17, color=color),
            ),
            ft.Text(
                title,
                size=14,
                weight=ft.FontWeight.W_600,
                color=COLOR_ON_SURFACE,
                expand=True,
            ),
        ]
        if trailing is not None:
            controls.append(trailing)
        return ft.Row(
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=controls,
        )

    def _status_chip(self, icon: str, label: str, color: str) -> ft.Container:
        return ft.Container(
            border_radius=8,
            bgcolor="#1A1E22",
            padding=ft.Padding.symmetric(horizontal=10, vertical=7),
            content=ft.Row(
                spacing=7,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(icon, size=14, color=color),
                    ft.Text(label, size=12, color=COLOR_ON_SURFACE, no_wrap=True),
                ],
            ),
        )

    def _metric_tile(
        self,
        icon: str,
        label: str,
        value: ft.Text,
        *,
        color: str = COLOR_ACCENT,
    ) -> ft.Container:
        return ft.Container(
            bgcolor="#171A1D",
            border_radius=8,
            padding=12,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Row(
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(icon, size=16, color=color),
                            ft.Text(label, size=12, color=COLOR_MUTED),
                        ],
                    ),
                    value,
                ],
            ),
        )

    def _workflow_step(
        self,
        label: str,
        *,
        active: bool = False,
        done: bool = False,
    ) -> ft.Container:
        color = COLOR_SUCCESS if done else COLOR_ACCENT if active else COLOR_MUTED
        return ft.Container(
            bgcolor="#171A1D" if not active else "#1D3034",
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            content=ft.Row(
                spacing=7,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(
                        ft.Icons.CHECK_CIRCLE if done else ft.Icons.RADIO_BUTTON_UNCHECKED,
                        size=14,
                        color=color,
                    ),
                    ft.Text(label, size=12, color=COLOR_ON_SURFACE if active else COLOR_MUTED),
                ],
            ),
        )

    def _build_dataset_manager_card(self) -> ft.Control:
        metrics = ft.ResponsiveRow(
            spacing=10,
            run_spacing=10,
            controls=[
                ft.Container(
                    content=self._metric_tile(
                        ft.Icons.FOLDER,
                        "Классы",
                        self._dataset_classes_value,
                    ),
                    col={"xs": 6, "md": 3},
                ),
                ft.Container(
                    content=self._metric_tile(
                        ft.Icons.DATA_ARRAY,
                        "Сэмплы",
                        self._dataset_samples_value,
                        color=COLOR_SUCCESS,
                    ),
                    col={"xs": 6, "md": 3},
                ),
                ft.Container(
                    content=self._metric_tile(
                        ft.Icons.AUTO_FIX_HIGH,
                        "real / aug",
                        self._dataset_real_aug_value,
                        color=COLOR_WARNING,
                    ),
                    col={"xs": 6, "md": 3},
                ),
                ft.Container(
                    content=self._metric_tile(
                        ft.Icons.BALANCE,
                        "Баланс",
                        self._dataset_balance_value,
                        color=COLOR_ACCENT,
                    ),
                    col={"xs": 6, "md": 3},
                ),
            ],
        )
        card = surface_card(
            ft.Column(
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(
                        ft.Icons.DATASET,
                        "Менеджер датасета",
                        trailing=ft.IconButton(
                            icon=ft.Icons.REFRESH,
                            tooltip="Обновить",
                            icon_color=COLOR_MUTED,
                            on_click=lambda _e: self._refresh_datasets(),
                        ),
                    ),
                    metrics,
                    self._datasets_column,
                ],
            ),
            padding=14,
            radius=8,
        )
        card.width = float("inf")
        return card

    def _build_workflow_card(self) -> ft.Control:
        card = surface_card(
            ft.Column(
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(
                        ft.Icons.ACCOUNT_TREE,
                        "Пайплайн обучения",
                        color=COLOR_SUCCESS,
                    ),
                    ft.Row(
                        spacing=8,
                        wrap=True,
                        controls=[
                            self._workflow_step("Запись", active=True),
                            self._workflow_step("Аугментация"),
                            self._workflow_step("Обучение"),
                            self._workflow_step("Проверка"),
                        ],
                    ),
                    ft.Row(
                        spacing=8,
                        wrap=True,
                        controls=[
                            self._status_chip(
                                ft.Icons.AUTO_FIX_HIGH,
                                "auto-augment",
                                COLOR_SUCCESS,
                            ),
                            self._status_chip(
                                ft.Icons.SECURITY,
                                "negative-защита",
                                COLOR_WARNING,
                            ),
                        ],
                    ),
                ],
            ),
            padding=14,
            radius=8,
        )
        card.width = float("inf")
        return card

    def _build_user_record_card(self) -> ft.Control:
        return surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    self._panel_title(
                        ft.Icons.FIBER_MANUAL_RECORD,
                        "1. Запись",
                        color=COLOR_DANGER,
                    ),
                    self._user_rec_label,
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        controls=[
                            self._user_rec_samples,
                            self._user_rec_two_hands,
                        ],
                    ),
                    self._user_rec_start_btn,
                ],
            ),
            padding=14,
            radius=8,
        )

    def _build_user_train_card(self) -> ft.Control:
        return surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    self._panel_title(
                        ft.Icons.MODEL_TRAINING,
                        "2. Обучение",
                        color=COLOR_ACCENT,
                    ),
                    self._user_tr_start_btn,
                ],
            ),
            padding=14,
            radius=8,
        )

    def _build_developer_record_card(self) -> ft.Control:
        return surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    self._panel_title(
                        ft.Icons.FIBER_MANUAL_RECORD,
                        "1. Запись",
                        color=COLOR_DANGER,
                    ),
                    self._rec_label,
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        controls=[
                            self._rec_samples,
                            self._rec_frames,
                            self._rec_two_hands,
                        ],
                    ),
                    self._rec_start_btn,
                ],
            ),
            padding=14,
            radius=8,
        )

    def _build_developer_train_card(self) -> ft.Control:
        return surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    self._panel_title(
                        ft.Icons.MODEL_TRAINING,
                        "2. Обучение",
                        color=COLOR_ACCENT,
                    ),
                    self._tr_data_root,
                    self._tr_out_path,
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        controls=[
                            ft.Container(content=self._tr_model_type, width=230),
                            self._tr_neighbors,
                        ],
                    ),
                    self._tr_start_btn,
                ],
            ),
            padding=14,
            radius=8,
        )

    def _build_dynamic_card(self) -> ft.Control:
        return surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    self._panel_title(
                        ft.Icons.TIMELINE,
                        "3. Динамика",
                        color=COLOR_SUCCESS,
                    ),
                    self._dyn_rec_label,
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        controls=[
                            self._dyn_rec_samples,
                            self._dyn_rec_frames,
                            self._dyn_rec_two_hands,
                        ],
                    ),
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            self._dyn_rec_start_btn,
                            self._dyn_tr_start_btn,
                        ],
                    ),
                    self._dyn_model_out,
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        controls=[
                            ft.Container(content=self._dyn_feature_mode, width=190),
                            ft.Container(content=self._dyn_model_type, width=150),
                            self._dyn_tr_neighbors,
                        ],
                    ),
                ],
            ),
            padding=14,
            radius=8,
        )

    def _build_negative_card(self) -> ft.Control:
        return surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    self._panel_title(
                        ft.Icons.FILTER_ALT,
                        "4. Negative",
                        color=COLOR_MUTED,
                    ),
                    ft.Row(
                        spacing=10,
                        wrap=True,
                        controls=[
                            self._neg_samples_per_label,
                            self._neg_seed,
                        ],
                    ),
                    self._neg_generate_btn,
                ],
            ),
            padding=14,
            radius=8,
        )

    def _build_recording_preview_card(self) -> ft.Control:
        camera_stage = ft.Container(
            content=ft.Stack(
                expand=True,
                controls=[
                    ft.Container(
                        expand=True,
                        bgcolor="#07090B",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Column(
                            spacing=8,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Icon(
                                    ft.Icons.VIDEO_CAMERA_FRONT,
                                    color=COLOR_SURFACE_HIGH,
                                    size=42,
                                ),
                                ft.Text(
                                    "Камера появится после старта записи",
                                    size=12,
                                    color=COLOR_MUTED,
                                ),
                            ],
                        ),
                    ),
                    ft.Container(
                        content=self._camera_image,
                        expand=True,
                        alignment=ft.Alignment.CENTER,
                    ),
                    ft.Container(
                        content=self._recording_badge,
                        left=12,
                        top=12,
                    ),
                ],
            ),
            bgcolor="#0B0D10",
            border_radius=8,
            padding=8,
            height=360,
            alignment=ft.Alignment.CENTER,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        progress_row = ft.Row(
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Container(content=self._recording_progress, expand=True),
                self._recording_progress_text,
            ],
        )
        return surface_card(
            ft.Column(
                spacing=12,
                controls=[
                    self._panel_title(
                        ft.Icons.VIDEO_CAMERA_FRONT,
                        "Камера и запись",
                    ),
                    self._recording_title,
                    camera_stage,
                    self._recording_detail,
                    progress_row,
                ],
            ),
            padding=14,
            radius=8,
        )

    def _build_training_body(self) -> ft.Control:
        def border_all(color: str) -> ft.Border:
            side = ft.BorderSide(1, color)
            return ft.Border(top=side, right=side, bottom=side, left=side)

        user_body = ft.Container(
            content=ft.Column(
                spacing=10,
                controls=[
                    self._build_user_record_card(),
                    self._build_user_train_card(),
                ],
            ),
            visible=True,
        )
        developer_body = ft.Container(
            content=ft.Column(
                spacing=10,
                controls=[
                    self._build_developer_record_card(),
                    self._build_developer_train_card(),
                    self._build_dynamic_card(),
                    self._build_negative_card(),
                ],
            ),
            visible=False,
        )

        segment_refs: dict[str, tuple[ft.Container, ft.Icon, ft.Text]] = {}

        def apply_mode(mode: str) -> None:
            user_body.visible = mode == "user"
            developer_body.visible = mode == "developer"
            for key, (container, icon, text) in segment_refs.items():
                active = key == mode
                container.bgcolor = "#223238" if active else "#171A1D"
                container.border = border_all(COLOR_ACCENT if active else COLOR_SURFACE_HIGH)
                icon.color = COLOR_ACCENT if active else COLOR_MUTED
                text.color = COLOR_ON_SURFACE if active else COLOR_MUTED
            try:
                user_body.update()
                developer_body.update()
                for container, icon, text in segment_refs.values():
                    container.update()
                    icon.update()
                    text.update()
            except Exception:
                pass

        def make_mode_segment(
            key: str,
            label: str,
            icon_name: str,
            *,
            selected: bool = False,
        ) -> ft.Container:
            icon = ft.Icon(
                icon_name,
                size=16,
                color=COLOR_ACCENT if selected else COLOR_MUTED,
            )
            text = ft.Text(
                label,
                size=12,
                weight=ft.FontWeight.W_600,
                color=COLOR_ON_SURFACE if selected else COLOR_MUTED,
                no_wrap=True,
            )
            container = ft.Container(
                expand=True,
                bgcolor="#223238" if selected else "#171A1D",
                border=border_all(COLOR_ACCENT if selected else COLOR_SURFACE_HIGH),
                border_radius=8,
                padding=ft.Padding.symmetric(horizontal=12, vertical=9),
                ink=False,
                on_click=lambda _e, value=key: apply_mode(value),
                content=ft.Row(
                    spacing=8,
                    tight=True,
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[icon, text],
                ),
            )
            segment_refs[key] = (container, icon, text)
            return container

        mode_picker = ft.Container(
            bgcolor="#111417",
            border=border_all(COLOR_SURFACE_HIGH),
            border_radius=8,
            padding=4,
            content=ft.Row(
                spacing=6,
                controls=[
                    make_mode_segment(
                        "user",
                        "Пользователь",
                        ft.Icons.PERSON,
                        selected=True,
                    ),
                    make_mode_segment(
                        "developer",
                        "Разработчик",
                        ft.Icons.CODE,
                    ),
                ],
            ),
        )

        return ft.Column(
            spacing=10,
            controls=[
                mode_picker,
                user_body,
                developer_body,
            ],
        )

    def build(self) -> ft.Control:
        workflow_strip = ft.Row(
            spacing=8,
            wrap=True,
            controls=[
                self._status_chip(ft.Icons.FIBER_MANUAL_RECORD, "запись", COLOR_DANGER),
                self._status_chip(ft.Icons.DATASET, "датасет", COLOR_WARNING),
                self._status_chip(ft.Icons.MODEL_TRAINING, "модель", COLOR_ACCENT),
            ],
        )
        title_group = ft.Row(
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Container(
                    width=36,
                    height=36,
                    border_radius=8,
                    bgcolor="#1A1E22",
                    alignment=ft.Alignment.CENTER,
                    content=ft.Icon(ft.Icons.MODEL_TRAINING, color=COLOR_ACCENT, size=21),
                ),
                ft.Column(
                    spacing=1,
                    expand=True,
                    controls=[
                        ft.Text(
                            "Студия обучения",
                            size=19,
                            weight=ft.FontWeight.BOLD,
                            color=COLOR_ON_SURFACE,
                        ),
                        ft.Text(
                            "запись, датасет, обучение, проверка",
                            size=12,
                            color=COLOR_MUTED,
                        ),
                    ],
                ),
            ],
        )
        header = ft.ResponsiveRow(
            spacing=10,
            run_spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Container(content=title_group, col={"xs": 12, "md": 5}),
                ft.Container(content=workflow_strip, col={"xs": 12, "md": 5}),
                ft.Container(
                    content=self._cancel_btn,
                    alignment=ft.Alignment.CENTER_RIGHT,
                    col={"xs": 12, "md": 2},
                ),
            ],
        )

        datasets_card = self._build_dataset_manager_card()

        log_card = surface_card(
            ft.Column(
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self._panel_title(
                        ft.Icons.TERMINAL,
                        "Журнал",
                        color=COLOR_MUTED,
                    ),
                    self._log_scroll,
                ],
            ),
            padding=14,
            radius=8,
        )
        log_card.width = float("inf")

        main_row = ft.ResponsiveRow(
            spacing=14,
            run_spacing=14,
            controls=[
                ft.Container(
                    content=self._build_recording_preview_card(),
                    col={"xs": 12, "lg": 7},
                ),
                ft.Container(
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            self._build_training_body(),
                            self._build_workflow_card(),
                        ],
                    ),
                    col={"xs": 12, "lg": 5},
                ),
            ],
        )
        lower_row = ft.ResponsiveRow(
            spacing=14,
            run_spacing=14,
            controls=[
                ft.Container(content=datasets_card, col={"xs": 12, "lg": 5}),
                ft.Container(content=log_card, col={"xs": 12, "lg": 7}),
            ],
        )

        return ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                surface_card(header, padding=14, radius=8),
                main_row,
                lower_row,
            ],
        )
