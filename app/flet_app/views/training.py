"""
Экран «Обучение» — единый пользовательский поток записи и обучения жестов.

Пользователь выбирает тип жеста (статический или динамический), записывает
примеры во встроенной камере и запускает обучение. Технические параметры модели
подбираются из проектных defaults, без отдельного режима разработчика в UI.

После успешного обучения static-модель попадает в ``models/knn.pkl``, а
dynamic-модель — в выбранный ``models/dynamic_*.pkl`` профиль. Главный экран
подхватит соответствующие артефакты при следующем запуске встроенного
распознавания.
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
_DEFAULT_STATIC_RECORD_FEATURE_DIM = 63
_DEFAULT_DYNAMIC_MODEL_TYPE = "dynamic_landmark_lstm_backbone"
_DEFAULT_DYNAMIC_MODEL_OUT = "models/dynamic_landmark_lstm_backbone.pkl"
_DYNAMIC_MODEL_OUT_BY_TYPE = {
    "dynamic_landmark_lstm_backbone": "models/dynamic_landmark_lstm_backbone.pkl",
    "sequence_mlp": "models/dynamic_sequence_mlp.pkl",
    "sequence_rocket": "models/dynamic_sequence_rocket.pkl",
    "sequence_multirocket": "models/dynamic_sequence_multirocket.pkl",
    "sequence_sprocket": "models/dynamic_sequence_sprocket.pkl",
    "sequence_shapelet": "models/dynamic_sequence_shapelet.pkl",
    "sequence_shapelet_72": "models/dynamic_sequence_shapelet_72.pkl",
    "sequence_phase_hmm": "models/dynamic_sequence_phase_hmm.pkl",
    "sequence_ensemble": "models/dynamic_sequence_ensemble.pkl",
    "sequence_gru_backbone": "models/dynamic_sequence_gru_backbone.pkl",
    "sequence_lstm_backbone": "models/dynamic_sequence_lstm_backbone.pkl",
    "dynamic_landmark_cnn": "models/dynamic_landmark_cnn.pkl",
}
_DEFAULT_DYNAMIC_RECORD_SAMPLES = 10
_DEFAULT_DYNAMIC_RECORD_FRAMES = 72
_DEFAULT_DYNAMIC_RECORD_FEATURE_DIM = 65
_DEFAULT_NEGATIVE_SAMPLES_PER_LABEL = 20
_DEFAULT_NEGATIVE_SEED = 42
_STATIC_TRAINING_SCOPE = "static,quasi_static,negative"
_DYNAMIC_TRAINING_SCOPE = "dynamic,negative"
_DEFAULT_STATIC_FEATURE_MODE = "static_craft_full_stats"
_DEFAULT_DYNAMIC_FEATURE_MODE = "dynamic_landmark_image"
_LEGACY_DYNAMIC_SEQUENCE_FEATURE_MODE = "dynamic_sequence"
_DEFAULT_SEQUENCE_GRU_OPTUNA_TRIALS = 8
_DEFAULT_SEQUENCE_GRU_OPTUNA_MAX_EPOCHS = 70
_DEFAULT_SEQUENCE_LSTM_OPTUNA_TRIALS = 8
_DEFAULT_SEQUENCE_LSTM_OPTUNA_MAX_EPOCHS = 70
_DEFAULT_MODEL_TYPE = "extra_trees"
_DEFAULT_EXTRA_TREES_OPTUNA_TRIALS = 20
_DEFAULT_EXTRA_TREES_OPTUNA_CV_FOLDS = 3
_DEFAULT_EXTRA_TREES_OPTUNA_TIMEOUT = 120
_DEFAULT_K_NEIGHBORS = 5
_PLACEHOLDER_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _dynamic_model_out_for_type(model_type: str) -> str:
    clean = str(model_type or _DEFAULT_DYNAMIC_MODEL_TYPE).strip().lower()
    return _DYNAMIC_MODEL_OUT_BY_TYPE.get(clean, _DEFAULT_DYNAMIC_MODEL_OUT)


def _known_dynamic_model_outputs() -> set[str]:
    return set(_DYNAMIC_MODEL_OUT_BY_TYPE.values())


def _dynamic_metadata_out_for_type(model_type: str) -> tuple[str, str, str]:
    clean = str(model_type or _DEFAULT_DYNAMIC_MODEL_TYPE).strip().lower()
    if clean == "dynamic_landmark_lstm_backbone":
        return (
            "models/dynamic_landmark_lstm_backbone_classes.json",
            "models/dynamic_landmark_lstm_backbone_feature_dim.txt",
            "models/dynamic_landmark_lstm_backbone_feature_mode.txt",
        )
    if clean == "sequence_mlp":
        return (
            "models/dynamic_sequence_mlp_classes.json",
            "models/dynamic_sequence_mlp_feature_dim.txt",
            "models/dynamic_sequence_mlp_feature_mode.txt",
        )
    if clean == "sequence_rocket":
        return (
            "models/dynamic_sequence_rocket_classes.json",
            "models/dynamic_sequence_rocket_feature_dim.txt",
            "models/dynamic_sequence_rocket_feature_mode.txt",
        )
    if clean == "sequence_multirocket":
        return (
            "models/dynamic_sequence_multirocket_classes.json",
            "models/dynamic_sequence_multirocket_feature_dim.txt",
            "models/dynamic_sequence_multirocket_feature_mode.txt",
        )
    if clean == "sequence_sprocket":
        return (
            "models/dynamic_sequence_sprocket_classes.json",
            "models/dynamic_sequence_sprocket_feature_dim.txt",
            "models/dynamic_sequence_sprocket_feature_mode.txt",
        )
    if clean == "sequence_shapelet":
        return (
            "models/dynamic_sequence_shapelet_classes.json",
            "models/dynamic_sequence_shapelet_feature_dim.txt",
            "models/dynamic_sequence_shapelet_feature_mode.txt",
        )
    if clean == "sequence_shapelet_72":
        return (
            "models/dynamic_sequence_shapelet_72_classes.json",
            "models/dynamic_sequence_shapelet_72_feature_dim.txt",
            "models/dynamic_sequence_shapelet_72_feature_mode.txt",
        )
    if clean == "sequence_phase_hmm":
        return (
            "models/dynamic_sequence_phase_hmm_classes.json",
            "models/dynamic_sequence_phase_hmm_feature_dim.txt",
            "models/dynamic_sequence_phase_hmm_feature_mode.txt",
        )
    if clean == "sequence_ensemble":
        return (
            "models/dynamic_sequence_ensemble_classes.json",
            "models/dynamic_sequence_ensemble_feature_dim.txt",
            "models/dynamic_sequence_ensemble_feature_mode.txt",
        )
    if clean == "sequence_gru_backbone":
        return (
            "models/dynamic_sequence_gru_backbone_classes.json",
            "models/dynamic_sequence_gru_backbone_feature_dim.txt",
            "models/dynamic_sequence_gru_backbone_feature_mode.txt",
        )
    if clean == "sequence_lstm_backbone":
        return (
            "models/dynamic_sequence_lstm_backbone_classes.json",
            "models/dynamic_sequence_lstm_backbone_feature_dim.txt",
            "models/dynamic_sequence_lstm_backbone_feature_mode.txt",
        )
    if clean == "dynamic_landmark_cnn":
        return (
            "models/dynamic_landmark_cnn_classes.json",
            "models/dynamic_landmark_cnn_feature_dim.txt",
            "models/dynamic_landmark_cnn_feature_mode.txt",
        )
    return _dynamic_metadata_out_for_type(_DEFAULT_DYNAMIC_MODEL_TYPE)


def _dynamic_feature_mode_for_model(model_type: str) -> str:
    clean = str(model_type or _DEFAULT_DYNAMIC_MODEL_TYPE).strip().lower()
    if clean == "sequence_shapelet_72":
        return "dynamic_sequence_72"
    if clean in {"dynamic_landmark_lstm_backbone", "dynamic_landmark_cnn"}:
        return "dynamic_landmark_image"
    return _LEGACY_DYNAMIC_SEQUENCE_FEATURE_MODE


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
        self._user_rec_type = ft.Dropdown(
            label="Тип жеста",
            value="static",
            border_color=COLOR_SURFACE_HIGH,
            width=180,
            options=[
                ft.DropdownOption(key="static", text="Статический"),
                ft.DropdownOption(key="dynamic", text="Динамический"),
            ],
            editable=False,
            on_select=self._on_user_gesture_type_changed,
        )
        self._user_rec_samples = ft.TextField(
            label="Реальных дублей",
            value=str(_DEFAULT_RECORD_SAMPLES),
            width=140,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._user_rec_frames = ft.TextField(
            label="Длина (кадров)",
            value=str(_DEFAULT_RECORD_FRAMES),
            width=160,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._user_rec_two_hands = ft.Switch(
            label="",
            value=False,
            active_color=COLOR_ACCENT,
            disabled=True,
            visible=False,
        )
        self._user_rec_start_btn = ft.FilledButton(
            content=ft.Text("Записать и обновить модель", weight=ft.FontWeight.BOLD),
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
            label="",
            value=False,
            active_color=COLOR_ACCENT,
            disabled=True,
            visible=False,
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
            label="",
            value=False,
            active_color=COLOR_ACCENT,
            disabled=True,
            visible=False,
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
                ft.DropdownOption(key="dynamic_sequence", text="dynamic_sequence"),
                ft.DropdownOption(
                    key="dynamic_sequence_72",
                    text="dynamic_sequence_72",
                ),
            ],
            disabled=True,
            editable=False,
        )
        self._dyn_model_type = ft.Dropdown(
            label="Модель",
            value=_DEFAULT_DYNAMIC_MODEL_TYPE,
            border_color=COLOR_SURFACE_HIGH,
            options=[
                ft.DropdownOption(
                    key="dynamic_landmark_lstm_backbone",
                    text="dynamic_landmark_lstm_backbone",
                ),
                ft.DropdownOption(key="sequence_mlp", text="sequence_mlp"),
                ft.DropdownOption(key="sequence_rocket", text="sequence_rocket"),
                ft.DropdownOption(
                    key="sequence_multirocket",
                    text="sequence_multirocket",
                ),
                ft.DropdownOption(key="sequence_sprocket", text="sequence_sprocket"),
                ft.DropdownOption(key="sequence_shapelet", text="sequence_shapelet"),
                ft.DropdownOption(
                    key="sequence_shapelet_72",
                    text="sequence_shapelet_72",
                ),
                ft.DropdownOption(
                    key="sequence_phase_hmm",
                    text="sequence_phase_hmm",
                ),
                ft.DropdownOption(
                    key="sequence_ensemble",
                    text="sequence_ensemble",
                ),
                ft.DropdownOption(
                    key="sequence_gru_backbone",
                    text="sequence_gru_backbone",
                ),
                ft.DropdownOption(
                    key="sequence_lstm_backbone",
                    text="sequence_lstm_backbone",
                ),
                ft.DropdownOption(
                    key="dynamic_landmark_cnn",
                    text="dynamic_landmark_cnn",
                ),
            ],
            editable=False,
            on_select=self._on_dynamic_model_type_changed,
        )
        self._dyn_model_out = ft.TextField(
            label="Файл dynamic-модели",
            value=_DEFAULT_DYNAMIC_MODEL_OUT,
            border_color=COLOR_SURFACE_HIGH,
            disabled=True,
        )
        self._dyn_tr_neighbors = ft.TextField(
            label="K",
            value=str(_DEFAULT_K_NEIGHBORS),
            width=100,
            border_color=COLOR_SURFACE_HIGH,
            disabled=True,
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
                ft.DropdownOption(key="static_stacking", text="static_stacking"),
                ft.DropdownOption(key="static_landmark_cnn", text="static_landmark_cnn"),
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
        self._activity_title = ft.Text(
            "Готово",
            size=15,
            weight=ft.FontWeight.W_600,
            color=COLOR_ON_SURFACE,
        )
        self._activity_detail = ft.Text(
            "Запиши примеры жеста или обучи модель на уже собранных данных.",
            size=12,
            color=COLOR_MUTED,
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
        self._dataset_mix_text = ft.Text(
            "Базовые 0 · Свои 0",
            size=12,
            color=COLOR_MUTED,
        )
        self._pending_delete_label = ""
        self._dataset_can_delete_labels: set[str] = set()
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
        self._dataset_can_delete_labels.clear()
        total_samples = 0
        class_counts: list[int] = []
        base_classes = 0
        custom_classes = 0
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
                can_delete = bool(row.get("canDelete", True))
                delete_reason = str(
                    row.get("deleteReason")
                    or "Можно удалять только классы, записанные пользователем"
                )
                if can_delete:
                    self._dataset_can_delete_labels.add(label)
                    custom_classes += 1
                else:
                    base_classes += 1
                total_samples += samples
                class_counts.append(samples)
                pending_delete = self._pending_delete_label == label and can_delete
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
                elif not can_delete:
                    actions = [
                        ft.IconButton(
                            icon=ft.Icons.LOCK_OUTLINE,
                            icon_color=COLOR_MUTED,
                            tooltip=delete_reason,
                            disabled=True,
                        )
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
                                    f"{samples} реальных сэмплов",
                                    size=12,
                                    color=COLOR_MUTED,
                                ),
                                ft.Container(
                                    bgcolor="#101316",
                                    border_radius=8,
                                    padding=ft.Padding.symmetric(horizontal=9, vertical=5),
                                    content=ft.Row(
                                        spacing=6,
                                        tight=True,
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                        controls=[
                                            ft.Icon(
                                                ft.Icons.EDIT if can_delete else ft.Icons.LOCK_OUTLINE,
                                                size=13,
                                                color=COLOR_SUCCESS if can_delete else COLOR_MUTED,
                                            ),
                                            ft.Text(
                                                "свой" if can_delete else "базовый",
                                                size=11,
                                                color=COLOR_SUCCESS if can_delete else COLOR_MUTED,
                                                no_wrap=True,
                                            ),
                                        ],
                                    ),
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
        self._dataset_real_aug_value.value = str(total_samples)
        self._dataset_balance_value.value = balance_label
        self._dataset_balance_value.color = balance_color
        self._dataset_mix_text.value = f"Базовые {base_classes} · Свои {custom_classes}"
        try:
            self._datasets_column.update()
            self._dataset_classes_value.update()
            self._dataset_samples_value.update()
            self._dataset_real_aug_value.update()
            self._dataset_balance_value.update()
            self._dataset_mix_text.update()
        except Exception:
            pass

    def _request_delete_samples(self, label: str) -> None:
        if label not in self._dataset_can_delete_labels:
            self._pending_delete_label = ""
            self._append_log(
                f"[i] «{label}» защищён: можно удалять только классы, записанные пользователем"
            )
            self._refresh_datasets()
            return
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

    def _set_activity(self, title: str, detail: str, *, color: str = COLOR_MUTED) -> None:
        self._page.run_thread(self._apply_activity, title, detail, color)

    def _apply_activity(self, title: str, detail: str, color: str = COLOR_MUTED) -> None:
        self._activity_title.value = title
        self._activity_detail.value = detail
        self._activity_detail.color = color
        for control in (self._activity_title, self._activity_detail):
            try:
                control.update()
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
        model_type = str(self._dyn_model_type.value or _DEFAULT_DYNAMIC_MODEL_TYPE)
        current = str(self._dyn_model_out.value or "").strip()
        suggested = _dynamic_model_out_for_type(model_type)
        if not current or current in _known_dynamic_model_outputs():
            self._dyn_model_out.value = suggested
            try:
                self._dyn_model_out.update()
            except Exception:
                pass
        self._dyn_feature_mode.value = _dynamic_feature_mode_for_model(model_type)
        try:
            self._dyn_feature_mode.update()
        except Exception:
            pass

    def _current_user_gesture_kind(self) -> str:
        clean = str(self._user_rec_type.value or "static").strip().lower()
        return "dynamic" if clean == "dynamic" else "static"

    def _on_user_gesture_type_changed(self, _e) -> None:
        kind = self._current_user_gesture_kind()
        if kind == "dynamic":
            if str(self._user_rec_samples.value or "").strip() in {
                "",
                str(_DEFAULT_RECORD_SAMPLES),
            }:
                self._user_rec_samples.value = str(_DEFAULT_DYNAMIC_RECORD_SAMPLES)
            if str(self._user_rec_frames.value or "").strip() in {
                "",
                str(_DEFAULT_RECORD_FRAMES),
            }:
                self._user_rec_frames.value = str(_DEFAULT_DYNAMIC_RECORD_FRAMES)
        else:
            if str(self._user_rec_samples.value or "").strip() in {
                "",
                str(_DEFAULT_DYNAMIC_RECORD_SAMPLES),
            }:
                self._user_rec_samples.value = str(_DEFAULT_RECORD_SAMPLES)
            if str(self._user_rec_frames.value or "").strip() in {
                "",
                str(_DEFAULT_DYNAMIC_RECORD_FRAMES),
            }:
                self._user_rec_frames.value = str(_DEFAULT_RECORD_FRAMES)
        for control in (self._user_rec_samples, self._user_rec_frames):
            try:
                control.update()
            except Exception:
                pass

    def _on_record_start(self, _e, *, mode: str = "developer") -> None:
        if mode == "user":
            kind = self._current_user_gesture_kind()
            label = (self._user_rec_label.value or "").strip()
            samples = max(
                1,
                self._parse_int(
                    self._user_rec_samples.value,
                    _DEFAULT_DYNAMIC_RECORD_SAMPLES
                    if kind == "dynamic"
                    else _DEFAULT_RECORD_SAMPLES,
                ),
            )
            frames = max(
                1,
                self._parse_int(
                    self._user_rec_frames.value,
                    _DEFAULT_DYNAMIC_RECORD_FRAMES
                    if kind == "dynamic"
                    else _DEFAULT_RECORD_FRAMES,
                ),
            )
            two_hands = False
            include_global_motion = kind == "dynamic"
            include_landmark_z = True
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
            two_hands = False
            include_global_motion = True
            include_landmark_z = True
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
            two_hands = False
            include_global_motion = False
            include_landmark_z = True

        if not label:
            self._append_log("[!] Укажи имя жеста")
            return

        self._append_log(
            f"[i] Запись «{label}»: {samples} реальных дублей, {frames} кадров"
            + (" + глобальное движение" if include_global_motion else "")
            + (" + landmark z" if include_landmark_z else "")
        )
        self._append_log(
            "Запись выполняется во встроенной камере: держи жест в кадре, "
            "сэмплы сохранятся автоматически."
        )
        self._set_activity(
            "Идет запись",
            f"Жест «{label}»: сохранится {samples} дублей по {frames} кадров.",
            color=COLOR_DANGER,
        )

        auto_train_mode = mode
        auto_user_kind = kind if mode == "user" else None
        ok = self._controller.start_recording(
            label=label,
            num_samples=samples,
            frames=frames,
            two_hands=two_hands,
            include_global_motion=include_global_motion,
            include_landmark_z=include_landmark_z,
            on_line=self._append_log,
            on_done=lambda code: self._on_recording_done(
                code,
                auto_train_mode=auto_train_mode,
                user_kind=auto_user_kind,
            ),
        )
        if not ok:
            self._append_log("[!] Не удалось запустить запись (возможно, уже идёт другая задача).")
            self._set_activity(
                "Запись не запустилась",
                "Другая операция еще выполняется или камера недоступна.",
                color=COLOR_DANGER,
            )
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

    def _on_train_start(
        self,
        _e,
        *,
        mode: str = "developer",
        auto: bool = False,
        user_kind: str | None = None,
    ) -> bool:
        if mode == "user":
            kind = user_kind or self._current_user_gesture_kind()
            if kind == "dynamic":
                data_root = _DEFAULT_DATA_ROOT
                out_path = _DEFAULT_DYNAMIC_MODEL_OUT
                neighbors = _DEFAULT_K_NEIGHBORS
                feature_mode = _DEFAULT_DYNAMIC_FEATURE_MODE
                expect_dim = _DEFAULT_DYNAMIC_RECORD_FEATURE_DIM
                model_type = _DEFAULT_DYNAMIC_MODEL_TYPE
                (
                    classes_out_path,
                    feature_dim_out_path,
                    feature_mode_out_path,
                ) = _dynamic_metadata_out_for_type(model_type)
                training_scope = _DYNAMIC_TRAINING_SCOPE
                training_extra_args: list[str] = ["--include-augmented"]
                self._append_log(
                    "[i] Автообучение dynamic-модели"
                    if auto
                    else "[i] Обучение dynamic-модели со стандартными параметрами проекта"
                )
            else:
                data_root = _DEFAULT_DATA_ROOT
                out_path = _DEFAULT_MODEL_OUT
                neighbors = _DEFAULT_K_NEIGHBORS
                feature_mode = _DEFAULT_STATIC_FEATURE_MODE
                expect_dim = _DEFAULT_STATIC_RECORD_FEATURE_DIM
                model_type = _DEFAULT_MODEL_TYPE
                classes_out_path = ""
                feature_dim_out_path = ""
                feature_mode_out_path = ""
                training_scope = _STATIC_TRAINING_SCOPE
                training_extra_args = [
                    "--include-augmented",
                    "--extra-trees-optuna-trials",
                    str(_DEFAULT_EXTRA_TREES_OPTUNA_TRIALS),
                    "--extra-trees-optuna-cv-folds",
                    str(_DEFAULT_EXTRA_TREES_OPTUNA_CV_FOLDS),
                    "--extra-trees-optuna-timeout",
                    str(_DEFAULT_EXTRA_TREES_OPTUNA_TIMEOUT),
                ]
                if auto:
                    self._append_log(
                        "[i] Автообучение static-модели "
                        f"+ Optuna trials={_DEFAULT_EXTRA_TREES_OPTUNA_TRIALS}"
                    )
                else:
                    self._append_log(
                        "[i] Обучение static-модели со стандартными параметрами проекта "
                        f"+ Optuna trials={_DEFAULT_EXTRA_TREES_OPTUNA_TRIALS}"
                    )
            self._append_log(
                "[i] GISLR-аугментации включены: старые aug_sample без "
                "метки gislr_landmark_v1 будут проигнорированы"
            )
        elif mode == "dynamic":
            data_root = _DEFAULT_DATA_ROOT
            neighbors = max(
                1,
                self._parse_int(self._dyn_tr_neighbors.value, _DEFAULT_K_NEIGHBORS),
            )
            selected_dynamic_model_type = str(
                self._dyn_model_type.value or ""
            ).strip().lower()
            feature_mode = _dynamic_feature_mode_for_model(selected_dynamic_model_type)
            expect_dim = None
            model_type = str(
                self._dyn_model_type.value or _DEFAULT_DYNAMIC_MODEL_TYPE
            ).strip()
            if model_type not in _DYNAMIC_MODEL_OUT_BY_TYPE:
                model_type = _DEFAULT_DYNAMIC_MODEL_TYPE
            train_model_type = (
                "sequence_shapelet"
                if model_type == "sequence_shapelet_72"
                else model_type
            )
            out_path = _dynamic_model_out_for_type(model_type)
            (
                classes_out_path,
                feature_dim_out_path,
                feature_mode_out_path,
            ) = _dynamic_metadata_out_for_type(model_type)
            training_scope = _DYNAMIC_TRAINING_SCOPE
            training_extra_args = []
            if train_model_type == "sequence_gru_backbone":
                training_extra_args = [
                    "--sequence-gru-optuna-trials",
                    str(_DEFAULT_SEQUENCE_GRU_OPTUNA_TRIALS),
                    "--sequence-gru-optuna-max-epochs",
                    str(_DEFAULT_SEQUENCE_GRU_OPTUNA_MAX_EPOCHS),
                ]
                self._append_log(
                    "[i] Для sequence_gru_backbone включен Optuna tuning: "
                    f"trials={_DEFAULT_SEQUENCE_GRU_OPTUNA_TRIALS}, "
                    f"epochs/trial={_DEFAULT_SEQUENCE_GRU_OPTUNA_MAX_EPOCHS}"
                )
            elif train_model_type in {
                "sequence_lstm_backbone",
                "dynamic_landmark_lstm_backbone",
            }:
                training_extra_args = [
                    "--sequence-lstm-optuna-trials",
                    str(_DEFAULT_SEQUENCE_LSTM_OPTUNA_TRIALS),
                    "--sequence-lstm-optuna-max-epochs",
                    str(_DEFAULT_SEQUENCE_LSTM_OPTUNA_MAX_EPOCHS),
                ]
                self._append_log(
                    f"[i] Для {train_model_type} включен Optuna tuning: "
                    f"trials={_DEFAULT_SEQUENCE_LSTM_OPTUNA_TRIALS}, "
                    f"epochs/trial={_DEFAULT_SEQUENCE_LSTM_OPTUNA_MAX_EPOCHS}"
                )
            self._append_log(
                f"[i] Обучение отдельной dynamic-модели: "
                f"model={train_model_type}, profile={model_type}, "
                f"feature_mode={feature_mode}, scope={training_scope}"
            )
            model_type = train_model_type
        else:
            data_root = (self._tr_data_root.value or _DEFAULT_DATA_ROOT).strip()
            out_path = (self._tr_out_path.value or _DEFAULT_MODEL_OUT).strip()
            neighbors = max(
                1,
                self._parse_int(self._tr_neighbors.value, _DEFAULT_K_NEIGHBORS),
            )
            expect_dim = _DEFAULT_STATIC_RECORD_FEATURE_DIM
            model_type = str(self._tr_model_type.value or _DEFAULT_MODEL_TYPE).strip()
            feature_mode = (
                "static_landmark_image"
                if model_type == "static_landmark_cnn"
                else _DEFAULT_STATIC_FEATURE_MODE
            )
            classes_out_path = ""
            feature_dim_out_path = ""
            feature_mode_out_path = ""
            training_scope = _STATIC_TRAINING_SCOPE
            training_extra_args = []

        self._append_log(
            f"[i] Обучение: data={data_root}, out={out_path}, "
            f"model={model_type}, k={neighbors}, feature_mode={feature_mode}"
        )
        self._set_activity(
            "Модель обучается",
            "Приложение соберет записанные примеры и обновит распознавание после завершения.",
            color=COLOR_ACCENT,
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
            extra_args=training_extra_args,
            on_line=self._append_log,
            on_done=self._on_subprocess_done,
        )
        if not ok:
            self._append_log("[!] Не удалось запустить обучение (возможно, уже идёт другая задача).")
            self._set_activity(
                "Обучение не запустилось",
                "Другая операция еще выполняется. Дождись завершения или останови ее.",
                color=COLOR_DANGER,
            )
            return False
        self._set_running(True)
        return True

    def _on_cancel(self, _e) -> None:
        self._controller.cancel_training()
        self._append_log("[i] Остановка процесса…")

    def _on_recording_done(
        self,
        code: int,
        *,
        auto_train_mode: str,
        user_kind: str | None = None,
    ) -> None:
        if code != 0:
            self._on_subprocess_done(code)
            return

        self._append_log("[i] Запись сохранена — запускаю автообучение модели")
        self._set_activity(
            "Автообучение",
            "Модель обновляется по новым примерам.",
            color=COLOR_ACCENT,
        )
        ok = self._on_train_start(
            None,
            mode=auto_train_mode,
            auto=True,
            user_kind=user_kind,
        )
        if not ok:
            self._set_running(False)

    def _on_subprocess_done(self, code: int) -> None:
        status = "✓ успешно" if code == 0 else f"⚠ код выхода {code}"
        self._append_log(f"[i] Процесс завершён ({status})")
        self._set_activity(
            "Готово" if code == 0 else "Нужно проверить",
            "Операция завершилась успешно." if code == 0 else "Операция завершилась с ошибкой.",
            color=COLOR_SUCCESS if code == 0 else COLOR_WARNING,
        )
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
                        "real samples",
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
                    self._dataset_mix_text,
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
                            self._workflow_step("Обучение"),
                            self._workflow_step("Проверка"),
                        ],
                    ),
                    ft.Row(
                        spacing=8,
                        wrap=True,
                        controls=[
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
                            self._user_rec_type,
                            self._user_rec_samples,
                            self._user_rec_frames,
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
                        "3. Защита",
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
        return ft.Column(
            spacing=10,
            controls=[
                self._build_user_record_card(),
            ],
        )

    def _build_activity_card(self) -> ft.Control:
        return surface_card(
            ft.Column(
                spacing=10,
                controls=[
                    self._panel_title(
                        ft.Icons.CHECK_CIRCLE,
                        "Состояние",
                        color=COLOR_SUCCESS,
                    ),
                    self._activity_title,
                    self._activity_detail,
                ],
            ),
            padding=14,
            radius=8,
        )

    def build(self) -> ft.Control:
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
                            "запись примеров и обучение модели",
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
                ft.Container(content=title_group, col={"xs": 12, "md": 9}),
                ft.Container(
                    content=self._cancel_btn,
                    alignment=ft.Alignment.CENTER_RIGHT,
                    col={"xs": 12, "md": 3},
                ),
            ],
        )

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
                            self._build_activity_card(),
                        ],
                    ),
                    col={"xs": 12, "lg": 5},
                ),
            ],
        )

        return ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                surface_card(header, padding=14, radius=8),
                main_row,
            ],
        )
