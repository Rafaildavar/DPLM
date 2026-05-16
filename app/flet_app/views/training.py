"""
Экран «Обучение» — UI-обёртка над CLI ``cv/record_gestures.py`` и
``cv/train_classifier.py``.

Шаг 1. Запись примеров. Поля «имя жеста / число сэмплов / длина / 2 руки»
→ запускает CLI как subprocess. CLI открывает отдельное OpenCV окно
(там клавиши s/n/q, как в скрипте), а live-лог транслируется в наше окно.

Шаг 2. Обучение KNN. Поля «папка датасета / выходной путь / соседи»
→ запускает CLI; live-лог + статус «модель сохранена в …».

После успешного обучения модель попадает в ``models/knn.pkl`` —
``GestureOnlineInfer`` подхватит её при следующем запуске встроенного
распознавания на Главной.
"""
from __future__ import annotations

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
    surface_card,
)


_MAX_LOG_LINES = 400


class TrainingView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller

        # --- Поля «Запись примеров» --------------------------------------
        self._rec_label = ft.TextField(
            label="Имя жеста",
            hint_text="например: zoom",
            border_color=COLOR_SURFACE_HIGH,
        )
        self._rec_samples = ft.TextField(
            label="Сэмплов",
            value="20",
            width=120,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._rec_frames = ft.TextField(
            label="Длина (кадров)",
            value="30",
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
            on_click=self._on_record_start,
        )

        # --- Поля «Обучение» ---------------------------------------------
        self._tr_data_root = ft.TextField(
            label="Папка датасета",
            value="data/gestures",
            border_color=COLOR_SURFACE_HIGH,
            expand=True,
        )
        self._tr_out_path = ft.TextField(
            label="Выходной файл модели",
            value="models/knn.pkl",
            border_color=COLOR_SURFACE_HIGH,
            expand=True,
        )
        self._tr_neighbors = ft.TextField(
            label="K (соседи)",
            value="5",
            width=120,
            border_color=COLOR_SURFACE_HIGH,
        )
        self._tr_start_btn = ft.FilledButton(
            content=ft.Text("Обучить модель", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.MODEL_TRAINING,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=20, vertical=14),
            ),
            on_click=self._on_train_start,
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
        )

        # --- Список записанных классов ------------------------------------
        self._datasets_text = ft.Text(
            "—", size=12, color=COLOR_MUTED, selectable=True
        )

    # ---- Жизненный цикл --------------------------------------------------

    def on_show(self) -> None:
        self._refresh_datasets()

    def on_hide(self) -> None:
        pass

    # ---- Хелперы ---------------------------------------------------------

    def _refresh_datasets(self) -> None:
        rows = self._controller.list_recorded_gestures()
        if not rows:
            self._datasets_text.value = (
                "В папке data/gestures/ пока ничего нет — запишите первый жест выше."
            )
        else:
            parts = [f"{r['label']}: {r['samples']} сэмплов" for r in rows]
            self._datasets_text.value = "\n".join(parts)
        try:
            self._datasets_text.update()
        except Exception:
            pass

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
        self._rec_start_btn.disabled = running
        self._tr_start_btn.disabled = running
        self._cancel_btn.disabled = not running
        try:
            self._rec_start_btn.update()
            self._tr_start_btn.update()
            self._cancel_btn.update()
        except Exception:
            pass

    # ---- Действия --------------------------------------------------------

    def _parse_int(self, value: str, default: int) -> int:
        try:
            return int((value or "").strip())
        except (TypeError, ValueError):
            return default

    def _on_record_start(self, _e) -> None:
        label = (self._rec_label.value or "").strip()
        if not label:
            self._append_log("[!] Укажи имя жеста")
            return
        samples = max(1, self._parse_int(self._rec_samples.value, 20))
        frames = max(1, self._parse_int(self._rec_frames.value, 30))
        two_hands = bool(self._rec_two_hands.value)

        self._append_log(
            f"[i] Запись «{label}»: {samples} сэмплов, {frames} кадров"
            + (" (две руки)" if two_hands else "")
        )
        self._append_log(
            "Откроется отдельное окно OpenCV. Клавиши: s — старт/стоп записи, "
            "n — сохранить семпл, q — выход."
        )

        ok = self._controller.start_recording(
            label=label,
            num_samples=samples,
            frames=frames,
            two_hands=two_hands,
            on_line=self._append_log,
            on_done=self._on_subprocess_done,
        )
        if not ok:
            self._append_log("[!] Не удалось запустить запись (возможно, уже идёт другая задача).")
            return
        self._set_running(True)

    def _on_train_start(self, _e) -> None:
        data_root = (self._tr_data_root.value or "data/gestures").strip()
        out_path = (self._tr_out_path.value or "models/knn.pkl").strip()
        neighbors = max(1, self._parse_int(self._tr_neighbors.value, 5))

        self._append_log(
            f"[i] Обучение KNN: data={data_root}, out={out_path}, k={neighbors}"
        )
        ok = self._controller.start_training(
            data_root=data_root,
            out_path=out_path,
            neighbors=neighbors,
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

    def build(self) -> ft.Control:
        header = ft.Row(
            controls=[
                ft.Icon(ft.Icons.MODEL_TRAINING, color=COLOR_ACCENT, size=28),
                ft.Text(
                    "Обучение",
                    size=20,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_ON_SURFACE,
                ),
                ft.Text(
                    "запись примеров + обучение KNN",
                    size=12,
                    color=COLOR_MUTED,
                ),
                ft.Container(expand=True),
                self._cancel_btn,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )

        record_card = surface_card(
            ft.Column(
                spacing=10,
                controls=[
                    ft.Text(
                        "1. Запись примеров",
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color=COLOR_ON_SURFACE,
                    ),
                    self._rec_label,
                    ft.Row(
                        spacing=10,
                        controls=[
                            self._rec_samples,
                            self._rec_frames,
                            self._rec_two_hands,
                        ],
                    ),
                    self._rec_start_btn,
                ],
            ),
            padding=16,
            radius=16,
        )

        train_card = surface_card(
            ft.Column(
                spacing=10,
                controls=[
                    ft.Text(
                        "2. Обучение KNN",
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color=COLOR_ON_SURFACE,
                    ),
                    self._tr_data_root,
                    self._tr_out_path,
                    ft.Row(spacing=10, controls=[self._tr_neighbors]),
                    self._tr_start_btn,
                ],
            ),
            padding=16,
            radius=16,
        )

        datasets_card = surface_card(
            ft.Column(
                spacing=8,
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(
                                "Накопленные классы",
                                size=14,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                            ),
                            ft.Container(expand=True),
                            ft.IconButton(
                                icon=ft.Icons.REFRESH,
                                tooltip="Обновить",
                                on_click=lambda _e: self._refresh_datasets(),
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self._datasets_text,
                ],
            ),
            padding=16,
            radius=16,
        )

        log_card = surface_card(
            ft.Column(
                spacing=6,
                controls=[
                    ft.Text(
                        "Журнал процесса",
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color=COLOR_ON_SURFACE,
                    ),
                    self._log_scroll,
                ],
            ),
            padding=14,
            radius=16,
        )

        return ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                surface_card(header, padding=16, radius=16),
                ft.Row(
                    spacing=14,
                    controls=[
                        ft.Container(content=record_card, expand=1),
                        ft.Container(content=train_card, expand=1),
                    ],
                ),
                datasets_card,
                log_card,
            ],
        )
