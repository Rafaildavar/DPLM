"""
Экран «Жесты» — список жестов из БД (карточки с привязанной командой).

Аналог QML ``GestureListScreen.qml``. Источник данных — таблица ``gestures``
через ``AppController.get_db_gestures()``. Здесь показываем только активные
и обученные жесты (model_class_id IS NOT NULL — это правило R3).
"""
from __future__ import annotations

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_SURFACE_HIGH,
    surface_card,
)


class GesturesView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller
        self._list_column = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO)
        self._info_text = ft.Text("", size=12, color=COLOR_MUTED)
        self._empty_state = ft.Container(
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
                controls=[
                    ft.Icon(ft.Icons.GESTURE, size=48, color=COLOR_MUTED),
                    ft.Text(
                        "В БД пока нет жестов",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=COLOR_ON_SURFACE,
                    ),
                    ft.Text(
                        "Запиши примеры на вкладке «Обучение» и нажми кнопку "
                        "«Импортировать в БД» ниже — папки из data/gestures/ "
                        "автоматически появятся в этом списке и сразу станут "
                        "доступны для привязки к команде.",
                        size=12,
                        color=COLOR_MUTED,
                        text_align=ft.TextAlign.CENTER,
                    ),
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
        rows = self._controller.get_db_gestures()
        self._list_column.controls.clear()
        for g in rows:
            bound = (g.get("boundCommandName") or "").strip()
            badge = (
                ft.Container(
                    content=ft.Text("привязано", size=11, color=COLOR_SUCCESS),
                    bgcolor="#1E3A2E",
                    border_radius=10,
                    padding=ft.Padding.symmetric(horizontal=10, vertical=4),
                )
                if bound
                else ft.Container(
                    content=ft.Text("без команды", size=11, color=COLOR_MUTED),
                    bgcolor=COLOR_SURFACE_HIGH,
                    border_radius=10,
                    padding=ft.Padding.symmetric(horizontal=10, vertical=4),
                )
            )
            hands_label = "две руки" if g.get("isTwoHands") else "одна рука"
            initial = (g["label"][:1] or "?").upper()
            card = ft.Container(
                bgcolor="#262636",
                border_radius=14,
                padding=14,
                content=ft.Row(
                    spacing=14,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(
                            width=48,
                            height=48,
                            border_radius=24,
                            bgcolor=COLOR_ACCENT,
                            alignment=ft.Alignment.CENTER,
                            content=ft.Text(
                                initial,
                                size=20,
                                weight=ft.FontWeight.BOLD,
                                color="#1a1a2e",
                            ),
                        ),
                        ft.Column(
                            spacing=4,
                            expand=True,
                            controls=[
                                ft.Row(
                                    spacing=10,
                                    controls=[
                                        ft.Text(
                                            g["label"],
                                            size=16,
                                            weight=ft.FontWeight.BOLD,
                                            color=COLOR_ON_SURFACE,
                                        ),
                                        badge,
                                    ],
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                ft.Text(
                                    f"{hands_label}"
                                    + (f"   ·   {g['description']}" if g.get("description") else ""),
                                    size=12,
                                    color=COLOR_MUTED,
                                ),
                                ft.Text(
                                    f"Команда: {bound}" if bound else "Команда не назначена — экран «Привязки»",
                                    size=12,
                                    color=COLOR_SUCCESS if bound else COLOR_MUTED,
                                ),
                            ],
                        ),
                    ],
                ),
            )
            self._list_column.controls.append(card)

        has_rows = len(rows) > 0
        self._empty_state.visible = not has_rows
        self._list_column.visible = has_rows
        try:
            self._page.update()
        except Exception:
            pass

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
                f"обновлено {summary['updated']}, всего классов: {summary['total']}"
            )
            self._info_text.color = COLOR_SUCCESS
        try:
            self._info_text.update()
        except Exception:
            pass
        self._refresh()

    def build(self) -> ft.Control:
        import_btn = ft.FilledButton(
            content=ft.Text("Импортировать из data/gestures/", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.CLOUD_UPLOAD,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=18, vertical=12),
            ),
            on_click=self._on_import_click,
            tooltip="Создать строки в таблице gestures для всех папок с npy",
        )

        header = ft.Row(
            controls=[
                ft.Icon(ft.Icons.GESTURE, color=COLOR_ACCENT, size=28),
                ft.Text(
                    "Жесты",
                    size=20,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_ON_SURFACE,
                ),
                ft.Text(
                    "из словаря БД (активные + обученные)",
                    size=12,
                    color=COLOR_MUTED,
                ),
                ft.Container(expand=True),
                import_btn,
                ft.IconButton(
                    icon=ft.Icons.REFRESH,
                    tooltip="Обновить",
                    on_click=lambda _e: self._refresh(),
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )
        return ft.Column(
            spacing=12,
            expand=True,
            controls=[
                surface_card(
                    ft.Column(spacing=8, controls=[header, self._info_text]),
                    padding=16,
                    radius=16,
                ),
                ft.Container(
                    content=ft.Stack(
                        controls=[self._list_column, self._empty_state],
                        expand=True,
                    ),
                    expand=True,
                ),
            ],
        )
