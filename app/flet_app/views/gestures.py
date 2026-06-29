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
    COLOR_SURFACE_HIGH,
    COLOR_WARNING,
    surface_card,
)


class GesturesView:
    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller
        self._rows: list[dict] = []
        self._selected_id: int | None = None
        self._filter = "all"

        self._list_column = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO)
        self._info_text = ft.Text("", size=12, color=COLOR_MUTED)
        self._summary_total = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=COLOR_ON_SURFACE)
        self._summary_bound = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=COLOR_SUCCESS)
        self._summary_unbound = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=COLOR_WARNING)
        self._detail_body = ft.Column(spacing=12)
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
            label="Фильтр",
            value="all",
            dense=True,
            height=46,
            border_color=COLOR_SURFACE_HIGH,
            focused_border_color=COLOR_ACCENT,
            options=[
                ft.DropdownOption(key="all", text="все"),
                ft.DropdownOption(key="bound", text="привязанные"),
                ft.DropdownOption(key="unbound", text="без команды"),
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
        self._summary_total.value = str(total)
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
        self._selected_id = self._row_id(row)
        self._render()

    def _row_id(self, row: dict) -> int:
        try:
            return int(row.get("id") or 0)
        except (TypeError, ValueError):
            return 0

    def _label(self, row: dict) -> str:
        return str(row.get("label") or "").strip() or "gesture"

    def _description(self, row: dict) -> str:
        return str(row.get("description") or "").strip()

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
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
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

    def _gesture_card(self, row: dict) -> ft.Container:
        label = self._label(row)
        bound = self._bound_command(row)
        selected = self._row_id(row) == self._selected_id
        samples = self._sample_count(row)
        hands = "две руки" if row.get("isTwoHands") else "одна рука"
        description = self._description(row)
        subtitle = description or hands
        if description:
            subtitle = f"{hands} · {description}"

        return ft.Container(
            bgcolor="#171A1D" if not selected else "#162A2E",
            border=self._border(COLOR_ACCENT if selected else COLOR_SURFACE_HIGH, 1.2 if selected else 1),
            border_radius=8,
            padding=12,
            on_click=lambda _e, item=row: self._select_row(item),
            content=ft.Row(
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        width=42,
                        height=42,
                        border_radius=8,
                        bgcolor="#101316",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(ft.Icons.BACK_HAND, color=COLOR_ACCENT, size=22),
                    ),
                    ft.Column(
                        spacing=6,
                        expand=True,
                        controls=[
                            ft.Row(
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    ft.Text(
                                        label,
                                        size=15,
                                        weight=ft.FontWeight.W_600,
                                        color=COLOR_ON_SURFACE,
                                        expand=True,
                                        no_wrap=True,
                                    ),
                                    self._status_chip(bool(bound)),
                                ],
                            ),
                            ft.Text(subtitle, size=12, color=COLOR_MUTED, no_wrap=True),
                            ft.Row(
                                spacing=8,
                                wrap=True,
                                controls=[
                                    self._chip(
                                        f"{samples} samples" if samples is not None else "trained",
                                        COLOR_ACCENT,
                                        icon=ft.Icons.DATASET,
                                    ),
                                    self._chip(hands, COLOR_MUTED, icon=ft.Icons.PAN_TOOL_ALT),
                                    self._chip(
                                        bound or "команда не назначена",
                                        COLOR_SUCCESS if bound else COLOR_MUTED,
                                        icon=ft.Icons.TERMINAL,
                                    ),
                                ],
                            ),
                        ],
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
        description = self._description(row) or "Описание не задано"
        self._detail_body.controls = [
            ft.Row(
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        width=46,
                        height=46,
                        border_radius=8,
                        bgcolor="#101316",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(ft.Icons.BACK_HAND, color=COLOR_ACCENT, size=24),
                    ),
                    ft.Column(
                        spacing=2,
                        expand=True,
                        controls=[
                            ft.Text(
                                label,
                                size=18,
                                weight=ft.FontWeight.BOLD,
                                color=COLOR_ON_SURFACE,
                                no_wrap=True,
                            ),
                            ft.Text(f"id #{self._row_id(row)}", size=12, color=COLOR_MUTED),
                        ],
                    ),
                ],
            ),
            ft.Text(description, size=12, color=COLOR_MUTED),
            self._detail_line(
                ft.Icons.TERMINAL,
                "Команда",
                bound or "Не назначена",
                COLOR_SUCCESS if bound else COLOR_WARNING,
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
            ft.ResponsiveRow(
                spacing=12,
                run_spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        col={"xs": 12, "md": 4},
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
                    ft.Container(
                        col={"xs": 12, "md": 5},
                        content=ft.Row(
                            spacing=8,
                            wrap=True,
                            controls=[
                                self._metric_tile("Всего", self._summary_total, ft.Icons.DATA_ARRAY, COLOR_ACCENT),
                                self._metric_tile("Привязано", self._summary_bound, ft.Icons.LINK, COLOR_SUCCESS),
                                self._metric_tile("Без команды", self._summary_unbound, ft.Icons.LINK_OFF, COLOR_WARNING),
                            ],
                        ),
                    ),
                    ft.Container(
                        col={"xs": 12, "md": 3},
                        content=ft.Row(
                            spacing=8,
                            alignment=ft.MainAxisAlignment.END,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[import_btn, refresh_btn],
                        ),
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
                    content=ft.Container(
                        content=ft.Stack(
                            controls=[self._list_column, self._empty_state],
                            expand=True,
                        ),
                        expand=True,
                    ),
                ),
                ft.Container(
                    col={"xs": 12, "lg": 4},
                    content=surface_card(
                        ft.Column(
                            spacing=12,
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
                                self._detail_body,
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
