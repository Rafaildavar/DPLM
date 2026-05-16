"""
Экран «Привязки жестов к командам ОС» — главная фича диплома.

Реализует правила из docs/BINDING_RULES.md:
    R1 — 1:1 жест↔команда (с переспросом при перезаписи)
    R3 — фильтр жестов: только is_active=True и model_class_id IS NOT NULL
    R6 — предупреждение для опасных действий, не привязанных к двуручному жесту
    R7 — валидация action_spec
    R8 — уникальность имени команды

Полностью реализует QML ``GestureCommandBindScreen.qml`` с теми же шагами:
    1. Выбрать жест из словаря (БД).
    2. Выбрать категорию команды (Запуск приложения / Системное действие / ...).
    3. Выбрать конкретное действие (open_app, open_url, volume_up, ...).
    4. Заполнить параметры (JSON).
    5. Дать имя команде.
    6. Сохранить — привязка попадает в БД, ``CommandExecutor`` синхронизируется.
"""
from __future__ import annotations

import json

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


class BindingsView:
    """Экран привязок: ComboBox жестов / категорий / действий + форма."""

    def __init__(self, page: ft.Page, controller: AppController) -> None:
        self._page = page
        self._controller = controller

        # Кэш данных
        self._gestures: list[dict] = []
        self._categories: list[dict] = []
        self._actions: list[dict] = []

        # Контролы
        # Flet 0.85: Dropdown использует on_select, а не on_change.
        self._gesture_dd = ft.Dropdown(
            label="Жест из словаря",
            hint_text="Выберите жест",
            border_color=COLOR_SURFACE_HIGH,
            on_select=self._on_gesture_changed,
            editable=False,
        )
        self._category_dd = ft.Dropdown(
            label="Категория команды",
            border_color=COLOR_SURFACE_HIGH,
            on_select=self._on_category_changed,
            editable=False,
        )
        self._action_dd = ft.Dropdown(
            label="Действие",
            border_color=COLOR_SURFACE_HIGH,
            on_select=self._on_action_changed,
            editable=False,
        )
        self._params_field = ft.TextField(
            label="Параметры (JSON)",
            value="{}",
            multiline=True,
            min_lines=1,
            max_lines=4,
            border_color=COLOR_SURFACE_HIGH,
            text_style=ft.TextStyle(font_family="Menlo", size=13),
        )
        self._name_field = ft.TextField(
            label="Имя команды",
            hint_text="Например: Открыть Safari",
            border_color=COLOR_SURFACE_HIGH,
            on_change=lambda _e: self._refresh_overwrite_hint(),
        )

        self._field_hints = ft.Text("", size=12, color=COLOR_MUTED)

        self._warn_two_hands = ft.Container(
            content=ft.Text(
                "⚠ Это «опасное» действие. Рекомендуется привязать его к "
                "двуручному жесту (правило R6).",
                color="#FFB300",
                size=12,
            ),
            bgcolor="#3a2a14",
            border=ft.Border.all(1, "#FFB300"),
            border_radius=8,
            padding=10,
            visible=False,
        )
        self._overwrite_hint = ft.Container(
            content=ft.Text("", color=COLOR_ACCENT, size=12),
            bgcolor="#1a2a3a",
            border=ft.Border.all(1, COLOR_ACCENT),
            border_radius=8,
            padding=10,
            visible=False,
        )

        self._save_btn = ft.FilledButton(
            content=ft.Text("Сохранить привязку", weight=ft.FontWeight.BOLD),
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                bgcolor=COLOR_ACCENT,
                color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=24, vertical=16),
            ),
            on_click=self._on_save_click,
        )

        self._test_btn = ft.OutlinedButton(
            content=ft.Text("Тестовый запуск"),
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_test_click,
        )

        self._refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Обновить список жестов из БД",
            on_click=lambda _e: self._refresh_gestures(),
        )

        self._error_text = ft.Text("", color=COLOR_DANGER, size=13, visible=False)
        self._info_text = ft.Text("", color=COLOR_SUCCESS, size=13, visible=False)

    # ---- Жизненный цикл ---------------------------------------------------

    def on_show(self) -> None:
        # Перечитываем словарь жестов из БД (R3) и категории при каждом показе.
        self._refresh_categories()
        self._refresh_gestures()

    def on_hide(self) -> None:
        pass

    # ---- Данные -----------------------------------------------------------

    def _refresh_categories(self) -> None:
        self._categories = self._controller.get_action_categories()
        self._category_dd.options = [
            ft.DropdownOption(key=c["id"], text=c["label"]) for c in self._categories
        ]
        try:
            self._category_dd.update()
        except Exception:
            pass

    def _refresh_gestures(self) -> None:
        self._gestures = self._controller.get_db_gestures()
        opts: list[ft.DropdownOption] = []
        for g in self._gestures:
            text = g["label"]
            bound = g.get("boundCommandName") or ""
            if bound:
                text = f"{text}  →  {bound}"
            opts.append(ft.DropdownOption(key=g["label"], text=text))
        self._gesture_dd.options = opts
        self._gesture_dd.hint_text = (
            "Нет жестов в БД — добавьте на экране «Жесты»"
            if not opts
            else "Выберите жест"
        )
        try:
            self._gesture_dd.update()
        except Exception:
            pass
        self._refresh_overwrite_hint()

    def _refresh_actions(self) -> None:
        cat_id = self._category_dd.value or ""
        self._actions = self._controller.get_actions_for_category(cat_id)
        self._action_dd.options = [
            ft.DropdownOption(key=a["action"], text=a["action"]) for a in self._actions
        ]
        self._action_dd.value = None
        self._field_hints.value = ""
        try:
            self._action_dd.update()
            self._field_hints.update()
        except Exception:
            pass

    # ---- Обработчики ------------------------------------------------------

    def _selected_gesture(self) -> dict | None:
        val = self._gesture_dd.value
        if not val:
            return None
        for g in self._gestures:
            if g["label"] == val:
                return g
        return None

    def _selected_action(self) -> dict | None:
        val = self._action_dd.value
        if not val:
            return None
        for a in self._actions:
            if a["action"] == val:
                return a
        return None

    def _on_gesture_changed(self, _e) -> None:
        self._refresh_overwrite_hint()
        self._refresh_warn_two_hands()

    def _on_category_changed(self, _e) -> None:
        self._refresh_actions()
        self._refresh_warn_two_hands()

    def _on_action_changed(self, _e) -> None:
        action = self._selected_action()
        if action is None:
            return
        # Подставить пример параметров в поле, очищая action/platform.
        example = dict(action.get("example") or {})
        example.pop("action", None)
        example.pop("platform", None)
        self._params_field.value = json.dumps(example, ensure_ascii=False)
        if not (self._name_field.value or "").strip():
            self._name_field.value = f"User: {action['action']}"
        # Подсказки по полям.
        hints = action.get("fieldHints") or {}
        if hints:
            self._field_hints.value = "\n".join(f"• {k}: {v}" for k, v in hints.items())
        else:
            self._field_hints.value = "Без дополнительных параметров"
        try:
            self._params_field.update()
            self._name_field.update()
            self._field_hints.update()
        except Exception:
            pass
        self._refresh_warn_two_hands()

    def _refresh_warn_two_hands(self) -> None:
        g = self._selected_gesture()
        a = self._selected_action()
        show = False
        if g is not None and a is not None and not g.get("isTwoHands"):
            if self._controller.is_action_dangerous(a["action"]):
                show = True
        if self._warn_two_hands.visible != show:
            self._warn_two_hands.visible = show
            try:
                self._warn_two_hands.update()
            except Exception:
                pass

    def _refresh_overwrite_hint(self) -> None:
        g = self._selected_gesture()
        if g is None:
            show = False
            msg = ""
        else:
            bound = (g.get("boundCommandName") or "").strip()
            current = (self._name_field.value or "").strip()
            if bound and bound != current:
                show = True
                msg = (
                    f"На этом жесте уже есть команда «{bound}». "
                    "При сохранении она будет заменена (правило R1)."
                )
            else:
                show = False
                msg = ""
        if msg and isinstance(self._overwrite_hint.content, ft.Text):
            self._overwrite_hint.content.value = msg
        self._overwrite_hint.visible = show
        try:
            self._overwrite_hint.update()
        except Exception:
            pass

    def _build_action_spec(self) -> dict | str:
        """Собрать action_spec из выбора + параметров. Возвращает dict или
        сообщение об ошибке (str)."""
        a = self._selected_action()
        if a is None:
            return "Выберите действие"
        spec: dict = {"action": a["action"], "platform": "macos"}
        raw = (self._params_field.value or "").strip()
        if raw:
            try:
                extra = json.loads(raw)
            except json.JSONDecodeError as e:
                return f"Параметры: некорректный JSON: {e}"
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if k not in ("action", "platform"):
                        spec[k] = v
        return spec

    def _show_message(self, error: str = "", info: str = "") -> None:
        self._error_text.value = error
        self._error_text.visible = bool(error)
        self._info_text.value = info
        self._info_text.visible = bool(info)
        try:
            self._error_text.update()
            self._info_text.update()
        except Exception:
            pass

    def _on_save_click(self, _e) -> None:
        self._show_message("", "")
        g = self._selected_gesture()
        if g is None:
            self._show_message(error="Выберите жест")
            return
        spec = self._build_action_spec()
        if isinstance(spec, str):
            self._show_message(error=spec)
            return
        name = (self._name_field.value or "").strip()
        bound_existing = (g.get("boundCommandName") or "").strip()
        # R8: уникальность имени — пропускаем проверку, если переписываем
        # привязку под существующим именем (overwrite candidate).
        if name and (not bound_existing or name != bound_existing):
            name_err = self._controller.validate_command_name(name)
            if name_err:
                self._show_message(error=name_err)
                return
        spec_err = self._controller.validate_action_spec_json(
            json.dumps(spec, ensure_ascii=False)
        )
        if spec_err:
            self._show_message(error=spec_err)
            return
        err = self._controller.save_binding(
            g["label"], name, json.dumps(spec, ensure_ascii=False)
        )
        if err:
            self._show_message(error=err)
            return
        self._show_message(info=f"Сохранено: «{g['label']}» → {name}")
        # Перечитать жесты, чтобы статус «уже привязано» обновился.
        self._refresh_gestures()

    def _on_test_click(self, _e) -> None:
        """Запускает текущую команду (если она уже сохранена в БД) — удобно
        для отладки привязки сразу после её создания."""
        g = self._selected_gesture()
        if g is None:
            self._show_message(error="Выберите жест")
            return
        # Передаём заведомо высокую уверенность, чтобы обойти R4.
        ok = self._controller.execute_for_gesture(g["label"], confidence=1.0)
        if ok:
            self._show_message(info=f"Тест запущен: «{g['label']}»")
        else:
            self._show_message(
                error="Не удалось выполнить — нет привязки в БД или R5 cooldown"
            )

    # ---- Сборка дерева ---------------------------------------------------

    def build(self) -> ft.Control:
        header = ft.Row(
            controls=[
                ft.Icon(ft.Icons.LINK, color=COLOR_ACCENT, size=28),
                ft.Text(
                    "Привязки жестов к командам ОС",
                    size=20,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_ON_SURFACE,
                ),
                ft.Container(expand=True),
                self._refresh_btn,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        hint = ft.Text(
            "Жест → команда. Выберите жест из словаря (R3), категорию действия и "
            "параметры. После сохранения система автоматически выполнит команду "
            "при следующем распознавании этого жеста (порог уверенности R4 и "
            "cooldown R5 — на экране «Настройки»).",
            color=COLOR_MUTED,
            size=12,
        )
        return ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                surface_card(
                    ft.Column(spacing=8, controls=[header, hint]),
                    padding=16,
                    radius=16,
                ),
                surface_card(
                    ft.Column(
                        spacing=12,
                        controls=[
                            ft.Text(
                                "1. Жест",
                                size=14,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                            ),
                            self._gesture_dd,
                            ft.Divider(color=COLOR_SURFACE_HIGH, thickness=1),
                            ft.Text(
                                "2. Категория и действие",
                                size=14,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                            ),
                            self._category_dd,
                            self._action_dd,
                            self._field_hints,
                            ft.Divider(color=COLOR_SURFACE_HIGH, thickness=1),
                            ft.Text(
                                "3. Параметры и имя",
                                size=14,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_ON_SURFACE,
                            ),
                            self._params_field,
                            self._name_field,
                            self._warn_two_hands,
                            self._overwrite_hint,
                            ft.Row(
                                spacing=10,
                                controls=[self._save_btn, self._test_btn],
                            ),
                            self._error_text,
                            self._info_text,
                        ],
                    ),
                    padding=20,
                    radius=16,
                ),
            ],
        )
