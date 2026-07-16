"""
Корневой shell GestureBind: боковая навигация, статус и содержимое.

Структура:

    Главная        — встроенное распознавание (камера + жест + команда).
    Жесты          — список жестов из БД.
    Датасет        — классы и записи в data/gestures.
    Обучение       — запись и обучение жестов.
    Привязки       — главная фича: жест → команда ОС.
    Настройки      — политика R4/R5/R6.

Все экраны создаются один раз при старте и переключаются через атрибут
``visible``. Это надёжнее, чем подмена ``Container.content`` (в Flet 0.85
такой подход иногда не подхватывает изменение и контент не перерисовывается).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import flet as ft

from app.flet_app.controller import AppController
from app.flet_app.theme import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_ON_SURFACE,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_SURFACE_HIGH,
    app_background,
)
from app.flet_app.views.bindings import BindingsView
from app.flet_app.views.gestures import DatasetGesturesView, GesturesView
from app.flet_app.views.home import HomeView
from app.flet_app.views.settings import SettingsView
from app.flet_app.views.training import TrainingView


@dataclass
class NavItem:
    key: str
    label: str
    icon: str
    icon_selected: str
    build: Callable[[], ft.Control]
    on_show: Optional[Callable[[], None]] = None
    on_hide: Optional[Callable[[], None]] = None


def build_shell(page: ft.Page, controller: AppController) -> ft.Control:
    # --- Хедер с глобальным статусом ---------------------------------------

    def border_all(color: str) -> ft.Border:
        side = ft.BorderSide(1, color)
        return ft.Border(top=side, right=side, bottom=side, left=side)

    def friendly_status(value: str, *, active: bool) -> str:
        clean = str(value or "").strip().lower()
        if any(
            token in clean
            for token in ("error", "failed", "ошиб", "не удалось", "перезапуст")
        ):
            return "Нужна проверка"
        if active:
            return "Распознавание включено"
        return "Готово"

    status_dot = ft.Container(
        width=12,
        height=12,
        border_radius=6,
        bgcolor=COLOR_SUCCESS if controller.is_recognizing else COLOR_MUTED,
    )
    status_text = ft.Text(
        friendly_status(controller.status, active=controller.is_recognizing),
        size=12,
        color=COLOR_ON_SURFACE,
    )
    title_text = ft.Text(
        "GestureBind",
        size=19,
        weight=ft.FontWeight.BOLD,
        color=COLOR_ON_SURFACE,
    )

    header = ft.Container(
        bgcolor=COLOR_SURFACE,
        border=border_all(COLOR_SURFACE_HIGH),
        border_radius=12,
        padding=12,
        content=ft.Row(
            controls=[
                ft.Icon(ft.Icons.GESTURE, color=COLOR_ACCENT, size=27),
                ft.Column(
                    spacing=2,
                    controls=[
                        title_text,
                        ft.Row(
                            controls=[status_dot, status_text],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                ),
            ],
            spacing=14,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )

    # --- Создаём views -----------------------------------------------------

    home = HomeView(page, controller)
    gestures = GesturesView(page, controller)
    dataset = DatasetGesturesView(page, controller)
    training = TrainingView(page, controller)
    bindings = BindingsView(page, controller)
    settings = SettingsView(page, controller)

    nav_items: list[NavItem] = [
        NavItem(
            "home", "Главная", ft.Icons.HOME_OUTLINED, ft.Icons.HOME,
            home.build, home.on_show, home.on_hide,
        ),
        NavItem(
            "gestures", "Жесты", ft.Icons.GESTURE, ft.Icons.GESTURE,
            gestures.build, gestures.on_show, gestures.on_hide,
        ),
        NavItem(
            "dataset", "Датасет", ft.Icons.DATASET, ft.Icons.DATASET,
            dataset.build, dataset.on_show, dataset.on_hide,
        ),
        NavItem(
            "training", "Обучение", ft.Icons.MODEL_TRAINING, ft.Icons.MODEL_TRAINING,
            training.build, training.on_show, training.on_hide,
        ),
        NavItem(
            "bindings", "Привязки", ft.Icons.LINK, ft.Icons.LINK,
            bindings.build, bindings.on_show, bindings.on_hide,
        ),
        NavItem(
            "settings", "Настройки", ft.Icons.SETTINGS_OUTLINED, ft.Icons.SETTINGS,
            settings.build, settings.on_show, settings.on_hide,
        ),
    ]

    # Все экраны строятся один раз и оборачиваются в Container с visible.
    panels: list[ft.Container] = []
    for i, item in enumerate(nav_items):
        try:
            body = item.build()
        except Exception as e:
            print(f"[!] failed to build {item.key}: {e}", flush=True)
            import traceback

            traceback.print_exc()
            body = ft.Text(f"Ошибка построения экрана: {e}")
        panels.append(
            ft.Container(
                content=body,
                expand=True,
                visible=(i == 0),
            )
        )

    content_stack = ft.Container(
        expand=True,
        bgcolor=COLOR_SURFACE,
        border_radius=16,
        padding=16,
        content=ft.Stack(expand=True, controls=panels),
    )

    state = {"current": 0, "show_token": 0}

    def show(index: int) -> None:
        prev = state["current"]
        if prev == index:
            return
        prev_item = nav_items[prev]
        if prev_item.on_hide:
            try:
                prev_item.on_hide()
            except Exception as e:
                print(f"[!] on_hide({prev_item.key}): {e}", flush=True)
        for i, p in enumerate(panels):
            p.visible = i == index
        state["current"] = index
        state["show_token"] += 1
        show_token = state["show_token"]
        try:
            page.update()
        except Exception as e:
            print(f"[!] page.update: {e}", flush=True)
        item = nav_items[index]
        if item.on_show:
            def run_on_show() -> None:
                try:
                    item.on_show()
                except Exception as e:
                    print(f"[!] on_show({item.key}): {e}", flush=True)
                if state.get("show_token") == show_token:
                    try:
                        page.update()
                    except Exception:
                        pass

            try:
                page.run_thread(run_on_show)
            except Exception:
                run_on_show()

    def on_nav_change(e) -> None:
        ctrl = getattr(e, "control", None)
        idx = getattr(ctrl, "selected_index", None) if ctrl is not None else None
        if idx is None:
            idx = getattr(e, "data", None)
        try:
            idx = int(idx) if idx is not None else 0
        except (TypeError, ValueError):
            idx = 0
        show(idx)

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=92,
        min_extended_width=200,
        bgcolor=COLOR_SURFACE,
        indicator_color=COLOR_ACCENT,
        selected_label_text_style=ft.TextStyle(
            color=COLOR_ACCENT, weight=ft.FontWeight.W_600
        ),
        unselected_label_text_style=ft.TextStyle(color=COLOR_MUTED),
        destinations=[
            ft.NavigationRailDestination(
                icon=item.icon,
                selected_icon=item.icon_selected,
                label=item.label,
            )
            for item in nav_items
        ],
        on_change=on_nav_change,
    )

    # --- Подписки на глобальный статус ------------------------------------

    def _apply_status() -> None:
        status_text.value = friendly_status(
            controller.status,
            active=controller.is_recognizing,
        )
        status_dot.bgcolor = COLOR_SUCCESS if controller.is_recognizing else COLOR_MUTED
        try:
            page.update()
        except Exception:
            pass

    controller.status_changed.connect(lambda _v: page.run_thread(_apply_status))
    controller.recognizing_changed.connect(lambda _v: page.run_thread(_apply_status))

    # --- Тост-уведомления (жест + команда) --------------------------------

    toast_text = ft.Text("", color=ft.Colors.WHITE, size=13)
    toast = ft.Container(
        content=toast_text,
        bgcolor=COLOR_ACCENT,
        border_radius=12,
        padding=12,
        visible=False,
    )
    toast_lock = threading.Lock()
    toast_token = {"id": 0}

    def _do_show_toast(msg: str, bg: str, tok: int) -> None:
        toast_text.value = msg
        toast.bgcolor = bg
        toast.visible = True
        try:
            page.update()
        except Exception:
            pass

        def hide() -> None:
            with toast_lock:
                if toast_token["id"] != tok:
                    return
            page.run_thread(_do_hide_toast)

        t = threading.Timer(2.0, hide)
        t.daemon = True
        t.start()

    def _do_hide_toast() -> None:
        toast.visible = False
        try:
            page.update()
        except Exception:
            pass

    def show_toast(msg: str, bg: str = COLOR_ACCENT) -> None:
        with toast_lock:
            toast_token["id"] += 1
            tok = toast_token["id"]
        page.run_thread(_do_show_toast, msg, bg, tok)

    controller.command_executed.connect(
        lambda name: show_toast(f"Команда выполнена: {name}", COLOR_ACCENT)
    )
    controller.gesture_detected.connect(
        lambda label: show_toast(f"Жест: {label}", COLOR_SURFACE_HIGH)
    )

    toast_row = ft.Row(
        controls=[toast],
        alignment=ft.MainAxisAlignment.CENTER,
    )

    return ft.Container(
        expand=True,
        gradient=app_background(),
        padding=16,
        content=ft.Column(
            expand=True,
            spacing=16,
            controls=[
                header,
                ft.Row(
                    expand=True,
                    controls=[
                        rail,
                        ft.VerticalDivider(width=1, color=COLOR_SURFACE_HIGH),
                        content_stack,
                    ],
                ),
                toast_row,
            ],
        ),
    )
