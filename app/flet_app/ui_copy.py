# -*- coding: utf-8 -*-
"""Пользовательские тексты интерфейса (RU)."""

from __future__ import annotations

APP_TITLE = "GestureBind — жестовое управление"
APP_HEADER = "GestureBind — жестовое управление"

STATUS_EXACT: dict[str, str] = {
    "Idle": "Готов к работе",
    "Stopped": "Остановлено",
    "Recognizing in background": "Распознавание активно",
    "Camera: streaming": "Камера включена",
    "Camera: OpenCV не установлен": "Камера недоступна: не установлен OpenCV",
    "CV: загрузка MediaPipe…": "Подготовка распознавания…",
    "CV: embedded gesture recognition": "Распознавание жестов включено",
    "CV init failed": "Не удалось запустить распознавание",
    "Recognition start failed": "Не удалось запустить распознавание",
    "Diagnostics: OK": "Проверка системы: всё в порядке",
}

STATUS_PREFIX: list[tuple[str, str]] = [
    ("Camera error:", "Ошибка камеры:"),
    ("Camera: open failed", "Не удалось открыть камеру — проверьте доступ в «Конфиденциальность → Камера»"),
    ("CV init error:", "Ошибка распознавания:"),
    ("Diagnostics:", "Диагностика:"),
    ("Pointer:", "Указатель:"),
]


def friendly_status(raw: str) -> str:
    """Перевести технический статус контроллера в понятную фразу."""
    text = (raw or "").strip()
    if not text:
        return "Готов к работе"
    if text in STATUS_EXACT:
        return STATUS_EXACT[text]
    for prefix, label in STATUS_PREFIX:
        if text.startswith(prefix):
            if prefix.endswith(":"):
                tail = text[len(prefix) :].strip()
                return f"{label} {tail}".strip() if tail else label.rstrip(":")
            return label
    return text
