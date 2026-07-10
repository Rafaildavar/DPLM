# -*- coding: utf-8 -*-
from app.flet_app.ui_copy import friendly_status


def test_friendly_status_known():
    assert friendly_status("Idle") == "Готов к работе"
    assert friendly_status("Camera: streaming") == "Камера включена"


def test_friendly_status_prefix():
    assert "Ошибка камеры" in friendly_status("Camera error: no device")
