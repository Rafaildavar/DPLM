# -*- coding: utf-8 -*-

from app.services.command_executor import CommandExecutor


def test_macos_default_commands_offer_demo_actions(monkeypatch):
    monkeypatch.setattr("app.services.command_executor.platform.system", lambda: "Darwin")

    commands = CommandExecutor().commands_registry

    assert commands["сделать снимок экрана"]["action"] == "screenshot"
    assert commands["прокрутить страницу вниз"]["clicks"] == -5
    assert commands["прокрутить страницу вверх"]["clicks"] == 5
    assert commands["выключить или включить звук"]["action"] == "mute_toggle"
    assert commands["пауза или продолжить музыку"]["kind"] == "play_pause"
    assert commands["открыть finder"]["app"] == "Finder"
    assert commands["открыть системные настройки"]["app"] == "System Settings"
