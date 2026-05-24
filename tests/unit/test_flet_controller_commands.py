# -*- coding: utf-8 -*-

from app.flet_app.controller import AppController


def test_list_commands_includes_action_category_and_config():
    controller = AppController.__new__(AppController)

    class FakeExecutor:
        commands_registry = {
            "open browser": {
                "action": "open_app",
                "app": "Safari",
                "platform": "macos",
            }
        }

    controller._command_executor = FakeExecutor()

    commands = controller.list_commands()

    assert commands == [
        {
            "name": "open browser",
            "description": "",
            "platform": "macos",
            "action": "open_app",
            "category": "launch",
            "config": {
                "action": "open_app",
                "app": "Safari",
                "platform": "macos",
            },
        }
    ]
