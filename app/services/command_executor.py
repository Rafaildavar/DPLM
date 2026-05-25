# -*- coding: utf-8 -*-
"""
Модуль выполнения команд системы
Command execution module

Выполняет команды пользователя через subprocess и pyautogui
Executes user commands via subprocess and pyautogui
"""

import subprocess
import platform
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("[WARN] pyautogui not installed - GUI automation disabled")

logger = logging.getLogger(__name__)


class CommandExecutor:
    """
    Исполнитель команд системы.
    System command executor.
    """
    
    def __init__(self):
        """Инициализация исполнителя команд / Initialize command executor"""
        self.system = platform.system().lower()
        self.commands_registry: Dict[str, Dict[str, Any]] = {}
        self._register_default_commands()

    def _host_platform_tag(self) -> str:
        """Тег платформы как в БД/UI: macos | windows | linux."""
        if self.system == "darwin":
            return "macos"
        if self.system == "windows":
            return "windows"
        return self.system

    def _config_platform_matches(self, config: Dict[str, Any]) -> bool:
        req = (config.get("platform") or "all")
        if isinstance(req, str):
            req = req.strip().lower()
        else:
            req = "all"
        if req in ("", "all"):
            return True
        if req in ("darwin", "mac", "macos"):
            req = "macos"
        if req in ("win32", "windows"):
            req = "windows"
        return req == self._host_platform_tag()
    
    def _register_default_commands(self):
        """Регистрация команд по умолчанию / Register default commands"""
        
        # Системные команды macOS / macOS system commands
        if self.system == "darwin":
            self.register_command("открыть браузер", {
                "action": "open_app",
                "app": "Safari",
                "platform": "macos"
            })
            self.register_command("open browser", {
                "action": "open_app",
                "app": "Safari",
                "platform": "macos"
            })
            self.register_command("увеличить громкость", {
                "action": "key_combination",
                "keys": ["volumeup"],
                "platform": "macos"
            })
            self.register_command("volume up", {
                "action": "volume_up",
                "platform": "macos"
            })
            self.register_command("уменьшить громкость", {
                "action": "volume_down",
                "platform": "macos"
            })
            self.register_command("volume down", {
                "action": "volume_down",
                "platform": "macos"
            })
            self.register_command("выключить или включить звук", {
                "action": "mute_toggle",
                "platform": "macos"
            })
            self.register_command("сделать снимок экрана", {
                "action": "screenshot",
                "platform": "macos"
            })
            self.register_command("прокрутить страницу вниз", {
                "action": "scroll",
                "clicks": -5,
                "platform": "macos"
            })
            self.register_command("прокрутить страницу вверх", {
                "action": "scroll",
                "clicks": 5,
                "platform": "macos"
            })
            self.register_command("пауза или продолжить музыку", {
                "action": "media_key",
                "kind": "play_pause",
                "platform": "macos"
            })
            self.register_command("следующий трек", {
                "action": "media_key",
                "kind": "next",
                "platform": "macos"
            })
            self.register_command("предыдущий трек", {
                "action": "media_key",
                "kind": "prev",
                "platform": "macos"
            })
            self.register_command("открыть finder", {
                "action": "open_app",
                "app": "Finder",
                "platform": "macos"
            })
            self.register_command("открыть системные настройки", {
                "action": "open_app",
                "app": "System Settings",
                "platform": "macos"
            })
        
        # Системные команды Windows / Windows system commands
        elif self.system == "windows":
            self.register_command("открыть браузер", {
                "action": "open_app",
                "app": "chrome",
                "platform": "windows"
            })
            self.register_command("open browser", {
                "action": "open_app",
                "app": "chrome",
                "platform": "windows"
            })
    
    def register_command(self, name: str, config: Dict[str, Any]):
        """
        Зарегистрировать команду.
        Register command.
        
        Args:
            name: название команды / command name
            config: конфигурация команды / command configuration
        """
        self.commands_registry[name.lower()] = config
    
    def execute_config(self, config: Dict[str, Any], **kwargs) -> bool:
        """
        Выполнить действие по словарю (в т.ч. JSON из БД ``Command.action_spec``).
        """
        if not self._config_platform_matches(config):
            logger.warning(
                "Команда не для этой платформы (хост=%s, в конфиге platform=%s)",
                self._host_platform_tag(),
                config.get("platform"),
            )
            return False
        try:
            return self._dispatch_action(config, **kwargs)
        except Exception as e:
            logger.error("Ошибка выполнения действия %s: %s", config.get("action"), e)
            return False

    def _dispatch_action(self, config: Dict[str, Any], **kwargs) -> bool:
        action = config.get("action")

        if action == "open_app":
            return self._open_application(config.get("app", ""))

        if action == "run_script":
            script_path = config.get("script_path")
            if script_path:
                return self._run_script(script_path, config.get("args", []))
            return False

        if action == "key_combination":
            keys = config.get("keys", [])
            return self._press_keys(keys)

        if action == "press":
            key = config.get("key") or config.get("keys")
            if isinstance(key, list) and key:
                key = key[0]
            if not key:
                return False
            return self._press_single(str(key))

        if action == "scroll":
            return self._scroll(int(config.get("clicks", -3)))

        if action == "volume_up":
            return self._volume_up()

        if action == "volume_down":
            return self._volume_down()

        if action == "open_url":
            url = (config.get("url") or "").strip()
            if not url:
                return False
            return self._open_url(url)

        if action == "mute_toggle":
            return self._press_single("volumemute")

        if action == "brightness_up":
            return self._brightness("up")

        if action == "brightness_down":
            return self._brightness("down")

        if action == "lock_screen":
            return self._lock_screen()

        if action == "screenshot":
            return self._screenshot()

        if action == "media_key":
            kind = (config.get("kind") or "").strip().lower()
            return self._media_key(kind)

        if action == "custom":
            handler = config.get("handler")
            if handler and callable(handler):
                return handler(**kwargs)
            return False

        logger.error("Неизвестное действие: %s", action)
        return False

    def execute(self, command_name: str, **kwargs) -> bool:
        """
        Выполнить команду по имени.
        Execute command by name.
        
        Args:
            command_name: название команды / command name
            **kwargs: дополнительные параметры / additional parameters
        
        Returns:
            True если команда выполнена успешно / True if command executed successfully
        """
        command_name_lower = command_name.lower()
        
        # Поиск команды / Find command
        if command_name_lower not in self.commands_registry:
            logger.warning(f"Команда не найдена: {command_name}")
            return False
        
        config = self.commands_registry[command_name_lower]
        return self.execute_config(config, **kwargs)
    
    def _open_application(self, app_name: str) -> bool:
        """
        Открыть приложение.
        Open application.
        
        Args:
            app_name: название приложения / application name
        
        Returns:
            True если приложение открыто / True if application opened
        """
        try:
            if self.system == "darwin":
                # macOS: используем open команду / macOS: use open command
                subprocess.Popen(["open", "-a", app_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            
            elif self.system == "windows":
                # Windows: используем start команду / Windows: use start command
                subprocess.Popen(["start", app_name], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            
            else:
                logger.warning(f"Открытие приложений не поддерживается на {self.system}")
                return False
        
        except Exception as e:
            logger.error(f"Ошибка открытия приложения '{app_name}': {e}")
            return False
    
    def _run_script(self, script_path: str, args: List[str] = None) -> bool:
        """
        Запустить скрипт.
        Run script.
        
        Args:
            script_path: путь к скрипту / script path
            args: аргументы скрипта / script arguments
        
        Returns:
            True если скрипт запущен / True if script started
        """
        try:
            script = Path(script_path)
            if not script.exists():
                logger.error(f"Скрипт не найден: {script_path}")
                return False
            
            cmd = [sys.executable, str(script)]
            if args:
                cmd.extend(args)
            
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        
        except Exception as e:
            logger.error(f"Ошибка запуска скрипта '{script_path}': {e}")
            return False
    
    def _press_keys(self, keys: List[str]) -> bool:
        """
        Нажать комбинацию клавиш.
        Press key combination.
        
        Args:
            keys: список клавиш / list of keys
        
        Returns:
            True если клавиши нажаты / True if keys pressed
        """
        if not PYAUTOGUI_AVAILABLE:
            logger.warning("pyautogui не установлен - автоматизация клавиатуры недоступна")
            return False
        
        try:
            pyautogui.hotkey(*keys)
            return True
        except Exception as e:
            logger.error(f"Ошибка нажатия клавиш {keys}: {e}")
            return False

    def _press_single(self, key: str) -> bool:
        if not PYAUTOGUI_AVAILABLE:
            logger.warning("pyautogui не установлен - автоматизация клавиатуры недоступна")
            return False
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            logger.error("Ошибка нажатия клавиши %s: %s", key, e)
            return False

    def _scroll(self, clicks: int) -> bool:
        if not PYAUTOGUI_AVAILABLE:
            return False
        try:
            pyautogui.scroll(clicks)
            return True
        except Exception as e:
            logger.error("Ошибка прокрутки: %s", e)
            return False

    def _open_url(self, url: str) -> bool:
        try:
            if self.system == "darwin":
                subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            if self.system == "windows":
                subprocess.Popen(["cmd", "/c", "start", "", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            logger.warning("open_url не поддерживается на %s", self.system)
            return False
        except Exception as e:
            logger.error("Ошибка open_url: %s", e)
            return False
    
    def _brightness(self, direction: str) -> bool:
        """
        Изменить яркость экрана.
        macOS: AppleScript ``key code`` 144 (вверх) и 145 (вниз) — это коды
        клавиш F2/F1 без модификаторов, обрабатываются системой как
        медиа-клавиши яркости.
        """
        direction = (direction or "").strip().lower()
        if direction not in ("up", "down"):
            logger.error("brightness: неизвестное направление %r", direction)
            return False
        if self.system == "darwin":
            key_code = 144 if direction == "up" else 145
            try:
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        f'tell application "System Events" to key code {key_code}',
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                )
                return True
            except (OSError, subprocess.SubprocessError) as e:
                logger.error("brightness osascript: %s", e)
                return False
        if not PYAUTOGUI_AVAILABLE:
            return False
        try:
            pyautogui.press("brightnessup" if direction == "up" else "brightnessdown")
            return True
        except Exception as e:
            logger.error("brightness pyautogui: %s", e)
            return False

    def _lock_screen(self) -> bool:
        """
        Заблокировать экран / усыпить дисплей.
        macOS: ``pmset displaysleepnow`` — гасит дисплей, после wake система
        запросит пароль, если он включён.
        """
        try:
            if self.system == "darwin":
                subprocess.Popen(
                    ["pmset", "displaysleepnow"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            logger.warning("lock_screen: поддерживается только на macOS")
            return False
        except OSError as e:
            logger.error("lock_screen: %s", e)
            return False

    def _screenshot(self) -> bool:
        """
        Сделать скриншот в ``~/Desktop/dplm_screenshot_<ts>.png``.
        macOS: системная утилита ``screencapture``.
        """
        if self.system != "darwin":
            logger.warning("screenshot: поддерживается только на macOS")
            return False
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = Path.home() / "Desktop" / f"dplm_screenshot_{ts}.png"
        try:
            subprocess.Popen(
                ["screencapture", "-x", str(out)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except OSError as e:
            logger.error("screenshot: %s", e)
            return False

    def _media_key(self, kind: str) -> bool:
        """
        Управление мультимедиа: play/pause, next, prev.
        Под капотом — клавиши PyAutoGUI ``playpause`` / ``nexttrack`` / ``prevtrack``.
        """
        mapping = {
            "play_pause": "playpause",
            "play": "playpause",
            "pause": "playpause",
            "next": "nexttrack",
            "prev": "prevtrack",
            "previous": "prevtrack",
        }
        key = mapping.get((kind or "").strip().lower())
        if not key:
            logger.error("media_key: неизвестный kind %r", kind)
            return False
        return self._press_single(key)

    def _volume_up(self) -> bool:
        """Увеличить громкость / Increase volume"""
        if not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            if self.system == "darwin":
                # macOS: F12 или специальная команда / macOS: F12 or special command
                pyautogui.press("volumeup")
            elif self.system == "windows":
                # Windows: Volume Up key / Windows: Volume Up key
                pyautogui.press("volumeup")
            return True
        except Exception as e:
            logger.error(f"Ошибка увеличения громкости: {e}")
            return False
    
    def _volume_down(self) -> bool:
        """Уменьшить громкость / Decrease volume"""
        if not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            if self.system == "darwin":
                pyautogui.press("volumedown")
            elif self.system == "windows":
                pyautogui.press("volumedown")
            return True
        except Exception as e:
            logger.error(f"Ошибка уменьшения громкости: {e}")
            return False


# Глобальный экземпляр исполнителя / Global executor instance
_executor_instance: Optional[CommandExecutor] = None


def get_executor() -> CommandExecutor:
    """
    Получить глобальный экземпляр исполнителя команд.
    Get global command executor instance.
    
    Returns:
        CommandExecutor instance
    """
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = CommandExecutor()
    return _executor_instance
