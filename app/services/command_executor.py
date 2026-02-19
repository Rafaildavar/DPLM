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
        
        # Проверка платформы / Platform check
        if "platform" in config:
            platform_req = config["platform"]
            if platform_req != "all" and platform_req != self.system:
                logger.warning(f"Команда '{command_name}' не поддерживается на {self.system}")
                return False
        
        # Выполнение действия / Execute action
        action = config.get("action")
        
        try:
            if action == "open_app":
                return self._open_application(config.get("app", ""))
            
            elif action == "run_script":
                script_path = config.get("script_path")
                if script_path:
                    return self._run_script(script_path, config.get("args", []))
            
            elif action == "key_combination":
                keys = config.get("keys", [])
                return self._press_keys(keys)
            
            elif action == "volume_up":
                return self._volume_up()
            
            elif action == "volume_down":
                return self._volume_down()
            
            elif action == "custom":
                handler = config.get("handler")
                if handler and callable(handler):
                    return handler(**kwargs)
            
            else:
                logger.error(f"Неизвестное действие: {action}")
                return False
        
        except Exception as e:
            logger.error(f"Ошибка выполнения команды '{command_name}': {e}")
            return False
    
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
