#!/usr/bin/env python3
"""Тестовый скрипт для проверки Qt"""
import sys
import os
from pathlib import Path

# Настройка путей ДО импорта PySide6
if sys.platform == "darwin":
    import PySide6
    p = Path(PySide6.__file__).parent / "Qt" / "plugins"
    os.environ['QT_PLUGIN_PATH'] = str(p.resolve())
    os.environ['QT_QPA_PLATFORM'] = 'cocoa'
    print(f"QT_PLUGIN_PATH: {os.environ['QT_PLUGIN_PATH']}")
    print(f"Path exists: {p.exists()}")

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QGuiApplication

if sys.platform == "darwin":
    import PySide6
    p = Path(PySide6.__file__).parent / "Qt" / "plugins"
    QCoreApplication.setLibraryPaths([str(p.resolve())])
    print(f"Library paths: {QCoreApplication.libraryPaths()}")

app = QGuiApplication(sys.argv)
print("✓ QGuiApplication created successfully!")
print("✓ Application is running!")
