#!/usr/bin/env python3
import os
import sys
import subprocess

print("=== Диагностика Qt на macOS ===")

# 1. Проверяем версию Python
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Machine: {os.uname().machine}")

# 2. Проверяем установку PySide6
try:
    import PySide6

    print(f"PySide6 version: {PySide6.__version__}")
    print(f"PySide6 path: {PySide6.__file__}")

    # Проверяем существование плагинов
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins')
    print(f"Plugin path: {plugin_path}")
    print(f"Plugin path exists: {os.path.exists(plugin_path)}")

    if os.path.exists(plugin_path):
        print("Contents of plugins directory:")
        for item in os.listdir(plugin_path):
            print(f"  {item}")

except ImportError as e:
    print(f"PySide6 import failed: {e}")
    sys.exit(1)

# 3. Проверяем переменные окружения
print("\nEnvironment variables:")
for key in ['QT_QPA_PLATFORM', 'QT_QPA_PLATFORM_PLUGIN_PATH']:
    value = os.environ.get(key, 'NOT SET')
    print(f"{key}: {value}")

# 4. Пробуем найти плагины через system find
print("\nSearching for cocoa plugin in system...")
try:
    result = subprocess.run(['find', '/', '-name', '*cocoa*', '-type', 'f', '-path', '*/Qt/plugins/*'],
                            timeout=10, capture_output=True, text=True)
    if result.returncode == 0:
        print("Found cocoa plugins:")
        for line in result.stdout.split('\n'):
            if line and 'cocoa' in line:
                print(f"  {line}")
except:
    print("System search failed or timed out")

print("\n=== Диагностика завершена ===")
# Проверим наличие cocoa плагина
platforms_path = os.path.join(plugin_path, 'platforms')
print(f"\nPlatforms path: {platforms_path}")
if os.path.exists(platforms_path):
    print("Contents of platforms directory:")
    for item in os.listdir(platforms_path):
        print(f"  {item}")
        if 'cocoa' in item:
            print(f"    ✓ FOUND COCOA PLUGIN: {item}")