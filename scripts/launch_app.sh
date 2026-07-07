#!/bin/bash
# Скрипт запуска приложения GestureBind
# Launch script for GestureBind application

cd "$(dirname "$0")/.."

# Активация виртуального окружения если есть
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Настройка переменных окружения для Qt на macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Найти плагины Qt через Python
    QT_PLUGINS=$(python3 -c 'import PySide6; import os; print(os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins"))')
    
    if [ -d "$QT_PLUGINS" ]; then
        export QT_PLUGIN_PATH="$QT_PLUGINS"
        echo "✓ Qt plugins path: $QT_PLUGIN_PATH"
    else
        echo "⚠ Warning: Qt plugins directory not found"
    fi
    
    export QT_QPA_PLATFORM=cocoa
    echo "✓ Qt platform: $QT_QPA_PLATFORM"
fi

# Запуск приложения
echo "Запуск GestureBind..."
python3 -m app.main
