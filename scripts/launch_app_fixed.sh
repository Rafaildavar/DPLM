#!/bin/bash
# Исправленный скрипт запуска приложения DPLM
# Fixed launch script for DPLM application

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
        # Также установим DYLD_LIBRARY_PATH для macOS
        export DYLD_LIBRARY_PATH="$QT_PLUGINS/../lib:$DYLD_LIBRARY_PATH"
        echo "✓ Qt plugins path: $QT_PLUGIN_PATH"
        echo "✓ DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
    else
        echo "⚠ Warning: Qt plugins directory not found"
    fi
    
    export QT_QPA_PLATFORM=cocoa
    echo "✓ Qt platform: $QT_QPA_PLATFORM"
    
    # Отключить проверку подписи для разработки (опционально)
    # export QT_MAC_DISABLE_APP_ICON_CHECK=1
fi

# Запуск приложения
echo "Запуск DPLM..."
python3 -m app.start
