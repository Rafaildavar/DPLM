# DPLM: Gesture & Voice Assistant

> **Статус: В активной разработке (полный редизайн)**  
> Текущая ветка: `redesign`  
> Старая версия: тег `v0.6.0-old`

## Описание проекта

Интеллектуальная система управления компьютером с помощью жестов и голосовых команд. Ключевая особенность: пользователь сам обучает систему жестам и привязывает их к любым командам (открытие приложений, системные действия, пользовательские скрипты).

### Основные возможности

1. **Обучение пользовательским жестам**
   - Интуитивный интерфейс для записи семплов жестов
   - Автоматическое обучение модели машинного обучения
   - Проверка точности распознавания в реальном времени

2. **Система команд**
   - Готовые шаблоны: системные, медиа, кастомные
   - Привязка жестов к любым действиям
   - Кроссплатформенное выполнение (Windows + macOS)

3. **Голосовой помощник**
   - Распознавание речи (Speech-to-Text)
   - Озвучка подтверждений и подсказок (Text-to-Speech)
   - Контекстные диалоги и помощь

4. **Современный интерфейс**
   - Минималистичная панель (прозрачность, drag&drop)
   - Анимированный аватар-ассистент
   - Material Design 3 (QML + Qt Quick)
   - Визуальная обратная связь при распознавании

## Архитектура

```
DPLM/
├── app/
│   ├── qml/                # QML интерфейсы (Material Design 3)
│   ├── backend/            # Бизнес-логика (команды, ML)
│   ├── models/             # Модели данных (PostgreSQL + SQLAlchemy)
│   ├── services/           # Сервисы (CV, TTS, STT, executor)
│   ├── resources/          # Ресурсы (иконки, аватар, звуки)
│   └── main.py             # Точка входа (Qt QML Application)
├── data/
│   └── gestures/           # Семплы жестов (NPY)
├── models/
│   └── user_gestures/      # Обученные модели пользователя
├── docs/                   # Документация (RU + EN)
└── requirements.txt        # Зависимости Python
```

## Технологический стек

| Компонент | Технология |
|-----------|-----------|
| **UI Framework** | PySide6 (QML + Qt Quick) |
| **UI Design** | Material Design 3 |
| **Computer Vision** | MediaPipe Hands (baseline), YOLO11 Pose (опционально) |
| **ML Classification** | scikit-learn (KNN/SVM), LSTM/GRU (для сложных жестов) |
| **Database** | PostgreSQL Embedded (pg_embed) |
| **ORM** | SQLAlchemy + Alembic (миграции) |
| **Speech-to-Text** | Временно отключено |
| **Text-to-Speech** | Временно отключено |
| **Command Execution** | subprocess, pyautogui |
| **Platforms** | Windows 10/11, macOS 11+ |

## Быстрый старт

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/your-username/DPLM.git
cd DPLM

# Создание и активация виртуального окружения (рекомендуется Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Запуск

```bash
# 1) Запустить PostgreSQL
docker compose up -d db

# 2) Запустить GUI (Flet — текущая основная версия)
PYTHONDONTWRITEBYTECODE=1 python -B -m app.flet_app.main
```

Голосовой помощник временно отключён: его логика пока не подключена к текущей
Flet-версии, поэтому вкладка «Помощник» убрана из интерфейса, чтобы не
блокировать запуск и не показывать нерабочие кнопки. Основной сценарий сейчас:
обучение жестов, список жестов, привязки жестов к командам и настройки.

Если на macOS приложение зависает на экране `Working...`, удалите старый
bytecode-кэш и запустите снова:

```bash
find app -name __pycache__ -type d -prune -exec rm -rf {} +
PYTHONDONTWRITEBYTECODE=1 python -B -m app.flet_app.main
```

### Legacy-запуск

```bash
# Запуск legacy QML/PySide6 (только если включён PySide6 в requirements.txt)
python -m app.main

# Запуск самой старой версии (для сравнения)
git checkout v0.6.0-old
python -m app.gui_main
```

### Почему Flet, а не PySide6

На macOS 26 (Tahoe) + Python 3.13 + Apple Silicon PySide6 регулярно ломался
с ошибкой `Could not find the Qt platform plugin "cocoa"` из-за конфликтов
загрузки `libqcocoa.dylib` через `@rpath/QtGui.framework`. Flet 0.85+
поставляет собственный Flutter-runtime внутри пакета `flet_desktop`, не
требует системных Qt-плагинов и устанавливается одним `pip install flet`.

Архитектура Flet-версии:

* `app/flet_app/main.py` — точка входа, конфигурация окна.
* `app/flet_app/controller.py` — GUI-агностичный контроллер (камера, инференс,
  команды). Заменяет PySide6-класс `AppController` без зависимостей от Qt.
* `app/flet_app/views/` — экраны основной логики (`home`, `gestures`,
  `training`, `bindings`, `settings`).

Существующие сервисы (`app/services/*`, `app/models/*`, `cv/*`) переиспользуются
без изменений — они и так были GUI-агностичными.

### Известная проблема на macOS 26 (Tahoe)

На macOS 26 wheel-ы `PySide6 6.10+` (включая 6.11) блокируются новой Gatekeeper-проверкой (`com.apple.provenance`), и Qt молча отбрасывает все плагины из директории `PySide6/Qt/plugins/platforms`. Симптом:

```
qt.qpa.plugin: Could not find the Qt platform plugin "cocoa"
This application failed to start because no Qt platform plugin could be initialized.
```

В `requirements.txt` PySide6 закреплён на `6.9.3` — этой версии достаточно, и она работает на macOS 26 + Python 3.13 без правок. Если ты вручную ставил более новую версию, откатись:

```bash
.venv/bin/pip install --force-reinstall 'PySide6==6.9.3' 'PySide6-Addons==6.9.3' 'PySide6-Essentials==6.9.3' 'shiboken6==6.9.3'
```

### Если PySide6 не запускается на macOS

Иногда `pip` оставляет повреждённую установку `PySide6` (пропадают `.dylib`/`.qm` файлы, например `libshiboken6...dylib`). В таком случае полностью переустановите Qt-стек в активном venv:

```bash
pip uninstall -y PySide6 PySide6-Addons shiboken6
pip cache purge
pip install --no-cache-dir -r requirements.txt
python scripts/check_env.py
python -m app.main
```

Если проблема повторяется, пересоздайте виртуальное окружение (`rm -rf .venv && python -m venv .venv`).

Для ошибки `Could not find the Qt platform plugin "cocoa"` дополнительно очистите конфликтующие переменные окружения и запустите снова:

```bash
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
python -m app.main
```

(Приложение выставляет корректные пути к плагинам Qt автоматически при запуске на macOS.)

## Рабочий процесс

1. **Создание команды**
   - Открыть панель команд
   - Нажать "Добавить команду"
   - Указать название, платформу, действие (приложение/скрипт)

2. **Обучение жесту**
   - Выбрать команду → "Назначить жест"
   - Выполнить жест 20-30 раз перед камерой
   - Система автоматически обучит модель
   - Проверить точность в режиме превью

3. **Использование**
   - Система работает в фоне (минималистичная панель)
   - Выполнить жест → команда выполняется
   - Аватар даёт визуальную обратную связь
   - Голосовые подсказки при необходимости

## Документация

- [Архитектура старой версии](docs/ARCHITECTURE_OLD.md) (RU + EN)
- [Руководство пользователя](docs/USER_GUIDE.md) (в разработке)
- [Руководство разработчика](docs/DEVELOPER.md) (в разработке)
- [Список команд](docs/COMMANDS.md) (в разработке)
- [Сравнение CV моделей](docs/CV_COMPARISON.md) (в разработке)
- [Тестирование](docs/TESTING.md) (в разработке)

## Разработка

### Структура веток

- `main` — стабильная версия (релизы)
- `dev` — разработка (интеграция фич)
- `redesign` — **текущая ветка** (полный редизайн)
- `v0.6.0-old` — тег старой версии

### Коммиты

Формат сообщений:
```
<тип>: краткое описание

Подробное описание (опционально)

<тип>: feat, fix, docs, refactor, test, chore
```

Пример:
```
feat: implement gesture training service with LSTM support

- Add app/services/gesture_trainer.py
- Support both KNN and LSTM classifiers
- Integrate with QML UI via Qt signals
```

## Требования

- Python 3.12 (рекомендуется для текущей Flet-версии)
- macOS 11+ (Apple Silicon / Intel) или Windows 10/11
- Веб-камера (1280×720 или выше)
- 4GB RAM минимум (8GB рекомендуется)
- 500MB свободного места на диске

## Лицензия

MIT License (см. LICENSE)

## Автор

Разработано в рамках дипломного проекта ГУАП (2025)

## Связь

- GitHub Issues: [проблемы и предложения]
- Email: rafaildavar@gmail.com

---

**Примечание**: Проект находится в активной разработке. Старая версия доступна в теге `v0.6.0-old` для ознакомления с исходной реализацией.
