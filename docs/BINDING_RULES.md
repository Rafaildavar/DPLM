# Правила привязки жестов к командам (BindingRules)

Документ фиксирует **контракт** между UI привязки (`GestureCommandBindScreen.qml`),
сервисом исполнения (`gesture_command_bridge.py`), исполнителем
(`command_executor.py`) и слоем БД (`commands`, `gestures`).

Цель: сделать привязку «жест → команда» предсказуемой, безопасной и
расширяемой без правки Python-кода (всё описывается строкой
`commands.action_spec` в БД).

> Платформа: **только macOS**. Поле `commands.platform` сохранено для
> совместимости со схемой БД, но в v1 всегда нормализуется в `macos`.

## 1. Категории привязок

Все привязки делятся на **шесть категорий**. Каждая категория = отдельная
вкладка в форме редактирования команды и отдельная иконка в
`CommandsPanel.qml`. В `action_spec` категория **выводится** из значения
поля `action`, поэтому хранить её отдельно не обязательно (см. раздел 4).

| # | Категория | Описание | `action` | Поля параметров |
|---|-----------|----------|----------|-----------------|
| 1 | **Запуск приложения** | Открыть приложение или URL | `open_app`, `open_url` | `app: str` или `url: str` |
| 2 | **Системное действие** | Громкость, яркость, блокировка экрана, скриншот, заглушение | `volume_up`, `volume_down`, `mute_toggle`, `brightness_up`, `brightness_down`, `lock_screen`, `screenshot` | — (для большинства) |
| 3 | **Прокрутка / навигация** | Колесо мыши и клавиши навигации | `scroll`, `press` | `clicks: int` (для `scroll`) или `key: str` (для `press`) |
| 4 | **Мультимедиа** | Управление плеером | `media_key` | `kind: "play_pause" \| "next" \| "prev"` |
| 5 | **Горячая клавиша** | Произвольная комбинация клавиш | `key_combination` | `keys: list[str]` |
| 6 | **Пользовательский скрипт** | Запуск Python-файла | `run_script` | `script_path: str`, `args?: list[str]` |

### 1.1. Категория 1 — Запуск приложения

```json
{"action": "open_app", "app": "Safari"}
{"action": "open_url", "url": "https://github.com"}
```

Семантика: вызов `open -a <app>` / `open <url>` через `subprocess.Popen`.

### 1.2. Категория 2 — Системное действие

```json
{"action": "volume_up"}
{"action": "volume_down"}
{"action": "mute_toggle"}
{"action": "brightness_up"}
{"action": "brightness_down"}
{"action": "lock_screen"}
{"action": "screenshot"}
```

Реализация на macOS:
| Действие | Под капотом |
|----------|-------------|
| `volume_up` / `volume_down` | `pyautogui.press("volumeup"/"volumedown")` |
| `mute_toggle` | `pyautogui.press("volumemute")` |
| `brightness_up` / `brightness_down` | `osascript -e 'tell application "System Events" to key code 144/145'` (F2/F1 без модификаторов) |
| `lock_screen` | `pmset displaysleepnow` |
| `screenshot` | `screencapture -i ~/Desktop/screenshot_<ts>.png` |

### 1.3. Категория 3 — Прокрутка / навигация

```json
{"action": "scroll", "clicks": -5}
{"action": "press", "key": "pagedown"}
```

`clicks > 0` — прокрутка вверх, `< 0` — вниз. `key` — любое имя из
PyAutoGUI: `up`, `down`, `pageup`, `pagedown`, `home`, `end`, `tab` и т.д.

### 1.4. Категория 4 — Мультимедиа

```json
{"action": "media_key", "kind": "play_pause"}
{"action": "media_key", "kind": "next"}
{"action": "media_key", "kind": "prev"}
```

Под капотом — PyAutoGUI: `playpause`, `nexttrack`, `prevtrack`.

### 1.5. Категория 5 — Горячая клавиша

```json
{"action": "key_combination", "keys": ["command", "space"]}
```

Любая комбинация PyAutoGUI: `command`, `option`, `control`, `shift`, `f1`–`f12`,
буквы и цифры. Полезно: `Cmd+C/V/X/Z`, `Cmd+Tab`, `Cmd+Space` (Spotlight),
`Cmd+Shift+3` (скриншот всего экрана).

### 1.6. Категория 6 — Пользовательский скрипт

```json
{"action": "run_script", "script_path": "/abs/path/tool.py", "args": ["--flag"]}
```

Только абсолютный путь к существующему `.py`. Запуск тем же интерпретатором,
которым работает приложение.

## 2. Правила привязки

Это инварианты, которые контролирует UI/сервис **до** сохранения
привязки и **во время** исполнения.

### R1. 1:1 жест ↔ команда

В схеме БД у `commands.gesture_id` **один** жест на одну команду
(см. ERD в `docs/DB_REPORT.md`). UI обязан:

- проверять, нет ли уже команды на выбранный жест;
- если есть — переспрашивать «Перезаписать привязку *X* → *Y*?» (не
  перезаписывать молча);
- при подтверждении — вызвать `bindGestureToCommand(gesture_label, command_name)`,
  который атомарно очищает старую и ставит новую привязку.

### R3. Жест должен быть активен и обучен

В выпадающем списке жестов на экране привязки показываются **только**
жесты, у которых:

- `gestures.is_active = TRUE`;
- `gestures.model_class_id IS NOT NULL` (то есть жест участвует в
  активной модели и реально может быть распознан).

Это исключает ситуацию «привязал, но никогда не сработает».

### R4. Порог уверенности (confidence threshold)

Команда исполняется только если уверенность распознавателя
≥ `binding.confidence_threshold`. Значение по умолчанию `0.65`,
хранится в `settings(key='binding.confidence_threshold')`.

При срабатывании ниже порога:
- запись в `recognition_logs` с `executed=FALSE`;
- команда **не** запускается.

### R5. Cooldown / антидребезг

Один и тот же жест не может выстреливать чаще, чем раз в
`binding.cooldown_ms` миллисекунд (по умолчанию `1500`). Хранится в
`settings(key='binding.cooldown_ms')`.

Реализация — таймер последнего срабатывания на каждую метку жеста
внутри `BindingPolicy` (in-memory, обновляется в `execute_for_gesture_label`).

### R6. Двуручные жесты для «опасных» команд (рекомендация)

Команды с `action ∈ {lock_screen, run_script}` **рекомендуется** привязывать
к жестам с `gestures.is_two_hands = TRUE`. UI показывает предупреждение
вида «Эту команду рекомендуем привязать к двуручному жесту, чтобы
избежать случайного срабатывания», но не блокирует сохранение —
пользователь сам принимает решение.

Список «опасных» action'ов задаётся в `gesture_command_bridge.DANGEROUS_ACTIONS`.

### R7. Валидация `script_path`

Для `action = "run_script"` UI обязан проверять перед сохранением:

- путь начинается с `/` (абсолютный);
- файл существует;
- расширение `.py`;
- размер файла > 0.

Не давать пользователю произвольный `shell:` через UI: префикс `shell:`
из `script_path` остаётся только для админ-режима / тестов.

### R8. Уникальность имени команды

`commands.name` имеет UNIQUE-ограничение в БД (см. `docs/DB_REPORT.md`).
UI должен:

- триммить введённое имя;
- запрещать пустое имя;
- проверять уникальность **до** отправки запроса;
- показывать ошибку «Команда с таким именем уже существует» прямо в форме.

## 3. Поток исполнения

```
gestureDetected(label, confidence)
        │
        ▼
BindingPolicy.should_execute(label, confidence)
        │   ├─ confidence < threshold → False (R4)
        │   └─ now - last_fired[label] < cooldown_ms → False (R5)
        ▼
resolve_command_for_gesture(session, label)
        │   └─ нет привязки → False
        ▼
execute_command_row(command)
        │   ├─ action_spec → CommandExecutor.execute_config()
        │   ├─ script_path → _run_script_path()
        │   └─ name        → CommandExecutor.execute()
        ▼
recognition_logs INSERT (executed=True/False)
gesture_history INSERT (если успешно)
```

## 4. Категория из `action`

UI определяет категорию по полю `action` без отдельного хранения:

```python
ACTION_TO_CATEGORY = {
    "open_app":         "launch",
    "open_url":         "launch",
    "volume_up":        "system",
    "volume_down":      "system",
    "mute_toggle":      "system",
    "brightness_up":    "system",
    "brightness_down":  "system",
    "lock_screen":      "system",
    "screenshot":       "system",
    "scroll":           "navigation",
    "press":            "navigation",
    "media_key":        "media",
    "key_combination":  "hotkey",
    "run_script":       "script",
}
```

## 5. Связанные настройки в `settings`

| Ключ | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `binding.confidence_threshold` | float (str в БД) | `"0.65"` | R4 |
| `binding.cooldown_ms` | int (str в БД) | `"1500"` | R5 |
| `binding.warn_two_hands` | bool (`"true"`/`"false"`) | `"true"` | R6 — показывать предупреждение |

Чтение/запись через `app/services/binding_settings.py` (см. ниже).

## 6. Как добавить новую команду из UI

1. Открыть `GestureCommandBindScreen.qml` → «+ Создать команду».
2. Ввести имя (R8 проверяется на лету), выбрать категорию.
3. Заполнить параметры по схеме (`ACTION_SPEC_SCHEMA`).
4. Выбрать жест из выпадающего списка (R3 фильтрует только активные обученные).
5. Если категория = «Системное» с опасным действием и жест одноручный — UI
   показывает предупреждение R6, но позволяет продолжить.
6. Нажать «Сохранить» → `appController.bindGestureToCommand(...)`:
   - валидация (R1, R7, R8);
   - запись в БД;
   - вызов `sync_db_commands_to_executor()` для регистрации в реестре.

## 7. Тестирование

Покрытие правил юнит-тестами:

| Правило | Тест |
|---------|------|
| R1 | `tests/unit/test_binding_rules.py::test_r1_overwrite_existing_binding` |
| R3 | `tests/unit/test_binding_rules.py::test_r3_filter_inactive_gestures` |
| R4 | `tests/unit/test_binding_policy.py::test_r4_confidence_threshold` |
| R5 | `tests/unit/test_binding_policy.py::test_r5_cooldown` |
| R6 | `tests/unit/test_binding_rules.py::test_r6_dangerous_action_warning` |
| R7 | `tests/unit/test_binding_rules.py::test_r7_script_path_validation` |
| R8 | `tests/unit/test_binding_rules.py::test_r8_unique_name` |

Все тесты — без реального выполнения системных действий: `pyautogui`,
`subprocess.Popen` и `osascript` мокируются.
