# Отчёт по базе данных системы распознавания жестов GestureBind

## 1. Назначение базы данных

База данных в дипломном проекте GestureBind выполняет роль **прикладного слоя
устойчивого хранения пользовательских данных** системы распознавания
жестов. Это **не «тяжёлый» центральный модуль**, а вспомогательное
хранилище, которое позволяет:

- сохранять пользовательский **словарь жестов** и их названия;
- сохранять **обучающие примеры** (метаданные сэмплов) для каждого жеста;
- хранить **привязки жестов к командам** системы;
- хранить **сведения об обученных моделях** распознавания;
- вести **журнал распознаваний** и **историю запусков** приложения;
- хранить **глобальные настройки** и **профили пользователей**.

Благодаря этому приложение становится персонализируемым: собранные
жесты, образцы и история действий не теряются после закрытия программы.

## 2. Технологический стек

| Компонент | Реализация |
|-----------|-----------|
| СУБД | **PostgreSQL 16** (через `psycopg2`) |
| Резервный режим | SQLite (для тестов и быстрой разработки) |
| ORM | **SQLAlchemy 2.x** (`declarative_base`) |
| Миграции | **Alembic** (`alembic upgrade head`) |
| Конфигурация | переменные окружения (`DATABASE_URL` или `DPLM_DB_*`), `.env` |
| Развёртывание | `docker-compose.yml` или `scripts/init_postgres.sh` |

PostgreSQL выбран как промышленный, кроссплатформенный, бесплатный
SQL-сервер с богатой поддержкой типов и хорошей интеграцией с Python
через `psycopg2`. SQLite оставлен как резерв для случаев, когда
запускать сервер БД нецелесообразно (юнит-тесты, демо в одну кнопку).

## 3. Структура хранения

База данных приложения состоит из **девяти** таблиц, разбитых по трём
логическим группам.

### 3.1. Группа «Словарь и обучение»

| Таблица | Назначение |
|---------|------------|
| `users` | пользователи системы (для персонализации словарей) |
| `gestures` | словарь жестов пользователя |
| `gesture_samples` | обучающие примеры (метаданные сэмплов) |
| `recognition_models` | обученные модели распознавания (KNN/MLP/...) |

### 3.2. Группа «Привязка жестов к действиям»

| Таблица | Назначение |
|---------|------------|
| `commands` | действия системы и их параметры |
| `gesture_history` | факты выполнения пары «жест → команда» |

### 3.3. Группа «Журналы и настройки»

| Таблица | Назначение |
|---------|------------|
| `recognition_logs` | сырые события распознавания (с уверенностью) |
| `app_sessions` | история запусков приложения |
| `settings` | глобальные настройки приложения (key-value) |

## 4. Концептуальная схема (ERD)

```mermaid
erDiagram
    USERS ||--o{ GESTURES : owns
    USERS ||--o{ APP_SESSIONS : has
    GESTURES ||--o{ GESTURE_SAMPLES : has
    GESTURES ||--o| COMMANDS : "bound to (1:1)"
    GESTURES ||--o{ GESTURE_HISTORY : triggered
    COMMANDS ||--o{ GESTURE_HISTORY : executed
    GESTURES ||--o{ RECOGNITION_LOGS : detected
    RECOGNITION_MODELS ||--o{ RECOGNITION_LOGS : produced
    APP_SESSIONS ||--o{ RECOGNITION_LOGS : during

    USERS {
      int id PK
      string username UK
      string display_name
      datetime created_at
    }
    GESTURES {
      int id PK
      string label UK
      text description
      text samples_path
      int model_class_id
      float accuracy
      bool is_two_hands
      bool is_active
      int user_id FK
      datetime created_at
      datetime updated_at
    }
    GESTURE_SAMPLES {
      int id PK
      int gesture_id FK
      int sample_index
      text features_path
      text features_json
      int frames
      int hand_count
      string source
      datetime created_at
    }
    RECOGNITION_MODELS {
      int id PK
      string name UK
      string algorithm
      text model_path
      text classes_path
      int feature_dim
      int n_classes
      int n_samples
      float accuracy
      text hyperparams_json
      bool is_active
      datetime created_at
    }
    COMMANDS {
      int id PK
      string name UK
      text description
      string platform
      text script_path
      text action_spec
      int gesture_id FK
      bool is_active
      datetime created_at
    }
    GESTURE_HISTORY {
      int id PK
      int gesture_id FK
      int command_id FK
      datetime executed_at
    }
    RECOGNITION_LOGS {
      int id PK
      string label
      float confidence
      int model_id FK
      int gesture_id FK
      int session_id FK
      bool executed
      datetime detected_at
    }
    APP_SESSIONS {
      int id PK
      int user_id FK
      string app_version
      string platform
      datetime started_at
      datetime ended_at
      text notes
    }
    SETTINGS {
      string key PK
      text value
    }
```

## 5. Подробное описание таблиц

### 5.1. `users` — пользователи системы

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `id` | INTEGER | PK, autoincrement | суррогатный ключ |
| `username` | VARCHAR(64) | NOT NULL, UNIQUE | логин (без зависимости от ОС) |
| `display_name` | VARCHAR(255) | NULL | отображаемое имя |
| `created_at` | TIMESTAMP | NOT NULL | дата создания записи |

Используется для персонализации словарей и истории. В одно­пользовательском
сценарии создаётся служебная запись `default`.

### 5.2. `gestures` — словарь жестов

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `id` | INTEGER | PK, autoincrement | суррогатный ключ |
| `label` | VARCHAR(255) | NOT NULL, UNIQUE | имя класса (используется в модели) |
| `description` | TEXT | NULL | человекочитаемое описание |
| `samples_path` | TEXT | NULL | путь к каталогу с сырыми примерами |
| `model_class_id` | INTEGER | NULL | индекс класса в `models/classes.json` |
| `accuracy` | FLOAT | NULL | оценка точности на валидации |
| `is_two_hands` | BOOLEAN | NOT NULL, DEFAULT FALSE | признак «две руки» |
| `is_active` | BOOLEAN | NOT NULL, DEFAULT TRUE | используется ли при инференсе |
| `user_id` | INTEGER | FK → users.id | владелец |
| `created_at` | TIMESTAMP | NOT NULL | дата добавления |
| `updated_at` | TIMESTAMP | NOT NULL | дата последней правки |

### 5.3. `gesture_samples` — обучающие примеры

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `id` | INTEGER | PK, autoincrement | суррогатный ключ |
| `gesture_id` | INTEGER | NOT NULL, FK → gestures.id, ON DELETE CASCADE | ссылка на жест |
| `sample_index` | INTEGER | NOT NULL | порядковый номер примера в рамках жеста |
| `features_path` | TEXT | NULL | путь к `.npy`/`.json` с признаками |
| `features_json` | TEXT | NULL | сериализованный вектор признаков (если хранится в БД) |
| `frames` | INTEGER | NULL | количество кадров |
| `hand_count` | INTEGER | NULL | обнаружено рук (1 или 2) |
| `source` | VARCHAR(32) | NOT NULL, DEFAULT 'camera' | источник примера |
| `created_at` | TIMESTAMP | NOT NULL | дата записи |

Уникальный составной ключ `(gesture_id, sample_index)` исключает дубли.
Большие массивы признаков (NumPy) допускается хранить на диске —
в БД лежит только относительный путь и метаданные.

### 5.4. `recognition_models` — обученные модели

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `id` | INTEGER | PK, autoincrement | суррогатный ключ |
| `name` | VARCHAR(128) | NOT NULL, UNIQUE | имя версии (`knn-2026-04-19`) |
| `algorithm` | VARCHAR(32) | NOT NULL, DEFAULT 'knn' | алгоритм |
| `model_path` | TEXT | NULL | путь к сериализованной модели (`models/knn.pkl`) |
| `classes_path` | TEXT | NULL | путь к `classes.json` |
| `feature_dim` | INTEGER | NULL | размерность вектора признаков |
| `n_classes` | INTEGER | NULL | количество классов |
| `n_samples` | INTEGER | NULL | общее количество обучающих примеров |
| `accuracy` | FLOAT | NULL | точность на валидации |
| `hyperparams_json` | TEXT | NULL | гиперпараметры в JSON |
| `is_active` | BOOLEAN | NOT NULL, DEFAULT FALSE | «текущая модель» |
| `created_at` | TIMESTAMP | NOT NULL | дата обучения |

При каждом цикле дообучения создаётся новая запись, активная — одна.

### 5.5. `commands` — действия системы

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `id` | INTEGER | PK, autoincrement | суррогатный ключ |
| `name` | VARCHAR(255) | NOT NULL, UNIQUE | название команды |
| `description` | TEXT | NULL | подсказка для UI |
| `platform` | VARCHAR(50) | NOT NULL, DEFAULT 'all' | платформа (`macos`/`windows`/`all`) |
| `script_path` | TEXT | NULL | путь/спецификация исполняемого ресурса |
| `action_spec` | TEXT | NULL | JSON-описание действия из UI |
| `gesture_id` | INTEGER | FK → gestures.id | привязка к жесту (1:1) |
| `is_active` | BOOLEAN | NOT NULL, DEFAULT TRUE | включена ли команда |
| `created_at` | TIMESTAMP | NOT NULL | дата создания |

Поле `action_spec` имеет приоритет перед `script_path`. Схема значений
описана в `app/services/user_command_sync.py::ACTION_SPEC_SCHEMA`.

### 5.6. `gesture_history` — история выполнений

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `id` | INTEGER | PK, autoincrement | суррогатный ключ |
| `gesture_id` | INTEGER | NOT NULL, FK → gestures.id | жест |
| `command_id` | INTEGER | NOT NULL, FK → commands.id | команда |
| `executed_at` | TIMESTAMP | NOT NULL | момент выполнения |

### 5.7. `recognition_logs` — журнал распознавания

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `id` | INTEGER | PK, autoincrement | суррогатный ключ |
| `label` | VARCHAR(255) | NOT NULL, INDEX | метка от классификатора |
| `confidence` | FLOAT | NULL | уверенность (0.0–1.0) |
| `model_id` | INTEGER | FK → recognition_models.id | какая модель распознала |
| `gesture_id` | INTEGER | FK → gestures.id | если метка нашлась в словаре |
| `session_id` | INTEGER | FK → app_sessions.id | в какой сессии произошло |
| `executed` | BOOLEAN | NOT NULL, DEFAULT FALSE | была ли запущена команда |
| `detected_at` | TIMESTAMP | NOT NULL, INDEX | момент распознавания |

Хранит «сырые» события — в том числе те, что были отброшены по порогу
уверенности или у которых нет привязки к команде. Используется для
аналитики качества и подсчёта статистики.

### 5.8. `app_sessions` — история запусков

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `id` | INTEGER | PK, autoincrement | суррогатный ключ |
| `user_id` | INTEGER | FK → users.id | текущий пользователь |
| `app_version` | VARCHAR(32) | NULL | версия приложения |
| `platform` | VARCHAR(32) | NULL | ОС (`macos`/`windows`/`linux`) |
| `started_at` | TIMESTAMP | NOT NULL | момент запуска |
| `ended_at` | TIMESTAMP | NULL | момент завершения |
| `notes` | TEXT | NULL | пометки/последняя ошибка |

### 5.9. `settings` — глобальные настройки

| Поле | Тип | Огр. | Описание |
|------|-----|------|----------|
| `key` | VARCHAR(255) | PK | имя настройки |
| `value` | TEXT | NULL | значение в виде строки/JSON |

## 6. Связи и целостность

| Связь | Кратность | Поведение при удалении родителя |
|-------|-----------|---------------------------------|
| `users.id` ← `gestures.user_id` | 1 : N | RESTRICT (по умолчанию) |
| `users.id` ← `app_sessions.user_id` | 1 : N | RESTRICT |
| `gestures.id` ← `gesture_samples.gesture_id` | 1 : N | **CASCADE** |
| `gestures.id` ← `commands.gesture_id` | 1 : 1 | RESTRICT |
| `gestures.id` ← `gesture_history.gesture_id` | 1 : N | RESTRICT |
| `commands.id` ← `gesture_history.command_id` | 1 : N | RESTRICT |
| `recognition_models.id` ← `recognition_logs.model_id` | 1 : N | RESTRICT |
| `gestures.id` ← `recognition_logs.gesture_id` | 1 : N | RESTRICT |
| `app_sessions.id` ← `recognition_logs.session_id` | 1 : N | RESTRICT |

Целостность гарантирована внешними ключами и уникальными ограничениями
(`UNIQUE (gesture_id, sample_index)`, `UNIQUE (label)`,
`UNIQUE (commands.name)` и т.п.). Для PostgreSQL внешние ключи проверяются
автоматически; для резервного SQLite модуль БД при подключении включает
`PRAGMA foreign_keys=ON`.

## 7. Конфигурация подключения

URL подключения собирается функцией
`app.models.database.resolve_database_url()` по приоритетам:

1. явно переданный аргумент `url`;
2. `DATABASE_URL` (полный SQLAlchemy URL);
3. `DPLM_DB_BACKEND=sqlite` — резерв на SQLite;
4. иначе — `postgresql+psycopg2://${DPLM_DB_USER}:${DPLM_DB_PASSWORD}@${DPLM_DB_HOST}:${DPLM_DB_PORT}/${DPLM_DB_NAME}`.

Значения по умолчанию: `dplm:dplm@localhost:5432/dplm`. Файл `.env`
(см. `.env.example`) подгружается автоматически без зависимости от
`python-dotenv`.

## 8. Миграции (Alembic)

Скрипты миграций лежат в `alembic/versions/`. Текущая цепочка:

```
4f809d88a774  →  b2c4e6d8a0f1  →  c3a1f0d24b9e  (head)
```

| Ревизия | Что добавляет |
|---------|---------------|
| `4f809d88a774` | стартовая схема: `commands`, `gestures`, `settings`, `gesture_history` |
| `b2c4e6d8a0f1` | поле `commands.action_spec` (JSON-действия из UI) |
| `c3a1f0d24b9e` | прикладной слой пользовательских данных: `users`, `gesture_samples`, `recognition_models`, `app_sessions`, `recognition_logs` + расширение `gestures` |

URL подключения для Alembic переопределяется в `alembic/env.py` той же
функцией `resolve_database_url()`, поэтому в `alembic.ini` ничего менять
не требуется.

Команды:

```bash
alembic upgrade head        # применить все миграции
alembic downgrade -1        # откатить одну ревизию
alembic current             # текущая ревизия
```

## 9. Развёртывание

### Вариант A: PostgreSQL в Docker

```bash
docker compose up -d db
alembic upgrade head
python -m scripts.seed_database
```

См. `docker-compose.yml`. Данные PostgreSQL в Docker хранятся в именованном томе
`dplm_postgres_data` (не в `./.db_data/postgres/` — bind-mount на macOS давал повреждения кластера).
Порт публикации задаётся переменной `DPLM_DB_PORT` (по умолчанию `5432`,
а в `.env` проекта поднят на `5433`, чтобы не конфликтовать с системным
PostgreSQL, если он установлен).

### Вариант B: системный PostgreSQL

```bash
./scripts/init_postgres.sh   # создаёт роль/БД и применяет миграции
python -m scripts.seed_database
```

### Вариант C: SQLite-разработка

```bash
DPLM_DB_BACKEND=sqlite python -m scripts.seed_database
```

## 10. Использование из приложения

```python
from app.models.database import init_database, get_db_session, Gesture, Command

init_database()              # читает .env / переменные окружения
session = get_db_session()
try:
    gestures = (
        session.query(Gesture)
        .filter(Gesture.is_active.is_(True))
        .order_by(Gesture.label)
        .all()
    )
    for g in gestures:
        print(g.label, "→", g.command.name if g.command else "—")
finally:
    session.close()
```

Сервисный слой (`app/services/gesture_command_bridge.py`,
`app/services/user_command_sync.py`) уже использует те же модели —
поэтому добавление новых таблиц **не ломает** существующую логику
исполнения команд и жестов.

## 11. Типичные запросы

```sql
-- Топ-5 самых часто выполняемых команд за неделю
SELECT c.name, COUNT(*) AS uses
FROM gesture_history h
JOIN commands c ON c.id = h.command_id
WHERE h.executed_at >= NOW() - INTERVAL '7 days'
GROUP BY c.name
ORDER BY uses DESC
LIMIT 5;

-- Средняя уверенность распознавания текущей модели по жестам
SELECT g.label, AVG(l.confidence) AS avg_conf, COUNT(*) AS n
FROM recognition_logs l
JOIN recognition_models m ON m.id = l.model_id
LEFT JOIN gestures g ON g.id = l.gesture_id
WHERE m.is_active = TRUE
GROUP BY g.label
ORDER BY avg_conf;

-- Сколько обучающих примеров у каждого жеста
SELECT g.label, COUNT(s.id) AS samples
FROM gestures g
LEFT JOIN gesture_samples s ON s.gesture_id = g.id
GROUP BY g.label
ORDER BY samples DESC;
```

## 12. Тестирование и фактический прогон

В `tests/unit/test_database.py` 12 юнит-тестов проверяют импорт моделей,
создание схемы in-memory, CRUD по каждой таблице и связи между ними.
Все 12 проходят (плюс смежные тесты `gesture_command_bridge` и
`user_command_sync`):

```
21 passed in 2.19s
```

Развёртывание **проверено вживую** на PostgreSQL 16 в Docker
(`postgres:16-alpine`, порт хоста `5433`):

```
$ docker compose up -d db
$ alembic upgrade head
INFO  [alembic.runtime.migration] Will assume transactional DDL.

$ alembic current
c3a1f0d24b9e (head)

$ docker exec dplm-postgres psql -U dplm -d dplm -c "\dt"
              List of relations
 Schema |        Name        | Type  | Owner
--------+--------------------+-------+-------
 public | alembic_version    | table | dplm
 public | app_sessions       | table | dplm
 public | commands           | table | dplm
 public | gesture_history    | table | dplm
 public | gesture_samples    | table | dplm
 public | gestures           | table | dplm
 public | recognition_logs   | table | dplm
 public | recognition_models | table | dplm
 public | settings           | table | dplm
 public | users              | table | dplm

$ python -m scripts.seed_database
[OK] Database initialized: postgresql+psycopg2://dplm:***@localhost:5433/dplm
[+] Создан пользователь: <User(id=1, username='default')>
[+] Импортирован жест: hello, palm, peace_sign, swipe_right,
    test_gesture, thumbs_up, zoom, zoom_one
[+] Зарегистрирована модель: knn-…
[+] Добавлено типовых команд: 3
[OK] Seed завершён.
```

Итоговое содержимое БД после первого запуска:

| Таблица | Записей |
|---------|---------|
| `users` | 1 |
| `gestures` | 8 |
| `gesture_samples` | 50 |
| `commands` | 3 |
| `recognition_models` | 1 |
| `settings` | 4 |

## 13. Состав изменений

| Файл | Изменение |
|------|-----------|
| `app/models/database.py` | переписан под Postgres-first, +5 моделей, env-конфиг |
| `alembic/env.py` | URL берётся из `resolve_database_url()` |
| `alembic.ini` | дефолт переключён на PostgreSQL |
| `alembic/versions/c3a1f0d24b9e_personalized_storage_layer.py` | новая миграция |
| `scripts/init_postgres.sh` | развёртывание ролей/БД через `psql` |
| `scripts/seed_database.py` | импорт жестов с диска и заполнение примеров |
| `docker-compose.yml` | контейнер PostgreSQL 16 для разработки |
| `.env.example` | пример конфигурации подключения |
| `.gitignore` | добавлен `.env` |

## 14. Итог

Реализованная база данных полностью соответствует поставленному
назначению: это **прикладное хранилище** пользовательских данных
системы распознавания жестов. Она:

- хранит **словарь жестов** и их обучающие примеры;
- хранит **привязки жестов к действиям**;
- хранит **сведения об обученных моделях** и **журнал распознавания**;
- хранит **историю работы приложения** и **настройки**;
- работает на **PostgreSQL** (с резервом на SQLite для тестов),
  управляется **Alembic-миграциями**, поднимается одной командой
  `docker compose up -d db`.
