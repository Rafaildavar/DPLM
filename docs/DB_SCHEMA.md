# GestureBind database schema

Фактическая схема берется из `app.models.database.Base.metadata`.

## Таблицы

### users

| Поле | Тип | Ограничения |
| --- | --- | --- |
| id | INTEGER | PK |
| username | VARCHAR(64) | NOT NULL, UNIQUE |
| display_name | VARCHAR(255) | NULL |
| created_at | DATETIME | NOT NULL |

### gestures

| Поле | Тип | Ограничения |
| --- | --- | --- |
| id | INTEGER | PK |
| label | VARCHAR(255) | NOT NULL, UNIQUE |
| description | TEXT | NULL |
| samples_path | TEXT | NULL |
| model_class_id | INTEGER | NULL |
| accuracy | FLOAT | NULL |
| is_two_hands | BOOLEAN | NOT NULL |
| is_active | BOOLEAN | NOT NULL |
| user_id | INTEGER | FK -> users.id, NULL |
| created_at | DATETIME | NOT NULL |
| updated_at | DATETIME | NOT NULL |

### gesture_samples

| Поле | Тип | Ограничения |
| --- | --- | --- |
| id | INTEGER | PK |
| gesture_id | INTEGER | FK -> gestures.id, NOT NULL, ON DELETE CASCADE |
| sample_index | INTEGER | NOT NULL |
| features_path | TEXT | NULL |
| features_json | TEXT | NULL |
| frames | INTEGER | NULL |
| hand_count | INTEGER | NULL |
| source | VARCHAR(32) | NOT NULL |
| created_at | DATETIME | NOT NULL |

Unique: `(gesture_id, sample_index)`.

### recognition_models

| Поле | Тип | Ограничения |
| --- | --- | --- |
| id | INTEGER | PK |
| name | VARCHAR(128) | NOT NULL, UNIQUE |
| algorithm | VARCHAR(32) | NOT NULL |
| model_path | TEXT | NULL |
| classes_path | TEXT | NULL |
| feature_dim | INTEGER | NULL |
| n_classes | INTEGER | NULL |
| n_samples | INTEGER | NULL |
| accuracy | FLOAT | NULL |
| hyperparams_json | TEXT | NULL |
| is_active | BOOLEAN | NOT NULL |
| created_at | DATETIME | NOT NULL |

### commands

| Поле | Тип | Ограничения |
| --- | --- | --- |
| id | INTEGER | PK |
| name | VARCHAR(255) | NOT NULL, UNIQUE |
| description | TEXT | NULL |
| platform | VARCHAR(50) | NOT NULL |
| script_path | TEXT | NULL |
| action_spec | TEXT | NULL |
| gesture_id | INTEGER | FK -> gestures.id, NULL |
| is_active | BOOLEAN | NOT NULL |
| created_at | DATETIME | NOT NULL |

### settings

| Поле | Тип | Ограничения |
| --- | --- | --- |
| key | VARCHAR(255) | PK |
| value | TEXT | NULL |

### gesture_history

| Поле | Тип | Ограничения |
| --- | --- | --- |
| id | INTEGER | PK |
| gesture_id | INTEGER | FK -> gestures.id, NOT NULL |
| command_id | INTEGER | FK -> commands.id, NOT NULL |
| executed_at | DATETIME | NOT NULL |

### recognition_logs

| Поле | Тип | Ограничения |
| --- | --- | --- |
| id | INTEGER | PK |
| label | VARCHAR(255) | NOT NULL |
| confidence | FLOAT | NULL |
| model_id | INTEGER | FK -> recognition_models.id, NULL |
| gesture_id | INTEGER | FK -> gestures.id, NULL |
| session_id | INTEGER | FK -> app_sessions.id, NULL |
| executed | BOOLEAN | NOT NULL |
| detected_at | DATETIME | NOT NULL |

### app_sessions

| Поле | Тип | Ограничения |
| --- | --- | --- |
| id | INTEGER | PK |
| user_id | INTEGER | FK -> users.id, NULL |
| app_version | VARCHAR(32) | NULL |
| platform | VARCHAR(32) | NULL |
| started_at | DATETIME | NOT NULL |
| ended_at | DATETIME | NULL |
| notes | TEXT | NULL |

## Основные связи

- `users.id` -> `gestures.user_id`, `app_sessions.user_id`
- `gestures.id` -> `gesture_samples.gesture_id`, `commands.gesture_id`, `gesture_history.gesture_id`, `recognition_logs.gesture_id`
- `commands.id` -> `gesture_history.command_id`
- `recognition_models.id` -> `recognition_logs.model_id`
- `app_sessions.id` -> `recognition_logs.session_id`
