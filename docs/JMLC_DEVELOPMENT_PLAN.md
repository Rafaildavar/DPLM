# JMLC Development Plan

План ведется под третью волну Junior ML Contest 2026:

- подача проекта: 3-20 июля;
- запись на защиту: 21-22 июля;
- питчинг: 27-29 июля;
- результаты: 31 июля.

## Принцип разработки

Каждая задача должна усиливать один из двух слоев:

1. **ML-доказательства**: данные, признаки, модели, метрики, валидация.
2. **Полноценная система**: GUI, команды, БД, тесты, CI, воспроизводимость.

Если задача не усиливает ни один слой, она откладывается.

## Неделя 1: стабилизация основы и перенос JMLC-ядра

Цель: текущая Flet-версия становится базой, а JMLC-исследования возвращаются в
репозиторий в новом виде.

P0:

- добавить свежий `docs/contest/JMLC.md` и этот plan-документ;
- восстановить/обновить `cv/gesture_features.py`;
- восстановить/обновить `scripts/compare_models.py`;
- создать `docs/experiments/README.md`;
- сделать `dataset_profile` для текущего `data/gestures`;
- разделить тестовый контур на core/Flet/legacy;
- зафиксировать один зеленый smoke-сценарий без камеры.

Acceptance:

- `python -m pytest tests/unit/test_*gesture* tests/unit/test_*command*` проходит;
- `python -m scripts.jmlc_dataset_profile` создает JSON и Markdown;
- `python -m scripts.compare_models` создает model comparison отчет;
- в `docs/experiments/` есть первые воспроизводимые артефакты.

## Неделя 2: датасет и model comparison

Цель: получить честный, сбалансированный benchmark под текущую модельную задачу.

P0:

- выбрать 5-6 жестов для конкурсного сценария;
- собрать/почистить минимум 25-40 семплов на класс;
- исключить пустые классы и несовместимые размерности из основного benchmark;
- сравнить `static_mean` и `dynamic_stats`;
- сравнить KNN, SVM, RandomForest/ExtraTrees;
- сохранить confusion matrix и per-class метрики;
- подготовить threshold report.

Acceptance:

- macro F1 основного кандидата >= 0.90 или есть честный разбор причин;
- у каждого класса есть precision/recall/F1;
- отчет показывает не только лучший score, но и ошибки модели;
- latency измерена отдельно от качества.

## Неделя 3: live validation и engineering

Цель: доказать, что offline-модель работает в живой системе.

P0:

- связать recommended threshold с настройками приложения;
- провести live-eval 10-15 минут;
- измерить false triggers, rejected predictions, command success rate;
- добавить smoke CLI/Makefile-команды;
- настроить GitHub Actions CI;
- вычистить README под текущую Flet-версию;
- убрать устаревшие QML-обещания из конкурсной документации.

Acceptance:

- есть `docs/experiments/live_eval.md`;
- CI запускает тесты и ML smoke;
- приложение запускается по инструкции из README;
- demo-flow работает без ручных правок БД.

## Неделя 4: упаковка и защита

Цель: проект готов к загрузке и защите.

P0:

- написать описание проекта до 3 страниц;
- подготовить презентацию на 5 минут;
- записать резервное demo-видео;
- создать `docs/ai_usage.md`;
- создать `docs/product_research.md`;
- собрать 3-5 пользовательских отзывов;
- подготовить Q&A по данным, моделям, leakage, live-ошибкам, MLOps и продукту.

Acceptance:

- полный прогон из свежего clone/venv воспроизводим;
- все P0-документы есть в репозитории;
- презентация укладывается в 5 минут;
- есть короткий список честных ограничений и next steps.

## Ближайший backlog

### P0. Вернуть ML-экспериментальный контур

- `cv/gesture_features.py`: единый API для `static_mean` и `dynamic_stats`.
- `scripts/jmlc_dataset_profile.py`: паспорт датасета.
- `scripts/compare_models.py`: benchmark моделей.
- `tests/unit/test_gesture_features.py`: тесты признаков.
- `tests/unit/test_compare_models.py`: тесты benchmark-логики.

### P0. Стабилизировать тесты

- legacy QML-тесты вынести в отдельную группу;
- заменить запись runtime-логов в `$HOME` на временные директории в тестах;
- оставить общий coverage-gate только для осмысленного core scope или перенести
  строгий gate на ML/services.

### P0. Пересобрать данные

- проверить текущие классы в `data/gestures`;
- выбрать конкурсные классы;
- удалить/исключить пустые и тестовые папки из benchmark;
- привести labels к единому стилю;
- переобучить модель и обновить `models/classes.json`, `models/knn.pkl`.

### P1. Инженерная упаковка

- Makefile: `test`, `profile-data`, `compare-models`, `benchmark-latency`,
  `seed-db`, `run-app`;
- CI workflow;
- model card;
- data card;
- demo guide.

### P1. Продуктовые доказательства

- competitor table;
- user feedback;
- time-to-first-command;
- false triggers / 10 min;
- command success rate.

## Definition of Done для JMLC-задач

Задача считается готовой, если:

- есть код или документ в репозитории;
- есть команда воспроизведения;
- результат не требует ручной догадки;
- для ML-результатов сохранены JSON/Markdown артефакты;
- ограничения явно описаны;
- задача привязана к критерию JMLC.
