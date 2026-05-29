# JMLC Stage 1 - Baseline Audit

Generated: `2026-05-29T16:12:12+00:00`

## Executive summary

- Dataset status: **medium** - есть база для первого baseline, но нужно явно обработать пустые классы, дисбаланс и разные размерности признаков.
- Test status: **recorded** - 75 passed, 1 skipped; full pytest fails only on coverage gate: total 26.40 < required 90.
- JMLC visibility: проект уже демонстрирует инженерную базу, но для победного трека нужны воспроизводимые ML-эксперименты и отчеты в `docs/experiments/`.

## Dataset snapshot

| Metric | Value |
|---|---:|
| Class directories | 13 |
| Active classes | 4 |
| Empty classes | 9 |
| Samples | 67 |
| Valid samples | 67 |
| Invalid samples | 0 |
| Active sample min | 4 |
| Active sample max | 28 |
| Imbalance ratio | 7.0 |
| Frame min | 30 |
| Frame max | 153 |
| Frame mean | 61.85 |
| Feature dimensions | 42, 84 |

## Classes

| Class | Samples | Valid | Frame min | Frame max | Frame mean | Feature dims | Issues |
|---|---:|---:|---:|---:|---:|---|---|
| `New` | 20 | 20 | 30 | 30 | 30.0 | 42 | - |
| `Newest` | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `Open` | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `hello` | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `new2` | 4 | 4 | 30 | 30 | 30.0 | 42 | - |
| `palm` | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `peace_sign` | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `swipe_right` | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `test_gesture` | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `thumbs_up` | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `zoom` | 15 | 15 | 34 | 62 | 43.33 | 84 | - |
| `zoom_one` | 28 | 28 | 64 | 153 | 99.07 | 42 | - |
| `ываы` | 0 | 0 | n/a | n/a | n/a | n/a | - |

## Engineering snapshot

| Artifact | Present |
|---|---|
| `JMLC.md` | yes |
| `docker-compose.yml` | yes |
| `pytest.ini` | yes |
| `alembic` | yes |
| `models/classes.json` | yes |
| `models/feature_dim.txt` | yes |
| `docs/jmlc_micro_presentation.html` | yes |

## Test snapshot

| Metric | Value |
|---|---:|
| Test files | 17 |
| Test functions | 87 |
| `pytest.ini` | yes |
| Coverage gate | 90 |

Pytest outcome: `75 passed, 1 skipped; full pytest fails only on coverage gate: total 26.40 < required 90`

Тесты проходят функционально, но quality gate по покрытию сейчас не закрыт.

## JMLC assessment

Что выглядит хорошо для комиссии:

1. Есть реальный датасет пользовательских жестов, а не только демо-код.
2. Есть unit-тесты и строгий coverage gate, даже если он пока не закрыт.
3. Есть инженерные артефакты: Docker для БД, Alembic, модели, документация, презентационный HTML.

Что пока выглядит рискованно:

1. Активных классов меньше, чем папок: пустые директории нельзя считать частью обучающего датасета.
2. Есть смешение размерностей 42/84, поэтому Stage 2 должен отдельно сравнивать one-hand и two-hand признаки или фильтровать совместимые классы.
3. Нужна таблица фактического сравнения `static_mean` vs `dynamic_stats` vs альтернативные модели.
4. Coverage gate 90% сейчас падает из-за низкого покрытия больших GUI/runtime модулей; это нужно либо закрывать тестами, либо честно объяснить и скорректировать стратегию покрытия для JMLC-ветки.

## Decision for Stage 2

Переходить к реализации `cv/gesture_features.py` и `scripts/compare_models.py`.
Минимальная цель Stage 2: получить воспроизводимую таблицу baseline-метрик и confusion matrix для текущего датасета.
