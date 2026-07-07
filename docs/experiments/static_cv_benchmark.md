# GestureBind Cross-Validation Model Comparison

## Краткий вывод

- Лучший общий результат: **`static_landmark_image` + `extra_trees`** с macro F1 `0.7389` и accuracy `0.8000`.
- Датасет: 260 семплов, 10 активных классов, target_dim 63.
- Рекомендация: Текущий датасет выглядит static-like по выбранному motion threshold. Пока рано доказывать отдельную dynamic-модель: нужно записать жесты с выраженным движением и повторить benchmark.

## Датасет

| Класс | Семплы |
|---|---:|
| `2finger` | 40 |
| `gun` | 20 |
| `hand` | 20 |
| `like` | 40 |
| `no_gesture_static` | 20 |
| `onefinger` | 40 |
| `partial_swipe` | 20 |
| `random_motion` | 20 |
| `return_motion` | 20 |
| `wrong_axis_motion` | 20 |

| Метрика | Значение |
|---|---:|
| Исходные размерности | 42, 44, 63 |
| Целевая размерность | 63 |
| CV folds | 5 |
| Motion threshold | 0.0150 |
| Augmented samples | included |

## Motion Profile

| Класс | Семплы | Median motion energy | Median displacement | Suggested type |
|---|---:|---:|---:|---|
| `2finger` | 40 | 0.0011 | 0.0021 | `static_like` |
| `gun` | 20 | 0.0045 | 0.0095 | `static_like` |
| `hand` | 20 | 0.0008 | 0.0015 | `static_like` |
| `like` | 40 | 0.0033 | 0.0094 | `static_like` |
| `no_gesture_static` | 20 | 0.0079 | 0.0158 | `static_like` |
| `onefinger` | 40 | 0.0014 | 0.0038 | `static_like` |
| `partial_swipe` | 20 | 0.0036 | 0.0036 | `static_like` |
| `random_motion` | 20 | 0.0036 | 0.0035 | `static_like` |
| `return_motion` | 20 | 0.0039 | 0.0035 | `static_like` |
| `wrong_axis_motion` | 20 | 0.0036 | 0.0312 | `static_like` |

## Результаты

| Признаки | Модель | Accuracy | Macro F1 | Static-like F1 | Dynamic-like F1 | Macro FPR | Latency ms/sample | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `static_landmark_image` | `extra_trees` | 0.8000 | 0.7389 | 0.7389 | n/a | 0.0217 | 0.096 | 0.9000 |
| `static_craft_full_stats` | `extra_trees` | 0.7769 | 0.7045 | 0.7045 | n/a | 0.0242 | 0.095 | 0.8000 |
| `static_landmark_image` | `static_stacking` | 0.7577 | 0.6861 | 0.6861 | n/a | 0.0263 | 0.568 | 0.4500 |
| `static_craft_full_stats` | `static_stacking` | 0.7577 | 0.6856 | 0.6856 | n/a | 0.0263 | 0.487 | 0.4500 |
| `static_landmark_image` | `static_landmark_cnn` | 0.5923 | 0.4599 | 0.4599 | n/a | 0.0462 | 5.683 | 0.5500 |

## Лучшая модель: per-class метрики

| Класс | Motion type | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| `2finger` | `static_like` | 40 | 1.0000 | 1.0000 | 1.0000 |
| `gun` | `static_like` | 20 | 1.0000 | 1.0000 | 1.0000 |
| `hand` | `static_like` | 20 | 1.0000 | 1.0000 | 1.0000 |
| `like` | `static_like` | 40 | 1.0000 | 1.0000 | 1.0000 |
| `no_gesture_static` | `static_like` | 20 | 1.0000 | 1.0000 | 1.0000 |
| `onefinger` | `static_like` | 40 | 1.0000 | 1.0000 | 1.0000 |
| `partial_swipe` | `static_like` | 20 | 0.4500 | 0.4500 | 0.4500 |
| `random_motion` | `static_like` | 20 | 0.3750 | 0.4500 | 0.4091 |
| `return_motion` | `static_like` | 20 | 0.2105 | 0.2000 | 0.2051 |
| `wrong_axis_motion` | `static_like` | 20 | 0.3529 | 0.3000 | 0.3243 |

## Матрица ошибок лучшей модели

| true \ pred | `2finger` | `gun` | `hand` | `like` | `no_gesture_static` | `onefinger` | `partial_swipe` | `random_motion` | `return_motion` | `wrong_axis_motion` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2finger` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gun` | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hand` | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `like` | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| `no_gesture_static` | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 |
| `onefinger` | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 |
| `partial_swipe` | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 4 | 6 | 1 |
| `random_motion` | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 9 | 4 | 4 |
| `return_motion` | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 4 | 6 |
| `wrong_axis_motion` | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 6 | 5 | 6 |

## Threshold для лучшей модели

| Метрика | Значение |
|---|---:|
| Threshold | 0.9000 |
| Coverage | 0.5962 |
| Accuracy на принятых предсказаниях | 1.0000 |
| Отклонено предсказаний | 105 |

## Наблюдения

- Активные классы в сравнении: 2finger, gun, hand, like, no_gesture_static, onefinger, partial_swipe, random_motion, return_motion, wrong_axis_motion
- CV folds: 5; минимальный размер класса: 20
- Augmented samples: included
- Пропущены несовместимые пары feature/model: `static_craft_full_stats+static_landmark_cnn`
- Обнаружены разные исходные размерности признаков; benchmark выравнивает семплы до target_dim 63.

## Следующий эксперимент

1. Если dynamic-like классов мало или нет, записать 2-3 жеста с выраженным
   движением и повторить benchmark.
2. Если static-like и dynamic-like группы выбирают разные пайплайны, проверить
   split-routing: сначала классифицировать тип жеста, затем запускать отдельную
   модель.
3. Подтвердить recommended threshold через live-eval в приложении.
