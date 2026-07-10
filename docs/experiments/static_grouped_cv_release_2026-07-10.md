# GestureBind Cross-Validation Model Comparison

## Краткий вывод

- Лучший общий результат: **`static_craft_full_stats` + `extra_trees`** с macro F1 `0.7600` и accuracy `0.8708`.
- Датасет: 480 семплов, 13 активных классов, target_dim 63.
- Рекомендация: Текущий датасет выглядит static-like по выбранному motion threshold. Пока рано доказывать отдельную dynamic-модель: нужно записать жесты с выраженным движением и повторить benchmark.

## Датасет

| Класс | Семплы |
|---|---:|
| `2Finger` | 40 |
| `Like` | 40 |
| `OneFInger` | 40 |
| `SwipeLeft` | 60 |
| `diagonal` | 60 |
| `gun` | 40 |
| `hand` | 40 |
| `no_gesture_static` | 20 |
| `partial_swipe` | 20 |
| `random_motion` | 20 |
| `return_motion` | 20 |
| `wrong_axis_motion` | 20 |
| `zoom` | 60 |

| Метрика | Значение |
|---|---:|
| Исходные размерности | 42, 44, 63, 65 |
| Целевая размерность | 63 |
| CV folds | 5 |
| Source groups | 260 |
| Grouped CV | yes |
| Motion threshold | 0.0150 |
| Augmented samples | included |

## Motion Profile

| Класс | Семплы | Median motion energy | Median displacement | Suggested type |
|---|---:|---:|---:|---|
| `2Finger` | 40 | 0.0011 | 0.0021 | `static_like` |
| `Like` | 40 | 0.0033 | 0.0094 | `static_like` |
| `OneFInger` | 40 | 0.0014 | 0.0038 | `static_like` |
| `SwipeLeft` | 60 | 0.0087 | 0.1441 | `static_like` |
| `diagonal` | 60 | 0.0146 | 0.3983 | `static_like` |
| `gun` | 40 | 0.0037 | 0.0087 | `static_like` |
| `hand` | 40 | 0.0008 | 0.0021 | `static_like` |
| `no_gesture_static` | 20 | 0.0079 | 0.0158 | `static_like` |
| `partial_swipe` | 20 | 0.0036 | 0.0036 | `static_like` |
| `random_motion` | 20 | 0.0036 | 0.0035 | `static_like` |
| `return_motion` | 20 | 0.0039 | 0.0035 | `static_like` |
| `wrong_axis_motion` | 20 | 0.0036 | 0.0312 | `static_like` |
| `zoom` | 60 | 0.0109 | 0.0119 | `static_like` |

## Результаты

| Признаки | Модель | Accuracy | Macro F1 | Static-like F1 | Dynamic-like F1 | Macro FPR | Latency ms/sample | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `static_craft_full_stats` | `extra_trees` | 0.8708 | 0.7600 | 0.7600 | n/a | 0.0104 | 0.087 | 0.9000 |

## Лучшая модель: per-class метрики

| Класс | Motion type | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| `2Finger` | `static_like` | 40 | 1.0000 | 1.0000 | 1.0000 |
| `Like` | `static_like` | 40 | 1.0000 | 1.0000 | 1.0000 |
| `OneFInger` | `static_like` | 40 | 1.0000 | 1.0000 | 1.0000 |
| `SwipeLeft` | `static_like` | 60 | 1.0000 | 1.0000 | 1.0000 |
| `diagonal` | `static_like` | 60 | 1.0000 | 1.0000 | 1.0000 |
| `gun` | `static_like` | 40 | 1.0000 | 1.0000 | 1.0000 |
| `hand` | `static_like` | 40 | 1.0000 | 1.0000 | 1.0000 |
| `no_gesture_static` | `static_like` | 20 | 1.0000 | 1.0000 | 1.0000 |
| `partial_swipe` | `static_like` | 20 | 0.3810 | 0.4000 | 0.3902 |
| `random_motion` | `static_like` | 20 | 0.2609 | 0.3000 | 0.2791 |
| `return_motion` | `static_like` | 20 | 0.1111 | 0.1000 | 0.1053 |
| `wrong_axis_motion` | `static_like` | 20 | 0.1111 | 0.1000 | 0.1053 |
| `zoom` | `static_like` | 60 | 1.0000 | 1.0000 | 1.0000 |

## Матрица ошибок лучшей модели

| true \ pred | `2Finger` | `Like` | `OneFInger` | `SwipeLeft` | `diagonal` | `gun` | `hand` | `no_gesture_static` | `partial_swipe` | `random_motion` | `return_motion` | `wrong_axis_motion` | `zoom` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2Finger` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Like` | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `OneFInger` | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `SwipeLeft` | 0 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `diagonal` | 0 | 0 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gun` | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hand` | 0 | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| `no_gesture_static` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 |
| `partial_swipe` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 3 | 5 | 4 | 0 |
| `random_motion` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 6 | 5 | 6 | 0 |
| `return_motion` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 2 | 6 | 0 |
| `wrong_axis_motion` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 8 | 6 | 2 | 0 |
| `zoom` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 |

## Threshold для лучшей модели

| Метрика | Значение |
|---|---:|
| Threshold | 0.9000 |
| Coverage | 0.7271 |
| Accuracy на принятых предсказаниях | 1.0000 |
| Отклонено предсказаний | 131 |

## Наблюдения

- Активные классы в сравнении: 2Finger, Like, OneFInger, SwipeLeft, diagonal, gun, hand, no_gesture_static, partial_swipe, random_motion, return_motion, wrong_axis_motion, zoom
- CV folds: 5; минимальный размер класса: 20
- Grouped CV: 260 source groups; originals and their augmentations stay in the same fold
- Augmented samples: included
- Обнаружены разные исходные размерности признаков; benchmark выравнивает семплы до target_dim 63.
- Дисбаланс классов высокий; macro-метрики важнее accuracy.

## Следующий эксперимент

1. Если dynamic-like классов мало или нет, записать 2-3 жеста с выраженным
   движением и повторить benchmark.
2. Если static-like и dynamic-like группы выбирают разные пайплайны, проверить
   split-routing: сначала классифицировать тип жеста, затем запускать отдельную
   модель.
3. Подтвердить recommended threshold через live-eval в приложении.
