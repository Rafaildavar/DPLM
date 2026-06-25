# JMLC Model Comparison

## Краткий вывод

- Лучший общий результат: **`dynamic_stats` + `extra_trees`** с macro F1 `0.8136` и accuracy `0.8209`.
- Датасет: 201 семплов, 10 активных классов, target_dim 44.
- Рекомендация: На текущем датасете достаточно единого пайплайна `dynamic_stats` + `extra_trees`: он не проигрывает отдельно static-like/dynamic-like группам.

## Датасет

| Класс | Семплы |
|---|---:|
| `CTRLZ` | 21 |
| `Hend` | 20 |
| `UP` | 20 |
| `gun` | 20 |
| `hand_left` | 30 |
| `sh3` | 20 |
| `swipe_down` | 10 |
| `swipe_left` | 10 |
| `swipe_up` | 30 |
| `three` | 20 |

| Метрика | Значение |
|---|---:|
| Исходные размерности | 42, 44 |
| Целевая размерность | 44 |
| CV folds | 5 |
| Motion threshold | 0.0150 |

## Motion Profile

| Класс | Семплы | Median motion energy | Median displacement | Suggested type |
|---|---:|---:|---:|---|
| `CTRLZ` | 21 | 0.0054 | 0.0118 | `static_like` |
| `Hend` | 20 | 0.0063 | 0.0457 | `static_like` |
| `UP` | 20 | 0.0056 | 0.0311 | `static_like` |
| `gun` | 20 | 0.0071 | 0.0490 | `static_like` |
| `hand_left` | 30 | 0.0185 | 0.1127 | `dynamic_like` |
| `sh3` | 20 | 0.0103 | 0.1080 | `static_like` |
| `swipe_down` | 10 | 0.0202 | 0.3460 | `dynamic_like` |
| `swipe_left` | 10 | 0.0126 | 0.1050 | `static_like` |
| `swipe_up` | 30 | 0.0365 | 0.1724 | `dynamic_like` |
| `three` | 20 | 0.0063 | 0.0296 | `static_like` |

## Результаты

| Признаки | Модель | Accuracy | Macro F1 | Static-like F1 | Dynamic-like F1 | Macro FPR | Latency ms/sample | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dynamic_stats` | `knn` | 0.6219 | 0.6166 | 0.5112 | 0.8625 | 0.0422 | 1.084 | 0.8500 |
| `dynamic_stats` | `svm` | 0.5124 | 0.5019 | 0.3350 | 0.8914 | 0.0545 | 0.038 | 0.6000 |
| `dynamic_stats` | `extra_trees` | 0.8209 | 0.8136 | 0.7563 | 0.9472 | 0.0200 | 0.125 | 0.6000 |

## Лучшая модель: per-class метрики

| Класс | Motion type | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| `CTRLZ` | `static_like` | 21 | 0.9500 | 0.9048 | 0.9268 |
| `Hend` | `static_like` | 20 | 0.6250 | 0.5000 | 0.5556 |
| `UP` | `static_like` | 20 | 0.7083 | 0.8500 | 0.7727 |
| `gun` | `static_like` | 20 | 0.5714 | 0.4000 | 0.4706 |
| `hand_left` | `dynamic_like` | 30 | 0.8529 | 0.9667 | 0.9062 |
| `sh3` | `static_like` | 20 | 0.6957 | 0.8000 | 0.7442 |
| `swipe_down` | `dynamic_like` | 10 | 0.9091 | 1.0000 | 0.9524 |
| `swipe_left` | `static_like` | 10 | 0.9091 | 1.0000 | 0.9524 |
| `swipe_up` | `dynamic_like` | 30 | 1.0000 | 0.9667 | 0.9831 |
| `three` | `static_like` | 20 | 0.8947 | 0.8500 | 0.8718 |

## Матрица ошибок лучшей модели

| true \ pred | `CTRLZ` | `Hend` | `UP` | `gun` | `hand_left` | `sh3` | `swipe_down` | `swipe_left` | `swipe_up` | `three` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `CTRLZ` | 19 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Hend` | 1 | 10 | 1 | 2 | 2 | 4 | 0 | 0 | 0 | 0 |
| `UP` | 0 | 0 | 17 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gun` | 0 | 4 | 5 | 8 | 1 | 2 | 0 | 0 | 0 | 0 |
| `hand_left` | 0 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 0 | 1 |
| `sh3` | 0 | 1 | 1 | 0 | 1 | 16 | 0 | 0 | 0 | 1 |
| `swipe_down` | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 |
| `swipe_left` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| `swipe_up` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 29 | 0 |
| `three` | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 17 |

## Threshold для лучшей модели

| Метрика | Значение |
|---|---:|
| Threshold | 0.6000 |
| Coverage | 0.5323 |
| Accuracy на принятых предсказаниях | 0.9907 |
| Отклонено предсказаний | 94 |

## Наблюдения

- Активные классы в сравнении: CTRLZ, Hend, UP, gun, hand_left, sh3, swipe_down, swipe_left, swipe_up, three
- CV folds: 5; минимальный размер класса: 10
- Обнаружены разные исходные размерности признаков; benchmark выравнивает семплы до target_dim 44.
- Дисбаланс классов высокий; macro-метрики важнее accuracy.

## Следующий эксперимент

1. Если dynamic-like классов мало или нет, записать 2-3 жеста с выраженным
   движением и повторить benchmark.
2. Если static-like и dynamic-like группы выбирают разные пайплайны, проверить
   split-routing: сначала классифицировать тип жеста, затем запускать отдельную
   модель.
3. Подтвердить recommended threshold через live-eval в приложении.
