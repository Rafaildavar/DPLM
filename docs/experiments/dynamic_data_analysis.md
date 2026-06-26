# Dynamic Gesture Data Analysis

Generated: `2026-06-26T11:43:26+00:00`

## Краткий вывод

- Недостаточно dynamic-сэмплов: swipe_down=10, swipe_left=10. Эти классы нужно дозаписать перед сильными выводами о модели.
- В сохраненных классах есть шум траекторий или конфликт направления: swipe_up. Чистый протокол записи сейчас важнее очередной смены модели.
- Самый информативный compact descriptor: `direction_sin` (MI=0.8027): нормализованное вертикальное направление, устойчивое к амплитуде.
- Offline KNN на сохраненных dynamic-сэмплах дает accuracy 0.9800 и macro F1 0.9785; live-ошибки поэтому указывают на distribution shift между записью и показом.
- Live-ошибки концентрируются в predicted labels: swipe_up:12.

## Объем данных

| Метрика | Значение |
|---|---:|
| Data root | `data/gestures` |
| Include prefix | `swipe_` |
| Классы | 3 |
| Сэмплы | 50 |
| Raw dimensions | 44 |
| Target dim | 44 |

| Класс | Сэмплы | Frames median | Raw dims | dx median | dy median | Path median | Straightness | Speed | Direction OK | Warnings |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `swipe_down` | 10 | 36.0000 | 44 | -0.0130 | 0.4876 | 0.5207 | 0.9194 | 0.0149 | 1.0000 | low_pose_motion_energy:2 |
| `swipe_left` | 10 | 36.0000 | 44 | -0.3824 | 0.0212 | 0.4467 | 0.8587 | 0.0128 | 1.0000 | low_global_path:1, low_pose_motion_energy:9 |
| `swipe_up` | 30 | 36.0000 | 44 | 0.0135 | -0.1820 | 0.9624 | 0.2062 | 0.0275 | 0.6667 | axis_drift:10, expected_up:10, low_pose_motion_energy:1, low_straightness:20 |

## Ключевые признаки

| Rank | Descriptor | Mutual information | Интерпретация |
|---:|---|---:|---|
| 1 | `direction_sin` | 0.8027 | нормализованное вертикальное направление, устойчивое к амплитуде |
| 2 | `dy` | 0.6806 | вертикальное смещение от старта к финишу; разделяет up/down |
| 3 | `vertical_ratio` | 0.6521 | доля вертикального движения относительно горизонтального |
| 4 | `horizontal_ratio` | 0.6521 | доля горизонтального движения относительно вертикального |
| 5 | `direction_cos` | 0.6463 | нормализованное горизонтальное направление, устойчивое к амплитуде |
| 6 | `straightness` | 0.6400 | чистый ли это штрих или траектория с петлей/возвратом |
| 7 | `motion_energy` | 0.4813 | среднее покадровое движение landmarks |
| 8 | `abs_dx` | 0.4597 | горизонтальная амплитуда |

## Offline-валидация

| Метрика | Значение |
|---|---:|
| Model | `knn(distance)` |
| Feature mode | `dynamic_stats` |
| CV folds | 5 |
| Сэмплы | 50 |
| Accuracy | 0.9800 |
| Macro F1 | 0.9785 |
| Ошибочных сэмплов | 1 |

### Confusion Matrix

| true \ pred | `swipe_down` | `swipe_left` | `swipe_up` |
|---|---:|---:|---:|
| `swipe_down` | 10 | 0 | 0 |
| `swipe_left` | 0 | 10 | 0 |
| `swipe_up` | 0 | 1 | 29 |

### Ошибочные sample

| Expected | Predicted | Confidence | Sample |
|---|---|---:|---|
| `swipe_up` | `swipe_left` | 1.0000 | `swipe_up/sample_0003.npy` |

### Correct vs Wrong Descriptor Gap

| Descriptor | Correct mean | Wrong mean | Effect size | Интерпретация |
|---|---:|---:|---:|---|
| `path_length` | 0.7509 | 0.2931 | 2.0978 | суммарный путь global wrist |
| `speed` | 0.0215 | 0.0084 | 2.0952 | path length, нормированный на число кадров |
| `motion_energy` | 0.0286 | 0.0105 | 1.7548 | среднее покадровое движение landmarks |
| `direction_cos` | -0.2297 | 0.3343 | -1.6226 | нормализованное горизонтальное направление, устойчивое к амплитуде |
| `straightness` | 0.5656 | 0.1748 | 1.6054 | чистый ли это штрих или траектория с петлей/возвратом |
| `abs_dy` | 0.2567 | 0.0483 | 1.4646 | вертикальная амплитуда |
| `displacement` | 0.2259 | 0.0798 | 1.4522 | start/end distance на уровне pose-признаков |
| `direction_sin` | -0.1647 | -0.9425 | 1.3353 | нормализованное вертикальное направление, устойчивое к амплитуде |

## Live-gap

| Метрика | Значение |
|---|---:|
| Source | `/Users/remi/.dplm/logs/live_evaluation.jsonl` |
| Runs | 9 |
| Attempts | 62 |
| Correct | 39 |
| Wrong | 12 |
| Missed | 11 |
| Accuracy | 0.6290 |
| Wrong labels | swipe_up:12 |

| Expected | Runs | Attempts | Correct | Wrong | Missed | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `swipe_down` | 7 | 42 | 21 | 11 | 10 | 0.5000 |
| `swipe_left` | 1 | 10 | 8 | 1 | 1 | 0.8000 |
| `swipe_up` | 1 | 10 | 10 | 0 | 0 | 1.0000 |

## Проблемные сохраненные sample

| Label | Sample | dx | dy | Path | Straightness | Energy | Warnings |
|---|---|---:|---:|---:|---:|---:|---|
| `swipe_up` | `swipe_up/sample_0003.npy` | 0.0171 | -0.0483 | 0.2931 | 0.1748 | 0.0105 | expected_up, axis_drift, low_pose_motion_energy, low_straightness |
| `swipe_up` | `swipe_up/sample_0023.npy` | -0.0350 | -0.0474 | 0.8661 | 0.0680 | 0.0419 | expected_up, axis_drift, low_straightness |
| `swipe_up` | `swipe_up/sample_0018.npy` | -0.0278 | -0.0174 | 1.0701 | 0.0306 | 0.0413 | expected_up, axis_drift, low_straightness |
| `swipe_up` | `swipe_up/sample_0013.npy` | -0.0205 | -0.0198 | 1.1987 | 0.0238 | 0.0344 | expected_up, axis_drift, low_straightness |
| `swipe_up` | `swipe_up/sample_0005.npy` | 0.0145 | -0.0240 | 1.2985 | 0.0216 | 0.0442 | expected_up, axis_drift, low_straightness |
| `swipe_up` | `swipe_up/sample_0001.npy` | -0.0277 | -0.0177 | 0.9587 | 0.0342 | 0.0309 | expected_up, axis_drift, low_straightness |
| `swipe_up` | `swipe_up/sample_0029.npy` | -0.0196 | 0.2717 | 1.2819 | 0.2125 | 0.0886 | expected_up, low_straightness |
| `swipe_up` | `swipe_up/sample_0022.npy` | 0.0269 | 0.1074 | 1.1254 | 0.0984 | 0.0433 | expected_up, low_straightness |
| `swipe_up` | `swipe_up/sample_0008.npy` | -0.1786 | -0.1178 | 1.0696 | 0.2000 | 0.0511 | axis_drift, low_straightness |
| `swipe_up` | `swipe_up/sample_0006.npy` | -0.0064 | 0.2373 | 1.1102 | 0.2138 | 0.0372 | expected_up, low_straightness |
| `swipe_up` | `swipe_up/sample_0004.npy` | -0.0479 | -0.1375 | 1.0816 | 0.1346 | 0.0352 | axis_drift, low_straightness |
| `swipe_up` | `swipe_up/sample_0002.npy` | 0.0027 | 0.0130 | 0.8682 | 0.0153 | 0.0357 | expected_up, low_straightness |
| `swipe_up` | `swipe_up/sample_0028.npy` | -0.0331 | -0.3093 | 1.0726 | 0.2900 | 0.0349 | low_straightness |
| `swipe_up` | `swipe_up/sample_0021.npy` | -0.0093 | -0.2206 | 0.8995 | 0.2455 | 0.0356 | low_straightness |
| `swipe_up` | `swipe_up/sample_0020.npy` | -0.0428 | -0.1848 | 1.2466 | 0.1522 | 0.0441 | low_straightness |
| `swipe_up` | `swipe_up/sample_0017.npy` | 0.1012 | -0.5569 | 1.0908 | 0.5189 | 0.0399 | low_straightness |
| `swipe_up` | `swipe_up/sample_0014.npy` | -0.0148 | -0.2337 | 1.2863 | 0.1821 | 0.0445 | low_straightness |
| `swipe_up` | `swipe_up/sample_0012.npy` | 0.0330 | -0.1776 | 1.0068 | 0.1794 | 0.0422 | low_straightness |
| `swipe_up` | `swipe_up/sample_0011.npy` | 0.1265 | -0.3708 | 0.6032 | 0.6494 | 0.0176 | axis_drift |
| `swipe_up` | `swipe_up/sample_0010.npy` | 0.0182 | -0.1533 | 0.9660 | 0.1598 | 0.0378 | low_straightness |
| `swipe_up` | `swipe_up/sample_0007.npy` | 0.0370 | -0.1792 | 0.9562 | 0.1914 | 0.0376 | low_straightness |
| `swipe_up` | `swipe_up/sample_0000.npy` | 0.6348 | -0.5013 | 1.1282 | 0.7169 | 0.0377 | axis_drift |
| `swipe_left` | `swipe_left/sample_0000.npy` | -0.1715 | -0.0056 | 0.2438 | 0.7040 | 0.0069 | low_pose_motion_energy, low_global_path |
| `swipe_left` | `swipe_left/sample_0009.npy` | -0.4756 | -0.0104 | 0.5796 | 0.8209 | 0.0148 | low_pose_motion_energy |

## Решения по предобработке

- Держать отдельный dynamic dataset/model для жестов движения, пока live-метрики не докажут стабильность единой модели.
- Использовать канон записи на уровне класса: фиксированный старт, чистое направление, видимый финиш, сброс между сэмплами.
- На сохранении применять quality gates: знак ожидаемого направления, minimum global path, axis alignment, straightness и pose motion energy.
- Балансировать dynamic-классы минимум до 20 сэмплов на класс перед сравнением моделей; основной offline-показатель - macro F1.
- Высокий offline CV при слабом live считать distribution shift; каждую candidate-модель подтверждать live evaluation с expected_label.
- Следующий preprocessing experiment: выделять active motion segment и resample до 36 кадров, чтобы медленные/быстрые жесты были сравнимы.

## Интерпретация для JMLC

- Анализ документирует понимание данных, критерии предобработки, протокол валидации и live/offline mismatch.
- Следующая измеримая гипотеза - не очередная смена модели, а более чистые dynamic-данные плюс live-evaluation confirmation.

## Воспроизведение

```bash
python -m scripts.dynamic_data_analysis --out-md docs/experiments/dynamic_data_analysis.md --out-json docs/experiments/dynamic_data_analysis.json
```
