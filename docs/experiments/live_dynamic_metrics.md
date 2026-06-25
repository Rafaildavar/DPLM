# Live Dynamic Metrics

Дата проверки: 2026-06-25.

Источник: PostgreSQL `recognition_logs`, база `dplm`, live-окно
`2026-06-25 14:42:18` - `14:44:00` UTC (`17:42:18` - `17:44:00` MSK).

Важно: эти метрики event-level. В логах есть предсказанная метка и confidence,
но нет ground truth попытки. Поэтому здесь нельзя честно посчитать attempt-level
accuracy или missed attempts без ручного протокола вида
`expected -> predicted`.

## Event Summary

| Label | Events | Avg confidence | Min | Max |
|---|---:|---:|---:|---:|
| `swipe_up` | 9 | 0.905 | 0.616 | 1.000 |
| `swipe_down` | 5 | 0.796 | 0.674 | 1.000 |
| `swipe_left` | 3 | 0.871 | 0.697 | 1.000 |
| `hand_left` | 1 | 0.503 | 0.503 | 0.503 |
| `sh3` | 1 | 0.407 | 0.407 | 0.407 |

Итого:
- events: `19`;
- target swipe events: `17`;
- non-target events: `2`;
- raw target purity: `17/19 = 0.895`.

## Confidence Threshold Sweep

| Threshold | Accepted | Target events | Non-target events | Coverage | Target purity | Target event recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 19 | 17 | 2 | 1.000 | 0.895 | 1.000 |
| 0.50 | 18 | 17 | 1 | 0.947 | 0.944 | 1.000 |
| 0.60 | 17 | 17 | 0 | 0.895 | 1.000 | 1.000 |
| 0.65 | 16 | 16 | 0 | 0.842 | 1.000 | 0.941 |
| 0.70 | 14 | 14 | 0 | 0.737 | 1.000 | 0.824 |
| 0.80 | 12 | 12 | 0 | 0.632 | 1.000 | 0.706 |
| 0.90 | 9 | 9 | 0 | 0.474 | 1.000 | 0.529 |

Предварительный вывод:
- для текущего small live-test оптимальный фильтр выглядит как
  `confidence >= 0.60`;
- `0.60` отсекает оба нецелевых события (`sh3`, `hand_left`) и не теряет
  target swipe events;
- `0.65` уже теряет один target event (`swipe_up`, confidence `0.616`).

## Raw Event Sequence

| UTC time | Label | Confidence |
|---|---|---:|
| 14:42:18.424434 | `swipe_left` | 0.915 |
| 14:42:24.976271 | `swipe_up` | 1.000 |
| 14:42:26.378454 | `swipe_down` | 0.674 |
| 14:42:27.542135 | `swipe_up` | 0.616 |
| 14:42:41.049368 | `swipe_up` | 1.000 |
| 14:42:41.186619 | `swipe_down` | 0.800 |
| 14:42:58.038745 | `swipe_down` | 1.000 |
| 14:43:01.704062 | `swipe_left` | 1.000 |
| 14:43:09.881839 | `swipe_up` | 1.000 |
| 14:43:46.277585 | `swipe_up` | 0.913 |
| 14:43:46.408967 | `swipe_left` | 0.697 |
| 14:43:52.077344 | `sh3` | 0.407 |
| 14:43:52.688200 | `hand_left` | 0.503 |
| 14:43:52.871341 | `swipe_up` | 1.000 |
| 14:43:55.303855 | `swipe_up` | 0.900 |
| 14:43:55.438395 | `swipe_down` | 0.708 |
| 14:43:56.107161 | `swipe_up` | 0.811 |
| 14:44:00.476466 | `swipe_down` | 0.798 |
| 14:44:00.739369 | `swipe_up` | 0.908 |

## Next Measurement

Для полноценной JMLC-метрики теперь используется Home -> `Live evaluation`.
Параметры режима:

- `Expected` — какой жест сейчас тестируется;
- `Attempts` — сколько попыток нужно собрать;
- `Timeout` — сколько секунд ждать accepted prediction до `Missed`;
- `Min conf` — минимальная confidence для зачета `Correct/Wrong`.

Результаты попыток пишутся в `~/.dplm/logs/live_evaluation.jsonl`.

Шаблон отчета:

| Expected | Attempts | Correct | Wrong label | Missed | Notes |
|---|---:|---:|---:|---:|---|
| `swipe_up` | 10 |  |  |  |  |
| `swipe_down` | 10 |  |  |  |  |
| `swipe_left` | 10 |  |  |  |  |
| `swipe_right` | 10 |  |  |  |  |
