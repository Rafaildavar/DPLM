# Live Evaluation Metrics

Source: `/Users/remi/.dplm/logs/live_evaluation.jsonl`

## Summary

| Attempts | Correct | Wrong | Missed | Accuracy |
|---:|---:|---:|---:|---:|
| 30 | 23 | 6 | 1 | 0.767 |

## By Label

| Expected | Attempts | Correct | Wrong | Missed | Accuracy | Avg confidence | Wrong labels |
|---|---:|---:|---:|---:|---:|---:|---|
| `swipe_down` | 10 | 5 | 5 | 0 | 0.500 | 0.971 | swipe_up:5 |
| `swipe_left` | 10 | 8 | 1 | 1 | 0.800 | 0.947 | swipe_up:1 |
| `swipe_up` | 10 | 10 | 0 | 0 | 1.000 | 1.000 |  |

## Manual Notes

One `swipe_left` missed attempt was an accidental `Пропуск` button press, not a
real model miss.

Adjusted view for discussion:

| Scope | Valid attempts | Correct | Wrong | Missed excluded | Adjusted accuracy |
|---|---:|---:|---:|---:|---:|
| all labels | 29 | 23 | 6 | 1 | 0.793 |
| `swipe_left` | 9 | 8 | 1 | 1 | 0.889 |
| `swipe_up` | 10 | 10 | 0 | 0 | 1.000 |
| `swipe_down` | 10 | 5 | 5 | 0 | 0.500 |

Main observed confusion: `swipe_down` -> `swipe_up` (`5/10`).
