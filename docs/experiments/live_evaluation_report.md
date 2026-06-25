# Live Evaluation Metrics

Source: `/Users/remi/.dplm/logs/live_evaluation.jsonl`

## Summary

| Attempts | Correct | Wrong | Missed | Accuracy |
|---:|---:|---:|---:|---:|
| 10 | 8 | 1 | 1 | 0.800 |

## By Label

| Expected | Attempts | Correct | Wrong | Missed | Accuracy | Avg confidence | Wrong labels |
|---|---:|---:|---:|---:|---:|---:|---|
| `swipe_left` | 10 | 8 | 1 | 1 | 0.800 | 0.947 | swipe_up:1 |

## Manual Note

One `swipe_left` missed attempt was an accidental `Пропуск` button press, not a
real model miss.

Adjusted view for discussion:

| Expected | Valid attempts | Correct | Wrong | Missed excluded | Adjusted accuracy |
|---|---:|---:|---:|---:|---:|
| `swipe_left` | 9 | 8 | 1 | 1 | 0.889 |

Rounded adjusted accuracy: `90%`.
