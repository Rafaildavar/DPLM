# Live Evaluation Metrics

Source: `/Users/remi/.dplm/logs/live_evaluation.jsonl`

## Summary

| Attempts | Correct | Wrong | Missed | Accuracy |
|---:|---:|---:|---:|---:|
| 52 | 38 | 11 | 3 | 0.731 |

## By Label

| Expected | Attempts | Correct | Wrong | Missed | Accuracy | Avg confidence | Wrong labels |
|---|---:|---:|---:|---:|---:|---:|---|
| `swipe_down` | 32 | 20 | 10 | 2 | 0.625 | 0.949 | swipe_up:10 |
| `swipe_left` | 10 | 8 | 1 | 1 | 0.800 | 0.947 | swipe_up:1 |
| `swipe_up` | 10 | 10 | 0 | 0 | 1.000 | 1.000 |  |

## Latest Completed Run By Label

| Expected | Model | Attempts | Correct | Wrong | Missed | Accuracy | Accepted accuracy | Avg confidence | Wrong labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `swipe_down` | `n/a` | 10 | 7 | 2 | 1 | 0.700 | 0.778 | 0.968 | swipe_up:2 |
| `swipe_left` | `n/a` | 10 | 8 | 1 | 1 | 0.800 | 0.889 | 0.947 | swipe_up:1 |
| `swipe_up` | `n/a` | 10 | 10 | 0 | 0 | 1.000 | 1.000 | 1.000 |  |

## Recent Runs

| Event | Expected | Model | Attempts | Correct | Wrong | Missed | Accuracy | Min conf | Timeout |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `completed` | `swipe_left` | `n/a` | 10 | 8 | 1 | 1 | 0.800 | 0.80 | 0.0 |
| `completed` | `swipe_up` | `n/a` | 10 | 10 | 0 | 0 | 1.000 | 0.80 | 0.0 |
| `completed` | `swipe_down` | `n/a` | 10 | 5 | 5 | 0 | 0.500 | 0.80 | 0.0 |
| `completed` | `swipe_down` | `n/a` | 10 | 8 | 1 | 1 | 0.800 | 0.80 | 0.0 |
| `completed` | `swipe_down` | `n/a` | 10 | 7 | 2 | 1 | 0.700 | 0.80 | 0.0 |
| `stopped` | `swipe_down` | `n/a` | 2 | 0 | 2 | 0 | 0.000 | 0.80 | 0.0 |
| `stopped` | `swipe_down` | `n/a` | 0 | 0 | 0 | 0 | 0.000 | 0.80 | 0.0 |
| `stopped` | `swipe_down` | `n/a` | 0 | 0 | 0 | 0 | 0.000 | 0.80 | 0.0 |
