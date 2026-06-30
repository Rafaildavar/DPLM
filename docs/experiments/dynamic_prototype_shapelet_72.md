# Dynamic Prototype Comparison

- Best method: `prototype_distance`
- Train samples: `472`
- Test samples: `158`
- External negatives: `True`

## Negative Conflict Filter

- Status: `ok`
- Method: `prototype_distance`
- Conflict margin: `1.20`
- External negatives: `440`
- Safe external negatives: `440`
- Conflicts removed: `0`
- Conflict rate: `0.0000`

## Method Metrics

| Method | Overall | Positive recall | Negative reject | Negative FP | Sequence accuracy | Edit distance |
|---|---:|---:|---:|---:|---:|---:|
| `prototype_distance` | 0.9810 | 0.9130 | 0.9926 | 0.0074 | 0.9130 | 2 |

## Per-Class Recall

### `prototype_distance`

| Label | Total | Correct | Wrong | Rejected |
|---|---:|---:|---:|---:|
| `UpAndLeft` | 5 | 5 | 0 | 0 |
| `swipe_down` | 5 | 5 | 0 | 0 |
| `swipe_left` | 8 | 8 | 0 | 0 |
| `swipe_up` | 5 | 3 | 0 | 2 |

## Interpretation

- `prototype_distance` is a fast flattened-sequence verifier.
- `prototype_dtw` is slower but more tolerant to timing variation.
- A method is useful for live only if it keeps positive recall high while lowering negative false positives.
