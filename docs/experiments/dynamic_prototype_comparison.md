# Dynamic Prototype Comparison

- Best method: `prototype_distance`
- Train samples: `277`
- Test samples: `93`
- External negatives: `True`

## Method Metrics

| Method | Overall | Positive recall | Negative reject | Negative FP | Sequence accuracy | Edit distance |
|---|---:|---:|---:|---:|---:|---:|
| `prototype_distance` | 0.9892 | 0.9444 | 1.0000 | 0.0000 | 0.9444 | 1 |
| `prototype_dtw` | 0.9892 | 0.9444 | 1.0000 | 0.0000 | 0.9444 | 1 |

## Per-Class Recall

### `prototype_distance`

| Label | Total | Correct | Wrong | Rejected |
|---|---:|---:|---:|---:|
| `swipe_down` | 5 | 5 | 0 | 0 |
| `swipe_left` | 8 | 7 | 0 | 1 |
| `swipe_up` | 5 | 5 | 0 | 0 |

### `prototype_dtw`

| Label | Total | Correct | Wrong | Rejected |
|---|---:|---:|---:|---:|
| `swipe_down` | 5 | 5 | 0 | 0 |
| `swipe_left` | 8 | 7 | 0 | 1 |
| `swipe_up` | 5 | 5 | 0 | 0 |

## Interpretation

- `prototype_distance` is a fast flattened-sequence verifier.
- `prototype_dtw` is slower but more tolerant to timing variation.
- A method is useful for live only if it keeps positive recall high while lowering negative false positives.
