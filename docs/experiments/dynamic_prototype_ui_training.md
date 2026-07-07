# Dynamic Prototype Comparison

- Best method: `prototype_distance`
- Train samples: `433`
- Test samples: `145`
- External negatives: `True`

## Negative Conflict Filter

- Status: `ok`
- Method: `prototype_distance`
- Conflict margin: `1.20`
- External negatives: `440`
- Safe external negatives: `438`
- Conflicts removed: `2`
- Conflict rate: `0.0045`

| Negative label | Conflicts |
|---|---:|
| `negative_external_ipn_dynamic` | 2 |

## Method Metrics

| Method | Overall | Positive recall | Negative reject | Negative FP | Sequence accuracy | Edit distance |
|---|---:|---:|---:|---:|---:|---:|
| `prototype_distance` | 0.9931 | 0.9000 | 1.0000 | 0.0000 | 0.9000 | 1 |

## Per-Class Recall

### `prototype_distance`

| Label | Total | Correct | Wrong | Rejected |
|---|---:|---:|---:|---:|
| `SwipeLeft` | 5 | 4 | 0 | 1 |
| `zoom` | 5 | 5 | 0 | 0 |

## Interpretation

- `prototype_distance` is a fast flattened-sequence verifier.
- `prototype_dtw` is slower but more tolerant to timing variation.
- A method is useful for live only if it keeps positive recall high while lowering negative false positives.
