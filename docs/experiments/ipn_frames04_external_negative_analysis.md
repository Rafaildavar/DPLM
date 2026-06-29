# IPN External Negative Analysis

Generated at: `1782730929.352`

## Recommendation

`existing` is safe as dynamic negative training/validation data for `prototype_distance`. `frames04` is safe as dynamic negative training/validation data for `prototype_distance`.

## Root Summary

| Root | Samples | Mean frames | Mean detection | Motion directions |
|---|---:|---:|---:|---|
| `existing` | 200 | 49.34 | 0.8943 | `{"diagonal_or_complex": 14, "down": 66, "left": 26, "right": 48, "up": 46}` |
| `frames04` | 240 | 37.02 | 0.8159 | `{"diagonal_or_complex": 20, "down": 102, "left": 22, "right": 54, "up": 42}` |

## Model Safety

### `existing`

| Model | Negative reject | False positive | Near positive | Accepted labels | Reject reasons |
|---|---:|---:|---:|---|---|
| `prototype_distance` | 1.0000 | 0.0000 | 0.0000 | `{}` | `{"far_from_prototype": 1, "nearest_negative": 199}` |
### `frames04`

| Model | Negative reject | False positive | Near positive | Accepted labels | Reject reasons |
|---|---:|---:|---:|---|---|
| `prototype_distance` | 1.0000 | 0.0000 | 0.0000 | `{}` | `{"far_from_prototype": 5, "nearest_negative": 235}` |

## Original IPN Labels

### `existing`

| Label | Samples |
|---|---:|
| `B0A` | 54 |
| `B0B` | 53 |
| `D0X` | 19 |
| `G01` | 10 |
| `G02` | 11 |
| `G07` | 11 |
| `G08` | 11 |
| `G09` | 10 |
| `G10` | 10 |
| `G11` | 11 |

### `frames04`

| Label | Samples |
|---|---:|
| `B0A` | 61 |
| `B0B` | 67 |
| `D0X` | 13 |
| `G01` | 14 |
| `G02` | 14 |
| `G07` | 14 |
| `G08` | 14 |
| `G09` | 14 |
| `G10` | 15 |
| `G11` | 14 |

## Warnings

- prototype model not found: /private/tmp/missing_dynamic_prototypes.json

## Interpretation

- Useful external negatives should have high negative reject rate and low false positive rate.
- `near_positive_rate` flags samples that are rejected but lie close to swipe prototypes; these are good validation cases, but risky training negatives.
- IPN throw-left/up/down/right classes should remain validation/reference data unless we intentionally model them as user commands.
