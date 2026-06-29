# IPN External Negative Analysis

Generated at: `1782724460.158`

## Recommendation

`existing` is safe as dynamic negative training/validation data for `prototype_distance`. `tar_expanded` is empty; do not use it yet.

## Root Summary

| Root | Samples | Mean frames | Mean detection | Motion directions |
|---|---:|---:|---:|---|
| `existing` | 200 | 49.34 | 0.8943 | `{"diagonal_or_complex": 14, "down": 66, "left": 26, "right": 48, "up": 46}` |
| `tar_expanded` | 0 | 0.00 | 0.0000 | `{}` |

## Model Safety

### `existing`

| Model | Negative reject | False positive | Near positive | Accepted labels | Reject reasons |
|---|---:|---:|---:|---|---|
| `prototype_distance` | 1.0000 | 0.0000 | 0.0000 | `{}` | `{"far_from_prototype": 1, "nearest_negative": 199}` |
| `prototype_dtw` | 1.0000 | 0.0000 | 0.0000 | `{}` | `{"far_from_prototype": 1, "nearest_negative": 199}` |
### `tar_expanded`

| Model | Negative reject | False positive | Near positive | Accepted labels | Reject reasons |
|---|---:|---:|---:|---|---|
| `prototype_distance` | 0.0000 | 0.0000 | 0.0000 | `{}` | `{}` |
| `prototype_dtw` | 0.0000 | 0.0000 | 0.0000 | `{}` | `{}` |

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

### `tar_expanded`

| Label | Samples |
|---|---:|
| none | 0 |

## Warnings

- external root not found: /Users/remi/Developer/GUAP/DPLM/data/external/ipn_hand_tar

## Interpretation

- Useful external negatives should have high negative reject rate and low false positive rate.
- `near_positive_rate` flags samples that are rejected but lie close to swipe prototypes; these are good validation cases, but risky training negatives.
- IPN throw-left/up/down/right classes should remain validation/reference data unless we intentionally model them as user commands.
