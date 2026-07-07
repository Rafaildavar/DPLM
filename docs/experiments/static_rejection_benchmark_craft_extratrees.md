# Rejection Method Benchmark

Generated at: `1783270945.985`

## Dataset

- data root: `/Users/remi/Developer/GUAP/DPLM/data/gestures`
- scope: `static`
- feature mode: `static_craft_full_stats`
- candidate model: `extra_trees`
- target dim: `63`
- augmented samples: `included`
- samples/classes/folds: `260` / `10` / `5`
- positive labels: `2finger, gun, hand, like, onefinger`
- negative labels: `no_gesture_static, partial_swipe, random_motion, return_motion, wrong_axis_motion`

## Results

| Method | Status | Overall | Pos recall | Pos reject | Neg reject | Neg FP | Accepted acc | Coverage | Wrong accepts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `negative_classes` | ok | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.6154 | 0 |
| `open_set_policy` | ok | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.6154 | 0 |
| `one_vs_rest_logreg` | ok | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.6154 | 0 |
| `confidence_threshold` | ok | 0.9885 | 0.9812 | 0.0187 | 1.0000 | 0.0000 | 1.0000 | 0.6038 | 0 |
| `isolation_forest` | ok | 0.9038 | 0.8438 | 0.1562 | 1.0000 | 0.0000 | 1.0000 | 0.5192 | 0 |
| `local_outlier_factor` | ok | 0.8731 | 0.7937 | 0.2062 | 1.0000 | 0.0000 | 1.0000 | 0.4885 | 0 |

## Recommendation

Use `negative_classes` first: overall_success=1.0000, positive_recall=1.0000, negative_false_positive_rate=0.0000.

## Method Notes

- `negative_classes`: supervised multiclass model rejects when predicted class is negative.
- `confidence_threshold`: rejects low-confidence positive predictions.
- `open_set_policy`: current runtime-style policy: negative probability, top1/top2 margin, prototype distance.
- `one_vs_rest_logreg`: binary verifier per positive class.
- `one_class_svm`, `isolation_forest`, `local_outlier_factor`: one-class/outlier verifier per positive class.
- `metric_nca_centroid`: metric-learning embedding + class prototype radius.
- `mlp_negative_classes`: nonlinear supervised classifier with negative classes.

## Best Method Negative Breakdown

Best method: `negative_classes`

| Negative label | Total | Rejected | False positive | Reject rate | FP rate | FP predictions |
|---|---:|---:|---:|---:|---:|---|
| `no_gesture_static` | 20 | 20 | 0 | 1.0000 | 0.0000 | `{}` |
| `partial_swipe` | 20 | 20 | 0 | 1.0000 | 0.0000 | `{}` |
| `random_motion` | 20 | 20 | 0 | 1.0000 | 0.0000 | `{}` |
| `return_motion` | 20 | 20 | 0 | 1.0000 | 0.0000 | `{}` |
| `wrong_axis_motion` | 20 | 20 | 0 | 1.0000 | 0.0000 | `{}` |

## Reject Reasons

### `negative_classes`
- reject reasons: `{"negative_class": 100}`
- false positive predictions: `{}`

### `confidence_threshold`
- reject reasons: `{"low_confidence": 3, "negative_class": 100}`
- false positive predictions: `{}`

### `open_set_policy`
- reject reasons: `{"negative_class": 100}`
- false positive predictions: `{}`

### `one_vs_rest_logreg`
- reject reasons: `{"negative_class": 100}`
- false positive predictions: `{}`

### `local_outlier_factor`
- reject reasons: `{"local_outlier_factor_outlier": 33, "negative_class": 100}`
- false positive predictions: `{}`

### `isolation_forest`
- reject reasons: `{"isolation_forest_outlier": 25, "negative_class": 100}`
- false positive predictions: `{}`
