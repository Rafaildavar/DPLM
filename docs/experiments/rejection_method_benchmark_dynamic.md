# Rejection Method Benchmark

Generated at: `1782593440.134`

## Dataset

- data root: `/Users/remi/Developer/GUAP/DPLM/data/gestures`
- scope: `dynamic`
- feature mode: `dynamic_stats`
- target dim: `44`
- samples/classes/folds: `170` / `8` / `3`
- positive labels: `swipe_down, swipe_left, swipe_up`
- negative labels: `no_gesture_static, partial_swipe, random_motion, return_motion, wrong_axis_motion`

## Results

| Method | Status | Overall | Pos recall | Pos reject | Neg reject | Neg FP | Accepted acc | Coverage | Wrong accepts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `open_set_policy` | ok | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.4118 | 0 |
| `one_vs_rest_logreg` | ok | 0.9941 | 0.9857 | 0.0143 | 1.0000 | 0.0000 | 1.0000 | 0.4059 | 0 |
| `confidence_threshold` | ok | 0.9882 | 1.0000 | 0.0000 | 0.9800 | 0.0200 | 0.9722 | 0.4235 | 2 |
| `negative_classes` | ok | 0.9647 | 1.0000 | 0.0000 | 0.9400 | 0.0600 | 0.9211 | 0.4471 | 6 |
| `isolation_forest` | ok | 0.9235 | 0.8143 | 0.1857 | 1.0000 | 0.0000 | 1.0000 | 0.3353 | 0 |
| `local_outlier_factor` | ok | 0.9235 | 0.8143 | 0.1857 | 1.0000 | 0.0000 | 1.0000 | 0.3353 | 0 |
| `mlp_negative_classes` | ok | 0.8824 | 0.7286 | 0.2714 | 0.9900 | 0.0100 | 0.9808 | 0.3059 | 1 |
| `one_class_svm` | ok | 0.6765 | 0.2143 | 0.7857 | 1.0000 | 0.0000 | 1.0000 | 0.0882 | 0 |
| `metric_nca_centroid` | ok | 0.4588 | 1.0000 | 0.0000 | 0.0800 | 0.9200 | 0.4321 | 0.9529 | 92 |

## Recommendation

Use `open_set_policy` first: overall_success=1.0000, positive_recall=1.0000, negative_false_positive_rate=0.0000.

## Method Notes

- `negative_classes`: supervised multiclass model rejects when predicted class is negative.
- `confidence_threshold`: rejects low-confidence positive predictions.
- `open_set_policy`: current runtime-style policy: negative probability, top1/top2 margin, prototype distance.
- `one_vs_rest_logreg`: binary verifier per positive class.
- `one_class_svm`, `isolation_forest`, `local_outlier_factor`: one-class/outlier verifier per positive class.
- `metric_nca_centroid`: metric-learning embedding + class prototype radius.
- `mlp_negative_classes`: nonlinear supervised classifier with negative classes.

## Reject Reasons

### `negative_classes`
- reject reasons: `{"negative_class": 94}`
- false positive predictions: `{"swipe_down": 1, "swipe_left": 1, "swipe_up": 4}`

### `confidence_threshold`
- reject reasons: `{"low_confidence": 4, "negative_class": 94}`
- false positive predictions: `{"swipe_down": 1, "swipe_left": 1}`

### `open_set_policy`
- reject reasons: `{"far_from_prototype": 6, "negative_class": 94}`
- false positive predictions: `{}`

### `one_vs_rest_logreg`
- reject reasons: `{"negative_class": 94, "verifier_rejected": 7}`
- false positive predictions: `{}`

### `one_class_svm`
- reject reasons: `{"negative_class": 94, "one_class_svm_outlier": 61}`
- false positive predictions: `{}`

### `isolation_forest`
- reject reasons: `{"isolation_forest_outlier": 19, "negative_class": 94}`
- false positive predictions: `{}`

### `local_outlier_factor`
- reject reasons: `{"local_outlier_factor_outlier": 19, "negative_class": 94}`
- false positive predictions: `{}`

### `metric_nca_centroid`
- reject reasons: `{"metric_distance_rejected": 8}`
- false positive predictions: `{"swipe_down": 16, "swipe_left": 74, "swipe_up": 2}`

### `mlp_negative_classes`
- reject reasons: `{"low_confidence": 15, "negative_class": 103}`
- false positive predictions: `{"swipe_up": 1}`
