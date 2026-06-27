# Rejection Method Benchmark

Generated at: `1782593354.068`

## Dataset

- data root: `/Users/remi/Developer/GUAP/DPLM/data/gestures`
- scope: `static`
- feature mode: `static_mean`
- target dim: `42`
- samples/classes/folds: `221` / `11` / `3`
- positive labels: `CTRLZ, Hend, UP, gun, sh3, three`
- negative labels: `no_gesture_static, partial_swipe, random_motion, return_motion, wrong_axis_motion`

## Results

| Method | Status | Overall | Pos recall | Pos reject | Neg reject | Neg FP | Accepted acc | Coverage | Wrong accepts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `one_vs_rest_logreg` | ok | 0.9774 | 0.9669 | 0.0165 | 0.9900 | 0.0100 | 0.9750 | 0.5430 | 3 |
| `negative_classes` | ok | 0.9774 | 0.9752 | 0.0000 | 0.9800 | 0.0200 | 0.9593 | 0.5566 | 5 |
| `confidence_threshold` | ok | 0.9774 | 0.9752 | 0.0083 | 0.9800 | 0.0200 | 0.9672 | 0.5520 | 4 |
| `open_set_policy` | ok | 0.9683 | 0.9587 | 0.0165 | 0.9800 | 0.0200 | 0.9587 | 0.5475 | 5 |
| `metric_nca_centroid` | ok | 0.9593 | 0.9752 | 0.0000 | 0.9400 | 0.0600 | 0.9291 | 0.5747 | 9 |
| `local_outlier_factor` | ok | 0.9457 | 0.9008 | 0.0992 | 1.0000 | 0.0000 | 1.0000 | 0.4932 | 0 |
| `isolation_forest` | ok | 0.8914 | 0.8182 | 0.1818 | 0.9800 | 0.0200 | 0.9802 | 0.4570 | 2 |
| `one_class_svm` | ok | 0.6742 | 0.4050 | 0.5950 | 1.0000 | 0.0000 | 1.0000 | 0.2217 | 0 |
| `mlp_negative_classes` | ok | 0.5973 | 0.2645 | 0.7355 | 1.0000 | 0.0000 | 1.0000 | 0.1448 | 0 |

## Recommendation

Use `one_vs_rest_logreg` first: overall_success=0.9774, positive_recall=0.9669, negative_false_positive_rate=0.0100.

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
- reject reasons: `{"negative_class": 98}`
- false positive predictions: `{"three": 2}`

### `confidence_threshold`
- reject reasons: `{"low_confidence": 1, "negative_class": 98}`
- false positive predictions: `{"three": 2}`

### `open_set_policy`
- reject reasons: `{"far_from_prototype": 2, "negative_class": 98}`
- false positive predictions: `{"three": 2}`

### `one_vs_rest_logreg`
- reject reasons: `{"negative_class": 98, "verifier_rejected": 3}`
- false positive predictions: `{"three": 1}`

### `one_class_svm`
- reject reasons: `{"negative_class": 98, "one_class_svm_outlier": 74}`
- false positive predictions: `{}`

### `isolation_forest`
- reject reasons: `{"isolation_forest_outlier": 22, "negative_class": 98}`
- false positive predictions: `{"three": 2}`

### `local_outlier_factor`
- reject reasons: `{"local_outlier_factor_outlier": 14, "negative_class": 98}`
- false positive predictions: `{}`

### `metric_nca_centroid`
- reject reasons: `{"metric_distance_rejected": 94}`
- false positive predictions: `{"Hend": 2, "sh3": 4}`

### `mlp_negative_classes`
- reject reasons: `{"low_confidence": 98, "negative_class": 91}`
- false positive predictions: `{}`
