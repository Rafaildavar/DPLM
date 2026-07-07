# Static Landmark CNN Benchmark

Date: `2026-07-05`

## Goal

Check the GISLR-inspired landmark-image idea for static gestures without
replacing the current production static model.

The production baseline remains:

```text
model = ExtraTreesClassifier
feature_mode = static_craft_full_stats
feature_dim = 1377
```

## Dataset

Training loaded `260` samples across `10` classes:

| Class | Samples |
|---|---:|
| `2finger` | 40 |
| `like` | 40 |
| `onefinger` | 40 |
| `gun` | 20 |
| `hand` | 20 |
| `no_gesture_static` | 20 |
| `partial_swipe` | 20 |
| `random_motion` | 20 |
| `return_motion` | 20 |
| `wrong_axis_motion` | 20 |

`--include-augmented` was enabled, so GISLR-marked `aug_sample_*` files were
included. Class balancing expanded the in-memory fit set to `270` samples; it
did not create new `.npy` files.

## Feature

`static_landmark_image` converts each static sample into a flattened tensor:

```text
30 frames x 21 points x 3 coordinates = 1890 features
```

Legacy `xy` samples are supported by filling `z = 0`.

## Model

`KerasStaticLandmarkCNNClassifier`:

- `Conv2D + DepthwiseConv2D`;
- weighted categorical cross entropy;
- label smoothing;
- dropout;
- early stopping;
- joblib-compatible numpy weight storage.

## Artifacts

```text
models/experiments/static_landmark_cnn/static_landmark_cnn.pkl
models/experiments/static_landmark_cnn/classes.json
models/experiments/static_landmark_cnn/feature_dim.txt
models/experiments/static_landmark_cnn/feature_mode.txt
models/experiments/static_landmark_cnn/gesture_rejection.json
```

## Result

| Model | Feature mode | Train accuracy | Validation accuracy |
|---|---|---:|---:|
| production ExtraTrees | `static_craft_full_stats` | `1.0000` | n/a |
| static CNN benchmark | `static_landmark_image` | `0.8000` | `0.7963` |

CNN recognizes the user gesture classes well on the train-set report, but it is
currently weaker on negative classes:

| Negative class | CNN F1 |
|---|---:|
| `partial_swipe` | `0.400` |
| `random_motion` | `0.500` |
| `return_motion` | `0.091` |
| `wrong_axis_motion` | `0.367` |

## Live Test Plan

Use the Flet model variant dropdown:

1. Select `production`.
2. Run the same 5 static gestures and no-command/partial-motion checks.
3. Select `static_landmark_cnn`.
4. Repeat the same attempts.

Success criteria:

- user gesture hit rate is not worse than production;
- false positives on no-command/partial movement are not worse than production;
- if CNN feels better on user gestures, add stronger negative training or
  image-branch rejection before making it production.
