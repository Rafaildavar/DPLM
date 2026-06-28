# External Public Datasets As Negative Evidence

Status: `experimental`

Owner: ML pipeline

## Why This Exists

GestureFlow is a personalized gesture system. The core positive dataset should
stay user-recorded because live quality depends on the user's camera, distance,
hand shape, speed and command semantics.

Public datasets are still useful, but mainly as:

- `negative` / out-of-distribution examples;
- reject-layer calibration;
- robustness checks for distance, background, hand shape and lighting;
- contest evidence that the project uses external data consciously instead of
blindly mixing domains.

## Candidate Datasets

| Dataset | Planned role | Why not direct positive training |
|---|---|---|
| IPN Hand | dynamic non-target motion, partial/ambiguous motion | classes and recording protocol do not equal the user's custom commands |
| HaGRID | static non-target hand poses, near-miss static gestures | many gestures are not GestureFlow commands, and some overlap semantically |

References:

- IPN Hand: `https://github.com/GibranBenitez/IPN-hand`
- HaGRID: `https://github.com/hukenovs/hagrid`

## Prepared Data Format

External data must be converted to the same landmark format as GestureFlow:

```text
data/external/ipn_hand/<original_label>/sample_0000.npy
data/external/hagrid/<original_label>/sample_0000.npy
```

Expected arrays:

- static/image-derived samples: `(frames, 42)`;
- dynamic/video-derived samples: `(frames, 44)`;
- values should be normalized landmark coordinates, not raw RGB pixels.

The experiment script maps these original labels into negative classes:

```text
negative_external_ipn_dynamic
negative_external_hagrid_static
```

This prevents a public class such as `call` or `stop` from becoming a new
GestureFlow command by accident.

## Variants

The current config lives in:

```text
configs/external_negative_datasets.json
```

Default variants:

| Variant | Meaning |
|---|---|
| `baseline_internal` | current GestureFlow personal + synthetic negatives |
| `ipn_external` | baseline + IPN dynamic negative samples |
| `hagrid_external` | baseline + HaGRID static negative samples |
| `combined_external` | baseline + IPN + HaGRID |

## Command

```bash
PYTHON=.venv/bin/python make external-negative-experiments
```

Outputs:

```text
docs/experiments/external_negative_dataset_experiments.json
docs/experiments/external_negative_dataset_experiments.md
data/experiments/external_negative/<variant>/gestures/
models/experiments/external_negative/<variant>/
```

MLflow runs:

```text
external-negative-<variant>-static
external-negative-<variant>-dynamic
external-negative-artifacts-<variant>-static
external-negative-artifacts-<variant>-dynamic:knn
```

## Model Strategy

For small personalized datasets, the default live model remains:

- static classifier: `KNN + static_mean`;
- dynamic classifier: `motion-first gate + KNN + dynamic_stats`;
- reject methods: `open_set_policy`, `one_vs_rest_logreg`,
  `mlp_negative_classes`, one-class/outlier verifiers.

External negatives mainly affect the second-stage reject problem:

- if external negatives are diverse but noisy, `one_vs_rest_logreg` and
  `open_set_policy` are safer;
- if there are many clean negative examples, `mlp_negative_classes` can model
  nonlinear boundaries;
- if positives are tight and negatives are broad, one-class methods can help;
- sequence models such as LSTM/TCN/Transformer should wait until there are
  enough real temporal samples per class and a clean live metric baseline.

## Live Test Protocol After Adding External Data

For each variant, launch the app with that variant's model root:

```bash
DPLM_MODELS_DIR=models/experiments/external_negative/hagrid_external \
DPLM_MODEL_PATH=models/experiments/external_negative/hagrid_external/knn.pkl \
DPLM_CLASSES_PATH=models/experiments/external_negative/hagrid_external/classes.json \
DPLM_FEATURE_DIM_PATH=models/experiments/external_negative/hagrid_external/feature_dim.txt \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m app.flet_app.main
```

Then run Live Evaluation:

- positive static: `gun`, `three`, `hend`, `up`, 20 attempts each;
- negative static: `no_gesture_static`, `partial_swipe`, `wrong_axis_motion`,
  20 attempts each;
- dynamic sanity: `swipe_left`, `swipe_up`, `swipe_down`, 20 attempts each.

Compare in MLflow:

- `live_accuracy`;
- `live_static_false_positive_rate`;
- `live_negative_false_positive_rate`;
- `live_static_reject_rate`;
- `live_dynamic_recall`;
- system metrics / latency;
- artifact charts under `live_evaluation/charts/`.

## Current Result

Date: `2026-06-28`

Local external files are not present yet, so only the internal baseline ran:

- `baseline_internal/static`: best `one_vs_rest_logreg`,
  overall `0.9774`, negative FP `0.0100`;
- `baseline_internal/dynamic`: best `open_set_policy`,
  overall `1.0000`, negative FP `0.0000`;
- `ipn_external`, `hagrid_external`, `combined_external`: `skipped`
  until converted samples are placed under `data/external/...`.

Conclusion:

The pipeline is ready for external negative datasets, but the actual quality
claim must wait for real converted IPN/HaGRID samples and live validation.
