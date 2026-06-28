# External Negative Dataset Experiments

Generated at: `1782667810.285`

## Goal

Проверить, улучшают ли публичные датасеты качество reject-layer без
подмены персонального датасета GestureFlow. Внешние данные используются
как negative / out-of-distribution evidence.

## Variants

| Variant | Status | Internal samples | External samples | Sources | Model root |
|---|---|---:|---:|---|---|
| `baseline_internal` | ok | 291 | 0 | none | `/Users/remi/Developer/GUAP/DPLM/models/experiments/external_negative/baseline_internal` |
| `ipn_external` | skipped | 291 | 0 | ipn_hand:missing:0 | `/Users/remi/Developer/GUAP/DPLM/models/experiments/external_negative/ipn_external` |
| `hagrid_external` | skipped | 291 | 0 | hagrid:missing:0 | `/Users/remi/Developer/GUAP/DPLM/models/experiments/external_negative/hagrid_external` |
| `combined_external` | skipped | 291 | 0 | ipn_hand:missing:0, hagrid:missing:0 | `/Users/remi/Developer/GUAP/DPLM/models/experiments/external_negative/combined_external` |

## Offline Benchmarks

| Variant | Scope | Status | Best method | Overall | Pos recall | Neg reject | Neg FP |
|---|---|---|---|---:|---:|---:|---:|
| `baseline_internal` | static | ok | `one_vs_rest_logreg` | 0.9774 | 0.9669 | 0.9900 | 0.0100 |
| `baseline_internal` | dynamic | ok | `open_set_policy` | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Live Artifacts

| Variant | Scope | Status | Samples | Classes | Feature mode | Model root |
|---|---|---|---:|---:|---|---|
| `baseline_internal` | `static` | ok | 221 | 11 | static_mean | `/Users/remi/Developer/GUAP/DPLM/models/experiments/external_negative/baseline_internal` |
| `baseline_internal` | `dynamic:knn` | ok | 170 | 8 | dynamic_stats | `/Users/remi/Developer/GUAP/DPLM/models/experiments/external_negative/baseline_internal` |

## How To Live-Test A Variant

Example for `hagrid_external`:

```bash
DPLM_MODELS_DIR=models/experiments/external_negative/hagrid_external \
DPLM_MODEL_PATH=models/experiments/external_negative/hagrid_external/knn.pkl \
DPLM_CLASSES_PATH=models/experiments/external_negative/hagrid_external/classes.json \
DPLM_FEATURE_DIM_PATH=models/experiments/external_negative/hagrid_external/feature_dim.txt \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m app.flet_app.main
```

Then use Live Evaluation exactly like the existing rejection protocol.

## Notes

- ipn_external: skipped - variant requires external sources, but none were imported
- hagrid_external: skipped - variant requires external sources, but none were imported
- combined_external: skipped - variant requires external sources, but none were imported
- External datasets are treated as negative/rejection data, not as replacement for personalized GestureFlow samples.
- Live validation still decides whether an external variant is useful for the user's camera and gesture style.
