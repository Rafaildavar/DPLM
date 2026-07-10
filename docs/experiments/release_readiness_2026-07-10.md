# Release Readiness Audit: 2026-07-10

## Frozen baseline

- Git tag: `pre-critical-ml-pipeline-2026-07-10`.
- Baseline commit: `784de9f`.
- The tag preserves the application and model artifacts exactly as they were
  before the release-hardening changes.
- No production `.pkl` model was replaced during this audit.

To inspect or create a rollback branch without changing the current branch:

```bash
git show pre-critical-ml-pipeline-2026-07-10
git branch codex/release-rollback pre-critical-ml-pipeline-2026-07-10
```

## Critical fixes

| Risk before release | Change | Product effect |
|---|---|---|
| An original capture and its augmentations could enter different CV folds | Source-grouped holdout and CV for ExtraTrees, recurrent tuning and benchmark scripts | Validation no longer sees transformed copies of its own test samples |
| LSTM normalization was fitted before the validation split | Split raw sequences first; fit normalization only on the training partition | Removes validation leakage and makes early stopping more honest |
| Runtime could emit `swipe_left` while the user's class is `SwipeLeft` | Unique punctuation/case normalization maps output back to the exact registered label | Live metrics and command bindings use the user's actual class name |
| Weak dynamic duplicates were saved and only marked with a warning | Gate now checks global displacement, path and pairwise hand-shape change before writing | Empty takes do not contaminate the next training run; zoom-like in-place motion still passes |
| ExtraTrees used all CPU workers for one live sample | Runtime changes loaded tree estimators to `n_jobs=1` | Current static `predict_proba` improved from `26.715 ms` to `8.690 ms` in the focused A/B |
| Model and sidecars were overwritten one by one | Temporary siblings, rollback on failure, sidecars first and model-last activation | A failed training process keeps the previous working bundle |
| Latency script averaged coordinates and could not test current 1377/4752-dimensional models | Benchmark now builds the real feature mode and runs production static/dynamic artifacts | Presentation numbers correspond to the code used in live inference |
| MLflow runs could not prove which data/code produced a model | Dataset SHA-256, model SHA-256, git commit/dirty state, source groups and grouped-validation fields | Runs are traceable and comparable instead of being anonymous accuracy rows |

## Verification

- Full automated suite: `592 passed, 2 skipped`.
- The two skipped modules are the optional legacy PySide6 UI in the active
  Flet environment.
- Atomic bundle rollback is tested with an injected model-activation failure.
- MLflow smoke run: `32866f9906ac4b3688b3f3f52b9d220d` in
  `GestureBind-Pipeline-Smoke`.
- Smoke run recorded dataset/model hashes, git state, `40` source groups and
  validation group overlap `0`.

## Grouped static CV

Source: [static_grouped_cv_release_2026-07-10.md](static_grouped_cv_release_2026-07-10.md).

- `300` files, including augmentations, but only `200` independent source groups.
- `5` folds with source-group isolation.
- `static_craft_full_stats + ExtraTrees`:
  - accuracy `0.7767`;
  - macro F1 `0.6613`;
  - accepted accuracy at threshold `0.75`: `1.0000`;
  - coverage: `0.7133`.
- All five user positive static classes and `no_gesture_static` have F1 `1.0`.
  The lower macro F1 comes from confusion inside the four similar negative
  motion subclasses, not from confusion between user commands.

## Production ML latency

Source: [release_latency_benchmark_2026-07-10.json](release_latency_benchmark_2026-07-10.json).

| Profile | Model / features | Mean | P95 | Capacity |
|---|---|---:|---:|---:|
| Static | ExtraTrees / `static_craft_full_stats` | `17.476 ms` | `18.100 ms` | `57.22 FPS` |
| Dynamic | LSTM / `dynamic_landmark_image` | `3.847 ms` | `4.053 ms` | `259.94 FPS` |

These values cover feature extraction plus the same `predict` and
`predict_proba` calls used by the recognition layer. They do not include camera
capture or MediaPipe; end-to-end webcam latency must be measured separately.

## Required fresh evidence before defense

1. Run at least `10` live attempts per positive static and dynamic class.
2. Run at least `30` no-command/background attempts.
3. Report correct, wrong, missed, per-class recall and negative false-positive rate.
4. Regenerate the MLflow/dashboard snapshot only after these runs.
5. Do not present training accuracy as test accuracy; label it
   `resubstitution accuracy` if it is shown at all.
