# GestureBind v0.8.0 MVP

GestureBind v0.8.0 is the first release candidate built around one coherent
desktop product path: record personal gestures, train local models, validate
them live and bind accepted gestures to safe macOS actions.

## Highlights

- Release-first, camera-centered home screen with one primary recognition action.
- One supported desktop runtime: Flet. The retired PySide/QML implementation
  and its skipped tests are no longer part of the release tree.
- Production dynamic profile retrained with source-grouped Optuna validation.
- Train-only normalization and augmentation groups prevent validation leakage.
- Label-safe routing and duplicate-motion rejection protect dynamic datasets.
- Atomic model bundle activation preserves the previous model on failure.
- MLflow training runs include dataset/model hashes, Git state and group overlap.
- Repository architecture and an automated tracked-tree hygiene gate are part of CI.
- Binding-agent golden cases are portable across macOS development and Linux CI.

## Reproducible Evidence

| Metric | Value |
|---|---:|
| Static grouped CV accuracy / macro F1 | `0.7767 / 0.6613` |
| Dynamic grouped validation accuracy | `0.8571` |
| Dynamic prototype positive recall | `0.9333` |
| Dynamic negative false-positive rate | `0.0000` |
| Dynamic ML latency mean / p95 | `13.979 / 16.013 ms` |

## Included

- Flet desktop source and current UI flows.
- Static, dynamic LSTM and intent-gate production artifacts.
- Taxonomy/config contracts, database migrations and command policies.
- Unit/integration tests, ML smoke checks and release automation.
- Curated architecture, training, binding and release-readiness documents.

## Not Included

- User recordings or personal gesture classes.
- `.env`, local SQLite/PostgreSQL state, runtime logs or MLflow runs.
- Experiment model folders, generated reports, IDE state or virtual environments.
- The retired Qt/QML desktop implementation.

## Known MVP Limits

- The release is macOS-first and is not yet notarized.
- A final live static/dynamic/no-command matrix is required before creating the
  public `v0.8.0` tag.
- New personal classes still require diverse recordings for reliable open-set
  behavior.
