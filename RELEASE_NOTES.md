# GestureBind v0.8.1 MVP

GestureBind v0.8.1 packages the validated v0.8.0 MVP as a native macOS app.
The product path remains the same: record personal gestures, train local
models, validate them live and bind accepted gestures to safe macOS actions.

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
- Static predictions in auto mode now require `0.80` confidence, based on the final controlled live run.
- Dynamic predictions and completed-event intent overrides now require at least `0.90` confidence.
- An adaptive per-class completion gate rejects unfinished dynamic gestures
  before amplitude normalization and can resume the same event after a pause.
- The tag workflow now publishes both a tracked source/model archive and a ZIP
  containing `GestureBind.app`, built and smoke-tested on macOS.
- Optional daily telemetry sends sanitized aggregates with offline retry and
  explicit consent; camera data, landmarks, labels and command text stay local.
- Correct, incorrect and missed controls provide user-confirmed quality signals.

## Reproducible Evidence

| Metric | Value |
|---|---:|
| Static grouped CV accuracy / macro F1 | `0.7767 / 0.6613` |
| Dynamic grouped validation accuracy | `0.8571` |
| Dynamic prototype positive recall | `0.9333` |
| Dynamic negative false-positive rate | `0.0000` |
| Dynamic ML latency mean / p95 | `13.979 / 16.013 ms` |
| Completion candidate full / prefix accepted | `60/60 / 6/240` |
| Production state-machine replay full / wrong class | `59/60 / 0` |
| Production state-machine replay prefix false accepts | `5/240 = 0.0208` |
| Post-gate controlled live static recall at threshold `0.80` | `20/20 = 1.0000` |
| Post-gate controlled live dynamic recall at threshold `0.90` | `38/40 = 0.9500` |
| Post-gate positive total / wrong class | `58/60 = 0.9667 / 0` |
| Partial/look-alike false-command rate | `1/20 = 0.0500` |
| Background/no-command false-command rate | `0/30 = 0.0000` |
| Aggregate safety false-command rate | `1/50 = 0.0200` |

The post-completion dynamic release run produced `SwipeLeft 20/20`,
`diagonal 9/10` and
`zoom 9/10`: two misses and no reported wrong-class predictions. The
command-binding confidence policy is independent from the static and dynamic
recognition thresholds.
The completion benchmark is source-recording replay, while the `38/40` dynamic
result is fresh controlled webcam evidence after enabling the gate.
The final safety matrix passed the release target: one false command across
`20` partial/look-alike attempts and zero across `30` background attempts.

## Included

- Flet desktop source and current UI flows.
- Static, dynamic LSTM and intent-gate production artifacts.
- Taxonomy/config contracts, database migrations and command policies.
- Unit/integration tests, ML smoke checks and release automation.
- Curated architecture, training, binding and release-readiness documents.
- Ad-hoc signed `GestureBind.app` in an install-friendly
  `GestureBind-macos-v0.8.1.dmg`, with ZIP kept as a fallback.
- Russian end-user guide included in both macOS artifacts.

## Not Included

- User recordings or personal gesture classes.
- `.env`, local SQLite/PostgreSQL state, runtime logs or MLflow runs.
- Experiment model folders, non-allowlisted generated reports, IDE state or virtual environments.
- The retired Qt/QML desktop implementation.

## Known MVP Limits

- The release is macOS-first and is not yet notarized.
- Live evidence is a controlled personalized protocol, not signer-independent
  benchmark quality.
- New personal classes still require diverse recordings for reliable open-set
  behavior.
