# Changelog

## v0.8.0 - 2026-07-10

- Rebuilt the home screen around the release camera and recognition workflow.
- Removed the retired PySide/QML runtime, duplicate launcher and skipped legacy tests.
- Retrained the production dynamic LSTM with grouped Optuna validation and zero group overlap.
- Added train-only normalization, label-safe routing and atomic model publication.
- Added reproducible MLflow provenance and production latency evidence.
- Added the root architecture contract and CI-enforced repository hygiene.
- Switched release source bundles to tracked files only.
- Made binding-agent path golden cases independent of a developer home directory.
- Raised the release auto-router static confidence threshold to `0.80` from live evidence.
- Raised the release dynamic confidence threshold and completed-event override floor to `0.90`.
- Added adaptive per-class dynamic completion verification with resumable
  segmentation and MLflow full/prefix replay evidence.
- Validated the post-gate controlled live matrix at `58/60` positive attempts,
  including `38/40` dynamic attempts and zero wrong-class predictions.
- Passed the final safety matrix with `1/20` partial/look-alike and `0/30`
  background false commands.

## v0.7.0 - 2026-07-08

- Prepared the repository for a public release: kept the project-focused source tree, current docs, CI/CD workflows and production model artifacts.
- Kept the production dynamic gesture model line on `dynamic_landmark_lstm_backbone`.
- Removed generated experiment materials, old release notes and unused model artifacts from the tracked repository snapshot.
- Simplified CI/CD documentation around the active checks: unit/integration tests, ML smoke checks, Docker runtime and release bundle publishing.
- Fixed the recurrent backbone optimizer path so the full CI suite can run reliably.
- Polished the command binding form for application launch actions.
- Made the legacy QML load smoke opt-in while keeping the current desktop UI test path active.

## v0.6.0

- Previous public project snapshot.
