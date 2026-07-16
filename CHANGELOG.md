# Changelog

## v0.8.1-beta.7 - 2026-07-16

- Added a native `hdiutil` fallback when the styled `create-dmg` step is not
  available, fails or completes without producing an artifact on GitHub Actions.
- Kept the drag-to-Applications flow and bundled guides in fallback DMGs while
  continuing to require both DMG and ZIP assets before publishing a release.
- Includes the configurable MAS LLM providers and Keychain-backed user API keys
  introduced in beta.5.

## v0.8.1-beta.6 - 2026-07-16

- Fixed the macOS release workflow so a failed DMG build can no longer be
  hidden by the packaging log pipeline.
- Released temporary PyInstaller directories before DMG creation to preserve
  runner disk space and made both the DMG and ZIP mandatory release assets.
- Includes the user-selectable LLM provider, model, endpoint and Keychain-backed
  API key settings introduced in beta.5.

## v0.8.1-beta.5 - 2026-07-16

- Added user-selectable LLM providers for the MAS binding assistant: OpenAI,
  Claude, Gemini, Mistral, OpenRouter, Groq, DeepSeek, Ollama and custom
  OpenAI-compatible endpoints.
- Added editable model and API endpoint settings with connection testing, while
  keeping the local MAS mode as the default.
- Stored API keys separately for each provider and endpoint in macOS Keychain;
  credentials are never written to the application config or release bundle.
- Added provider-specific environment overrides and retained compatibility with
  existing Mistral settings.

## v0.8.1-beta.4 - 2026-07-12

- Fixed a native MediaPipe crash when recognition was stopped while the camera
  thread was processing a frame.
- Bundled and branded the Flet desktop client so GestureBind has one Dock icon
  instead of separate launcher and Flet icons.
- Added a native installation warning when GestureBind is launched directly
  from a mounted DMG.
- Kept camera shutdown and model disposal ordered and thread-safe across model
  changes as well as the main Stop action.

## v0.8.1-beta.3 - 2026-07-12

- Reworked the macOS DMG as a native drag-and-drop installer with a custom
  Finder layout and a clean app-to-Applications flow.
- Added a branded GestureBind macOS icon and placed the installation, user and
  project guides directly in the DMG.
- Added clear Gatekeeper and Accessibility recovery instructions for beta users.
- Fixed cursor mode so missing macOS Accessibility permission is reported and
  the correct System Settings page opens automatically.
- Published the first camera frame before ML warm-up to reduce perceived startup
  delay in the packaged app.
- Fixed static user recordings being mislabeled as dynamic when their name also
  exists in the production dynamic model.
- Documented that the public beta contains no Mistral key and uses the local MAS.

## v0.8.1 - 2026-07-10

- Added a tracked PyInstaller macOS application bundle with Flet, MediaPipe,
  production model artifacts and local SQLite runtime paths.
- Unified source and macOS packaging under one tag workflow so a release is
  published only after both artifacts build successfully.
- Added macOS packaging files to the repository hygiene contract.
- Added an install-friendly macOS DMG with an `Applications` shortcut, while
  keeping the ZIP package as a fallback.
- Added a Russian beta user guide covering installation, permissions, gesture
  training, command binding, privacy and troubleshooting.
- Added opt-in daily usage telemetry with persistent cursors, offline retry,
  idempotent reports and a self-hosted collector.
- Added correct, incorrect and missed feedback controls for real-use quality.

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
