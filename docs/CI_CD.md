# GestureFlow CI/CD

## Status

Implemented first production-oriented CI/CD layer:

- GitHub Actions CI for pull requests and pushes.
- Separate ML smoke workflow for model artifact checks.
- Docker runtime workflow for reproducible headless ML/runtime validation.
- macOS app workflow for `.app`/`.zip` delivery artifacts.
- Tag-based desktop release bundle.
- Local Makefile entrypoints for the same checks.

## Local Commands

```bash
PYTHON=.venv/bin/python make ci
```

Runs unit tests without camera/GUI and then validates ML artifacts.

```bash
PYTHON=.venv/bin/python make ml-smoke
```

Loads tracked model files, rebuilds synthetic feature vectors from metadata, runs
`predict_proba`, validates finite probabilities and writes:

```text
outputs/ci/ml_smoke_report.json
```

```bash
make docker-build
make docker-ml-smoke
make docker-ci
```

Builds the local `gestureflow-runtime:local` image and runs the same checks in a
clean Linux container. This validates the project outside the developer's macOS
virtualenv and makes CD more realistic.

The Docker runtime is built as `linux/amd64`. This is intentional: current
MediaPipe wheels required by the project are available for Linux x86_64, while
Linux arm64 Docker builds on Apple Silicon do not provide the required
`mediapipe>=0.10.33` package. Docker Desktop runs the amd64 image through
emulation on Apple Silicon Macs.

The image installs `torch` separately from the PyTorch CPU wheel index and then
installs the remaining project requirements. This avoids pulling CUDA/GPU
packages into a headless ML smoke image and keeps the CD artifact aligned with
the actual CPU inference path.

The Dockerfile is multi-stage: compiler/build tooling is used only in the
builder stage, while the final `runtime` stage contains the installed Python
environment, system runtime libraries and project files.

```bash
make docker-mlflow
```

Starts an MLflow UI service from the Docker runtime image on
`http://127.0.0.1:5000`.

```bash
make macos-app
```

Builds a macOS `GestureFlow.app` archive through
`packaging/macos/build_app.sh`. This target is intended for macOS machines and
GitHub-hosted macOS runners.

## GitHub Actions

### `.github/workflows/ci.yml`

Trigger:

- pull request;
- push to `contest_version`, `dev`, `jmlc`, `main`.

Checks:

- install project dependencies;
- run `make ci`;
- upload `outputs/ci` as CI artifacts.

Purpose:

- protect code quality;
- prove the project can be installed from a clean checkout;
- catch broken imports, tests and model metadata before merge.

### `.github/workflows/ml-smoke.yml`

Trigger:

- manual `workflow_dispatch`;
- daily schedule;
- push to `contest_version` when ML/runtime/model files change.

Checks:

- load static model artifact;
- load dynamic `sequence_mlp` artifact;
- load dynamic `sequence_lstm_backbone` artifact;
- validate classes, feature dim/mode, rejection/prototype files;
- run inference without camera.

Purpose:

- protect the ML pipeline separately from UI work;
- detect broken serialized models or metadata drift;
- keep a machine-readable report as artifact.

### `.github/workflows/docker-runtime.yml`

Trigger:

- manual `workflow_dispatch`;
- push to `contest_version` when Docker/runtime/model files change;
- tag push `v*`.

Checks:

- build `gestureflow-runtime:<sha>` from `Dockerfile`;
- run `make ml-smoke` inside the container;
- upload Docker image metadata as artifact.

Purpose:

- prove the ML/runtime layer is reproducible outside the local machine;
- detect missing Linux system packages before release;
- make CD relevant for ML/edge/server handoff scenarios.

### `.github/workflows/desktop-release.yml`

Trigger:

- tag push `v*`;
- manual run.

Output:

- `gestureflow-<version>.tar.gz`;
- draft GitHub Release for tag builds.

Current release is a handoff bundle, not yet a native installer. It contains
source code, tracked model artifacts, configs and docs, while excluding local
raw datasets, MLflow DB, virtualenvs and generated outputs.

### `.github/workflows/macos-app.yml`

Trigger:

- manual `workflow_dispatch`;
- push to `contest_version` when app/runtime/model/package files change;
- tag push `v*`.

Checks:

- install project dependencies on a macOS runner;
- run `scripts.ml_smoke` before packaging;
- pre-bundle the Flet desktop client archive when available;
- build `GestureFlow.app` with PyInstaller/Flet;
- add macOS camera/microphone/automation usage descriptions to `Info.plist`;
- ad-hoc sign the app;
- upload a `.zip` artifact and attach it to draft releases on tags.

Purpose:

- create a user-facing macOS artifact, not just a source bundle;
- verify that model files and runtime dependencies can be packaged together;
- make CD relevant for real desktop delivery and demos.

## Desktop Deployment Logic

GestureFlow is currently a desktop Flet application with local camera access and
local model files. CI cannot honestly test a real webcam stream on GitHub-hosted
runners, so the pipeline is split:

- CI verifies code and ML artifacts without hardware.
- Docker verifies a clean Linux runtime for headless ML checks.
- Live camera quality is verified through the project live-evaluation mode.
- Release workflow packages a reproducible project bundle.
- macOS app workflow produces a downloadable `.app/.zip` artifact.

Docker is not the primary way to distribute the macOS desktop camera app to
friends. Docker Desktop on macOS does not provide a smooth native webcam/window
experience for this Flet application because the camera and desktop window live
outside the Linux container. For sharing the actual app, the better CD target is
a native macOS `.app`/`.zip` bundle. Docker remains valuable as a reproducible
runtime artifact for ML smoke, MLOps, CI and future edge/server variants.

Next desktop CD step:

- test the macOS artifact on a second physical Mac;
- add Developer ID signing and Apple notarization;
- optionally add Windows artifact later;
- attach richer model metrics to release assets.

## Mobile Deployment Logic

For mobile, CI/CD should keep ML core independent from the UI shell:

- train/export model artifacts as versioned assets;
- run ML smoke on the exported assets;
- mobile app consumes the model bundle;
- live camera tests run on physical devices or device farm.

The same `scripts.ml_smoke` concept remains useful: it validates the model
contract before the mobile app ships it.

## Robotics Deployment Logic

For robotics or edge devices, deployment should be artifact-based:

- CI validates Python ML/runtime code;
- ML smoke validates model compatibility;
- CD publishes a model bundle or Docker image;
- robot updates model/runtime by version;
- hardware-in-the-loop tests run on a self-hosted runner with camera/robot
  attached.

GitHub-hosted runners are not enough for robotics behavior; they only validate
the software contract.

## JMLC Value

This layer directly supports the contest criteria:

- engineering: GitHub Actions, repeatable checks, release artifacts;
- ML: model artifact validation, feature/metadata consistency;
- MLOps: automated reports for model health;
- product: safer delivery path for desktop now and mobile/robotics later.
