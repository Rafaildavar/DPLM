# GestureFlow CI/CD

## Status

Implemented first production-oriented CI/CD layer:

- GitHub Actions CI for pull requests and pushes.
- Separate ML smoke workflow for model artifact checks.
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

## Desktop Deployment Logic

GestureFlow is currently a desktop Flet application with local camera access and
local model files. CI cannot honestly test a real webcam stream on GitHub-hosted
runners, so the pipeline is split:

- CI verifies code and ML artifacts without hardware.
- Live camera quality is verified through the project live-evaluation mode.
- Release workflow packages a reproducible project bundle.

Next desktop CD step:

- add PyInstaller/Flet native packaging;
- build `.app`/`.zip` for macOS;
- optionally add Windows artifact later;
- attach ML smoke report and model metrics to release.

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
