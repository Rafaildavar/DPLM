# GestureBind CI/CD

## Status

GestureBind uses a compact release-oriented pipeline:

- `ci.yml` for pull requests and active development branches;
- `main-pipeline.yml` as the ordered quality gate for `main`;
- `ml-smoke.yml` for scheduled and manual model artifact checks;
- `docker-runtime.yml` for reproducible headless runtime validation;
- `desktop-release.yml` for tag-based handoff bundles.

## Local Commands

```bash
PYTHON=.venv/bin/python make ci
```

Runs unit tests without camera/GUI access and then validates release ML
artifacts.

```bash
PYTHON=.venv/bin/python make ml-smoke
```

Loads tracked model files, rebuilds synthetic feature vectors from metadata,
runs `predict_proba`, validates finite probabilities and writes:

```text
outputs/ci/ml_smoke_report.json
```

The release smoke set contains:

- `static_default`: `models/knn.pkl`;
- `dynamic_landmark_lstm_backbone`: production dynamic LSTM-backbone model;
- intent-gate metadata used by the runtime router.

```bash
make docker-build
make docker-ml-smoke
make docker-ci
```

Builds the local `gesturebind-runtime:local` image and runs the same checks in a
clean container. This catches missing system packages and dependency drift
before release.

```bash
make docker-mlflow
```

Starts an MLflow UI service from the Docker runtime image on
`http://127.0.0.1:5000`.

## GitHub Actions

### `.github/workflows/ci.yml`

Trigger:

- pull request;
- push to `contest_version` or `dev`.

Checks:

- install project dependencies;
- run `make ci`;
- upload `outputs/ci` as artifacts.

### `.github/workflows/main-pipeline.yml`

Trigger:

- manual `workflow_dispatch`;
- push to `main`.

Sequence:

1. Unit tests run `make test-unit-ci`.
2. ML smoke runs `make ml-smoke`.
3. Docker runtime builds the image and runs ML smoke inside the container.

The sequence keeps `main` as an ordered quality gate: Docker validation starts
only after unit and direct model checks pass.

### `.github/workflows/ml-smoke.yml`

Trigger:

- manual `workflow_dispatch`;
- daily schedule;
- push to `contest_version` when runtime/model files change.

Purpose:

- detect broken serialized models or metadata drift;
- keep a machine-readable model-health report as an artifact;
- validate release model files without camera or GUI access.

### `.github/workflows/docker-runtime.yml`

Trigger:

- manual `workflow_dispatch`;
- push to `contest_version` when Docker/runtime/model files change;
- tag push `v*`.

Purpose:

- prove the ML/runtime layer is reproducible from a clean checkout;
- run release smoke checks inside the runtime image;
- upload image metadata for debugging failed builds.

### `.github/workflows/desktop-release.yml`

Trigger:

- tag push `v*`;
- manual run.

Output:

- `gesturebind-<version>.tar.gz`;
- draft GitHub Release for tag builds.

The bundle contains source code, tracked model artifacts, configs and release
documentation. Local datasets, generated reports, virtual environments, MLflow
state and output folders are excluded.

## Release Gate

Before creating a tag:

1. Run `PYTHON=.venv/bin/python make ci`.
2. Check that `models/` contains only release artifacts.
3. Check that `git status` has no unexpected local files staged.
4. Tag the release with `vX.Y.Z`.
5. Let `desktop-release.yml` create the draft release bundle.
