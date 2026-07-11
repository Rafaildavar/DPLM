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

Validates the tracked repository contract, runs tests without camera/GUI access
and then validates release ML artifacts.

```bash
PYTHON=.venv/bin/python make repo-hygiene
```

Fails when generated/private data, a retired UI path or an unexpected model
artifact enters the Git index. It also checks that every required release file
is present.

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

1. Repository hygiene and unit tests run `make repo-hygiene test-unit-ci`.
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

- push to `contest_version` for a pre-release artifact build;
- tag push `v*`;
- manual run.

Output:

- `gesturebind-<version>.tar.gz`;
- `GestureBind-macos-<version>.zip` containing `GestureBind.app`;
- GitHub Release for tag builds.

The workflow produces both a `git archive` source/model bundle and a macOS ZIP
containing `GestureBind.app`. The source archive contains only source code,
allowlisted tracked model artifacts, configs and release documentation
from the tagged commit. Local datasets, generated reports, virtual
environments, MLflow state and output folders cannot enter the archive.

The macOS job reads optional repository variables
`GESTUREBIND_TELEMETRY_ENDPOINT` and `GESTUREBIND_TELEMETRY_PROJECT_KEY` and
embeds them as the release telemetry destination. They do not enable collection
without user consent. Deployment details are in `docs/TELEMETRY.md`.

## Release Gate

Before creating a tag:

1. Run `PYTHON=.venv/bin/python make ci`.
2. Confirm that `python -m scripts.check_repository_hygiene` succeeds.
3. Check that `git status` has no unexpected local files staged.
4. Tag the release with `vX.Y.Z`.
5. Let `desktop-release.yml` create the GitHub release bundle.

For the current release branch:

```bash
git push origin contest_version
git tag -a v0.8.1 -m "GestureBind v0.8.1"
git push origin v0.8.1
```

First push `contest_version` and confirm that both release jobs are green. The
tag command must then run on that exact commit. A branch build uploads temporary
workflow artifacts but does not create a GitHub Release; the tag build publishes
both artifacts together with `RELEASE_NOTES.md`.
