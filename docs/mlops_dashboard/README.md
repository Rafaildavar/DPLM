# GestureFlow MLOps Dashboard

This folder contains the local HTML observability dashboard for the JMLC branch.

Install project dependencies first:

```bash
pip install -r requirements.txt
```

Generate it after recording live-evaluation runs:

```bash
make mlops-dashboard
```

Open:

```text
docs/mlops_dashboard/index.html
```

The dashboard tracks:

- dataset classes and sample counts by taxonomy type;
- live-evaluation accuracy by label;
- route and dynamic decision sources;
- negative rejection count;
- runtime latency from `runtime_performance.jsonl`;
- model artifact size, modified time and SHA-256 fingerprint.

MLflow tracks training and live-evaluation experiments separately:

```bash
PYTHON=.venv/bin/python make mlflow-ui
```

Then open `http://127.0.0.1:5000`.

Training runs are logged to the local `GestureFlow` experiment with
`sqlite:///mlflow.db` as the tracking backend. MLflow stores run parameters,
sample/class counts, train accuracy and model artifacts.

Live tests from the Flet interface are logged automatically when a test reaches
its target attempt count or is stopped. Look for runs named
`live-<expected_label>-<mode>-<dynamic_profile>`. The key live metrics are:

- `live_accuracy` and `live_recall`;
- `live_static_hijack_rate` for dynamic gestures routed as static;
- `live_wrong_dynamic_direction_rate` for swipe direction confusion;
- `live_negative_false_positive_rate` for negative tests;
- `live_latency_avg_s`, `live_latency_p50_s`, `live_latency_p95_s`;
- `live_end_reason_*` and `live_decision_*` counters.

Each live run also stores `live_evaluation_run.json` with raw attempts and route
metadata. New live runs also store an artifact bundle under
`Artifacts / live_evaluation /`:

- `index.html` with a readable run report;
- `charts/*.svg` for quality, outcomes, routes, end reasons, runtime and
  attempt timeline;
- `attempts.csv` and `metrics.csv`;
- `runtime_performance.json` when runtime windows match the live run.

MLflow's built-in System metrics tab tracks per-run resource metrics when
`psutil` is installed. For always-on system monitoring the planned production
path is Prometheus + Grafana; MLflow remains the experiment tracker. The HTML
dashboard is the current system snapshot; MLflow is the experiment history.
