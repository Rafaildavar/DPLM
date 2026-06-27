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

MLflow tracks training experiments separately:

```bash
PYTHON=.venv/bin/python make mlflow-ui
```

Then open `http://127.0.0.1:5000`.

Training runs are logged to the local `GestureFlow` experiment with
`sqlite:///mlflow.db` as the tracking backend. MLflow stores run parameters,
sample/class counts, train accuracy and model artifacts. The HTML dashboard is
the current system snapshot; MLflow is the experiment history.
