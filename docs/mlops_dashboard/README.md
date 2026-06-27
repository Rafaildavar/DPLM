# GestureFlow MLOps Dashboard

This folder contains the local HTML observability dashboard for the JMLC branch.

Generate it after recording live-evaluation runs:

```bash
python -m scripts.mlops_dashboard
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

This is the first local MLOps layer. The next step is adding experiment runs
with model parameters, dataset version and metrics snapshots for every training.
