# Live Rejection Test Protocol

Date: `2026-06-28`

Purpose: собрать реальные live-метрики из приложения для выбора static
rejection strategy. Offline benchmark используется только как гипотеза; решение
принимаем по live runs в MLflow.

## 1. Start MLflow

```bash
cd /Users/remi/Developer/GUAP/DPLM
PYTHON=.venv/bin/python make mlflow-ui
```

Open:

```text
http://127.0.0.1:5000
```

If port `5000` is busy:

```bash
lsof -i :5000
kill <PID>
```

## 2. Prepare verifier artifact

Run this after retraining static data or changing negative samples:

```bash
PYTHON=.venv/bin/python make static-rejection-verifiers
```

Expected local artifact:

```text
models/static_rejection_verifiers.pkl
```

This file is intentionally local/ignored by git. It is used by live inference.

## 3. Start the app

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m app.flet_app.main
```

Recommended live settings:

| Field | Value |
|---|---|
| Model | `auto` |
| Dynamic | `knn` |
| Threshold | `0.60` |
| Attempts | `20` |

## 4. Static positive tests

Goal: проверить, что rejection layer не убивает настоящие static gestures.

Test labels:

```text
gun
three
hend
up
```

For each label:

1. Select `Reject=open_set_policy`.
2. Run `20` attempts, `Timeout=0`.
3. Repeat with `Reject=one_vs_rest_logreg`.

Primary metrics:

| Metric | Target |
|---|---:|
| `live_accuracy` | `>= 0.80` |
| `live_static_reject_rate` | explain if high |
| `live_static_false_positive_rate` | should be `0` for positive tests |

## 5. Negative / false trigger tests

Goal: проверить, что система не распознает случайные или почти-жесты.

Test labels:

```text
no_gesture_static
partial_swipe
wrong_axis_motion
```

For each label:

1. Select `Reject=open_set_policy`.
2. Set `Timeout=1.2`.
3. Do random hand poses, half-gestures, almost-gun, almost-three.
4. Do not press skip when the system is silent. Wait for timeout: silence is a
   correct rejection.
5. Repeat with `Reject=one_vs_rest_logreg`.

Primary metrics:

| Metric | Target |
|---|---:|
| `live_negative_false_positive_rate` | `<= 0.10` |
| `live_static_false_positive_rate` | `<= 0.10` |
| `live_static_reject_rate` | high is good here |

## 6. Dynamic sanity tests

Goal: проверить, что static rejection changes did not break auto routing.

Test labels:

```text
swipe_left
swipe_up
swipe_down
```

Use:

| Field | Value |
|---|---|
| Model | `auto` |
| Dynamic | `knn` |
| Reject | `open_set_policy`, then `one_vs_rest_logreg` |
| Timeout | `0` |
| Attempts | `20` |

Primary metrics:

| Metric | Target |
|---|---:|
| `live_static_hijack_rate` | `0` |
| `live_wrong_dynamic_direction_rate` | as low as possible |
| `live_dynamic_recall` | `>= 0.80` |

## 7. What to inspect in MLflow

Each completed live test creates a run:

```text
live-<expected>-<mode>-<dynamic_profile>-<static_rejection_method>
```

Open the run and inspect:

| Tab | What to check |
|---|---|
| Model metrics | `live_accuracy`, `live_negative_false_positive_rate`, `live_static_false_positive_rate`, `live_static_reject_rate` |
| System metrics | CPU/RAM/runtime metrics collected by MLflow |
| Artifacts | `live_evaluation/index.html` |

Artifact charts:

```text
live_evaluation/charts/quality.svg
live_evaluation/charts/static_rejection.svg
live_evaluation/charts/static_verifier_signals.svg
live_evaluation/charts/routes.svg
live_evaluation/charts/runtime.svg
live_evaluation/charts/attempt_timeline.svg
```

The most important new charts:

| Chart | Why it matters |
|---|---|
| `static_rejection.svg` | shows selected method, decision source and reject reasons |
| `static_verifier_signals.svg` | shows verifier probability/confidence/distance vs thresholds |

## 8. Generate local report after testing

```bash
.venv/bin/python -m scripts.live_evaluation_report \
  --out-json docs/experiments/live_evaluation_report.json \
  --out-md docs/experiments/live_evaluation_report.md
```

Optional local dashboard:

```bash
PYTHON=.venv/bin/python make mlops-dashboard
```

Open:

```text
docs/mlops_dashboard/index.html
```

## 9. Decision rule

Pick the method that wins on live data, not offline data.

Recommended winner criteria:

| Scenario | Rule |
|---|---|
| Static positive gestures | accuracy/recall stays `>= 80%` |
| Negative tests | false positive rate `<= 10%` |
| Dynamic tests | static hijack remains `0%` |
| UX | no obvious latency or unstable flickering |

Current hypothesis:

```text
one_vs_rest_logreg should reduce random static false triggers better than open_set_policy,
but it must prove this in live evaluation.
```
