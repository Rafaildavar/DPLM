# IPN Hand Research Analysis For GestureFlow

Status: `research accepted`

Date: `2026-06-28`

Sources:

- GitHub: `https://github.com/GibranBenitez/IPN-hand`
- Project page: `https://gibranbenitez.github.io/IPN_Hand/`
- Paper: `https://arxiv.org/abs/2005.02134`
- Online test script:
  `https://raw.githubusercontent.com/GibranBenitez/IPN-hand/master/online_test.py`
- Online dataset loader:
  `https://raw.githubusercontent.com/GibranBenitez/IPN-hand/master/datasets/ipn_online.py`
- Online run config:
  `https://raw.githubusercontent.com/GibranBenitez/IPN-hand/master/tests/run_online_ipnTest.sh`

## Why IPN Hand Matters

IPN Hand is much closer to GestureFlow than language-sign datasets or large
static image datasets. It is designed for real-time continuous hand gesture
recognition from RGB video, with non-gesture intervals and start/end frame
annotations.

Dataset details from the project README:

- RGB-only videos;
- `640x480`, `30 fps`;
- subjects record gestures with their own PCs, so distance to camera varies;
- 13 gesture classes plus non-gesture;
- 5649 total instances:
  - 4218 gesture instances;
  - 1431 non-gesture instances.

The important match for GestureFlow:

- IPN contains directional dynamic gestures:
  `Throw up`, `Throw down`, `Throw left`, `Throw right`;
- it also contains non-gesture and unrelated dynamic gestures;
- it evaluates continuous recognition, not only isolated classification.

## What We Should Take

### 1. Dataset Subset For Dynamic Negatives

Use IPN as external evidence for dynamic reject-layer:

| IPN class | GestureFlow role |
|---|---|
| `D0X Non-gesture` | `negative_external_ipn_dynamic` |
| `Open twice`, `Double click`, `Zoom in/out` | other dynamic motion negatives |
| `Pointing one/two fingers` | hold/static-ish negatives for dynamic model |
| `Throw up/down/left/right` | optional validation/pretraining, not production positives by default |

Do not import all IPN classes as new GestureFlow commands. The raw class label
should be preserved in metadata, but the training label should be mapped to a
controlled negative or experimental auxiliary class.

### 2. Start/End Frame Segmentation

IPN annotations contain gesture intervals. We should use them to build
automatic samples instead of manually deciding where a dynamic gesture begins
and ends.

GestureFlow conversion target:

```text
data/external/ipn_hand/<ipn_label>/sample_0000.npy
```

Expected converted array:

```text
(frames, 44)
```

Where:

- first 42 values: MediaPipe normalized hand landmarks;
- last 2 values: global wrist/screen coordinate used by dynamic trajectory
  features.

### 3. Two-Stage Online Recognition

IPN uses a detector model as a switch and a classifier model only while the
detector is active.

GestureFlow equivalent:

```text
hand landmarks
-> dynamic intent detector: active / inactive / reject
-> dynamic class model: swipe_left/up/down/right
-> router + cooldown + command execution
```

This is more useful than adding a heavier classifier first. It directly targets
our current live problem: avoid firing a dynamic gesture when the user is only
moving their hand casually.

### 4. Temporal Probability Smoothing

IPN keeps queues for detector and classifier probabilities and supports:

- raw probabilities;
- median;
- moving average;
- exponentially weighted moving average.

Their online test config uses:

- detector window: `8` frames;
- classifier window: `32` frames;
- detector moving-average queue: `4`;
- classifier moving-average queue: `16`;
- stride: `1`.

GestureFlow adaptation:

- keep our natural swipe segmenter;
- add smoothed detector confidence over a short queue;
- add smoothed class confidence over an active segment;
- emit only when top1 confidence and top1/top2 margin are stable.

### 5. Cumulative Evidence Over The Active Segment

IPN accumulates classifier evidence while a gesture is active and supports early
prediction if the leading class is far enough ahead.

GestureFlow adaptation:

- for every active dynamic segment, accumulate:
  - motion direction confidence;
  - model probability;
  - top1/top2 margin;
  - negative probability;
  - segment length;
- emit at segment end if final confidence passes threshold;
- optionally emit early only when margin is high and direction is stable.

### 6. Continuous-Session Metric

IPN evaluates predicted gesture sequences with Levenshtein distance.

GestureFlow should add a live continuous-session metric:

```text
expected sequence: swipe_up, swipe_left, swipe_down
predicted sequence: swipe_up, swipe_down
sequence_edit_distance = 1
sequence_accuracy = 1 - distance / len(expected)
```

This is stronger for JMLC than only per-attempt accuracy because it measures
real command-stream behavior.

## What We Should Not Copy

Do not copy IPN's heavy RGB 3D-CNN architecture now:

- it is GPU-oriented;
- it needs raw RGB frames and large training data;
- it uses ResNet/ResNeXt-style video models;
- it would complicate the app and contest demo;
- our live feature space is MediaPipe landmarks, not RGB clips.

Better approach:

- keep MediaPipe as feature extractor;
- use IPN to improve segmentation, negative evidence and evaluation;
- only consider TCN/LSTM/Transformer after enough real dynamic samples exist.

## Model Experiments To Add

### Immediate

| Experiment | Model | Data |
|---|---|---|
| Dynamic intent detector | LogisticRegression / SVM / ExtraTrees | our swipes vs IPN non-gesture/other-dynamic negatives |
| Dynamic reject classifier | current KNN + negative IPN labels | our swipes + IPN negatives |
| Smoothed class decision | current dynamic KNN + MA/EWMA queues | live stream only |

### Next

| Experiment | Model | Why |
|---|---|---|
| MLP dynamic rejector | small MLP over dynamic_stats | nonlinear boundary without huge model |
| TCN over landmarks | small temporal conv net | natural sequence model, lighter than Transformer |
| LSTM/GRU | sequence baseline | useful only after enough real samples |

### Later

| Experiment | Model | Condition |
|---|---|---|
| Transformer | tiny landmark transformer | only if dataset grows and current models plateau |
| RGB video model | 3D CNN / VideoMAE | only if MediaPipe landmarks are the bottleneck |

## Proposed Implementation Plan

1. Add IPN converter:
   - read annotation intervals;
   - sample only a small subset;
   - run MediaPipe;
   - save `(frames, 44)` arrays;
   - write metadata with original IPN label and frame interval.

   Implemented entrypoint:

   ```bash
   PYTHON=.venv/bin/python make ipn-convert
   ```

   Default expected local layout:

   ```text
   data/raw/ipn_hand/frames/<video_id>/<video_id>_000001.jpg
   data/raw/ipn_hand/annotations/ipnall.json
   ```

2. Add external dynamic negative variant:
   - `baseline_internal`;
   - `ipn_external`;
   - `combined_external`.

3. Train and compare:
   - current dynamic KNN;
   - dynamic KNN + IPN negatives;
   - binary dynamic intent detector;
   - MLP negative classifier if enough samples exist.

4. Add live metrics:
   - `live_dynamic_intent_precision`;
   - `live_dynamic_intent_recall`;
   - `live_false_dynamic_activation_rate`;
   - `live_sequence_edit_distance`;
   - `live_sequence_accuracy`;
   - `live_dynamic_start_latency_frames`;
   - `live_dynamic_end_latency_frames`;
   - `live_dynamic_detector_active_ratio`.

5. Promote a variant only if live metrics improve:
   - false dynamic activation decreases;
   - dynamic recall does not fall below current baseline;
   - command latency remains acceptable.

## Recommendation

Use IPN Hand as the main external dataset for dynamic recognition research.

Priority:

1. Convert a small IPN subset to landmarks.
2. Use `D0X` and unrelated dynamic gestures as automatic negatives.
3. Add a trained dynamic intent detector before the dynamic classifier.
4. Add IPN-style smoothing and cumulative evidence.
5. Add sequence-level live metrics in MLflow.

This gives GestureFlow a stronger ML story without abandoning the personalized
MediaPipe-landmark pipeline.
