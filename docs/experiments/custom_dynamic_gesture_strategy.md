# Custom Dynamic Gesture Strategy

Status: `accepted architecture direction`

Date: `2026-06-28`

## Core Requirement

GestureBind must support user-defined dynamic gestures, not only predefined
swipes.

Current swipes are useful for debugging and early evaluation, but they must not
define the final recognition architecture. A user should be able to create a
gesture such as:

- draw a small circle;
- move hand in a zigzag;
- open and close fingers while moving;
- move from left to right and then up;
- any personalized command gesture.

## Consequence

Swipe-specific logic such as:

```text
dx < 0 -> swipe_left
dy > 0 -> swipe_down
```

can only be an auxiliary rule for known directional primitives. It must not be
the universal dynamic classifier.

The universal pipeline should be:

```text
MediaPipe landmarks
-> generic dynamic intent detector
-> sequence normalization
-> user-trained dynamic classifier
-> open-set reject layer
-> router / command execution
```

## Data Representation

For arbitrary dynamic gestures, the saved sample should remain a full sequence:

```text
(frames, 44)
```

Where:

- `42` values are hand landmarks;
- `2` values are global wrist/screen position;
- no temporal averaging is allowed before saving.

Training-time features should include:

- resampled landmark sequence;
- wrist trajectory;
- velocity and acceleration statistics;
- path length and displacement;
- direction changes;
- pose deltas between start/middle/end;
- optional pairwise landmark distances/angles for hand-shape changes.

## Model Strategy

### Few-shot baseline

When the user records only `5-20` examples per gesture, use models that work
with small data:

| Model | Role |
|---|---|
| KNN on normalized sequence features | baseline personalized classifier |
| prototype/centroid distance | explainable reject and nearest-example view |
| DTW-like sequence distance | useful for gestures with timing variation |
| SVM / LogisticRegression | simple supervised baseline |
| ExtraTrees | robust non-linear baseline for tabular features |

### After more data

Use neural sequence models only after enough real samples exist:

| Model | When |
|---|---|
| small MLP over dynamic stats | enough negatives and positives |
| TCN over landmarks | dozens of samples per dynamic class |
| GRU/LSTM | sequence baseline after dataset growth |
| tiny Transformer | later research, not first contest dependency |

## Dynamic Intent Detector

The first stage should not answer "which swipe is this?".

It should answer:

```text
is there an intentional dynamic gesture segment right now?
```

Possible labels:

- `active_dynamic`;
- `idle_hand`;
- `static_hold`;
- `random_motion`;
- `partial_motion`;
- `unknown_dynamic`.

This detector can be trained with:

- user's positive dynamic samples;
- synthetic dynamic negatives;
- IPN Hand non-gesture / unrelated gesture intervals;
- live negative attempts from MLflow logs.

## IPN Hand Role

IPN Hand should not turn GestureBind into a fixed IPN-class recognizer.

Use IPN as:

- external dynamic negative evidence;
- segmentation/continuous-recognition reference;
- robustness validation against camera distance and natural motion;
- source for non-gesture and unrelated-motion examples.

Do not treat IPN `Throw left` as the definitive production definition of
`swipe_left`. User-recorded examples remain the source of truth.

## Runtime Decision

For known directional swipes, the system may combine:

```text
motion primitive score + model probability + reject score
```

For arbitrary gestures, the system must rely on:

```text
learned sequence/prototype classifier + reject layer
```

This keeps swipe quality high without blocking future custom gestures.

## Live Metrics

Required metrics for arbitrary dynamic gestures:

- per-class `live_recall`;
- `live_false_dynamic_activation_rate`;
- `live_unknown_dynamic_reject_rate`;
- `live_dynamic_open_set_false_positive_rate`;
- `live_sample_efficiency`: quality after 5/10/20 samples;
- `live_sequence_accuracy` for command streams;
- `live_start_latency_frames`;
- `live_end_latency_frames`;
- `live_confusion_static_vs_dynamic`.

## Implementation Plan

1. Keep current swipe-specific motion rule as an optional primitive scorer.
2. Add a generic dynamic intent detector.
3. Add normalized sequence/prototype features for arbitrary dynamic gestures.
4. Train dynamic models on all user-defined dynamic classes, not only swipes.
5. Add open-set rejection for unknown dynamic motion.
6. Use IPN Hand only as negatives/validation unless explicitly running an
   external pretraining experiment.
7. Report both swipe-specific and arbitrary-dynamic metrics in MLflow.

## Decision

The final contest story should be:

GestureBind is a personalized gesture-recognition system. Swipes are the first
test classes, but the ML pipeline is designed for user-defined static,
quasi-static and dynamic gestures. Public datasets such as IPN Hand are used to
improve rejection and robustness, while user-recorded samples define the actual
commands.
