# Live Direction Error Analysis

Date: `2026-06-27`

Source:

- `~/.dplm/logs/live_evaluation.jsonl`
- MLflow experiment `GestureBind`
- Runs:
  - `live-swipe_up-auto-knn`
  - `live-swipe_down-auto-knn`
  - `live-swipe_left-auto-knn`

## Summary

Fresh dynamic tests show that the main failure mode changed.

Before natural swipe and route fixes, `swipe_up` often fell through to static
labels such as `gun`. In the latest runs, all `30/30` dynamic attempts were
routed as `dynamic`.

Current bottleneck:

| Gesture | Score | Static hijack | Main error |
|---|---:|---:|---|
| `swipe_up` | `7/10` | `0%` | predicted as `swipe_down` |
| `swipe_down` | `7/10` | `0%` | predicted as `swipe_up` or `swipe_left` |
| `swipe_left` | `10/10` | `0%` | none |

Interpretation: routing is good enough for the next stage. The remaining issue
is direction confusion in vertical dynamic gestures.

## `swipe_up`

All attempts used `route=dynamic`, `dynamic_axis=vertical`.

| Group | Count | Predicted | End reason | Scale mean | Axis ratio mean | Straightness mean | Frames mean |
|---|---:|---|---|---:|---:|---:|---:|
| correct | `7` | `swipe_up` | `velocity_drop=7` | `0.585` | `24.641` | `0.956` | `13.4` |
| wrong | `3` | `swipe_down` | `hand_lost=2`, `velocity_drop=1` | `0.171` | `13.607` | `0.810` | `11.0` |

Important attempts:

| Attempt | Result | Predicted | End reason | Axis ratio | Straightness | Scale | Frames |
|---:|---|---|---|---:|---:|---:|---:|
| 5 | wrong | `swipe_down` | `hand_lost` | `32.231` | `0.977` | `0.000` | `10` |
| 7 | wrong | `swipe_down` | `velocity_drop` | `4.496` | `0.470` | `0.514` | `14` |
| 8 | wrong | `swipe_down` | `hand_lost` | `4.093` | `0.984` | `0.000` | `9` |

Conclusion:

- Correct `swipe_up` is usually clean: high straightness and consistent
  `velocity_drop`.
- One wrong case has very low straightness (`0.470`), which means the path was
  noisy or curved.
- Two wrong cases ended by `hand_lost` with `dynamic_motion_scale=0.000`.
  This suggests that when the hand leaves the frame too abruptly, the final
  direction can be captured from an unstable segment edge.

## `swipe_down`

All attempts used `route=dynamic`; `9/10` were vertical, `1/10` became
horizontal and was predicted as `swipe_left`.

| Group | Count | Predicted | End reason | Scale mean | Axis ratio mean | Straightness mean | Frames mean |
|---|---:|---|---|---:|---:|---:|---:|
| correct | `7` | `swipe_down` | `hand_lost=5`, `still=1`, `velocity_drop=1` | `0.149` | `10.037` | `0.941` | `11.3` |
| wrong | `3` | `swipe_up=2`, `swipe_left=1` | `velocity_drop=3` | `0.525` | `4.202` | `0.785` | `9.7` |

Important attempts:

| Attempt | Result | Predicted | Axis | Direction | Axis ratio | Straightness | Scale |
|---:|---|---|---|---|---:|---:|---:|
| 7 | wrong | `swipe_up` | vertical | up | `4.283` | `0.866` | `0.522` |
| 8 | wrong | `swipe_left` | horizontal | left | `2.908` | `0.870` | `0.520` |
| 10 | wrong | `swipe_up` | vertical | up | `5.414` | `0.618` | `0.534` |

Conclusion:

- Wrong `swipe_down` attempts have lower axis ratio and lower straightness than
  correct attempts.
- The `swipe_left` error is especially informative: the segment became
  `dynamic_axis=horizontal`, so the user motion likely had a horizontal tail or
  diagonal component.
- The model confidence is not useful as a guard here: `dynamic_model_confidence`
  is `1.0` for both correct and wrong attempts.

## Main Findings

1. Static hijack is not the current problem.
   - Latest `live_static_hijack_rate=0.0` for `swipe_up`, `swipe_down`,
     `swipe_left`.

2. Direction quality is the current problem.
   - `swipe_up/down` both have `live_wrong_dynamic_direction_rate=0.3`.

3. KNN confidence is not calibrated.
   - Wrong attempts often have `dynamic_model_confidence=1.0`.
   - Raising the confidence threshold will not solve the vertical confusion.

4. Trajectory quality matters more than model confidence.
   - Wrong vertical attempts tend to have lower `dynamic_axis_ratio` and/or
     lower `dynamic_straightness`.

5. Segment ending matters.
   - `hand_lost` works for natural swipe, but abrupt hand removal can create
     unstable final direction for `swipe_up`.
   - `velocity_drop` is generally clean when the movement is straight, but it
     can also capture a return/bounce motion.

## Hypothesis For Next 20-Attempt Test

If the user performs vertical swipes with a clear straight path and avoids a
return movement through the camera frame, then:

- `swipe_up/down` recall should improve above `80%`;
- `live_wrong_dynamic_direction_rate` should drop below `20%`;
- `live_static_hijack_rate` should remain `0%`.

If the error remains around `30%`, the next code change should strengthen the
direction gate:

- require higher vertical dominance for vertical labels;
- reject or delay segments with low straightness;
- add special handling for return-motion/bounce after a swipe;
- compare first-half vs second-half trajectory direction before accepting a
  vertical label.

## Protocol For Next Test

Run in `auto` mode with dynamic profile `knn`.

For each class:

| Label | Attempts | Goal |
|---|---:|---|
| `swipe_up` | `20` | check vertical direction confusion |
| `swipe_down` | `20` | check vertical direction confusion |
| `swipe_left` | `10` | baseline regression check |

Execution protocol:

1. Keep the hand fully visible at the start.
2. Move in one straight stroke.
3. Do not hold the final pose.
4. Do not immediately return through the camera frame.
5. For `swipe_up`, start lower and finish higher.
6. For `swipe_down`, start higher and finish lower.
7. Keep the same camera distance and roughly the same screen zone for all
   attempts.

After the run, compare:

- `live_accuracy`
- `live_recall`
- `live_wrong_dynamic_direction_rate`
- `live_static_hijack_rate`
- correct vs wrong `dynamic_axis_ratio`
- correct vs wrong `dynamic_straightness`
- `dynamic_end_reason` distribution
