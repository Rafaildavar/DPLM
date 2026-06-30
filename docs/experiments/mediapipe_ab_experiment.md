# MediaPipe A/B Experiment

Дата: `2026-06-30`

Цель: проверить, является ли качество live-распознавания ограниченным не
классификатором, а слоем `camera -> MediaPipe -> landmarks -> segmenter`.

## Что добавлено

- Реальный timestamp кадра передается из camera capture loop в MediaPipe.
- Добавлены A/B-профили MediaPipe thresholds через `DPLM_MEDIAPIPE_PROFILE`.
- Добавлено EMA-сглаживание landmarks через `DPLM_MEDIAPIPE_SMOOTHING_ALPHA`.
- Добавлен `hand_lost_grace`: короткая потеря руки на 1-2 кадра не завершает
  dynamic-сегмент сразу.
- В runtime logs и MLflow live evaluation пишутся MediaPipe quality metrics:
  `hand_detected_rate`, `world_landmarks_available_rate`,
  `landmark_z_available_rate`, `hand_bbox_diag_avg`,
  `primary_wrist_step_avg`, `hand_lost_streak_max`,
  `mediapipe_real_timestamp_rate`.

## A/B профили

| Профиль | detection | presence | tracking | Гипотеза |
|---|---:|---:|---:|---|
| `baseline_06` | 0.6 | 0.6 | 0.6 | Текущий стабильный baseline |
| `recall_05` | 0.5 | 0.5 | 0.5 | Больше кадров с рукой, но больше шума |
| `strict_tracking` | 0.7 | 0.5 | 0.7 | Строже tracking, меньше дрожания/ложных рук |
| `redetect_presence` | 0.5 | 0.7 | 0.5 | Чаще переобнаруживать при слабой presence |

Дополнительный параметр:

```bash
export DPLM_MEDIAPIPE_SMOOTHING_ALPHA=0.35
```

`0` выключает сглаживание. `0.25-0.40` - мягкое EMA. Сильно выше пока не
использовать: можно убить быстрые жесты.

Hand-lost policy:

```bash
export DPLM_HAND_LOST_GRACE_FRAMES=2
```

## Как тестировать

Для каждого профиля:

```bash
export DPLM_MEDIAPIPE_PROFILE=baseline_06
export DPLM_MEDIAPIPE_SMOOTHING_ALPHA=0
export DPLM_HAND_LOST_GRACE_FRAMES=2
```

1. Перезапустить Flet-приложение.
2. Включить `auto` routing и текущую production dynamic-модель.
3. Запустить live evaluation:
   - `upandleft`: 20 попыток;
   - `swipe_up`: 20 попыток;
   - `swipe_left`: 20 попыток;
   - `partial_swipe`: 10 попыток;
   - `random_motion`: 10 попыток.
4. После каждого теста смотреть MLflow run:
   - Model metrics: `live_accuracy`, `live_recall`, `live_miss_rate`,
     `live_negative_false_positive_rate`.
   - System/MediaPipe metrics:
     `system_runtime_inference_ms_avg`,
     `system_mediapipe_detection_ms_avg`,
     `system_hand_detected_rate_avg`,
     `system_world_landmarks_available_rate_avg`,
     `system_hand_lost_streak_max`.
   - Artifacts: `live_evaluation/index.html`,
     `charts/mediapipe_quality.svg`, `attempts.csv`.

## Что считаем улучшением

Профиль считается лучше baseline, если:

- live accuracy/recall не ниже baseline;
- miss rate ниже или равен baseline;
- negative false positive rate не растет;
- `hand_lost_streak_max` ниже или стабильнее;
- `system_runtime_inference_ms_avg` и `system_mediapipe_detection_ms_avg`
  не ломают интерактивность.

## Про z / world landmarks

MediaPipe Tasks API возвращает 2D landmarks, image-space `z` и world landmarks.
Сейчас старые обучающие `.npy` содержат в основном 2D + global motion
`(x/y)`, поэтому 3D нельзя честно добавить в модель без новой записи данных.

На этом шаге 3D используется как диагностика:

- проверяем, доступен ли `world_landmarks` в live;
- смотрим `landmark_z_range` и `world_z_range`;
- если 3D стабильно доступен и коррелирует с качеством, следующий эксперимент:
  `dynamic_sequence_world_72` с признаками `x/y/z + global dx/dy/path`.
