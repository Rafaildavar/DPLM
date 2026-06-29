# IPN New Packs Deep Analysis

Дата: `2026-06-29`

## Цель

Проверить, помогают ли новые IPN Hand frame packs для GestureFlow:

- как external negative data для динамического rejection;
- как validation/reference data для жестов, похожих на пользовательские свайпы;
- как часть ML/MLOps истории проекта для JMLC.

## Что добавлено в data

В проект добавлены frame packs:

| Pack | Размер | Видео | Файлов в архиве | Подходящих candidate-сегментов |
|---|---:|---:|---:|---:|
| `frames02.tar` | `2.0G` | `40` | `161625` | входит в общий pool |
| `frames03.tar` | `2.2G` | `40` | `166388` | входит в общий pool |
| `frames04.tar` | `1.8G` | `40` | `158484` | `1078` |
| `frames05.tar` | `2.0G` | `40` | `155433` | входит в общий pool |

Общий pool по четырем архивам:

- videos in archives: `160`;
- annotated segments found: `5649`;
- negative candidate segments: `3906`;
- selected smoke-test segments: `240`.

Общий candidate label breakdown:

| IPN label | Candidate | Selected for smoke test |
|---|---:|---:|
| `B0A` | `805` | `40` |
| `B0B` | `802` | `40` |
| `D0X` | `1179` | `40` |
| `G01` | `160` | `17` |
| `G02` | `160` | `17` |
| `G07` | `160` | `17` |
| `G08` | `160` | `18` |
| `G09` | `160` | `17` |
| `G10` | `160` | `17` |
| `G11` | `160` | `17` |

Итеративная распаковка выполнена для `frames04.tar`:

- extract root: `data/raw/ipn_hand/frames04_extracted`;
- extracted size: `2.0G`;
- extracted files: `158444` JPEG;
- extracted videos: `40`;
- free disk after extraction: `6.2G`.

## Как устроен IPN pack

Архив содержит не готовые ML-признаки, а покадровые изображения:

```text
frames/<video_id>/<video_id>_<frame_index>.jpg
```

Аннотация `Annot_List.txt` задает только интервалы:

```text
video,label,id,t_start,t_end,frames
```

То есть для GestureFlow нужно выполнить отдельный feature extraction:

1. выбрать сегмент по `video_id`, `t_start`, `t_end`;
2. взять равномерно sampled кадры;
3. найти руку через MediaPipe;
4. нормализовать landmarks в наш формат `T x 44`;
5. сохранить компактный `.npy` + metadata;
6. проверить sample через prototype/rejection pipeline.

## Маппинг IPN labels

Безопасные external negatives:

| IPN label | Роль | Почему можно использовать как negative |
|---|---|---|
| `D0X` | non-gesture | фоновые интервалы без целевого жеста |
| `B0A`, `B0B` | pointing hold | статичное/квазистатичное указание, не dynamic command |
| `G01`, `G02` | click motion | клики не равны пользовательским свайпам |
| `G07` | open twice | динамика руки, но не текущая команда |
| `G08`, `G09` | double click | динамика, которая должна отвергаться |
| `G10`, `G11` | zoom motion | сложная динамика, не текущая команда |

Опасные классы, которые не включаем в negative training:

| IPN label | Роль | Почему не берем как negative |
|---|---|---|
| `G03` | throw up | слишком похож на `swipe_up` |
| `G04` | throw down | слишком похож на `swipe_down` |
| `G05` | throw left | слишком похож на `swipe_left` |
| `G06` | throw right | потенциальный будущий `swipe_right` |

Эти классы лучше держать как `validation_reference`: они помогают проверить,
не слишком ли агрессивно reject-policy отвергает реальные пользовательские жесты.

## Что уже доказано на существующем IPN subset

Существующий compact subset:

- root: `data/external/ipn_hand/negative_external_ipn_dynamic`;
- samples: `200`;
- mean frames: `49.34`;
- mean detection rate: `0.8943`;
- original labels: `B0A`, `B0B`, `D0X`, `G01`, `G02`, `G07`, `G08`, `G09`, `G10`, `G11`.

Проверка против динамических prototype-моделей:

| Model | Negative reject rate | False positive rate | Near positive rate |
|---|---:|---:|---:|
| `prototype_distance` | `1.0000` | `0.0000` | `0.0000` |
| `prototype_dtw` | `1.0000` | `0.0000` | `0.0000` |

Вывод: текущий IPN subset полезен как external negative validation/training data
для dynamic rejection. Он не ломает позитивные классы по offline safety check.

## Что показал frames04

Для `frames04.tar` найдено:

| IPN label | Candidate | Selected for test |
|---|---:|---:|
| `B0A` | `201` | `40` |
| `B0B` | `200` | `40` |
| `D0X` | `397` | `40` |
| `G01` | `40` | `17` |
| `G02` | `40` | `17` |
| `G07` | `40` | `17` |
| `G08` | `40` | `17` |
| `G09` | `40` | `17` |
| `G10` | `40` | `17` |
| `G11` | `40` | `18` |

Это хороший источник новых external negatives: pack покрывает не только idle/noise,
но и активные движения рукой, которые система должна уметь отвергать.

## Блокер

Headless MediaPipe extraction сейчас недоступен в CLI:

```text
DrishtiMetalHelper initWithCalculatorContext
graph_service.h:139 Check failed: service_ Service is unavailable
```

Проверки:

- stream-конвертер из tar получил статус `mediapipe_unavailable`;
- конвертер по распакованным кадрам получил статус `mediapipe_unavailable`;
- значит проблема не в tar format и не в путях, а в native MediaPipe runtime.

Чтобы не портить ML pipeline, эти кадры пока не добавлены в обучение.

## Решение в коде

Добавлены инструменты:

- `scripts/convert_ipn_hand_tars.py` - streaming conversion из `.tar` без полной распаковки;
- `scripts/analyze_ipn_external_negatives.py` - safety-анализ external negatives против prototype models;
- safe MediaPipe preflight в `scripts/convert_ipn_hand.py`, чтобы CLI больше не падал process abort-ом;
- Make targets:
  - `ipn-tar-convert`;
  - `ipn-external-analysis`.

Артефакты:

- `docs/experiments/ipn_tar_frames04_report.md`;
- `docs/experiments/ipn_frames04_extracted_conversion_report.md`;
- `docs/experiments/ipn_external_negative_analysis.md`.

## ML вывод

Данные IPN нам помогают, но не как replacement пользовательских gestures.

Правильная роль IPN:

1. external negatives для stage-1 "intentional dynamic gesture or not";
2. hard validation на похожих внешних движениях;
3. измерение false positive rate и near-positive rate;
4. защита от лишних срабатываний на чужих динамических движениях.

Неправильная роль IPN:

1. обучать пользовательские классы вместо user-recorded gestures;
2. добавлять `G03-G06` как negative, потому что они похожи на swipe;
3. без проверки смешивать внешние данные с live pipeline.

## Следующий шаг

1. Оставить `prototype_distance` как основной live verifier.
2. Исправить MediaPipe CLI extraction одним из способов:
   - запускать extraction через тот же runtime, что Flet/live camera;
   - подобрать совместимую версию `mediapipe`/tasks для macOS;
   - вынести extraction в отдельный Docker/Linux runtime.
3. После успешного extraction из одного pack:
   - rerun `ipn-external-analysis`;
   - сравнить `baseline_internal` vs `ipn_external_expanded`;
   - логировать conversion + safety metrics в MLflow.
4. Если metrics безопасные:
   - использовать expanded IPN только в dynamic rejection stage;
   - не трогать пользовательские positive gestures.
