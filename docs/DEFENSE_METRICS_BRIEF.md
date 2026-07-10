# Defense Metrics Brief

Дата подготовки: 2026-07-10

Этот документ - короткая выжимка по метрикам и экспериментам GestureBind для
обсуждения на защите. Главная мысль: проект оценивается не одной accuracy, а
цепочкой метрик, потому что система управляет командами ОС и должна уметь
молчать, когда жест не надежен.

## Релизное уточнение 10.07

Старые live-цифры ниже сохранены как история экспериментов. Для финальной
защиты после изменения классов нужно записать новый live run; нельзя смешивать
старые классы и текущую production LSTM.

Новые воспроизводимые факты:

| Проверка | Результат |
|---|---:|
| Grouped static CV | accuracy `0.7767`, macro F1 `0.6613`, `200` source groups, overlap `0` |
| Positive static classes | F1 `1.0000` для пяти пользовательских классов |
| Accepted static predictions | accuracy `1.0000` при coverage `0.7133`, threshold `0.75` |
| Static ML latency | mean `18.939 ms`, p95 `21.721 ms` |
| Dynamic LSTM ML latency | mean `13.979 ms`, p95 `16.013 ms` |
| Dynamic grouped retrain | Optuna `0.8857`, final validation `0.8571`, group overlap `0` |
| Dynamic prototype safety | positive recall `0.9333`, negative FP `0.0000` |
| Automated checks | `592 passed`, `2` optional Qt modules skipped |

Источник: [release readiness audit](experiments/release_readiness_2026-07-10.md).

## 1. Главный тезис для защиты

GestureBind - это не просто классификатор жестов. Это end-to-end ML-система:
пользователь записывает жест, модель обучается на персональных данных, качество
проверяется в offline CV и live evaluation, а команда выполняется только после
routing и rejection policy.

Что важно подчеркнуть:

- offline accuracy показывает, насколько модель выучила сохраненные samples;
- macro F1 важнее raw accuracy, потому что классы несбалансированы;
- accepted accuracy и coverage показывают trade-off между безопасностью и
  молчаливостью системы;
- negative false positive rate критичен для команд ОС;
- live evaluation важнее красивой offline-цифры, потому что ловит шум камеры,
  тайминг, missed detections и distribution shift.

## 2. Ключевые цифры одним экраном

| Блок | Главная цифра | Интерпретация |
|---|---:|---|
| Static CV benchmark | `static_landmark_image + extra_trees`: accuracy `0.8000`, macro F1 `0.7389` | Лучший static CV на текущем mixed dataset из `260` samples / `10` classes |
| Static rejection | best `negative_classes`: pos recall `1.0000`, neg FP `0.0000`, accepted acc `1.0000` | Для ОС-команд reject layer так же важен, как классификатор |
| Dynamic model comparison | `dynamic_stats + extra_trees`: accuracy `0.8209`, macro F1 `0.8136`, dynamic-like F1 `0.9472` | Хороший baseline для direction-sensitive и temporal gestures |
| Dynamic rejection | `open_set_policy`: overall `1.0000`, pos recall `1.0000`, neg FP `0.0000` | Runtime-style policy хорошо отсекает dynamic negative examples |
| Dynamic prototype + external negatives | overall `0.9873`, pos recall `0.9565`, neg FP `0.0074`, sequence acc `0.9565` | Прототипный verifier устойчив при добавлении external negative evidence |
| Intent gate | accuracy `0.9672`, macro F1 `0.9480` | Gate отделяет `static`, `dynamic` и `none`, чтобы не запускать лишние модели |
| Live evaluation | `52` attempts: `38` correct, `11` wrong, `3` missed, accuracy `0.731` | Реальная webcam-метрика показывает, где offline модель расходится с live |
| Static CNN benchmark | validation accuracy `0.7963`, слабые negative F1 | Нейросетевая ветка проверена, но пока не лучше production-подхода |

## 3. Как объяснять эксперименты

### 3.1 Static gesture recognition

Источник: [static_cv_benchmark.md](experiments/static_cv_benchmark.md)

Текущий static benchmark:

- dataset: `260` samples, `10` classes, target dim `63`;
- augmented samples включены;
- 5-fold CV;
- лучший общий вариант: `static_landmark_image + extra_trees`;
- accuracy `0.8000`, macro F1 `0.7389`;
- latency `0.096 ms/sample`;
- threshold `0.9000`;
- accepted accuracy `1.0000`, coverage `0.5962`.

Что говорить:

> На static-жестах я сравнивал несколько feature/model choices. Лучший CV
> результат дал `static_landmark_image + extra_trees`, но я не делаю вывод
> только по raw accuracy. Для desktop-команд важнее, чтобы модель умела
> отклонять сомнительные предсказания. Поэтому я отдельно проверяю threshold,
> accepted accuracy и negative false positives.

Нюанс:

- production baseline остается conservative: ExtraTrees на crafted features;
- image/CNN branches - это проверенные гипотезы, а не автоматическая замена
  production-модели.

### 3.2 Rejection layer для static

Источник:
[static_rejection_benchmark_craft_extratrees.md](experiments/static_rejection_benchmark_craft_extratrees.md)

Эксперимент:

- scope: `static`;
- feature mode: `static_craft_full_stats`;
- candidate model: `extra_trees`;
- dataset: `260` samples, `10` classes, `5` folds;
- positive labels: `2finger`, `gun`, `hand`, `like`, `onefinger`;
- negative labels: `no_gesture_static`, `partial_swipe`, `random_motion`,
  `return_motion`, `wrong_axis_motion`.

Лучшие методы:

| Method | Overall | Pos recall | Neg reject | Neg FP | Accepted acc | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| `negative_classes` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `1.0000` | `0.6154` |
| `open_set_policy` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `1.0000` | `0.6154` |
| `confidence_threshold` | `0.9885` | `0.9812` | `1.0000` | `0.0000` | `1.0000` | `0.6038` |

Что говорить:

> Для управления ОС нельзя оценивать только "угадал класс или нет". Нужна
> метрика "не сработал ли жест на не-жест". Поэтому я добавил negative labels и
> сравнил несколько rejection methods. В лучшем варианте negative false positive
> rate равен `0.0000`, то есть в offline benchmark все negative samples были
> отклонены.

### 3.3 Threshold trade-off

Источник: [threshold_report.md](experiments/threshold_report.md)

Лучший threshold experiment:

- model: `static_stats + svm`;
- dataset: `121` samples, `6` active classes;
- base accuracy `0.9752`;
- base macro F1 `0.9756`;
- threshold `0.50`;
- coverage `0.9587`;
- accepted accuracy `1.0000`;
- rejected predictions `5`.

Как объяснять:

> Threshold - это ручка безопасности. Если поднять порог, меньше ложных
> срабатываний, но система чаще молчит. Поэтому я смотрю две метрики вместе:
> `accepted accuracy` и `coverage`. Хороший порог не просто повышает точность,
> он сохраняет приемлемую долю срабатываний.

### 3.4 Dynamic gesture model comparison

Источник: [model_comparison.md](experiments/model_comparison.md)

Эксперимент:

- dataset: `201` samples, `10` active classes, target dim `44`;
- 5-fold CV;
- классы включают static-like и dynamic-like gestures;
- лучший общий результат: `dynamic_stats + extra_trees`.

Результаты:

| Model | Accuracy | Macro F1 | Static-like F1 | Dynamic-like F1 | Macro FPR | Latency |
|---|---:|---:|---:|---:|---:|---:|
| `knn` | `0.6219` | `0.6166` | `0.5112` | `0.8625` | `0.0422` | `1.084 ms` |
| `svm` | `0.5124` | `0.5019` | `0.3350` | `0.8914` | `0.0545` | `0.038 ms` |
| `extra_trees` | `0.8209` | `0.8136` | `0.7563` | `0.9472` | `0.0200` | `0.125 ms` |

Что говорить:

> На dynamic-сценариях ExtraTrees на `dynamic_stats` оказался сильнее KNN и SVM
> по macro F1. Особенно важно, что dynamic-like F1 получился `0.9472`, то есть
> direction-sensitive жесты в сохраненном датасете отделяются хорошо. Но я не
> утверждаю, что это окончательное качество в продукте: live evaluation показал
> distribution shift.

### 3.5 Dynamic rejection

Источник:
[rejection_method_benchmark_dynamic.md](experiments/rejection_method_benchmark_dynamic.md)

Эксперимент:

- scope: `dynamic`;
- feature mode: `dynamic_stats`;
- target dim `44`;
- `170` samples, `8` classes, `3` folds;
- positive labels: `swipe_down`, `swipe_left`, `swipe_up`;
- negative labels: `no_gesture_static`, `partial_swipe`, `random_motion`,
  `return_motion`, `wrong_axis_motion`.

Лучший вариант:

- method: `open_set_policy`;
- overall `1.0000`;
- pos recall `1.0000`;
- neg reject `1.0000`;
- neg FP `0.0000`;
- accepted acc `1.0000`;
- coverage `0.4118`.

Что говорить:

> Для dynamic-жестов лучший reject дал runtime-style `open_set_policy`: negative
> probability, top1/top2 margin и distance-to-prototype. Он полностью сохранил
> positive recall и отсеял negative samples в этом benchmark. Низкая coverage -
> сознательная цена безопасности: лучше промолчать, чем случайно выполнить
> команду.

### 3.6 Dynamic prototype experiments

Источники:
[dynamic_prototype_sequence_ensemble.md](experiments/dynamic_prototype_sequence_ensemble.md),
[dynamic_prototype_lstm_backbone.md](experiments/dynamic_prototype_lstm_backbone.md)

Ключевой результат:

- best method: `prototype_distance`;
- train samples `472`;
- test samples `158`;
- external negatives: `True`;
- external negatives checked: `440`;
- conflicts removed: `0`;
- conflict rate `0.0000`;
- overall `0.9873`;
- positive recall `0.9565`;
- negative reject `0.9926`;
- negative FP `0.0074`;
- sequence accuracy `0.9565`;
- edit distance `1`.

Что говорить:

> Я проверял не только обычный classifier, но и prototype verifier для
> последовательностей. Он полезен для live, потому что сравнивает завершенный
> motion segment с эталонными траекториями и дает понятную reject-логику.
> External negatives не конфликтовали с моими positive gestures: conflict rate
> `0.0000`.

### 3.7 Live evaluation и live gap

Источники:
[live_evaluation_report.md](experiments/live_evaluation_report.md),
[dynamic_data_analysis.md](experiments/dynamic_data_analysis.md)

Attempt-level live evaluation:

- attempts `52`;
- correct `38`;
- wrong `11`;
- missed `3`;
- accuracy `0.731`.

По labels:

| Expected | Attempts | Correct | Wrong | Missed | Accuracy | Main issue |
|---|---:|---:|---:|---:|---:|---|
| `swipe_up` | `10` | `10` | `0` | `0` | `1.000` | stable in this run |
| `swipe_left` | `10` | `8` | `1` | `1` | `0.800` | occasional `swipe_up` confusion |
| `swipe_down` | `32` | `20` | `10` | `2` | `0.625` | often confused with `swipe_up` |

Dynamic data analysis:

- saved dynamic dataset: `50` samples, `3` classes;
- offline KNN: accuracy `0.9800`, macro F1 `0.9785`;
- live-gap report: `62` attempts, accuracy `0.6290`;
- wrong labels concentrated in `swipe_up:12`;
- key descriptor: `direction_sin`, mutual information `0.8027`.

Что говорить:

> Это самый честный результат проекта: offline CV высокий, но live качество
> ниже. Значит проблема не только в выборе модели, а в distribution shift между
> записью и реальным показом жеста. Поэтому следующий шаг - quality gates при
> записи: направление, minimum global path, straightness, active motion segment
> и баланс классов минимум до 20 samples на dynamic class.

Эту часть лучше подавать как плюс:

> Я не спрятал live-gap. Я добавил live-evaluation protocol, чтобы видеть
> correct/wrong/missed и понимать, какие именно жесты конфликтуют.

### 3.8 Intent gate

Источник: [intent_gate_training.md](experiments/intent_gate_training.md)

Результаты:

- samples `731`;
- feature dim `25`;
- accuracy `0.9672`;
- macro F1 `0.9480`;
- records by intent: `dynamic=70`, `static=121`, `none=540`.

Confusion matrix:

| expected | static | dynamic | none |
|---|---:|---:|---:|
| `static` | `28` | `1` | `1` |
| `dynamic` | `0` | `17` | `1` |
| `none` | `3` | `0` | `132` |

Что говорить:

> Intent gate нужен, чтобы система не заставляла классификатор всегда выбирать
> жест. Сначала решаем, есть ли static gesture, dynamic gesture или вообще
> `none`. Это снижает ложные срабатывания и делает routing расширяемым.

### 3.9 Static CNN benchmark

Источник: [static_landmark_cnn_benchmark.md](experiments/static_landmark_cnn_benchmark.md)

Гипотеза:

- проверить GISLR-inspired landmark-image representation;
- не заменять production model без доказательств.

Результат:

- dataset `260` samples, `10` classes;
- `static_landmark_image`: `30 frames x 21 points x 3 coordinates = 1890 features`;
- CNN train accuracy `0.8000`;
- CNN validation accuracy `0.7963`;
- negative F1 слабые:
  - `partial_swipe`: `0.400`;
  - `random_motion`: `0.500`;
  - `return_motion`: `0.091`;
  - `wrong_axis_motion`: `0.367`.

Что говорить:

> Я проверил нейросетевую гипотезу, но не стал подменять надежную baseline-модель
> модной архитектурой. На маленьком персональном датасете CNN не дал уверенного
> выигрыша и хуже вел себя на negative classes. Поэтому production-путь остается
> conservative: crafted features + tree models + rejection.

### 3.10 External negative datasets

Источник:
[external_negative_dataset_experiments.md](experiments/external_negative_dataset_experiments.md)

Цель:

- использовать публичные данные как negative / out-of-distribution evidence;
- не заменять персональный датасет GestureBind.

Варианты:

- baseline internal: `291` internal samples;
- IPN external: `291` internal + `80` external samples;
- Hagrid skipped: source отсутствовал;
- combined external: фактически IPN + missing Hagrid.

Offline results:

| Variant | Scope | Best method | Overall | Pos recall | Neg reject | Neg FP |
|---|---|---|---:|---:|---:|---:|
| baseline internal | static | `one_vs_rest_logreg` | `0.9774` | `0.9669` | `0.9900` | `0.0100` |
| baseline internal | dynamic | `open_set_policy` | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| ipn external | static | `one_vs_rest_logreg` | `0.9774` | `0.9669` | `0.9900` | `0.0100` |
| ipn external | dynamic | `one_vs_rest_logreg` | `1.0000` | `1.0000` | `1.0000` | `0.0000` |

Что говорить:

> Внешние датасеты я использую аккуратно: не как чужие positive gestures, а как
> negative/OOD evidence. На текущих offline-цифрах IPN не дал сильного прироста,
> но он проверяет важную гипотезу: внешние движения можно использовать для
> усиления reject-layer.

## 4. Что показывать комиссии как исследовательский путь

1. Сначала был baseline: MediaPipe landmarks + KNN/простые признаки.
2. Потом появились dynamic features и сравнение KNN/SVM/ExtraTrees.
3. Потом стало понятно, что для команд ОС нужна не только classification, но и
   rejection.
4. Были добавлены negative classes, open-set policies и threshold curves.
5. Для live были добавлены attempt-level метрики: correct, wrong, missed.
6. Live-gap показал, что слабое место - не только модель, а протокол записи и
   distribution shift.
7. Поэтому следующие эксперименты идут в data quality: active motion segment,
   direction gates, straightness, class balance, external negatives.

## 5. Какие метрики называть главными

Для offline model selection:

- `macro F1`;
- per-class recall;
- false positive rate;
- latency per sample.

Для safety/rejection:

- accepted accuracy;
- coverage;
- negative false positive rate;
- positive recall after rejection;
- reject reasons.

Для live product quality:

- attempt accuracy;
- missed rate;
- wrong-label distribution;
- accepted accuracy;
- confidence distribution;
- command execution success / no-command false triggers.

Короткая формулировка:

> В моем проекте accuracy - это не одна метрика. Offline я смотрю macro F1,
> для безопасности - negative false positives и accepted accuracy, а для
> продукта - live correct/wrong/missed.

## 6. Слабые места, которые стоит назвать самому

- Static benchmark уже неплохой, но current mixed dataset содержит сложные
  negative-like classes, из-за которых raw macro F1 ниже, чем на ранних
  чистых датасетах.
- Dynamic offline quality высокая, но live ниже: это distribution shift между
  сохраненными samples и естественным исполнением жеста.
- `swipe_down` чаще всего конфликтует со `swipe_up`; это указывает на проблему
  сегментации/направления, а не просто на "плохой классификатор".
- CNN-ветка проверена, но пока не production, потому что хуже работает на
  negative classes.
- External negatives пока больше полезны как safety evidence, чем как прямое
  улучшение итоговых метрик.

Такой способ подачи выглядит сильнее, чем попытка продать одну красивую цифру.

## 7. Вероятные вопросы и короткие ответы

### Почему live accuracy ниже offline?

Offline CV проверяет сохраненные samples. Live добавляет шум камеры, другой
тайминг, неполный жест, задержки и реальные движения пользователя. Поэтому я
добавил live evaluation с expected label и считаю correct/wrong/missed отдельно.

### Почему не взять CNN как основную модель?

CNN был проверен как гипотеза. На текущем персональном датасете он дал validation
accuracy `0.7963`, но слабые F1 на negative classes. Для команд ОС важнее
безопасность, поэтому production-путь пока остается crafted features +
ExtraTrees/rejection.

### Почему macro F1, а не только accuracy?

Классы несбалансированы: например, в dynamic comparison часть классов имеет
`10` samples, часть `30`. Accuracy может скрыть просадку редкого жеста, а macro
F1 усредняет качество по классам.

### Зачем negative classes?

Система должна уметь не выполнять команду. Negative classes моделируют
no-command, partial movements, random motion и wrong-axis motion. Это снижает
риск ложных запусков команд ОС.

### Что означает coverage?

Coverage - доля предсказаний, которые система принимает после threshold/reject.
Можно добиться accepted accuracy `1.0000`, но если coverage слишком низкая,
приложение будет слишком часто молчать. Поэтому accuracy и coverage нужно
обсуждать вместе.

### Что будет следующим экспериментом?

Главный следующий шаг - улучшить dynamic data quality: активный сегмент движения,
resample до фиксированной длины, direction/straightness gates, баланс минимум
`20` real samples на dynamic class, затем повторить live evaluation.

## 8. Финальная формулировка на 30 секунд

> Я построил не просто модель распознавания жестов, а полный ML-loop для
> desktop-команд. Offline я сравнил static и dynamic признаки, tree models, SVM,
> CNN и prototype methods. Лучшие offline результаты: static benchmark
> `0.8000` accuracy / `0.7389` macro F1 на сложном mixed dataset, dynamic
> ExtraTrees `0.8209` accuracy / `0.8136` macro F1, intent gate `0.9672`
> accuracy. Но для команд ОС главная часть - rejection: static и dynamic
> benchmark показывают `0.0000` negative false positives в лучших вариантах.
> Live evaluation честно показал gap: `52` attempts, accuracy `0.731`, и
> главная проблема - direction confusion у dynamic gestures. Поэтому следующий
> шаг не просто заменить модель, а улучшить протокол записи, segmentation и
> live validation.

## 9. Источники

- [Static CV benchmark](experiments/static_cv_benchmark.md)
- [Static rejection benchmark](experiments/static_rejection_benchmark_craft_extratrees.md)
- [Dynamic model comparison](experiments/model_comparison.md)
- [Dynamic rejection benchmark](experiments/rejection_method_benchmark_dynamic.md)
- [Dynamic prototype sequence ensemble](experiments/dynamic_prototype_sequence_ensemble.md)
- [Live evaluation report](experiments/live_evaluation_report.md)
- [Dynamic data analysis](experiments/dynamic_data_analysis.md)
- [Intent gate training](experiments/intent_gate_training.md)
- [Static landmark CNN benchmark](experiments/static_landmark_cnn_benchmark.md)
- [External negative dataset experiments](experiments/external_negative_dataset_experiments.md)
- [ML system design](contest/ML_SYSTEM_DESIGN.md)
