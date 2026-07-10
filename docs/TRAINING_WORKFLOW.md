# Training Workflow

## Единый интерфейс

Экран `Обучение` больше не должен делить сценарии на "пользователь" и
"разработчик". В UI остается один поток:

1. выбрать тип жеста: `Статический` или `Динамический`;
2. ввести имя жеста;
3. выбрать число реальных дублей, длину записи и режим двух рук;
4. нажать `Записать примеры`;
5. нажать `Обучить модель`.

Технические пути моделей, feature mode, scope обучения и параметры модели
выбираются автоматически из проектных defaults.

На экране `Обучение` обычный пользователь больше не видит developer-панели:
`Менеджер датасета`, `Пайплайн обучения`, `3. Защита` и терминальный `Журнал`
убраны из основного интерфейса. Вместо этого рядом с камерой остается короткий
статус операции: запись идет, модель обучается, операция завершилась или требует
проверки.

Управление данными перенесено на экран `Gesture Library`. Там пользователь
видит стандартные и свои жесты, число реальных записей и аугментаций, может
обновить список и удалить только собственные записи. Служебные negative-классы
вроде `no_gesture_static`, `partial_swipe`, `random_motion`, `return_motion` и
`wrong_axis_motion` скрыты из пользовательской библиотеки.

## Как сейчас идет запись

Запись выполняется во встроенной Flet-камере через
`AppController.start_recording`.

- Для статического жеста единый UI сохраняет `21 * xyz = 63` признака на руку.
  `x/y` берутся из нормализованной кисти, `z` — из MediaPipe `landmarks_xyz`.
  Старые static samples `21 * xy = 42` остаются совместимыми и при обучении
  дополняются `z = 0`.
- Для динамического жеста включается `include_global_motion`: к `42`
  нормализованным координатам добавляется global wrist `x/y`, итого `44`
  признака на руку за кадр.
- Новый dynamic-формат включает `include_landmark_z`: тогда на руку сохраняется
  `21 * xyz + wrist_xy = 65` признаков. `wrist_xy` сохраняет экранную
  траекторию для direction-sensitive команд.
- Единый UI включает `include_landmark_z` для static и dynamic. Static-модель
  обучается с `--expect-dim 63`, dynamic-модель — с `--expect-dim 65`, чтобы
  старые `42/44` samples конвертировались в новый layout.
- Перед записью quality gate ждет стабильную стартовую позу.
- После заполнения нужного числа кадров создается `sample_XXXX.npy`.
- Рядом пишется `sample_XXXX.meta.json` с формой, типом записи и quality
  metrics.

## GISLR-style аугментация

После каждого реального sample создаются отдельные `aug_sample_*` файлы:

- static: `1` augmented sample на реальный;
- dynamic: `2` augmented samples на реальный.

Аугментация живет в `cv/gesture_augmentation.py` и использует безопасную для
команд часть подхода GISLR:

- `time_warp`, вероятность `0.45`: вся последовательность слегка растягивается
  или сжимается по времени (`scale 0.86..1.18`, `shift -0.35..0.35`) и
  интерполируется обратно в исходное число кадров. Длина sample не меняется.
- `affine`, вероятность `0.85` для каждой найденной руки: координаты кисти
  масштабируются (`0.86..1.16`), немного сдвигаются (`-0.045..0.045`) и
  поворачиваются (`-8..8` градусов).
- `part_scale`, вероятность `0.45`: кисть масштабируется относительно своего
  среднего центра (`0.90..1.12`), имитируя другой размер руки или расстояние до
  камеры.
- `joint_rotate`, вероятность `0.65`: поддеревья пальцев вращаются вокруг
  суставов на `-4..4` градуса. Для каждого суставного дерева применяется
  дополнительная вероятность `0.35`.
- `noise`, вероятность `0.55`: добавляется небольшой Gaussian noise с
  `sigma = 0.004`.
- `point_mask`, вероятность `0.20`: случайные landmarks зануляются на всей
  последовательности. Доля точек `0.02..0.08`, минимум одна точка.
- `time_mask`, вероятность `0.18`: короткий фрагмент времени заменяется
  соседним кадром или нулями, чтобы модель была устойчивее к кратким провалам
  трекинга.
- Для dynamic-жестов отдельно меняется `global wrist trajectory`: траектория
  запястья масштабируется относительно первого кадра (`0.88..1.14`), получает
  небольшой сдвиг (`-0.018..0.018`) и шум `sigma = 0.002`.

Перед трансформациями sample приводится к плоской матрице `frames x features`.
Одна рука может занимать `42` (`21 x xy`), `44` (`21 x xy + wrist_xy`), `63`
(`21 x xyz`) или `65` (`21 x xyz + wrist_xy`) признаков. Если записаны две
руки, аугментация применяется к каждому hand-блоку отдельно.

Каждый augmented-файл получает стабильный seed из имени жеста, исходного sample
и номера варианта. Поэтому результат воспроизводим для одного и того же
исходного файла.

Horizontal flip намеренно не используется: он может превратить `swipe_left` в
`swipe_right` и поменять смысл команды.

## Обучение с augmented samples

`aug_sample_*` отделены от реальных записей. Единый экран обучения передает
`--include-augmented`, поэтому новая модель учится на реальных дублях и
безопасных GISLR-аугментациях.

Защита от прошлой неудачной аугментации находится в `cv.train_classifier`:
даже с `--include-augmented` в обучение попадают только `aug_sample_*`, у
которых рядом есть `.meta.json` с `transform = gislr_landmark_v1` или
`transform_metadata.policy = gislr_landmark_v1`. Старые augmented-файлы без
этой метки игнорируются.

В CLI baseline остается управляемым явно:

```bash
python -m cv.train_classifier --include-augmented ...
```

## Static model default

Для статических и quasi-static жестов кнопка `Обучить модель` теперь использует:

```text
feature_mode = static_craft_full_stats
model_type = extra_trees
expect_dim = 63
optuna = 20 trials, 3-fold stratified macro-F1 CV, 120s timeout
```

Это заменяет старый baseline `static_mean + knn`. Новый pipeline ближе к двум
референсам:

- GISLR показывает, что геометрические признаки кисти важны даже рядом с
  тяжелыми моделями;
- Gstrl использует hand landmarks и геометрические правила для pinch/hold/fist,
  то есть форма кисти важнее ближайшего сырого sample.

`static_craft_full_stats` включает:

- raw pose stats: `mean`, `std`, `min`, `max` по `xyz`;
- все `210` pairwise distances между `21` точкой кисти;
- `15` углов фаланг;
- агрегаты по времени для геометрии: `mean`, `std`, `min`, `max`, `p95`.

Для одной руки и `--expect-dim 63` итоговый размер признака:

```text
63 * 4 + (210 + 15) * 5 = 1377
```

Модель `extra_trees` выбрана как production default для текущего UX:
пользователь записывает небольшой набор стандартных и своих жестов, а ансамбль
деревьев хорошо работает на широких hand-crafted признаках, меньше зависит от
масштабирования признаков и обычно проще переживает шумные дубли, чем KNN.
`svm`, `rf` и `logreg` остаются доступными для benchmark и developer training.

Через UI static-обучение запускает Optuna перед финальным fit:

```bash
--extra-trees-optuna-trials 20
--extra-trees-optuna-cv-folds 3
--extra-trees-optuna-timeout 120
```

Optuna подбирает `n_estimators`, `max_depth`, `min_samples_split`,
`min_samples_leaf`, `max_features`, `criterion` и `bootstrap`. Целевая метрика
— `macro-F1` на stratified CV, чтобы редкие или похожие жесты не терялись за
общей accuracy. Summary сохраняется рядом с моделью как `*_optuna.json`.
Для этого в активном venv должен быть установлен пакет `optuna` из
`requirements.txt`.

LSTM/sequence-backbone стоит рассматривать отдельным экспериментом после
записи полного набора классов: для статических жестов он учит не только форму
кисти, но и микродвижения во время удержания позы, поэтому требует больше
данных и аккуратной validation-схемы.

## GISLR-style handcrafted features

В `cv.gesture_features` добавлены feature modes `static_craft_full_stats`,
`dynamic_craft_stats` и `dynamic_craft_full_stats`. Это альтернативы тяжелой
Transformer-ветке GISLR: признаки считаются из landmark-последовательности и
остаются совместимыми со scikit-learn моделями.

`dynamic_craft_stats` включает базовый `dynamic_stats`, а затем добавляет
агрегаты по handcrafted-сигналам кисти:

- расстояния между ключевыми точками: wrist-tip, tip-tip, MCP-MCP и ширина
  кисти;
- углы фаланг по пяти finger routes;
- forward/backward motion: signed progress вдоль общего направления жеста,
  forward-шаги, backward-шаги и lateral motion;
- агрегаты по времени для каждого сигнала: `mean`, `std`, `min`, `max`, `p95`.

`dynamic_craft_full_stats` устроен так же, но вместо короткого списка
расстояний использует полный GISLR-набор для кисти:

- все `210` pairwise distances между `21` точкой кисти;
- `15` углов фаланг по пяти finger routes;
- те же `4` motion-сигнала;
- те же `5` агрегатов по времени.

Для одной руки это `229` handcrafted-сигналов на кадр. При `--expect-dim 65`
итоговый вектор имеет `1542` признака: `65 * 6 + 7 + 229 * 5`. Поэтому режим
лучше сначала прогонять как benchmark рядом с `dynamic_craft_stats`, а не
автоматически считать production-дефолтом.

Формат понимает старые `42/44` samples и новые `63/65` samples. При обучении с
`--expect-dim 63` старый static `42` layout конвертируется в `63`: `x/y`
остаются, `z` заполняется нулями. При обучении с `--expect-dim 65` старый
dynamic `44` layout конвертируется в `65`: `x/y` остаются, `z` заполняется
нулями, `wrist_xy` переносится в последние две координаты.

Пример CLI-эксперимента:

```bash
python -m cv.train_classifier \
  --feature-mode dynamic_craft_full_stats \
  --model-type sequence_mlp \
  --expect-dim 65 \
  --include-augmented
```

## Class balancing

В `cv.train_classifier` включена политика `--class-balance auto`. Она сохраняет
GISLR-идею усиления слабых/редких классов, но подстраивается под тип модели:

- модели с `class_weight="balanced"` или weighted loss используют встроенную
  балансировку;
- KNN, `sequence_mlp` и `sequence_phase_hmm` получают oversampling train set до
  размера самого крупного класса;
- классы `hend` и `gun` по умолчанию дополнительно усиливаются через
  `--boost-factor 1.5`, если такие labels есть в датасете.

Можно управлять этим явно:

```bash
python -m cv.train_classifier \
  --class-balance auto \
  --boost-label Hend \
  --boost-label gun \
  --boost-factor 1.5
```

Для отключения:

```bash
python -m cv.train_classifier --class-balance none
```

## Landmark image benchmark

В GISLR CNN-ветка рассматривала landmarks как изображение: ось времени,
ось точек и каналы `x/y/z`. В проект добавлены два режима с этой идеей.

`static_landmark_image` предназначен для статических поз и превращает запись в
плоский вектор из тензора:

- `30` кадров;
- `21` точка кисти на руку;
- `3` координаты на точку, где для старых `xy` samples `z = 0`.

Для одной руки и `--expect-dim 63` размер входа:

```text
30 * 21 * 3 = 1890
```

Поверх этого режима добавлена экспериментальная TensorFlow-модель
`static_landmark_cnn`: компактная Depthwise/Conv2D CNN с weighted categorical
cross-entropy, label smoothing, dropout и early stopping. Она сохраняется через
`joblib`, как остальные модели, но требует установленный `tensorflow`.

Пример запуска:

```bash
python -m cv.train_classifier \
  --feature-mode static_landmark_image \
  --model-type static_landmark_cnn \
  --expect-dim 63 \
  --include-augmented
```

Обычная кнопка `Обучить модель` пока остается на `extra_trees +
static_craft_full_stats`: при маленьком пользовательском датасете это обычно
стабильнее. `static_landmark_cnn` стоит использовать как benchmark после
записи полного набора стандартных и пользовательских жестов.

`dynamic_landmark_image` превращает динамический жест в плоский sklearn-вектор
из тензора `time x points x xyz`:

- `72` кадра;
- `21` точка кисти + дополнительная `wrist_xy` точка, если она есть;
- `3` координаты на точку, где для старых `xy` samples `z = 0`.

В production этот тензор использует `dynamic_landmark_lstm_backbone`: общий
per-frame MLP-backbone кодирует landmarks, затем двухслойная bidirectional LSTM
и `final/mean/max` pooling классифицируют всю последовательность. Обычная
кнопка dynamic-обучения публикует модель и sidecars под префиксом
`models/dynamic_landmark_lstm_backbone.*`.

В ту же транзакцию обучения входит adaptive completion gate. Для каждого
положительного пользовательского класса автоматически строится grouped
logistic profile:

- positive: полные реальные записи целевого класса;
- hard negatives: его префиксы по cumulative motion progress;
- cross-class hard negatives: полные записи и префиксы остальных классов;
- доступные negative classes тоже используются, если имеют совместимый raw
  layout.

Профиль сохраняется в model-specific `*_rejection.json`. Он работает по raw
траектории до amplitude normalization, поэтому переименование или добавление
жеста не требует ручных alias/rules. Для проверки текущего production bundle:

```bash
python -m scripts.evaluate_dynamic_completion --log-mlflow
```
