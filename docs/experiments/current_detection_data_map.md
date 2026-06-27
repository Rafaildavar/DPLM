# Current Detection Data Map

Актуально для ветки `contest_version` после H-033.

## 1. Общая схема детектирования

| Шаг | Вход | Обработка | Выход | ML или правило |
|---|---|---|---|---|
| 1. Camera | BGR-кадр `1280x720`, target `30 FPS` | Flip, RGB, resize до ширины `640` для CV | RGB-кадр | Инженерный pipeline |
| 2. Hand detection | RGB-кадр | Один общий MediaPipe detector для static и dynamic | До двух рук, по `21` точке `x/y` | Готовая AI-модель MediaPipe |
| 3. Pose normalization | `21 x 2` landmarks | Wrist переносится в `(0,0)`, координаты делятся на размер руки | `42` pose-признака | Предобработка |
| 4a. Static branch | `42` pose-признака | Среднее по доступному окну | Вектор `42` | KNN |
| 4b. Dynamic branch | Pose `42` + wrist `x/y` | Motion segmentation, trim, resample, position/scale normalization | Последовательность `36 x 44` | Предобработка |
| 5. Dynamic motion-first | Completed trajectory | Ось, направление, прямолинейность, dominance | `swipe_*` label + confidence | Feature rule |
| 6. Dynamic ML fallback | `36 x 44` | Delta, velocity statistics, path statistics, trajectory | Вектор `271` | Feature engineering |
| 7. Classification | Static `42` или dynamic `271` | Static KNN; dynamic KNN/SVM/ExtraTrees как diagnostic/fallback/rejection | Label + probability | ML |
| 8. Guards | Label, probability, trajectory | Confidence, taxonomy, direction, finger count, negative rejection | Accepted/rejected candidate | Детерминированные правила |
| 9. Router | Static и dynamic candidates | Dynamic имеет приоритет; static блокируется во время движения | `static`, `dynamic` или `none` | Routing agent |
| 10. Confirmation | Последовательные outputs | Dynamic `2` кадра, static в auto `15` кадров | Подтвержденный gesture event | Temporal policy |

## 2. Данные одного кадра

| Поле | Размер | Диапазон | Что означает | Попадает в модель |
|---|---:|---|---|---|
| Raw landmarks | `21 x 2` на руку | Обычно `0..1` | Положение точек руки в кадре | Нет, сначала нормализуется |
| Normalized pose | `42` | Относительные значения | Форма руки без абсолютной позиции и размера | Да |
| Wrist `x/y` | `2` | Обычно `0..1` | Глобальная траектория руки | Только dynamic |
| Projected hand scale | `1` | Положительное число | Диагональ bbox руки; proxy дистанции до камеры | Не входит в KNN, управляет порогами segmenter |
| Handedness | Строка | `Left/Right` | Стабильный порядок рук | Не является ML-признаком |
| Finger count | Целое | `0..4` без большого пальца | Проверка совместимости static pose | Guard после KNN |

Dynamic frame имеет `44` ML-признака:

| Индексы | Содержимое |
|---|---|
| `0..41` | Нормализованная форма руки |
| `42` | Wrist `x` |
| `43` | Wrist `y` |

`projected_hand_scale` хранится отдельно и не загрязняет model vector.

## 3. Dynamic state machine

| Состояние | Что происходит | Условие перехода |
|---|---|---|
| `warming_up` | Набираются стартовые кадры | Собрано `5` кадров |
| `idle` | Рука наблюдается, prediction отсутствует | Началось достаточное движение |
| `active` | Собирается один жест переменной длины | Рука остановилась или достигнут лимит |
| `completed` | Segment нормализуется, выполняется один prediction | Prediction передан router |
| `cooldown` | Возврат руки игнорируется | Прошло `8` кадров |

| Параметр | Текущее значение | Distance-aware формула |
|---|---:|---|
| Pre-roll | `5` кадров | Фиксировано |
| Minimum active length | `8` кадров | Фиксировано |
| Maximum active length | `60` кадров | Фиксировано |
| End stillness | `5` кадров | Step threshold зависит от hand scale |
| Onset path fallback | `0.040` | `min(0.040, max(0.012, scale * 0.15))` |
| Onset displacement fallback | `0.025` | `min(0.025, max(0.008, scale * 0.10))` |
| Still-step fallback | `0.006` | `min(0.006, max(0.002, scale * 0.025))` |
| Canonical duration | `36` кадров | Linear resampling |
| Canonical displacement | `0.5` | Wrist path переносится в `(0,0)` и масштабируется |

## 4. Dynamic model vector

Dynamic branch теперь разделен на два слоя:

1. Motion-first classifier для `swipe_*`:

| Поле | Смысл |
|---|---|
| `dx`, `dy` | Направление completed wrist trajectory |
| `path_length`, `displacement` | Сила и завершенность движения |
| `axis_ratio` | Насколько главная ось сильнее поперечной |
| `straightness` | Насколько траектория похожа на прямой свайп |

2. ML fallback/diagnostics/rejection. `dynamic_stats` формирует `271` признаков:

| Блок | Размер | Смысл |
|---|---:|---|
| `last - first` | `44` | Итоговое изменение pose и wrist |
| Mean velocity | `44` | Средняя скорость |
| Velocity standard deviation | `44` | Неравномерность скорости |
| Mean absolute velocity | `44` | Средняя интенсивность движения |
| Absolute path sum | `44` | Накопленное движение по каждому признаку |
| Maximum step | `44` | Максимальный покадровый скачок |
| Trajectory descriptors | `7` | `dx`, `dy`, `abs_dx`, `abs_dy`, path, direction cos/sin |
| **Итого** | **271** | Вход `dynamic_knn.pkl` |

Trajectory block умножается на `8.0`, чтобы направление не проигрывало
сотням pose/velocity компонентов в евклидовом расстоянии KNN. После H-031 KNN
не является главным решателем для простых `swipe_*`: direction classifier
выбирает completed motion label, а ML-модель сохраняется как
diagnostic/fallback. После H-032/H-034 negative labels дают ML-модели право
отклонить событие, если оно похоже на `no_gesture`/`random_motion`.
Negative samples генерируются автоматически из positive dynamic samples, а не
записываются пользователем вручную.

## 5. Модели и классы

| Модель | Feature dim | Алгоритм | Классы | Threshold router |
|---|---:|---|---|---:|
| `models/knn.pkl` | `42` | KNN, `k=5`, distance weights | `ctrlz`, `hend`, `up`, `gun`, `sh3`, `three` | `0.50` |
| `models/dynamic_knn.pkl` | `271` | KNN, `k=5`, distance weights | `swipe_*` + 5 negative labels | `0.60` |
| Motion-first dynamic | trajectory | Direction classifier | `swipe_*` из `dynamic_classes.json` | `>=0.60` |

Static-модель сохранена sklearn `1.8.0`, а текущее окружение использует
`1.7.2`. Она загружается с warning и должна быть переобучена текущей версией.
Dynamic-модель уже пересохранена текущим pipeline.

## 6. Текущий датасет

| Класс | Samples | Используется моделью | Тип |
|---|---:|---|---|
| `CTRLZ` | 21 | Static | static |
| `Hend` | 20 | Static | static |
| `UP` | 20 | Static | static |
| `gun` | 20 | Static | static |
| `sh3` | 20 | Static | static |
| `three` | 20 | Static | static |
| `swipe_down` | 20 | Dynamic | dynamic |
| `swipe_left` | 30 | Dynamic | dynamic |
| `swipe_up` | 20 | Dynamic | dynamic |
| `no_gesture_static` | 20 auto | Dynamic rejection | negative |
| `random_motion` | 20 auto | Dynamic rejection | negative |
| `partial_swipe` | 20 auto | Dynamic rejection | negative |
| `return_motion` | 20 auto | Dynamic rejection | negative |
| `wrong_axis_motion` | 20 auto | Dynamic rejection | negative |
| `background_no_hand` | reserved | Reserved for no-hand pipeline | negative |
| Остальные taxonomy labels | 0 | Нет | static/dynamic |

Всего обучающих samples: `291`, из них dynamic positive: `70`,
negative auto: `100`.

### Position profile dynamic-классов

| Класс | Start X range | Start Y range | Displacement range | Median displacement |
|---|---|---|---|---:|
| `swipe_up` | `0.640..0.792` | `0.893..1.011` | `0.300..0.649` | `0.542` |
| `swipe_down` | `0.688..0.784` | `0.301..0.465` | `0.261..0.654` | `0.493` |
| `swipe_left` | `0.713..0.878` | `0.697..0.873` | `0.330..0.579` | `0.465` |

Старые NPY не содержат hand scale, поэтому реальная дистанция старых записей
неизвестна. После H-030 новые dynamic samples получают
`sample_XXXX.meta.json` с:

| Metadata field | Значение |
|---|---|
| `projected_hand_scale_median` | Медианный размер руки в sample |
| `projected_hand_scale_min/max` | Диапазон масштаба |
| `frames` | Число кадров |
| `raw_feature_dim` | Обычно `44` |
| `recorded_at` | Время записи |

Сейчас sidecar-файлов `0`: новые distance-aware samples еще не записывались.

## 7. Правила принятия решения

| Проверка | Dynamic | Static |
|---|---|---|
| Taxonomy | Label обязан иметь тип `dynamic` | Dynamic label запрещен |
| Confidence | `>=0.60` | `>=0.50` |
| Motion gate | Path `>=0.12`, displacement `>=0.06` после normalization | Нет |
| Direction sign | Left/right по `dx`, up/down по `dy` | Нет |
| Axis dominance | Главная ось минимум в `1.2` раза сильнее поперечной | Нет |
| Negative rejection | Negative probability `>=0.72` отклоняет dynamic event | Нет |
| Finger-count guard | Не применяется | Проверяет pose signature |
| Confirmation | `2` outputs одного event | `15` кадров в auto |

Для `swipe_*` label выбирается motion-first. Если направление неоднозначно или
класс отсутствует в dynamic classes, используется ML fallback с прежней
direction compatibility проверкой.

## 8. Какие данные пишутся в логи

| Файл | Основные поля | Для чего |
|---|---|---|
| `~/.dplm/logs/live_evaluation.jsonl` | expected, predicted, result, confidence, route | Accuracy и confusion |
| `live_evaluation.jsonl` | static/dynamic labels и confidence | Разделение ошибки router/model |
| `live_evaluation.jsonl` | `dynamic_phase`, `dynamic_segment_frames` | Проверка segmenter |
| `live_evaluation.jsonl` | `dynamic_motion_scale` | Accuracy по дистанции |
| `live_evaluation.jsonl` | `dynamic_decision_source`, `dynamic_motion_label`, `dynamic_model_label` | Разделение ошибки motion-first и ML fallback |
| `live_evaluation.jsonl` | `dynamic_negative_label`, `dynamic_negative_confidence` | Анализ false positives и rejection layer |
| `~/.dplm/logs/runtime_performance.jsonl` | inference avg/p95, detection avg, FPS capacity | Производительность |
| `docs/mlops_dashboard/index.html` | HTML dashboard | MLOps-метрики без терминала |
| `docs/mlops_dashboard/summary.json` | Dataset/model/live/runtime snapshot | Версионируемый MLOps-снимок |
| `docs/experiments/negative_sampling_manifest.json` | Generated negative summary | Контроль synthetic negative данных |
| `mlflow.db` | MLflow tracking backend | Параметры, метрики и история запусков |
| `mlruns/` | MLflow artifacts | Артефакты моделей из MLflow runs |

## 9. Что уже проверено

| Проверка | Результат |
|---|---:|
| 5-fold CV dynamic KNN | Accuracy `1.0`, macro F1 `1.0` |
| Position/scale/speed augmentation | `210/210 correct` |
| Motion-first dynamic unit tests | Direction, ambiguous axis, missing class |
| Negative rejection unit tests | Auto generation, taxonomy scope, runtime rejection |
| MLOps dashboard unit test | HTML + summary generation |
| Shared MediaPipe detection | Rate `1.0` |
| ML/runtime targeted unit tests | `47 passed` после H-034/H-035 |
| Real near/mid/far live validation | Еще не выполнена |

## 10. Что модель пока не знает

| Отсутствующие данные | Последствие | Как закрываем |
|---|---|---|
| Реальный hand scale старых samples | Нельзя построить accuracy по старой дистанции | Sidecar для новых записей |
| Несколько пользователей | Не измерена person generalization | Записать отдельные user/session groups |
| Разные камеры и освещение | Возможен MediaPipe distribution shift | Session metadata и live runs |
| Negative/no-gesture samples | Confidence KNN не является вероятностью отсутствия жеста | Auto negative generation + rejection layer |
| `swipe_right` samples | Класс нельзя распознавать | Записать минимум `20` samples |

После H-034 hand-based negative/no-gesture samples строятся автоматически
скриптом `scripts.generate_negative_samples`. Пользователь записывает только
реальные dynamic gestures. `background_no_hand` остается reserved-классом:
текущий pipeline использует hand landmarks, а не пустые кадры без руки.

В live-evaluation для labels типа `negative` отсутствие prediction по timeout
или ручному skip считается `correct`: это позволяет измерять false positive
rate как долю `wrong` на negative-runs.
