# ML Pipeline Log

Рабочий журнал гипотез и решений по ML-пайплайну GestureFlow для JMLC.
Каждая запись фиксирует, что проверяли, что получилось, что не получилось и
какое инженерное решение принято.

## Правила ведения

- Все новые ML-гипотезы фиксируются до или сразу после эксперимента.
- Для каждой гипотезы указываем статус: `planned`, `testing`, `validated`,
  `rejected`, `needs-more-data`.
- Коммит должен содержать код, тесты и обновление этого журнала, если менялась
  логика ML-пайплайна.
- Локальные модели `models/*.pkl` и сырые записи жестов считаются артефактами:
  коммитим их только отдельным осознанным решением.

## Текущее состояние

- Основной static-пайплайн: `models/knn.pkl`, `classes.json`, `feature_dim.txt`.
- Отдельный dynamic-пайплайн: `models/dynamic_knn.pkl`,
  `dynamic_classes.json`, `dynamic_feature_dim.txt`,
  `dynamic_feature_mode.txt`.
- Dynamic live-инференс сейчас грузит именно `models/dynamic_knn.pkl`, даже
  если внутри сохранен не KNN, а, например, `ExtraTreesClassifier`.
- Новый dynamic-формат записи: `36` кадров, `44` признака на кадр:
  `42` нормализованных признака позы + `2` координаты запястья в кадре.

## Гипотезы и результаты

### H-001: Одной static-модели достаточно для всех жестов

Статус: `needs-more-data`

Гипотеза: если признаки позы достаточно устойчивые, одна модель может
распознавать и статические, и динамические жесты.

Что проверяли:
- `scripts.compare_models` на текущем датасете.
- Feature modes: `static_mean`, `static_stats`, `dynamic_stats`,
  `hybrid_stats`.
- Models: `knn`, `svm`, `rf`, `extra_trees`, `logreg`.

Что получилось:
- На датасете после добавления `swipe_up` лучший CV-результат показал
  `static_mean + knn`: macro F1 около `0.9876`.
- Для классов `hand_left` и `swipe_up` в CV получилось F1 `1.0`.

Что не получилось / риск:
- CV завышает уверенность, если dynamic-классы записаны в разных форматах.
- Старый `hand_left` записан как `(60, 21, 2)` без глобального движения, а
  новый `swipe_up` записан как `(36, 44)`.
- В live-проверке сохраненная dynamic-модель распознавала `swipe_up`, но
  относила старый `hand_left` к `swipe_up`.

Решение:
- Не делать окончательный вывод про единую модель до перезаписи всех
  dynamic-жестов в едином формате `(36, 44)`.

Следующий шаг:
- Перезаписать `hand_left` в новом dynamic-формате.
- Повторить benchmark и live-проверку.

### H-002: Для свайпов нужны глобальные координаты движения

Статус: `validated`

Гипотеза: нормализация `wrist_scale` хорошо подходит для статических поз, но
выбрасывает абсолютное перемещение руки по экрану, поэтому для свайпов нужно
добавить глобальные координаты.

Источник идеи:
- Анализ подхода Gestop: для динамических жестов важны не только относительные
  landmark-позы, но и изменение положения руки во времени.

Что проверяли:
- Старый `hand_left` был записан без глобального движения.
- Новый `swipe_up` записан с координатами запястья.
- Сравнили motion profile и live-распознавание.

Что получилось:
- `swipe_up` записался как `(36, 44)`.
- Median global `dy` для `swipe_up`: примерно `-0.182`, то есть в данных есть
  явное движение вверх.
- Live dynamic-режим корректно распознал `swipe_up`.

Что не получилось / риск:
- Старые dynamic-сэмплы без глобальных координат несовместимы по смыслу с
  новыми dynamic-сэмплами.

Решение:
- Dynamic-запись в Flet теперь включает `include_global_motion=True`.
- Static-запись оставлена без глобальных координат.
- Live-инференс умеет строить 44 признака на кадр, если модель обучена на
  такой размерности.

### H-003: Dynamic-окно 60 кадров слишком медленное для свайпа

Статус: `validated`

Гипотеза: окно 60 кадров делает жест похожим на длинное удержание и ухудшает
ощущение live-сценария.

Что проверяли:
- Dynamic-запись и live-инференс с окном 60 кадров.
- Затем сократили окно до 36 кадров.

Что получилось:
- `swipe_up` в 36 кадрах распознался корректно.
- Live-сценарий стал ближе к реальному свайпу.

Решение:
- `DYNAMIC_RECOGNITION_WINDOW = 36`.
- `_DEFAULT_DYNAMIC_RECORD_FRAMES = 36`.

### H-004: Dynamic-жесты нельзя подтверждать как статические позы

Статус: `validated`

Гипотеза: подтверждение `6` одинаковых кадров подряд подходит для статических
жестов, но слишком строго для короткого свайпа.

Что проверяли:
- Dynamic live-confirm уменьшен до `2` кадров.

Что получилось:
- `swipe_up` распознался корректно в live.

Решение:
- Для static оставлено `GESTURE_CONFIRM_FRAMES = 6`.
- Для dynamic добавлено `DYNAMIC_GESTURE_CONFIRM_FRAMES = 2`.

### H-005: Модель важна меньше, чем согласованный формат данных

Статус: `validated`

Гипотеза: замена KNN на SVM/дерево не исправит плохие данные, если dynamic
классы записаны в разных форматах.

Что проверяли:
- Пользователь обучил dynamic-модель на KNN, SVM и дереве.
- Текущий `models/dynamic_knn.pkl` оказался `ExtraTreesClassifier`, потому что
  разные модели сохранялись в один и тот же путь.
- Offline-check текущей saved-модели:
  - `swipe_up`: `30/30` правильно.
  - `hand_left`: `0/30`, все предсказано как `swipe_up`.

Что получилось:
- `ExtraTreesClassifier` хорошо ловит новый `swipe_up`, но не спасает старый
  `hand_left`.

Решение:
- Сначала привести dynamic-данные к единому формату.
- Затем сравнить `knn`, `svm`, `extra_trees` на одинаковых данных.
- Сохранять разные модели в разные файлы:
  - `models/dynamic_knn.pkl`
  - `models/dynamic_svm.pkl`
  - `models/dynamic_extra_trees.pkl`

### H-006: Candidate-модели нельзя сохранять в один файл

Статус: `validated`

Гипотеза: если KNN, SVM и деревья сохраняются в один и тот же
`models/dynamic_knn.pkl`, мы теряем воспроизводимость эксперимента и не можем
честно собирать live-метрики по кандидатам.

Что проверяли:
- Пользователь обучил dynamic-модель на KNN, SVM и дереве.
- Последняя обученная модель перезаписала `models/dynamic_knn.pkl`.
- При проверке файл `models/dynamic_knn.pkl` оказался `ExtraTreesClassifier`.

Что получилось:
- UI теперь подставляет файл модели по выбранному типу:
  - `knn` -> `models/dynamic_knn.pkl`;
  - `svm` -> `models/dynamic_svm.pkl`;
  - `extra_trees` -> `models/dynamic_extra_trees.pkl`;
  - `rf` -> `models/dynamic_rf.pkl`;
  - `logreg` -> `models/dynamic_logreg.pkl`.
- Если пользователь вручную ввел кастомный путь, UI его не перезаписывает.

Решение:
- Dynamic-candidate модели сохраняются в отдельные файлы.
- Метрики по кандидатам собираем после обучения одинакового набора данных и
  фиксируем в этом журнале.

### H-007: Нужен формальный протокол записи жестов

Статус: `validated`

Гипотеза: качество live-распознавания упирается не только в модель, но и в
воспроизводимость записи: если каждый dynamic-жест записывать "на глаз", CV и
live-проверка будут расходиться.

Что сделали:
- Добавлен протокол записи и проверки:
  `docs/experiments/gesture_protocol.md`.
- Для каждого типа жестов описаны:
  - формат данных;
  - длительность записи;
  - правила выполнения;
  - ошибки записи;
  - live-evaluation template.
- Для `swipe_up` зафиксирован ожидаемый global signal: `dy < 0`.
- Для `hand_left` зафиксирован ожидаемый global signal: `dx < 0`.

Что получилось:
- Следующий ML-эксперимент теперь имеет четкий входной критерий: dynamic
  классы должны быть в формате `(36, 44)`.
- Старый `hand_left` явно помечен как `rerecord-required`.

Решение:
- Новые static/quasi-static/dynamic жесты добавлять только через протокол.
- Live-метрики собирать по таблице из `gesture_protocol.md`.

### H-008: Интерфейс записи должен валидировать сэмпл до обучения

Статус: `in-progress`

Гипотеза: если Flet-запись стартует сразу после таймера, пользователь легко
получает кривые dynamic-сэмплы: рука может быть не в кадре, стартовая поза
плывет, а записанный `swipe_up/hand_left` не содержит нужного глобального
смещения.

Что сделали:
- Перед записью каждого сэмпла добавлен quality gate:
  - рука должна быть найдена в кадре;
  - стартовая поза должна быть стабильной несколько кадров;
  - после стабильной позы запускается короткий отсчет.
- Если рука пропала во время сэмпла, частичная запись сбрасывается.
- После сохранения dynamic-сэмпла выводится отчет качества:
  - `motion`;
  - `disp`;
  - `dx`;
  - `dy`;
  - `OK/WARN`.
- Для directional labels проверяется ожидаемое глобальное движение:
  - `left` -> `dx < -0.05`;
  - `right` -> `dx > 0.05`;
  - `up` -> `dy < -0.05`;
  - `down` -> `dy > 0.05`.

Что ожидаем:
- Перезапись `hand_left` станет воспроизводимее.
- Плохой сэмпл будет виден сразу в журнале обучения, до переобучения модели.

Следующая проверка:
- Записать `hand_left` заново через Flet.
- Проверить, что каждая строка сохранения имеет `dx < -0.05` и `OK`.
- После этого обучить `dynamic_knn`, `dynamic_svm`, `dynamic_extra_trees` и
  собрать live-метрики.

### H-009: Сократить число записей через few-shot прототипы и аугментации

Статус: `proposed`

Наблюдение из Gestop:
- Gestop разделяет static и dynamic gestures.
- Для dynamic-жестов используются признаки движения ладони: абсолютные
  координаты основания ладони и `timediff`.
- Начало/конец dynamic-жеста отделяются явным trigger-сигналом, чтобы модель
  не училась на случайных кусках видео.

Гипотеза для нашего проекта:
- Для первых версий нового dynamic-жеста можно записывать не `30`, а `5-10`
  хороших сэмплов, если:
  - каждый сэмпл проходит interface quality gate;
  - training pipeline генерирует аугментации;
  - live pipeline сравнивает последовательность не только классификатором, но
    и с прототипом жеста.

План эксперимента:
1. Записать `5`, `10`, `20`, `30` сэмплов одного dynamic-жеста.
2. Для каждого набора обучить:
   - обычный classifier baseline;
   - classifier + аугментации;
   - prototype matcher с DTW/sequence distance.
3. Аугментации:
   - time stretch / speed jitter;
   - небольшое смещение глобальной траектории;
   - small landmark noise;
   - mirror only for paired gestures (`left` <-> `right`) with label swap.
4. Метрики:
   - offline accuracy/F1;
   - live attempts;
   - false positives;
   - median delay.

Решение пока не принято:
- В текущей итерации сначала проверяем quality gate на `hand_left`.
- Если `5-10` сэмплов после аугментаций дают стабильный live-result, уменьшаем
  recommended recording count для новых gestures.

### H-010: Новые `swipe_down` и `swipe_left` надо проверить на семантику движения

Статус: `fixed`

Что записано:
- `swipe_up`: `30` сэмплов, shape `(36, 44)`;
- `swipe_down`: `10` сэмплов, shape `(36, 44)`;
- `swipe_left`: `10` сэмплов, shape `(36, 44)`;
- `swipe_right`: `0` сэмплов.

Проблема:
- пользователь подтвердил, что при записи перепутал названия
  `swipe_down` и `swipe_left`;
- папки `data/gestures/swipe_down` и `data/gestures/swipe_left` были
  аккуратно поменяны местами без удаления сэмплов;
- `models/dynamic_knn.pkl` переобучен после swap.

Диагностика global wrist motion:

| Label | Expected | Median dx | Median dy | Interpretation |
|---|---|---:|---:|---|
| `swipe_up` | `dy < -0.05` | `+0.0135` | `-0.1820` | OK |
| `swipe_down` | `dy > +0.05` | `-0.0130` | `+0.4876` | OK |
| `swipe_left` | `dx < -0.05` | `-0.3823` | `+0.0212` | OK |

Offline prediction with current `models/dynamic_knn.pkl`:
- `swipe_up`: `30/30` -> `swipe_up`;
- `swipe_down`: `10/10` -> `swipe_down`;
- `swipe_left`: `10/10` -> `swipe_left`;
- old `hand_left`: mostly predicted as `swipe_up`/`swipe_left`, not reliable
  because shape is `(60, 42)`.

Вывод:
- Семантика `swipe_down` и `swipe_left` исправлена.
- Следующий шаг: live-test этих двух классов, затем запись `swipe_right`.

### H-011: Live dynamic logs дают первичный confidence threshold

Статус: `validated-small-sample`

Источник:
- PostgreSQL `recognition_logs`;
- live-окно: `2026-06-25 14:42:18` - `14:44:00` UTC;
- подробный отчет:
  `docs/experiments/live_dynamic_metrics.md`.

Что увидели:
- всего events: `19`;
- target swipe events: `17`;
- non-target events: `2` (`sh3`, `hand_left`);
- raw target purity: `17/19 = 0.895`.

Threshold sweep:
- `confidence >= 0.50`: target purity `0.944`, target event recall `1.000`;
- `confidence >= 0.60`: target purity `1.000`, target event recall `1.000`;
- `confidence >= 0.65`: target purity `1.000`, target event recall `0.941`.

Вывод:
- Для текущего small live-test лучший кандидат порога: `0.60`.
- При `0.60` оба non-target события отсекаются, а все target swipe events
  остаются.
- Это пока event-level метрика: в `recognition_logs` нет ground truth
  `expected_label`, поэтому attempt-level accuracy и missed attempts нельзя
  честно посчитать автоматически.

Следующий шаг:
- Добавить или вручную вести live-evaluation protocol:
  `expected_label`, `attempt`, `predicted_label`, `confidence`, `result`.
- Повторить тест по `10` попыток на каждый dynamic class.

### H-012: Нужен attempt-level live evaluation mode

Статус: `implemented`

Гипотеза:
- Event-level `recognition_logs` недостаточно для JMLC-метрик: там есть
  predicted label и confidence, но нет `expected_label`, поэтому невозможно
  честно посчитать `Missed` и attempt-level accuracy.

Что сделали:
- На Home добавлен блок `Live evaluation`.
- Пользователь выбирает:
  - `Expected`;
  - `Attempts`;
  - `Timeout` (`0` = ждать жест без авто-пропуска);
  - `Min conf`.
- Контроллер считает попытки автоматически:
  - `Correct`, если accepted prediction совпал с expected label;
  - `Wrong`, если accepted prediction не совпал;
  - `Missed`, если пользователь нажал `Пропуск`; опционально можно включить
    авто-missed через timeout больше `0`.
- Индикация текущего теста перенесена в overlay внутри camera preview, чтобы
  результат считывался рядом с live-кадром.
- Во время evaluation auto-execute временно отключается, чтобы тест метрик не
  запускал системные команды.
- Подробные попытки пишутся в:
  `~/.dplm/logs/live_evaluation.jsonl`.
- Отчет по попыткам собирается командой:
  `python scripts/live_evaluation_report.py --out-md docs/experiments/live_evaluation_report.md --out-json docs/experiments/live_evaluation_report.json`.

Правила текущего протокола:
- recommended attempts: `10` на класс;
- initial min confidence: `0.60`;
- timeout: `0` для ручной проверки без гонки с таймером;
- между попытками убрать руку из кадра, чтобы тот же label не засчитался
  повторно.

Следующий тест:
- `swipe_up`: 10 attempts;
- `swipe_down`: 10 attempts;
- `swipe_left`: 10 attempts;
- после записи `swipe_right`: 10 attempts.

### H-013: Первый attempt-level live-test для `swipe_left`

Статус: `validated-small-sample`

Источник:
- Home -> `Live evaluation`;
- expected label: `swipe_left`;
- attempts: `10`;
- min confidence: `0.80`;
- timeout: `0` (ручной режим без авто-пропуска);
- отчет: `docs/experiments/live_evaluation_report.md`.

Результат:
- raw correct: `8/10`;
- raw wrong: `1/10`;
- raw missed: `1/10`;
- raw accuracy: `0.800`;
- manual note: `missed=1` был случайным нажатием `Пропуск`, не реальным
  пропуском модели;
- adjusted valid attempts: `9`;
- adjusted accuracy: `8/9 = 0.889` (округленно `90%`);
- avg confidence accepted predictions: `0.947`;
- wrong label: `swipe_up` (`1` раз).

Вывод:
- `swipe_left` уже распознается live на уровне около `90%` в ручном
  attempt-level тесте.
- Реальная ошибка пока одна: смешение с `swipe_up`; это сигнал проверить пару
  `swipe_left`/`swipe_up`, а не переписывать весь класс немедленно.

Следующий шаг:
- Повторить такой же test для `swipe_up` и `swipe_down`.
- Если `swipe_up` тоже конфликтует с `swipe_left`, усилить датасет:
  добрать `swipe_left` до `20-30` сэмплов и записывать более чистое движение
  по горизонтали.
- После этого сравнить `dynamic_knn`, `dynamic_svm`, `dynamic_extra_trees` на
  одинаковом наборе.

## Текущий ML-пайплайн

1. Запись:
   - static: `(frames, 21, 2)`;
   - dynamic: `(36, 44)`.
2. Feature extraction:
   - static baseline: `static_mean`;
   - dynamic baseline: `dynamic_stats`.
3. Обучение:
   - CLI: `cv.train_classifier`;
   - поддерживаемые модели: `knn`, `svm`, `extra_trees`, `rf`, `logreg`.
4. Live:
   - Home dropdown `Модель`: `static` / `dynamic`;
   - dynamic live currently reads `models/dynamic_knn.pkl`.

## Следующие эксперименты

1. Перезаписать `hand_left` в новом dynamic-формате `(36, 44)`.
2. Переобучить dynamic-модели в отдельные файлы:
   - `dynamic_knn.pkl`;
   - `dynamic_svm.pkl`;
   - `dynamic_extra_trees.pkl`.
3. Добавить в Home выбор конкретного dynamic model file или model profile.
4. Повторить `scripts.compare_models --target-dim 44`.
5. Зафиксировать live-результаты по каждому dynamic-жесту:
   `swipe_up`, `hand_left`, затем добавить `swipe_down` и `swipe_right`.
