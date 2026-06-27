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

### H-014: Live-test `swipe_up` и `swipe_down` при пороге `0.80`

Статус: `needs-data-fix`

Источник:
- Home -> `Live evaluation`;
- attempts: `10` на класс;
- min confidence: `0.80`;
- timeout: `0` (ручной режим без авто-пропуска);
- отчет: `docs/experiments/live_evaluation_report.md`;
- скрины пользователя от 2026-06-25 20:07 и 20:09.

Raw summary по актуальным live-тестам:
- total attempts: `30`;
- correct: `23`;
- wrong: `6`;
- missed: `1`;
- raw accuracy: `23/30 = 0.767`.

Adjusted summary:
- `missed=1` у `swipe_left` исключен как случайное нажатие `Пропуск`;
- valid attempts: `29`;
- adjusted accuracy: `23/29 = 0.793`.

По классам:
- `swipe_up`: `10/10`, accuracy `1.000`, avg confidence `1.000`;
- `swipe_left`: raw `8/10`, adjusted `8/9 = 0.889` (`90%`);
- `swipe_down`: `5/10`, accuracy `0.500`, avg confidence `0.971`;
- все ошибки `swipe_down` ушли в `swipe_up` (`5/10`).

Вывод:
- Текущий dynamic-пайплайн уже способен стабильно распознавать отдельные
  динамические классы: `swipe_up` прошел live-test на `100%`.
- Главный дефект сейчас не общий, а направленный: модель путает `swipe_down`
  с `swipe_up` при высокой уверенности, значит confidence threshold сам по
  себе проблему не решит.
- Вероятная причина: текущие признаки/датасет недостаточно кодируют знак
  вертикального смещения или `swipe_down` записан несимметрично относительно
  `swipe_up`.

Следующий шаг:
- Не добавлять новые классы до фикса `swipe_down`.
- Проверить feature vector для `swipe_up`/`swipe_down`: отдельно вывести
  агрегаты `dy`, `path`, `dominant_axis`, `start_y`, `end_y`.
- Добавить в dynamic features явные trajectory-признаки:
  `delta_x`, `delta_y`, `abs_delta_x`, `abs_delta_y`, `path_length`,
  `direction_cos`, `direction_sin`.
- Переобучить `dynamic_knn`, `dynamic_svm`, `dynamic_extra_trees` и повторить
  live-test `swipe_down` 10 раз.

### H-015: Trajectory-признаки для разделения `swipe_up`/`swipe_down`

Статус: `offline-validated`

Гипотеза:
- Ошибка `swipe_down -> swipe_up` возникала не из-за низкого confidence, а из-за
  недостаточно явного кодирования направления движения в feature vector.
- Если добавить компактные признаки траектории запястья, модель должна лучше
  различать знак вертикального смещения.

Что сделали:
- В `dynamic_stats` добавлены признаки:
  `delta_x`, `delta_y`, `abs_delta_x`, `abs_delta_y`, `path_length`,
  `direction_cos`, `direction_sin`.
- Размерность dynamic-признаков для raw `44` стала `271` вместо `264`.
- Сохранена совместимость со старыми dynamic-моделями `raw * 6`: инференс
  умеет вывести raw dimension и для старой, и для новой размерности.
- Исправлен баг обучения: `--expect-dim 44` теперь трактуется как raw
  per-frame dimension и не обрезает итоговый `dynamic_stats` vector до `44`.

Диагностика данных:
- `swipe_up`: `n=30`, mean `dy=-0.2172`, median `dy=-0.1820`,
  median `path=0.9624`;
- `swipe_down`: `n=10`, mean `dy=+0.4501`, median `dy=+0.4876`,
  median `path=0.5207`;
- `swipe_left`: `n=10`, mean `dx=-0.3618`, median `dx=-0.3823`.

Offline comparison:
- команда:
  `python scripts/compare_models.py --feature-modes dynamic_stats --models knn,svm,extra_trees --target-dim 44 --min-samples-per-class 10`;
- лучший кандидат: `dynamic_stats + extra_trees`;
- accuracy: `0.8209`;
- macro F1: `0.8136`;
- `swipe_down`: precision `0.9091`, recall `1.0000`, F1 `0.9524`;
- `swipe_up`: precision `1.0000`, recall `0.9667`, F1 `0.9831`;
- в CV-матрице лучшей модели `swipe_down` больше не путается со
  `swipe_up`.

Модели:
- `models/dynamic_knn.pkl`: feature dim `271`;
- `models/dynamic_svm.pkl`: feature dim `271`;
- `models/dynamic_extra_trees.pkl`: feature dim `271`;
- metadata: `models/dynamic_feature_dim.txt = 271`,
  `models/dynamic_feature_mode.txt = dynamic_stats`.

Вывод:
- Гипотеза подтверждена offline: траекторные признаки явно разделяют
  `swipe_up` и `swipe_down` на записанном датасете.
- `extra_trees` сейчас сильнее `knn` и `svm` по cross-validation, но live
  подтверждение еще нужно.

Следующий шаг:
- Перезапустить приложение, чтобы оно подхватило новую модель и feature dim.
- Повторить `Live evaluation` для `swipe_down`: `10` attempts,
  threshold `0.80`, timeout `0`.
- Цель: минимум `8/10`, идеал `9/10+`.
- Если `swipe_down` останется ниже `8/10`, следующий фикс: либо переключить
  live dynamic profile на `dynamic_extra_trees.pkl`, либо дозаписать
  `swipe_down` до `20` сэмплов с разной скоростью/амплитудой.

### H-016: Live-validation `swipe_down` после trajectory-признаков

Статус: `live-validated-small-sample`

Источник:
- Home -> `Live evaluation`;
- expected label: `swipe_down`;
- attempts: `10`;
- min confidence: `0.80`;
- timeout: `0`;
- модель: обновленный `models/dynamic_knn.pkl`;
- feature mode: `dynamic_stats`, feature dim `271`;
- скрин пользователя от 2026-06-25 20:28;
- отчет: `docs/experiments/live_evaluation_report.md`.

Результат последнего completed run:
- raw correct: `8/10`;
- raw wrong: `1/10`;
- raw missed: `1/10`;
- raw accuracy: `0.800`;
- accepted attempts: `9`;
- accepted accuracy: `8/9 = 0.889` (`89%`, в UI округлено до `90%`);
- avg confidence accepted predictions: `0.924`;
- wrong label: `swipe_up` (`1` раз).

Сравнение с H-014:
- было: `swipe_down = 5/10`, wrong `5/10`, все ошибки в `swipe_up`;
- стало: `swipe_down = 8/10`, wrong `1/10`, missed `1/10`;
- improvement raw accuracy: `+30 pp`;
- target `>= 8/10` достигнут на первом live-прогоне после фикса.

Вывод:
- Trajectory-признаки подтверждены не только offline, но и live: конфликт
  `swipe_down -> swipe_up` заметно снизился.
- Оставшаяся ошибка все еще уходит в `swipe_up`, значит пара `up/down`
  остается главным кандидатом для следующего улучшения.
- Один `missed` показывает, что кроме классификации важен gate/сегментация
  попытки: иногда жест не засчитывается как accepted prediction.

Следующий шаг:
- Повторить `swipe_down` еще один run на `10` attempts для устойчивости.
- Если снова будет `8/10+`, считать `swipe_down` временно стабилизированным и
  переходить к `swipe_right`.
- Если результат упадет ниже `8/10`, проверить live на
  `models/dynamic_extra_trees.pkl`, потому что offline он лучший.

### H-017: Повторный live-test `swipe_down` на dynamic KNN

Статус: `model-unstable-live`

Источник:
- Home -> `Live evaluation`;
- expected label: `swipe_down`;
- attempts: `10`;
- min confidence: `0.80`;
- timeout: `0`;
- модель: `models/dynamic_knn.pkl`;
- feature mode: `dynamic_stats`, feature dim `271`;
- скрин пользователя от 2026-06-25 20:48;
- отчет: `docs/experiments/live_evaluation_report.md`.

Результат последнего completed run:
- raw correct: `7/10`;
- raw wrong: `2/10`;
- raw missed: `1/10`;
- raw accuracy: `0.700`;
- accepted attempts: `9`;
- accepted accuracy: `7/9 = 0.778`;
- avg confidence accepted predictions: `0.968`;
- wrong label: `swipe_up` (`2` раза).

Сравнение:
- до trajectory-признаков: `5/10`;
- первый run после trajectory-признаков: `8/10`;
- второй run после trajectory-признаков: `7/10`;
- итог: стало лучше baseline, но `dynamic_knn` нестабилен для `swipe_down`.

Вывод:
- Trajectory-признаки помогли, но KNN не закрывает целевую стабильность
  `>= 8/10` на двух подряд live-run.
- Ошибки по-прежнему концентрируются в `swipe_up`, значит проблема именно в
  границе пары `up/down`, а не в случайной путанице всех классов.
- Следующий честный ML-шаг: проверить live на другой модели, а не добирать
  данные вслепую.

Что сделали:
- Добавлен выбор dynamic model profile в Home: `knn`, `svm`, `extra_trees`.
- Embedded live inference теперь может использовать:
  - `models/dynamic_knn.pkl`;
  - `models/dynamic_svm.pkl`;
  - `models/dynamic_extra_trees.pkl`.
- При смене profile текущий embedded infer пересоздается, чтобы следующий
  live-run шел уже на выбранной модели.

Следующий шаг:
- Переключить на главной:
  - `Модель`: `dynamic`;
  - `Dynamic`: `extra_trees`;
  - `Live evaluation`: `swipe_down`, `10` attempts, threshold `0.80`,
    timeout `0`.
- Если `extra_trees >= 8/10`, использовать его как основной dynamic-profile.
- Если `extra_trees` тоже ниже `8/10`, добрать `swipe_down` до `20` сэмплов.

### H-018: Live-проверка SVM/ExtraTrees не подтверждает offline-CV

Статус: `rejected-live`

Наблюдение:
- Пользователь проверил `dynamic_svm.pkl` и `dynamic_extra_trees.pkl` в live.
- Результаты субъективно и по quick-run хуже KNN: `swipe_down` продолжает
  уходить в `swipe_up`.
- В текущем логе есть остановленный run `swipe_down`: `0/2`, wrong `2/2`,
  predicted `swipe_up`, но старый формат лога еще не сохранял
  `dynamic_model_profile`, поэтому этот фрагмент нельзя строго привязать к
  SVM или ExtraTrees.

Вывод:
- Offline-CV переоценил `extra_trees` для реального live-потока.
- Для текущей демонстрации основной кандидат остается `dynamic_knn.pkl`, но он
  требует усиления данных/сегментации.
- Следующий ML-шаг не смена модели, а улучшение датасета `swipe_down` и
  измерение live-run с сохраненным `dynamic_model_profile`.

Что сделали:
- В live-evaluation JSONL добавлено логирование:
  - `recognition_model_mode`;
  - `dynamic_model_profile`.
- Отчет `docs/experiments/live_evaluation_report.md` теперь показывает модель
  в Latest Completed Run и Recent Runs.
- Старые записи без профиля отображаются как `n/a`; новые будут иметь вид
  `dynamic:knn`, `dynamic:svm`, `dynamic:extra_trees`.

Следующий шаг:
- Вернуться на `Dynamic: knn`.
- Добрать `swipe_down` с `10` до `20` сэмплов:
  - 5 медленных чистых вниз;
  - 5 быстрых вниз;
  - рука каждый раз стартует выше и заканчивает ниже;
  - между сэмплами убрать руку из кадра/вернуть в стартовую позу.
- Переобучить `dynamic_knn.pkl`.
- Повторить live-test `swipe_down`, target `>= 8/10` на двух run подряд.

### H-019: Нужен канон движения для каждого dynamic-жеста

Статус: `planned`

Гипотеза:
- Часть ошибок live-распознавания может появляться не из-за модели, а из-за
  нестабильной формы жеста при записи и тестировании.
- Для dynamic-классов модель учит траекторию: старт, направление, амплитуду,
  скорость и окончание движения.
- Если `swipe_down` сегодня записан как короткий диагональный жест, а завтра
  тестируется как длинный вертикальный, это становится label noise.

Канон для `swipe_down` перед новой записью:
- старт: открытая ладонь/указатель в верхней части кадра;
- движение: заметно вниз, почти без ухода влево/вправо;
- амплитуда: не маленький рывок, а движение примерно на `25-40%` высоты кадра;
- финал: рука остается ниже стартовой точки на долю секунды;
- сброс: между сэмплами убрать руку из кадра или вернуть ее в верхнюю стартовую
  позицию, чтобы следующий sample не начинался из середины прошлого.

Протокол перезаписи `swipe_down`:
- удалить старый класс/семплы `swipe_down`;
- записать `20` новых сэмплов в одном стиле:
  - `10` обычных чистых движений;
  - `5` чуть медленнее;
  - `5` чуть быстрее;
- не добавлять диагональные и слишком короткие движения в первый новый датасет;
- после обучения проверить два live-run подряд с `Dynamic: knn`, target
  `>= 8/10`.

Следующая продуктовая доработка:
- Добавить в интерфейс запись "gesture guide" для каждого класса: короткое
  описание канона, пример стартовой/конечной позиции и подсказку перед записью.
- Это должно снизить зависимость качества ML от памяти пользователя и сделать
  демонстрацию более воспроизводимой.

### H-020: Dynamic data analysis показывает offline/live distribution shift

Статус: `accepted-analysis`

Артефакты:
- `scripts/dynamic_data_analysis.py`;
- `docs/experiments/dynamic_data_analysis.md`;
- `docs/experiments/dynamic_data_analysis.json`.

Команда воспроизведения:

```bash
python -m scripts.dynamic_data_analysis --out-md docs/experiments/dynamic_data_analysis.md --out-json docs/experiments/dynamic_data_analysis.json
```

Данные:
- анализировались классы `swipe_*`;
- классы: `3`;
- сэмплы: `50`;
- raw dim: `44`;
- target dim: `44`;
- `swipe_down`: `10` сэмплов;
- `swipe_left`: `10` сэмплов;
- `swipe_up`: `30` сэмплов.

Ключевые признаки по mutual information:
- `direction_sin`: `0.8027`;
- `dy`: `0.6806`;
- `vertical_ratio`: `0.6521`;
- `horizontal_ratio`: `0.6521`;
- `direction_cos`: `0.6463`;
- `straightness`: `0.6400`;
- `motion_energy`: `0.4813`;
- `abs_dx`: `0.4597`.

Offline:
- модель: `knn(distance)`;
- feature mode: `dynamic_stats`;
- CV folds: `5`;
- accuracy: `0.9800`;
- macro F1: `0.9785`;
- единственная offline-ошибка: `swipe_up/sample_0003.npy` ->
  `swipe_left`.

Live:
- источник: `~/.dplm/logs/live_evaluation.jsonl`;
- runs: `9`;
- attempts: `62`;
- correct: `39`;
- wrong: `12`;
- missed: `11`;
- accuracy: `0.6290`;
- все wrong labels: `swipe_up:12`;
- `swipe_down` live: `21/42`, accuracy `0.5000`;
- `swipe_left` live: `8/10`, accuracy `0.8000`;
- `swipe_up` live: `10/10`, accuracy `1.0000`.

Вывод:
- На сохраненных файлах KNN почти идеально разделяет dynamic-классы, но live
  распознавание слабее. Это distribution shift между записанными sample и тем,
  как жест показывается в live.
- `swipe_down` не обязательно "плохой" в файлах: saved `swipe_down` выглядит
  достаточно прямым (`dy median = 0.4876`, `straightness median = 0.9194`).
- Главная зона риска: `swipe_up` стал слишком широким/шумным классом:
  `direction_ok_rate = 0.6667`, `low_straightness = 20/30`,
  `expected_up = 10/30`, `axis_drift = 10/30`.
- Поэтому live `swipe_down -> swipe_up` может быть не только проблемой
  `swipe_down`, но и следствием слишком широкого decision region у `swipe_up`.

Решения по обработке данных:
- держать отдельный dynamic pipeline до подтверждения unified model в live;
- балансировать dynamic-классы минимум до `20` сэмплов на класс;
- использовать quality gates при записи:
  - знак ожидаемого направления;
  - minimum global path;
  - axis alignment;
  - straightness;
  - pose motion energy;
- считать высокий offline CV при слабом live как distribution shift;
- следующий preprocessing experiment: выделять active motion segment и
  resample до `36` кадров, чтобы скорость меньше ломала признаки.

### H-021: Handedness может быть скрытым фактором для dynamic-жестов

Статус: `planned`

Гипотеза:
- То, какой рукой пользователь показывает жест, может влиять на распознавание.
- В текущем dynamic sample сохраняются landmarks и global wrist movement, но
  не сохраняется явная метка `left_hand` / `right_hand`.
- Даже если направление `swipe_down` одинаковое, форма руки может отличаться:
  зеркальность пальцев, угол кисти, закрытие landmark-точек, стартовая позиция
  в кадре и траектория wrist могут сместить feature vector.

Как это влияет на модель:
- KNN сравнивает новый live-вектор с ближайшими сохраненными sample.
- Если класс записан правой рукой, а live жест показан левой рукой, новый
  вектор может оказаться дальше от своего класса и ближе к шумному соседнему
  классу.
- Для свайпов это особенно важно, потому что модель использует и motion
  признаки (`dx`, `dy`, `path_length`, `direction_*`), и pose-признаки кисти.

Решение для текущей перезаписи:
- Сейчас записывать все dynamic-классы одной и той же рукой.
- Использовать ту же руку при live-тестах.
- В журнале записать, какой рукой сделана новая версия датасета.
- Не смешивать левую и правую руку внутри одного класса, пока нет отдельной
  метки/аугментации handedness.

Контрольный эксперимент:
- После стабильного baseline одной рукой записать мини-набор другой рукой:
  `5` sample на `swipe_down`, `5` на `swipe_up`, `5` на `swipe_left`.
- Обучить временную модель и сравнить live:
  - train right / test right;
  - train right / test left;
  - mixed train / test both.
- Если mixed train улучшит обе руки без падения точности, добавить
  handedness-balanced recording protocol.
- Если mixed train ухудшит качество, разделять profile или явно хранить
  handedness metadata.

### H-022: Training taxonomy защищает dynamic-модель от static-классов

Статус: `implemented`

Проблема:
- Dynamic-модель обучалась через общий список активных DB-жестов.
- Из-за этого в `models/dynamic_*.pkl` могли попадать static/quasi-static
  классы, и dynamic-пайплайн начинал распознавать жест даже на статичной руке.

Что сделали:
- Добавлен `configs/gesture_taxonomy.json` с типами:
  - `static`;
  - `quasi_static`;
  - `dynamic`.
- Добавлен сервис `app/services/gesture_taxonomy.py`.
- В training pipeline добавлен `training_scope`.
- Dynamic-вкладка обучения теперь запускает обучение с
  `training_scope="dynamic"`.
- Controller передает в `cv.train_classifier` только labels из dynamic-scope.

Текущее правило:
- `swipe_down`, `swipe_left`, `swipe_right`, `swipe_up` -> `dynamic`;
- `hand_left` временно вынесен в `quasi_static`, пока не будет перезаписан в
  новом dynamic-формате;
- неизвестные labels по умолчанию считаются `static`, чтобы они не попадали в
  dynamic-модель случайно.

Проверка:
- unit tests: `38 passed`;
- `configs/gesture_taxonomy.json` валиден;
- `py_compile` для `gesture_taxonomy.py`, `controller.py`, `training.py`
  проходит.

Следующий шаг:
- Добавить motion gate / Recognition Router Agent, чтобы dynamic-модель вообще
  не вызывалась на статичном или неканоничном движении.

### H-023: Motion gate защищает live dynamic inference от статичных кадров

Статус: `implemented`

Проблема:
- Даже после фильтрации training labels dynamic-модель может получать окно,
  где рука почти не двигалась.
- Для KNN это опасно: модель все равно ищет ближайший известный dynamic-класс
  и может вернуть `swipe_*` на статичной руке.

Что сделали:
- В `GestureOnlineInfer` добавлен runtime dynamic motion gate для новых
  global dynamic samples с размерностью `44`.
- Gate считается по trajectory features:
  - `dx`, `dy`;
  - `path_length`;
  - `displacement`;
  - направление движения.
- Если `path_length < 0.12` или `displacement < 0.06`, dynamic-модель не
  вызывается вообще.
- Если модель предсказала направление, которое противоречит фактическому
  движению (`swipe_up` при положительном `dy`, `swipe_left` при положительном
  `dx` и т.д.), результат очищается до no gesture.

Что получилось:
- Статичная рука больше не должна исполнять dynamic-жесты.
- Ошибки вида `swipe_down -> swipe_up` теперь частично блокируются на уровне
  признаков движения, а не только классификатором.
- Legacy dynamic samples с `42` признаками не ломаются: gate включается только
  для global motion формата `>= 44`.

Проверка:
- unit tests: `36 passed`;
- `py_compile` для runtime inference и связанных Flet модулей проходит.

Следующий шаг:
- Доделать Recognition Router Agent: единый live-интерфейс без ручного выбора
  static/dynamic, route metrics и явный `no_gesture`.

### H-024: Recognition Router Agent объединяет static и dynamic inference

Статус: `implemented`

Проблема:
- На главной странице пользователь вручную выбирал `static` или `dynamic`.
- Для полноценной системы это слабое место: пользователь не должен заранее
  знать, какой тип жеста он сейчас показывает.
- Dynamic-модель также не должна побеждать, если она вдруг вернула
  quasi-static/static label.

Что сделали:
- Добавлен `GestureRecognitionRouter` как runtime routing-agent.
- Добавлен режим `recognition_model_mode="auto"` и он стал режимом по
  умолчанию в Flet controller.
- В `auto` controller создает два канала:
  - static inference: `models/knn.pkl`;
  - dynamic inference: выбранный `models/dynamic_*.pkl`.
- Router выбирает dynamic-result только если:
  - есть label;
  - confidence `>= 0.60`;
  - taxonomy подтверждает, что label относится к `dynamic`.
- Если dynamic не прошел guard, используется static-result.
- В output добавляется route metadata:
  - `route`;
  - `static_label`, `static_confidence`;
  - `dynamic_label`, `dynamic_confidence`;
  - `dynamic_threshold`.

Что получилось:
- Главная страница теперь может работать как unified live interface через
  `auto`.
- Dynamic-классы больше не смешиваются с quasi-static/static на уровне
  runtime routing.
- Для live-evaluation можно собирать метрики именно в `auto` режиме.

Ограничение текущей версии:
- Static и dynamic inference пока используют два отдельных MediaPipe detector.
- Это правильно для MVP и тестируемо, но следующим инженерным улучшением надо
  вынести shared detection, чтобы не платить двойную стоимость на каждый кадр.

Проверка:
- unit tests: `60 passed`;
- `py_compile` для router, controller, home view и online inference проходит.

Следующий шаг:
- Добавить route metrics / MLOps-срез: сколько раз router выбрал
  `static`, `dynamic`, `none`, и какие confidence/ошибки были в live-test.

### H-025: Route metrics делают live-evaluation пригодным для MLOps

Статус: `implemented`

Проблема:
- После добавления `auto` недостаточно видеть только `correct/wrong/missed`.
- Для анализа ошибок нужно понимать, какая ветка дала решение:
  `static`, `dynamic` или `none`.
- Иначе невозможно отличить ошибку модели от ошибки router-policy.

Что сделали:
- `_dispatch_infer_result` передает `router` metadata в live-evaluation.
- Attempt rows в `live_evaluation.jsonl` теперь могут содержать:
  - `route`;
  - `static_label`, `static_confidence`;
  - `dynamic_label`, `dynamic_confidence`.
- Run rows получают `route_counts`.
- `scripts/live_evaluation_report.py` читает route metadata и показывает
  routes в:
  - `By Label`;
  - `Latest Completed Run By Label`;
  - `Recent Runs`.

Что получилось:
- После live-test можно увидеть не только accuracy, но и то, чем она
  объясняется: например, `dynamic:8, none:2`.
- Это закрывает первый практический MLOps-срез для unified inference.

Проверка:
- unit tests: `63 passed`;
- `py_compile` для controller, router и live evaluation report проходит.

Следующий шаг:
- Добавить команду/кнопку генерации markdown/json отчета по текущему
  `live_evaluation.jsonl` прямо из интерфейса или developer-панели.

### H-026: Router policy должна запрещать dynamic-label через static fallback

Статус: `implemented`

Анализ:
- Router не подключен к LLM и не должен вызывать LLM на каждом кадре:
  per-frame распознавание должно быть быстрым, детерминированным и
  воспроизводимым.
- LLM полезнее использовать вне hot path: анализ качества записей,
  рекомендации по перезаписи классов, генерация отчетов и подсказки
  пользователю при сборе датасета.
- Слабое место текущего router было в fallback-логике: если static-модель
  случайно вернула `swipe_*`, этот dynamic-label мог пройти через static-route.
- Еще одна проблема: у router не было явных причин отказа, поэтому в live-log
  было видно `static/dynamic/none`, но не было видно, почему dynamic был
  отклонен.

Что сделали:
- Добавлен static confidence threshold `0.50`.
- Static-route теперь запрещает labels с типом `dynamic`.
- Dynamic labels (`swipe_*`) могут пройти только через dynamic-route после:
  - motion gate;
  - dynamic confidence threshold `0.60`;
  - taxonomy check.
- Router payload теперь содержит:
  - `selected_reason`;
  - `static_type`, `static_reject_reason`, `static_threshold`;
  - `dynamic_type`, `dynamic_reject_reason`, `dynamic_threshold`.
- Live-evaluation JSONL сохраняет эти reason-поля.
- В `auto` режиме dropdown dynamic-профиля теперь видим, чтобы было понятно,
  какой `dynamic_knn/svm/extra_trees` участвует в unified routing.

Что получилось:
- Статичная рука больше не должна получать `swipe_*` через static fallback,
  даже если static-модель загрязнена dynamic-классами.
- Ошибки router-policy теперь можно анализировать по JSONL, а не только по
  итоговой accuracy.

Проверка:
- unit tests: `65 passed`;
- `py_compile` для router, controller, home view и live evaluation report
  проходит.

Следующий шаг:
- Прогнать live-evaluation в `auto` режиме и посмотреть распределение:
  `route`, `selected_reason`, `dynamic_reject_reason`.

### H-027: Auto routing должен ждать движение, а KNN должен опираться на траекторию

Статус: `implemented`, требуется повторная live-валидация

Наблюдение:
- Live-run `swipe_up` в `auto`: `2/10 correct`, `8/10 wrong`.
- Семь ошибок были не ошибками dynamic KNN, а перехватом static-route:
  `gun` с confidence около `1.0` выбирался, пока dynamic-окно еще набирало
  `36` кадров.
- Live-run `swipe_left` был остановлен после `5` попыток:
  `1 correct`, `4 wrong`; все решения пришли через dynamic-route, но модель
  выбирала `swipe_up` или `swipe_down`.
- На сохраненных `70` dynamic-сэмплах KNN дает `accuracy=1.0` и
  `macro F1=1.0`. Это подтверждает distribution shift между записью и live,
  а не отсутствие разделимости в train dataset.

Гипотезы:
1. Static fallback срабатывает раньше, чем temporal-модель успевает увидеть
   движение.
2. В `dynamic_stats` семь global trajectory features проигрывают по вкладу
   сотням pose/velocity features в евклидовом расстоянии KNN.
3. После распознавания в rolling window остается обратное движение руки, из-за
   чего оно может стать следующей попыткой.

Что сделали:
- Dynamic inference отдает temporal phase:
  `warming_up`, `active` или `idle`.
- Router удерживает static fallback в фазах `warming_up` и `active`.
  Статический жест остается доступен после короткого удержания до `idle`.
- Global trajectory block в `dynamic_stats` получает вес `8.0` перед KNN.
- Направление проходит axis-dominance guard: горизонтальный swipe не может
  быть принят как `up/down`, если `abs(dx)` доминирует над `abs(dy)`.
- Вероятности модели rerank-ятся только среди классов, совместимых с
  измеренным направлением.
- После подтвержденного dynamic-жеста temporal window очищается, чтобы возврат
  руки не считался новым свайпом.
- `dynamic_knn.pkl` переобучен только на:
  `swipe_down=20`, `swipe_left=30`, `swipe_up=20`.

Offline-проверка:
- `70` samples, `3` classes, feature dimension `271`;
- `5-fold CV accuracy=1.0`, `macro F1=1.0`;
- профильные unit tests: `56 passed`.

Критерий live-приемки:
- для каждого класса два run по `10` попыток в `auto`;
- `accuracy >= 80%`, `dynamic route rate >= 80%`;
- `static hijack rate = 0%`;
- directional confusion не более `1/10`.

Следующий шаг:
- Повторить baseline без переобучения между прогонами и сравнить новые run с
  зафиксированными результатами H-027.

### H-028: Lag в auto может снижать качество temporal detection

Статус: `implemented`, требуется camera benchmark и повторный live-test

Наблюдение:
- Видео в Flet заметно отстает во время `auto`.
- Dynamic inference использует окно `36` кадров. Если pipeline вместо `30 FPS`
  обрабатывает, например, `10 FPS`, окно описывает уже `3.6` секунды вместо
  `1.2`. В него могут одновременно попасть сам свайп, остановка и возврат руки.
- До оптимизации static и dynamic `GestureOnlineInfer` независимо запускали
  MediaPipe на одном кадре.
- `HomeView` и скрытый `TrainingView` были постоянно подписаны на camera event.
  Каждый кадр мог дважды кодироваться в base64 и создавать несколько
  `page.run_thread` задач без backpressure.

Гипотезы:
1. Два MediaPipe detector в `auto` уменьшают effective FPS и растягивают
   temporal window.
2. Обновление скрытой вкладки и неограниченная очередь Flet создают визуальный
   lag даже при приемлемой скорости ML.
3. Старые кадры в backend-буфере камеры увеличивают end-to-end latency.

Что сделали:
- Router запускает hand detection один раз и передает один набор
  `DetectedHand` в static и dynamic классификаторы.
- Secondary dynamic infer создается без собственного MediaPipe detector.
- Preview ограничен `20 FPS`, но ML продолжает работать на configured target
  FPS. JPEG, overlay и Flet event выполняются только для preview-кадров.
- `HomeView` и `TrainingView` игнорируют кадры, пока скрыты.
- Для каждой видимой вкладки разрешен максимум один pending UI frame; лишние
  кадры отбрасываются вместо накопления задержки.
- На главной обновляется только camera `Image`, а не вся Flet-страница.
- Для OpenCV запрашивается camera buffer size `1`, если backend это
  поддерживает.
- Каждые `5` секунд приложение пишет
  `~/.dplm/logs/runtime_performance.jsonl`:
  - `shared_detection_rate`;
  - `inference_ms_avg`;
  - `inference_ms_p95`;
  - `detection_ms_avg`;
  - `inference_fps_capacity`.

Проверка:
- Unit-test подтверждает один detector call и два classifier calls на кадр.
- Unit-test подтверждает coalescing: два camera events создают одну UI-задачу.
- Unit-test подтверждает, что скрытая training-вкладка не ставит frame update.
- Headless MediaPipe benchmark не принят как доказательство: в тестовом
  окружении detector не создал GL context и вернул ранний empty result.
- Фактическая latency должна измеряться в запущенном приложении с камерой.

Критерий runtime-приемки:
- `shared_detection_rate = 1.0` в режиме `auto`;
- `inference_ms_p95 <= 33.3 ms` для target `30 FPS`, либо capacity не ниже
  фактически выбранного FPS;
- preview не накапливает заметную задержку относительно движения;
- после оптимизации повторить H-027: по два run на каждый dynamic-класс.

Следующий шаг:
- Запустить приложение минимум на `15` секунд в `auto`, затем сравнить
  `runtime_performance.jsonl` с live accuracy и route metrics.

## Текущий ML-пайплайн

1. Запись:
   - static: `(frames, 21, 2)`;
   - dynamic: `(36, 44)`.
2. Feature extraction:
   - static baseline: `static_mean`;
   - dynamic baseline: `dynamic_stats` with trajectory features weighted by
     `8.0` for distance-based KNN.
3. Обучение:
   - CLI: `cv.train_classifier`;
   - поддерживаемые модели: `knn`, `svm`, `extra_trees`, `rf`, `logreg`.
4. Live:
   - Home dropdown `Модель`: `auto` / `static` / `dynamic`;
   - Home dropdown `Dynamic`: `knn` / `svm` / `extra_trees`;
   - `auto` запускает static + dynamic channels и выбирает route через
     `GestureRecognitionRouter`;
   - current live candidate: `models/dynamic_knn.pkl`.

## Следующие эксперименты

1. Прогнать live-evaluation в `auto` режиме по `swipe_up`, `swipe_down`,
   `swipe_left`.
2. Перезаписать `swipe_down`, `swipe_up`, `swipe_left` по канону H-019/H-021:
   одна и та же рука, `20` сэмплов на класс.
3. Переобучить `dynamic_knn.pkl`.
4. Повторить live-test по каждому классу: target `>= 8/10` на двух run подряд.
5. Если `swipe_down` все еще уходит в `swipe_up`, почистить/перезаписать
   шумные `swipe_up` sample из `docs/experiments/dynamic_data_analysis.md`.
6. После стабильного baseline проверить handedness experiment из H-021.
7. Если стабильно, добавить `swipe_right` и записать `10-20` sample.
8. Перезаписать `hand_left` в новом dynamic-формате `(36, 44)`.
