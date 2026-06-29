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

### H-029: Dynamic gesture должен быть событием, а не произвольным окном

Статус: `implemented`, требуется повторная live-валидация

Результат H-028 на реальной камере:
- `shared_detection_rate=1.0` во всех пяти окнах.
- Четыре окна: `inference_ms_avg=14.9-18.4`,
  `inference_ms_p95=19.3-26.6`, capacity `54-67 FPS`.
- Одно окно со spike: average `28.7 ms`, p95 `58.5 ms`, но average capacity
  осталась `34.85 FPS` при target `30`.
- Вывод: производительность могла усиливать ошибки, но не является основной
  причиной нестабильного распознавания.
- Сообщение `hend expected=4 current=0` означает, что finger-count guard
  отклонил несовместимый static prediction. Это защитный отказ, а не
  подтвержденный жест.

Концептуальная проблема:
- Train sample записывается как один управляемый фрагмент после стабильной
  стартовой позы.
- Старый live inference классифицировал последние `36` кадров на каждом кадре.
- Такое окно могло содержать ожидание, только часть свайпа, остановку и возврат
  руки одновременно.
- Скорость пользователя меняла не только velocity, но и долю полезного
  движения внутри фиксированного окна.

Гипотеза:
- Сначала нужно детерминированно выделить одно законченное движение, и только
  затем передавать его ML-модели.
- Train и live должны применять одинаковые trim/resample операции.

Что сделали:
- Добавлен `DynamicMotionSegmenter` со состояниями:
  `warming_up -> idle -> active -> completed -> cooldown`.
- Начало определяется по global wrist path/displacement.
- Конец определяется по пяти последовательным кадрам покоя.
- Variable-length active segment обрезается от статических краев и линейно
  ресемплируется до `36` кадров.
- `dynamic_stats` применяет ту же canonical normalization при обучении и
  inference.
- Модель делает один prediction после `completed`, а не prediction на каждом
  sliding window.
- Prediction повторяется два кадра только для confirmation policy, затем
  emitted prediction подтверждается без сброса cooldown state.
- Cooldown блокирует немедленный возврат руки как новый жест.
- В `auto` статический класс требует deliberate dwell `15` кадров.
- Во время live-test dynamic-класса static-route показывается, но не
  засчитывается как ошибочная попытка.
- Route log дополнен `dynamic_phase` и `dynamic_segment_frames`.
- `dynamic_knn.pkl` переобучен на canonical active segments.

Offline-проверка:
- `5-fold CV accuracy=1.0`, `macro F1=1.0` на `70` исходных samples.
- Создано `210` speed/padding augmentations: `18`, `36`, `60` motion frames с
  разной длиной покоя по краям.
- На этих вариантах сохранено `210/210 correct`, robustness accuracy `1.0`.
- Unit tests покрывают static hand, разные скорости, motion stop, resampling и
  cooldown возврата.
- Расширенный ML/runtime набор: `92 passed`.

Ограничение:
- Жест теперь считается завершенным после короткой остановки руки в финальной
  точке. Это осознанная event boundary, а не таймер попытки.
- Live accuracy еще не измерена после H-029, поэтому production-гипотеза пока
  не считается подтвержденной.

Критерий live-приемки:
- По два run `10` попыток для `swipe_up`, `swipe_down`, `swipe_left`.
- После свайпа удерживать руку в финальной точке примерно `0.2` секунды.
- `accuracy >= 80%`, `static hijack=0%`, directional confusion `<=1/10`.
- В correct/wrong rows ожидается `dynamic_phase=completed`.

Следующий шаг:
- Перезапустить приложение, чтобы загрузить новую модель и state machine, и
  провести первый run для каждого dynamic-класса без переобучения между ними.

### H-030: Dynamic representation должна быть инвариантна к позиции и дистанции

Статус: `implemented`, требуется live near/mid/far validation

Наблюдение пользователя:
- Качество сильно зависит от начальной области кадра.
- Если обучение выполнялось на одной дистанции от камеры, на другой дистанции
  распознавание заметно ухудшается.

Профиль текущего датасета:
- `swipe_up`: start x `0.640-0.792`, start y `0.893-1.011`,
  displacement median `0.542`.
- `swipe_down`: start x `0.688-0.784`, start y `0.301-0.465`,
  displacement median `0.493`.
- `swipe_left`: start x `0.713-0.878`, start y `0.697-0.873`,
  displacement median `0.465`.
- Начальная позиция сильно коррелирует с label: up записан снизу, down сверху,
  left справа.
- Legacy sample `(36, 44)` содержит normalized pose и wrist `x/y`, но не
  projected hand size. Поэтому дистанцию старых записей нельзя измерить
  напрямую. Это зафиксированное ограничение датасета.

Разбор причин:
- Pose landmarks уже нормализованы относительно wrist и hand scale функцией
  `normalize_landmarks`, поэтому pose block инвариантен к переносу и масштабу.
- Global trajectory использовала delta, но сохраняла абсолютную экранную
  амплитуду.
- Fixed onset/path/still thresholds работали в координатах кадра. При удалении
  от камеры физически одинаковый жест становился короче и мог не запустить
  segmenter.

Гипотеза:
- ML должен видеть направление и форму траектории, но не место старта и не
  projected amplitude.
- Порог выделения движения должен измеряться относительно размера руки в
  текущем кадре.

Что сделали:
- После active trim/resample wrist trajectory переносится в `(0, 0)`.
- Полная displacement-норма приводится к reference `0.5`; направление и
  относительная форма сохраняются.
- Segmenter получает bbox diagonal руки как `motion_scale`.
- Onset path, onset displacement и still-step автоматически уменьшаются для
  маленькой руки вдали от камеры; для старых вызовов остается absolute
  fallback.
- Dynamic KNN переобучен на новой canonical representation.
- Live route metadata и `live_evaluation.jsonl` дополнены
  `dynamic_motion_scale`.
- Новые dynamic samples сохраняют sidecar `sample_XXXX.meta.json`:
  median/min/max projected hand scale. NPY остается `(frames, 44)`, поэтому
  training compatibility не ломается.
- При удалении sample его metadata-sidecar также удаляется.

Offline-проверка:
- Position invariance unit-test переносит gesture в другую часть кадра.
- Scale invariance unit-test уменьшает амплитуду в `5` раз и получает тот же
  canonical sequence.
- Far-camera unit-test подтверждает onset для projected displacement `0.08`,
  который не проходил прежний absolute threshold `0.04` на пяти кадрах.
- `5-fold CV accuracy=1.0`, `macro F1=1.0` на `70` samples.
- Проверено `210` combinations:
  - amplitude factors `0.2`, `0.5`, `1.0`;
  - start positions `(0.2,0.2)`, `(0.5,0.5)`, `(0.8,0.75)`;
  - duration `18`, `36`, `60` frames плюс разный static padding.
- Результат: `210/210 correct`, robustness accuracy `1.0`.
- Расширенный ML/runtime набор после H-030: `94 passed`.

Live-протокол проверки:
1. Для каждого класса выполнить по `10` попыток на обычной дистанции.
2. Повторить по `10` попыток примерно в `1.5` раза дальше от камеры.
3. Повторить по `10` попыток ближе к камере.
4. В каждой серии менять start position: центр, левее/правее, выше/ниже.
5. Не переобучать модель между сериями.

Критерий приемки:
- Accuracy каждого класса и каждой дистанции `>=80%`.
- Разница accuracy между near/mid/far не больше `10` процентных пунктов.
- `dynamic_motion_scale` заметно различается между дистанциями, иначе тест
  фактически проведен на одинаковом масштабе.
- Directional confusion `<=1/10`.

Следующий шаг:
- Перезапустить приложение и выполнить сначала три run `swipe_left`:
  mid, far, near. Это был наиболее проблемный класс и самый быстрый тест H-030.

## Текущий ML-пайплайн

1. Запись:
   - static: `(frames, 21, 2)`;
   - dynamic: `(36, 44)`.
2. Feature extraction:
   - static baseline: `static_mean`;
   - dynamic baseline: active motion trim + resample to `36` frames;
   - wrist trajectory: origin translation + amplitude normalization;
   - segmentation thresholds: projected hand-scale aware;
   - `dynamic_stats` with trajectory features weighted by `8.0` for
     distance-based KNN.
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

### H-031: Dynamic pipeline нужно разделить на motion-first detector и ML fallback

Статус: `implemented`, требуется live validation

Наблюдение пользователя:
- KNN плохо показал себя при формировании/распознавании dynamic gestures.
- В `auto` режиме статический класс вроде `gun` и pose-heavy KNN могут
  конкурировать с реальным свайпом.

Разбор причин:
- Для `swipe_*` основной сигнал — направление завершенной траектории wrist.
- KNN использует евклидово расстояние по `271` признаку. Даже после weighting
  часть расстояния остается связанной с позой руки, скоростью и шумом записи.
- Поэтому KNN хорош как baseline/diagnostic model, но не должен быть главным
  решателем для простых directional gestures.

Гипотеза:
- Если сначала выделять motion event, затем классифицировать направление
  траектории, а KNN использовать как fallback/диагностику, то live accuracy
  свайпов станет устойчивее к позе кисти, стартовой позиции и дистанции.

Что сделали:
- Добавлен `cv/dynamic_direction.py`.
- Dynamic branch теперь делится на этапы:
  1. MediaPipe landmarks;
  2. dynamic motion segmentation;
  3. canonical trajectory normalization;
  4. motion-first direction classifier для `swipe_left/right/up/down`;
  5. KNN/SVM/ExtraTrees probability как diagnostic/fallback.
- В router/live-evaluation добавлены поля:
  `dynamic_decision_source`, `dynamic_motion_label`,
  `dynamic_motion_confidence`, `dynamic_model_label`,
  `dynamic_model_confidence`, `dynamic_axis`, `dynamic_direction`,
  `dynamic_axis_ratio`, `dynamic_straightness`.

Offline-проверка:
- Unit-test проверяет, что горизонтальный completed motion выбирает
  `swipe_left`, даже если pose-heavy classifier выше оценивает `swipe_up`.
- Unit-test подтверждает, что `swipe_*` может быть принят motion-first
  классификатором даже без доступного estimator-файла.
- Unit-test отклоняет диагональный ambiguous motion, чтобы не принимать
  случайное движение как свайп.
- Целевой regression suite после H-031: `60 passed`.

Live-протокол проверки:
1. Перезапустить приложение.
2. В `auto` режиме прогнать `swipe_left`, `swipe_up`, `swipe_down` по `10`
   attempts.
3. Для каждой ошибки смотреть в `live_evaluation.jsonl`:
   - `dynamic_decision_source=motion_first` или `motion_and_model_agree`;
   - `dynamic_motion_label`;
   - `dynamic_model_label`.
4. Если `motion_label` верный, а `model_label` неверный — KNN действительно
   не должен решать этот класс.
5. Если `motion_label` неверный — проблема в записи/сегментации/траектории, а
   не в ML-классификаторе.

Критерий приемки:
- Для каждого swipe-класса live accuracy `>=80%` на `10` attempts.
- Static hijack в dynamic run: `0/10`.
- Ошибки должны быть объяснимы через новые diagnostic fields.

### H-032: Negative examples нужны для rejection layer

Статус: `implemented`, требуется генерация данных, обучение и live validation

Наблюдение пользователя:
- Dynamic-модель все еще может плохо работать в live.
- Нужен способ отличать настоящий жест от случайного движения, частичного
  свайпа, возврата руки и статичной руки.

Гипотеза:
- Качество live-распознавания сильнее вырастет от negative examples и
  rejection layer, чем от немедленного перехода на LSTM/Transformer.
- Для текущего маленького датасета deep sequence model может переобучиться, а
  negative examples сразу уменьшают false positives.

Что сделали:
- В taxonomy добавлен тип `negative`.
- Negative labels:
  - `no_gesture_static`;
  - `random_motion`;
  - `partial_swipe`;
  - `return_motion`;
  - `wrong_axis_motion`.
- `background_no_hand` остается reserved-сценарием для следующего synthetic /
  no-hand pipeline: текущая запись через интерфейс требует landmarks руки и
  не сохраняет пустой кадр как sample.
- В developer UI на экране обучения добавлена секция `Negative examples`.
- Пользователь больше не записывает negative examples руками: кнопка
  генерирует synthetic negative samples из уже записанных dynamic samples.
- Negative samples сохраняются в dynamic-формате с global wrist motion:
  `(36, 44)` и префиксом `sample_auto_`.
- Генератор пишет sidecar metadata `sample_auto_XXXX.meta.json` и manifest
  `docs/experiments/negative_sampling_manifest.json`.
- Dynamic training scope изменен с `dynamic` на `dynamic,negative`, поэтому
  dynamic-модель учит и жесты, и отрицательные примеры.
- Runtime rejection:
  - если estimator уверенно предсказывает negative label с confidence `>=0.72`,
    dynamic event отклоняется;
  - в live logs пишутся `dynamic_negative_label`,
    `dynamic_negative_confidence`, `dynamic_negative_threshold`;
  - negative labels не становятся исполняемыми командами.
- Live-evaluation для negative labels считает отсутствие предсказания по
  timeout/manual missed как `correct`, чтобы можно было измерять false
  positive rate без ручных пересчетов.

Offline-проверка:
- Taxonomy умеет фильтровать `dynamic,negative`.
- UI-тест подтверждает, что negative generation не вызывает ручную запись
  через камеру.
- Runtime-тест подтверждает, что `no_gesture_static` с confidence `0.85`
  блокирует motion-first `swipe_left`.
- Live-evaluation test подтверждает, что no-prediction для
  `no_gesture_static` засчитывается как correct.
- Целевой regression suite: `47 passed` после H-034/H-035.

Как генерировать negative examples:
1. Записать реальные dynamic gestures: `swipe_up`, `swipe_down`,
   `swipe_left` и следующие классы.
2. Открыть `Обучение -> Разработчик -> Negative examples`.
3. Нажать `Сгенерировать negative`.
4. Нажать `Обучить dynamic модель`.
5. Перезапустить live-recognition.

Live-протокол проверки:
- `swipe_left`, `swipe_up`, `swipe_down`: по `10` attempts.
- `no_gesture_static`, `random_motion`, `partial_swipe`: по `10` attempts,
  timeout `1.0-1.5` секунды.
- Для negative-run правильное поведение: `route=none`, команда не исполняется.
- Отдельная метрика: false positive rate на negative classes.

Критерий приемки:
- Swipe recall `>=80%`.
- Negative false positive rate `<=10%`.
- Static hijack rate `0`.
- В логах есть объяснение rejection через `dynamic_negative_label/confidence`.

### H-033: Первый MLOps dashboard для JMLC

Статус: `implemented`, локальный HTML dashboard

Цель:
- Уйти от просмотра метрик только через терминал.
- Дать конкурсному жюри видимую MLOps-петлю: dataset, live metrics, runtime,
  model versions.

Что сделали:
- Добавлен `scripts/mlops_dashboard.py`.
- Dashboard генерируется в `docs/mlops_dashboard/index.html`.
- Дополнительно пишется machine-readable snapshot:
  `docs/mlops_dashboard/summary.json`.

Что показывает dashboard:
- live accuracy;
- количество live attempts;
- negative rejection count;
- dataset classes/samples by taxonomy type;
- route counts и dynamic decision sources;
- latest runtime latency/FPS capacity;
- model artifact size, modified time, SHA-256 fingerprint.

Команда генерации:

```bash
python -m scripts.mlops_dashboard
```

Проверка:
- Unit-test `tests/unit/test_mlops_dashboard.py`;
- целевой regression suite: `47 passed` после H-034/H-035.

Следующий MLOps-шаг:
- Добавить versioned training runs: dataset hash, model hash, params,
  offline metrics, live metrics, acceptance status.

### H-034: Automatic negative sampling вместо ручной записи negative

Статус: `implemented`, ожидает live validation

Проблема:
- Negative examples должны быть данными, которые точно не являются жестами.
- Если пользователь сам записывает negative, он начинает решать
  ML-задачу руками и может случайно записать настоящий жест как negative.

Решение:
- Добавлен `scripts/generate_negative_samples.py`.
- Генератор берет только taxonomy labels типа `dynamic`.
- Для каждого negative label строятся контролируемые сценарии:
  `static_hold`, `closed_random_walk`, `aborted_partial_motion`,
  `out_and_back_return`, `ambiguous_diagonal`.
- Существующие ручные файлы `sample_*.npy` не удаляются.
- При повторном запуске чистятся только auto-файлы `sample_auto_*.npy`.
- Локально сгенерировано `100` negative samples из `70` dynamic source
  samples: по `20` на `no_gesture_static`, `random_motion`,
  `partial_swipe`, `return_motion`, `wrong_axis_motion`.
- `models/dynamic_knn.pkl` переобучена на scope `dynamic,negative`:
  `170` samples, `8` classes, feature dim `271`, train accuracy `1.0000`.

Проверка:
- `tests/unit/test_generate_negative_samples.py` проверяет создание `.npy`,
  metadata, manifest и сохранность ручных samples.
- `tests/unit/test_flet_training_view.py` проверяет, что UI запускает
  generation flow, а не camera recording.
- Целевой regression suite: `47 passed`.

Критерий приемки:
- После генерации и переобучения dynamic model:
  - swipe recall `>=80%`;
  - negative false positive rate `<=10%`;
  - static hijack rate `0`.

### H-035: MLflow tracking для промышленного MLOps-следа

Статус: `implemented`, локальный backend `sqlite:///mlflow.db`

Зачем:
- HTML dashboard показывает состояние системы, но не является полноценным
  experiment tracker.
- Для JMLC нужно показать, что каждое обучение имеет параметры, метрики,
  артефакты и историю запусков.

Что добавлено:
- `cv/train_classifier.py` логирует MLflow run при каждом обучении.
- Default experiment: `GestureFlow`.
- Default tracking URI: `sqlite:///mlflow.db`.
- Логируются параметры: `model_type`, `feature_mode`, `neighbors`,
  `weights`, `include_labels`, `classes`.
- Логируются метрики: `sample_count`, `class_count`, `feature_dim`,
  `train_accuracy`.
- Логируются артефакты: model pickle, classes json, feature dim и feature mode.
- Если MLflow не установлен, обучение не падает, а пишет warning.
- После установки `mlflow==3.14.0` файловый backend `file:./mlruns` был
  заменен на SQLite backend: MLflow 3 переводит filesystem tracking backend в
  maintenance mode и требует либо env-флаг, либо database backend.
- Smoke-run `dynamic-negative-smoke` успешно записан в experiment
  `GestureFlow`: `170` samples, `8` classes, `train_accuracy=1.0000`.

Команды:

```bash
make negative-samples
make mlops-dashboard
PYTHON=.venv/bin/python make mlflow-ui
```

Роль в JMLC:
- `docs/mlops_dashboard/index.html` — витрина текущего качества и runtime.
- `sqlite:///mlflow.db` + MLflow UI — трекинг экспериментов и версий моделей.

### H-036: Natural swipe segmentation без финальной позы

Статус: `implemented`, ожидает live validation

Наблюдение пользователя:
- Чтобы dynamic gesture распознался, приходилось оставлять руку в финальной
  точке.
- Это превращало UX в `swipe + pose`, хотя продуктовая идея требует
  естественный swipe без удержания начальной и финальной позиции.

Причина:
- `DynamicMotionSegmenter` завершал segment только после нескольких почти
  неподвижных кадров в конце движения.
- Контроллер дополнительно требовал `2` подтверждающих кадра для dynamic
  label. Если рука уходила из кадра сразу после свайпа, событие могло
  потеряться.

Что изменено:
- Завершение segment теперь поддерживает причины:
  - `velocity_drop`: скорость движения упала после достаточного swipe;
  - `hand_lost`: рука ушла из кадра после достаточного active motion;
  - `still`: обратная совместимость, если рука все же остановилась;
  - `max_frames`: fallback по лимиту длительности.
- `end_still_frames` снижен с `5` до `2`, но финальная пауза больше не
  является обязательной.
- Dynamic confirmation снижено до `1` кадра; static confirmation оставлено
  dwell-based, чтобы статические жесты не стали случайными.
- В live logs добавлено поле `dynamic_end_reason`.

Проверка:
- `tests/unit/test_dynamic_motion.py`:
  - segment завершается по `velocity_drop` без финального hold;
  - segment завершается по `hand_lost`, если рука ушла после swipe.
- `tests/unit/test_gesture_online_infer_no_hand.py`:
  - `GestureOnlineInfer` распознает `swipe_left`, когда рука исчезает после
    движения.
- `tests/unit/test_flet_controller_commands.py`:
  - dynamic dispatch подтверждается одним кадром.
- Целевой regression suite:
  `tests/unit/test_dynamic_motion.py`,
  `tests/unit/test_gesture_online_infer_no_hand.py`,
  `tests/unit/test_flet_controller_commands.py`,
  `tests/unit/test_recognition_router.py` -> `71 passed`.

Live-протокол проверки:
1. Включить `auto` mode.
2. Для `swipe_up`, `swipe_down`, `swipe_left` сделать по `10` естественных
   свайпов без удержания финальной точки.
3. После каждого свайпа можно сразу увести руку из кадра.
4. В логах ожидать `dynamic_end_reason=velocity_drop` или `hand_lost`, а не
   только `still`.

Критерий приемки:
- Dynamic recall по каждому swipe `>=80%`.
- Median субъективная задержка распознавания должна стать ниже: жест
  срабатывает после движения, а не после позы.
- False positive rate на negative-сценариях не должен вырасти выше `10%`.

### H-037: Live evaluation должна автоматически попадать в MLflow

Статус: `implemented`, ожидает новых live прогонов пользователя

Наблюдение:
- `live_evaluation.jsonl` уже давал attempt-level логи, но для JMLC нужен
  промышленный experiment tracker, где можно сравнивать live-runs между собой.
- Важные ошибки auto-модели раньше могли теряться: если в dynamic-тесте router
  выбирал static route (`gun`, `hend`, etc.), live evaluation игнорировала это
  событие вместо того, чтобы считать его ошибкой.

Что изменено:
- После завершения live-теста Flet controller автоматически пишет MLflow run
  в experiment `GestureFlow`.
- Run name строится как `live-<expected>-<mode>-<dynamic_profile>`.
- Tracking URI берется из `MLFLOW_TRACKING_URI`, fallback:
  `sqlite:////Users/remi/Developer/GUAP/DPLM/mlflow.db`.
- Каждый run получает artifact `live_evaluation_run.json` с session summary,
  attempts и route counts.
- Static route внутри expected dynamic теперь считается `wrong`, чтобы
  метрика `static_hijack_rate` была честной.

Метрики MLflow:
- базовые: `live_accuracy`, `live_recall`, `live_error_rate`,
  `live_miss_rate`, `live_completion_rate`;
- счетчики: `live_total`, `live_correct`, `live_wrong`, `live_missed`;
- latency: `live_latency_avg_s`, `live_latency_p50_s`,
  `live_latency_p95_s`;
- routing: `live_route_dynamic_count`, `live_route_static_count`,
  `live_route_none_count`;
- dynamic diagnostics: `live_dynamic_recall`,
  `live_static_hijack_rate`, `live_wrong_dynamic_direction_rate`;
- natural swipe segmentation: `live_end_reason_hand_lost_count`,
  `live_end_reason_velocity_drop_count`, `live_end_reason_still_count`;
- decision source: `live_decision_motion_first_count`,
  `live_decision_motion_and_model_agree_count`,
  `live_decision_negative_rejected_count`;
- negative tests: `live_negative_false_positive_rate`,
  `live_negative_rejected_count`.

Как проверять:
1. Запустить MLflow UI:

```bash
PYTHON=.venv/bin/python make mlflow-ui
```

2. В интерфейсе GestureFlow пройти live-test, например `swipe_left 10/10`.
3. Обновить `http://127.0.0.1:5000`, открыть experiment `GestureFlow`.
4. Найти run `live-swipe_left-auto-knn`.
5. Смотреть:
   - `live_accuracy` и `live_recall` — общее качество;
   - `live_static_hijack_rate` — dynamic ушел в static;
   - `live_wrong_dynamic_direction_rate` — перепутано направление;
   - `live_end_reason_*` — завершился ли natural swipe без финальной позы;
   - `live_evaluation_run.json` — сырые попытки и route metadata.

Проверка:
- `python -m py_compile app/flet_app/controller.py`
- `python -m pytest --no-cov tests/unit/test_flet_controller_commands.py -q`
  -> `36 passed`.

### H-038: Natural swipe + MLflow показали устранение static hijack

Статус: `validated`, нужны дополнительные данные для `swipe_up/down`

Дата live-прогона: `2026-06-27 17:25-17:26`

Наблюдение:
- После natural swipe segmentation и MLflow live logging auto-модель стала
  значительно лучше отделять dynamic gestures от static gestures.
- Ранее `swipe_up` часто уходил в `gun`: `7/10` попыток были `route=static`.
- В свежих run'ах все `30/30` попыток для `swipe_up`, `swipe_down`,
  `swipe_left` прошли через `route=dynamic`.

Результаты MLflow / `live_evaluation.jsonl`:

| Expected | Score | Accuracy/Recall | Static hijack | Wrong direction | Avg latency | End reasons |
|---|---:|---:|---:|---:|---:|---|
| `swipe_up` | `7/10` | `70%` | `0%` | `30%` | `2.02s` | `velocity_drop=8`, `hand_lost=2` |
| `swipe_down` | `7/10` | `70%` | `0%` | `30%` | `2.60s` | `velocity_drop=4`, `hand_lost=5`, `still=1` |
| `swipe_left` | `10/10` | `100%` | `0%` | `0%` | `1.46s` | `velocity_drop=9`, `hand_lost=1` |

MLflow runs:
- `live-swipe_up-auto-knn`
- `live-swipe_down-auto-knn`
- `live-swipe_left-auto-knn`

Вывод:
- Критичная проблема `dynamic -> static` на свежем прогоне ушла:
  `live_static_hijack_rate=0.0` для всех трех классов.
- `swipe_left` достиг целевого качества и может считаться текущим baseline.
- Оставшаяся ошибка для `swipe_up/down` — не routing, а directional confusion:
  `swipe_up -> swipe_down` и `swipe_down -> swipe_up/swipe_left`.
- Natural swipe работает: большинство событий завершается по
  `velocity_drop` или `hand_lost`, то есть без обязательной финальной позы.

Следующая гипотеза:
- Для вертикальных жестов нужно усилить axis/direction gate и проверить
  канон записи: `swipe_up/down` должны иметь больше вертикальной доминанты и
  меньше диагонального/горизонтального хвоста.
- Следующий эксперимент: повторить `swipe_up/down` по `20` попыток с
  одинаковой стартовой зоной и затем сравнить `dynamic_axis_ratio`,
  `dynamic_straightness`, `dynamic_motion_scale` между correct/wrong.

### H-039: Ошибки `swipe_up/down` связаны с качеством траектории, а не confidence

Статус: `analysis_ready`, ждет повторного live-теста по `20` попыток

Источник:
- `~/.dplm/logs/live_evaluation.jsonl`;
- MLflow runs `live-swipe_up-auto-knn`, `live-swipe_down-auto-knn`,
  `live-swipe_left-auto-knn`;
- подробный отчет:
  `docs/experiments/live_direction_error_analysis.md`.

Наблюдение:
- `swipe_up`: `7/10`, все `route=dynamic`, все ошибки были
  `predicted=swipe_down`.
- `swipe_down`: `7/10`, все `route=dynamic`, ошибки:
  `swipe_up=2`, `swipe_left=1`.
- `swipe_left`: `10/10`, используется как baseline regression check.

Выводы:
- Static hijack больше не является текущей проблемой:
  `live_static_hijack_rate=0.0`.
- KNN confidence не подходит как защитный порог: wrong attempts часто имеют
  `dynamic_model_confidence=1.0`.
- Ошибки вертикальных жестов коррелируют с trajectory quality:
  - у wrong `swipe_down` ниже `dynamic_axis_ratio` и
    `dynamic_straightness`;
  - один `swipe_down` стал `dynamic_axis=horizontal` и ушел в `swipe_left`;
  - wrong `swipe_up` часто связан с `hand_lost` или низкой straightness.

Гипотеза перед следующим прогоном:
- Если пользователь делает вертикальные свайпы одним прямым движением без
  возврата руки через кадр, recall для `swipe_up/down` должен подняться выше
  `80%`, а wrong direction rate должен упасть ниже `20%`.
- Если качество останется около `70%`, следующий кодовый шаг - усилить
  direction gate: vertical dominance, straightness threshold и анализ
  first-half/second-half direction.

Протокол следующей проверки:
- `swipe_up`: `20` попыток;
- `swipe_down`: `20` попыток;
- `swipe_left`: `10` попыток как baseline;
- одинаковая дистанция от камеры, похожая стартовая зона;
- без финальной позы, но без немедленного возврата руки через кадр.

### H-040: MLflow должен хранить не только числа, но и читаемые live artifacts

Статус: `implemented`, ждет следующего live-run

Наблюдение:
- MLflow auto-charts показывают отдельные метрики, но для демо и анализа
  удобнее иметь готовый artifact bundle внутри каждого run.
- System metrics в индустриальном смысле лучше выносить в Grafana/Prometheus,
  а MLflow использовать как experiment tracker.

Что добавлено:
- Live runs включают `log_system_metrics=True` для MLflow per-run resource
  metrics, если установлен `psutil`.
- В live-run добавляются attempt-level time-series metrics:
  `attempt_latency_s`, `attempt_axis_ratio`, `attempt_straightness`,
  `attempt_motion_scale`, `attempt_wrong_direction`, `attempt_static_hijack`.
- В aggregate metrics добавлены runtime показатели из
  `runtime_performance.jsonl`:
  `system_runtime_inference_ms_avg`,
  `system_runtime_inference_ms_p95_max`,
  `system_runtime_detection_ms_avg`,
  `system_runtime_fps_capacity_avg`.
- Каждый live-run сохраняет artifact bundle `live_evaluation/`:
  - `index.html`;
  - `charts/quality.svg`;
  - `charts/outcomes.svg`;
  - `charts/routes.svg`;
  - `charts/end_reasons.svg`;
  - `charts/runtime.svg`;
  - `charts/attempt_timeline.svg`;
  - `attempts.csv`, `metrics.csv`, `runtime_performance.json`.

MLOps boundary:
- MLflow: сравнение моделей, live-runs, метрики и артефакты экспериментов.
- Grafana/Prometheus: следующий слой для always-on runtime/system monitoring.

### H-041: Static negative dataset + open-set rejection должны снизить false triggers

Статус: `implemented`, ждет live-теста static/negative

Дата реализации: `2026-06-27`

Проблема:
- Static-модель всегда выбирает ближайший известный класс, поэтому при
  "почти жестах" или случайной позе рукой она может выдавать `gun`, `hend`,
  `three` и т.д.
- Повышение одного confidence threshold решает проблему грубо: можно потерять
  нормальные static-жесты, но все равно принимать уверенные ложные классы.

Гипотеза:
- Если добавить автоматические static hard negatives и rejection policy, модель
  сможет не только классифицировать жест, но и отказываться от небезопасного
  вывода.

Что добавлено:
- `scripts/generate_negative_samples.py` теперь генерирует static hard
  negatives для `no_gesture_static` из уже записанных static/quasi-static
  классов:
  - `static_pose_jitter`;
  - `static_closed_pose`;
  - `static_pose_mixup`;
  - `static_partial_pose`.
- Пользователь negative не записывает вручную: source sample выбирается
  seed-based из существующих real samples.
- Static обучение в Flet теперь использует scope
  `static,quasi_static,negative` и `expect_dim=42`, чтобы dynamic labels не
  попадали в `models/knn.pkl`.
- `cv.train_classifier` сохраняет `models/gesture_rejection.json`:
  - negative labels;
  - confidence threshold для negative class;
  - top1/top2 margin threshold;
  - centroid/prototype radius по каждому классу.
- `GestureOnlineInfer` применяет static reject policy:
  - `negative_class`;
  - `low_margin`;
  - `far_from_prototype`;
  - плюс существующий `finger_count_mismatch`.
- Router и live evaluation прокидывают static decision metadata в
  `live_evaluation.jsonl` и MLflow.

MLflow metrics:
- `live_static_accept_rate`;
- `live_static_reject_rate`;
- `live_static_false_positive_rate`;
- `live_static_rejection_reason_*_count`;
- per-attempt:
  `attempt_static_margin`, `attempt_static_negative_confidence`,
  `attempt_static_prototype_distance`,
  `attempt_static_prototype_threshold`.

Локальный прогон:
- Negative generation:
  `100` auto samples, `191` source samples.
- `no_gesture_static` source distribution:
  `CTRLZ=4`, `Hend=2`, `UP=3`, `gun=1`, `sh3=3`, `three=7`.
- Static training:
  `221` samples, `11` classes, `feature_dim=42`,
  `train_accuracy=1.0000`.
- MLflow run:
  `static-knn-negative-reject`.

Проверка кода:
- `python -m pytest --no-cov tests/unit/test_generate_negative_samples.py -q`
  -> `2 passed`.
- `python -m pytest --no-cov tests/unit/test_flet_training_view.py tests/unit/test_flet_controller_commands.py tests/unit/test_train_classifier.py -q`
  -> `49 passed`.
- Полный targeted набор:
  `81 passed`.

Следующий live-протокол:
- Обычные static-жесты: `gun`, `three`, `hend`, `up` по `10` попыток.
- "Почти жесты" и случайные позы: выбрать `no_gesture_static` и делать
  случайные движения/неполные позы по `20` попыток.
- Целевые метрики:
  - static recall для настоящих жестов `>=80%`;
  - `live_static_false_positive_rate <=10%` на negative;
  - rejection reasons должны быть объяснимыми:
    `negative_class`, `low_margin`, `far_from_prototype`.

### H-042: Reject strategy нужно выбирать сравнением ML-методов, а не вручную

Статус: `implemented`, offline benchmark готов

Дата реализации: `2026-06-27`

Проблема:
- Есть несколько способов решить false trigger:
  - negative classes;
  - confidence threshold;
  - current open-set policy;
  - one-vs-rest verifier;
  - one-class/outlier detectors;
  - metric learning;
  - nonlinear classifier.
- Без единого benchmark нельзя честно сказать, какой метод лучше для текущего
  персонального датасета.

Что добавлено:
- CLI benchmark:
  `scripts/rejection_method_benchmark.py`.
- Make target:
  `make rejection-benchmark`.
- Unit test:
  `tests/unit/test_rejection_method_benchmark.py`.
- Отчеты:
  - `docs/experiments/rejection_method_benchmark_static.md`;
  - `docs/experiments/rejection_method_benchmark_static.json`;
  - `docs/experiments/rejection_method_benchmark_dynamic.md`;
  - `docs/experiments/rejection_method_benchmark_dynamic.json`.

Сравниваемые методы:
- `negative_classes`;
- `confidence_threshold`;
- `open_set_policy`;
- `one_vs_rest_logreg`;
- `one_class_svm`;
- `isolation_forest`;
- `local_outlier_factor`;
- `metric_nca_centroid`;
- `mlp_negative_classes`.

Static benchmark:
- Dataset: `221` samples, `11` classes, positives:
  `CTRLZ`, `Hend`, `UP`, `gun`, `sh3`, `three`;
  negatives:
  `no_gesture_static`, `partial_swipe`, `random_motion`,
  `return_motion`, `wrong_axis_motion`.
- Лучший метод: `one_vs_rest_logreg`.
- Метрики:
  - `overall_success=0.9774`;
  - `positive_recall=0.9669`;
  - `negative_false_positive_rate=0.0100`.
- Текущий `open_set_policy`:
  - `overall_success=0.9683`;
  - `positive_recall=0.9587`;
  - `negative_false_positive_rate=0.0200`.

Dynamic benchmark:
- Dataset: `170` samples, `8` classes, positives:
  `swipe_down`, `swipe_left`, `swipe_up`;
  negatives:
  `no_gesture_static`, `partial_swipe`, `random_motion`,
  `return_motion`, `wrong_axis_motion`.
- Лучший метод: `open_set_policy`.
- Метрики:
  - `overall_success=1.0000`;
  - `positive_recall=1.0000`;
  - `negative_false_positive_rate=0.0000`.
- `one_vs_rest_logreg` тоже силен:
  - `overall_success=0.9941`;
  - `positive_recall=0.9857`;
  - `negative_false_positive_rate=0.0000`.

Вывод:
- Для static следующий кандидат на live integration:
  `KNN candidate + one-vs-rest LogisticRegression verifier`.
- Для dynamic текущий `open_set_policy` пока лучше, но это offline оценка;
  live-тест все равно обязателен.
- One-class методы (`one_class_svm`, `isolation_forest`, `LOF`) хорошо режут
  false positives, но часто слишком агрессивно reject'ят реальные gestures.
- `metric_nca_centroid` на текущих negative данных нестабилен, особенно для
  dynamic: слишком много false positives.

Проверка:
- `python -m pytest --no-cov tests/unit/test_rejection_method_benchmark.py -q`
  -> `1 passed`.
- Static benchmark:
  `python -m scripts.rejection_method_benchmark --scope static --feature-mode static_mean --target-dim 42`.
- Dynamic benchmark:
  `python -m scripts.rejection_method_benchmark --scope dynamic --feature-mode dynamic_stats --target-dim 44`.

Следующий шаг:
- Добавить runtime-профиль `static_verifier=one_vs_rest_logreg`.
- Затем в live evaluation сравнить:
  - current `open_set_policy`;
  - `one_vs_rest_logreg`;
  - target: `live_static_false_positive_rate <= 10%` без падения recall ниже
    `80%` на настоящих static gestures.

### H-043: Rejection method нужно проверять live, а не только offline

Статус: `implemented`, готово к ручному live A/B тесту

Дата реализации: `2026-06-28`

Проблема:
- Offline benchmark полезен для отбора кандидатов, но конкурсно и продуктово
  важнее real-camera behavior.
- Один и тот же static жест может выглядеть иначе в live из-за дистанции,
  скорости, угла руки, освещения и естественных “почти жестов”.
- Поэтому выбор reject strategy должен подтверждаться live метриками:
  false positive rate, recall, reject reason, latency.

Что добавлено:
- Runtime-переключатель static rejection method в Flet Live Evaluation.
- Поддержанные live methods:
  - `open_set_policy`;
  - `one_vs_rest_logreg`;
  - `negative_classes`;
  - `confidence_threshold`;
  - `one_class_svm`;
  - `isolation_forest`;
  - `local_outlier_factor`;
  - `metric_nca_centroid`;
  - `mlp_negative_classes`.
- Обучаемый verifier layer:
  `scripts/train_static_rejection_verifiers.py`.
- Make target:
  `make static-rejection-verifiers`.
- Runtime artifact:
  `models/static_rejection_verifiers.pkl`.
- Live logs now include:
  - `static_rejection_method`;
  - `static_verifier_probability`;
  - `static_verifier_confidence`;
  - `static_verifier_score`;
  - `static_verifier_distance`;
  - `static_verifier_threshold`.
- MLflow live run name now includes reject method:
  `live-<expected>-<mode>-<dynamic_profile>-<static_rejection_method>`.
- MLflow params include:
  `static_rejection_method`.
- MLflow metrics include:
  `live_static_rejection_method_<method>_count`.
- HTML dashboard now shows:
  static rejection method, decision source and rejection reason counters.

Локальный прогон:
- `make static-rejection-verifiers`
  -> `models/static_rejection_verifiers.pkl`.
- Ready methods: `6/6`:
  `one_vs_rest_logreg`, `one_class_svm`, `isolation_forest`,
  `local_outlier_factor`, `metric_nca_centroid`, `mlp_negative_classes`.
- MLflow training run:
  `static-rejection-verifiers`.

Проверка:
- `python -m py_compile app/gesture_online_infer.py app/services/recognition_router.py app/flet_app/controller.py app/flet_app/views/home.py scripts/train_static_rejection_verifiers.py`
  -> passed.
- `python -m pytest --no-cov tests/unit/test_gesture_online_infer_no_hand.py tests/unit/test_flet_controller_commands.py tests/unit/test_static_rejection_verifier_training.py -q`
  -> `59 passed`.

Live protocol:
- Для настоящих static жестов:
  `gun`, `three`, `hend`, `up` по `20` попыток на метод.
- Для negative/live false trigger теста:
  выбрать `no_gesture_static`, `partial_swipe`, `wrong_axis_motion`
  и делать “почти жесты”/случайные движения по `20` попыток.
- Для каждого сценария прогнать минимум:
  - `open_set_policy`;
  - `one_vs_rest_logreg`;
  - при времени: `confidence_threshold`.
- Сравнивать:
  - `live_accuracy`;
  - `live_static_false_positive_rate`;
  - `live_negative_false_positive_rate`;
  - `live_static_reject_rate`;
  - `live_static_rejection_reason_*`;
  - per-attempt verifier score/probability/distance.

Критерий выбора:
- Лучший метод не тот, у которого выше offline score, а тот, который в live:
  - держит static recall не ниже `80%`;
  - снижает false positives на negative до `<=10%`;
  - имеет объяснимые rejection reasons;
  - не ухудшает latency/UX.

### H-044: MLflow live runs должны иметь графики именно для reject-layer

Статус: `implemented`

Дата реализации: `2026-06-28`

Проблема:
- `live_accuracy` и route counts показывают итог, но не объясняют, почему
  static жест был принят или отклонен.
- Для JMLC нужно показать не только метрики, но и MLOps-подход:
  traceable live runs, artifacts, explainability for rejection decisions.

Что добавлено:
- Протокол тестирования:
  `docs/experiments/live_rejection_test_protocol.md`.
- В MLflow artifact bundle каждого live run добавлены графики:
  - `live_evaluation/charts/static_rejection.svg`;
  - `live_evaluation/charts/static_verifier_signals.svg`.
- В `live_evaluation/index.html` добавлены эти графики и attempt columns:
  - `Static method`;
  - `Static reject`;
  - `Verifier signal`.

Как использовать:
- Запустить MLflow:
  `PYTHON=.venv/bin/python make mlflow-ui`.
- Запустить приложение:
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m app.flet_app.main`.
- Провести live tests по протоколу и сравнить runs:
  `open_set_policy` vs `one_vs_rest_logreg`.

Критерий проверки:
- В каждом live-run должны быть:
  - model metrics;
  - system metrics;
  - `live_evaluation/index.html`;
  - SVG charts для quality/static rejection/verifier signals/routes/runtime.

### H-045: Публичные датасеты стоит использовать как negative evidence, а не как замену персонального датасета

Статус: `implemented`, ожидает реальные внешние файлы

Дата реализации: `2026-06-28`

Гипотеза:
- IPN Hand и HaGRID могут улучшить reject-layer: система должна чаще молчать
  на чужих/почти жестах и не принимать их за пользовательские команды.
- При этом они не должны заменять основной датасет GestureFlow, потому что
  задача проекта персонализированная: камера, дистанция, рука, скорость и
  смысл команд отличаются от публичных датасетов.

Что добавлено:
- Конфиг внешних источников:
  `configs/external_negative_datasets.json`.
- Отдельный сценарий экспериментов:
  `scripts/external_negative_dataset_experiments.py`.
- Make target:
  `make external-negative-experiments`.
- Документ протокола:
  `docs/experiments/external_negative_datasets.md`.
- Генерируемые отчеты:
  - `docs/experiments/external_negative_dataset_experiments.json`;
  - `docs/experiments/external_negative_dataset_experiments.md`.
- Отдельные model roots для live A/B:
  `models/experiments/external_negative/<variant>/`.

Проверяемые варианты:
- `baseline_internal`;
- `ipn_external`;
- `hagrid_external`;
- `combined_external`.

Как устроена обработка:
- Внешние samples кладутся в формате GestureFlow:
  `data/external/<source>/<original_label>/sample_*.npy`.
- Скрипт не добавляет оригинальные labels как новые команды.
- Все внешние классы мапятся в negative-классы:
  - `negative_external_ipn_dynamic`;
  - `negative_external_hagrid_static`.
- Для каждого варианта строится временный датасет:
  `data/experiments/external_negative/<variant>/gestures/`.
- Для live создаются отдельные артефакты:
  `knn.pkl`, `classes.json`, `feature_dim.txt`,
  `static_rejection_verifiers.pkl`, `dynamic_knn.pkl`,
  `dynamic_classes.json`.

MLflow:
- Offline benchmark runs:
  `external-negative-<variant>-static`,
  `external-negative-<variant>-dynamic`.
- Live artifact training runs:
  `external-negative-artifacts-<variant>-static`,
  `external-negative-artifacts-<variant>-dynamic:knn`.
- Логируются:
  - `external_sample_count`;
  - best offline method;
  - `best_overall_success`;
  - `best_positive_recall`;
  - `best_negative_reject_rate`;
  - `best_negative_false_positive_rate`;
  - model artifacts.

Локальный результат:
- `baseline_internal/static`: best `one_vs_rest_logreg`,
  `overall_success=0.9774`, `negative_false_positive_rate=0.0100`.
- `baseline_internal/dynamic`: best `open_set_policy`,
  `overall_success=1.0000`, `negative_false_positive_rate=0.0000`.
- `ipn_external`, `hagrid_external`, `combined_external`:
  `skipped`, потому что локально еще нет конвертированных файлов
  `data/external/...`.

Вывод:
- Pipeline для внешних датасетов готов.
- Качество внешних данных пока не заявляем: сначала нужно положить реальные
  IPN/HaGRID samples, затем повторить offline benchmark и live evaluation.

Проверка:
- `.venv/bin/python -m py_compile scripts/external_negative_dataset_experiments.py`
  -> passed.
- `.venv/bin/python -m pytest --no-cov tests/unit/test_external_negative_dataset_experiments.py -q`
  -> `3 passed`.
- `.venv/bin/python -m scripts.external_negative_dataset_experiments`
  -> baseline logged, external variants skipped as missing.

### H-046: Полный HaGRID слишком большой, нужен targeted external sampling

Статус: `accepted`

Дата: `2026-06-28`

Проблема:
- HaGRID полезен как источник внешних static negatives, но полный датасет и
  даже resized/light archive слишком тяжелые для быстрой конкурсной итерации.
- Скачивание десятков или сотен гигабайт ради reject-layer ухудшает скорость
  разработки и не дает гарантии live-качества на персональных жестах.

Решение:
- Не использовать full/light HaGRID как обязательный шаг.
- Для текущего этапа брать только маленький targeted subset:
  - `no_gesture` если доступен отдельным архивом;
  - либо `100-300` изображений из 1-2 нерелевантных классов после конвертации
    в landmarks;
  - не хранить raw images/videos в git.
- Для dynamic negatives приоритетнее IPN Hand subset, потому что его классы
  ближе к real-time continuous gesture recognition.

Что это значит для ML:
- Основной personalized positive dataset остается пользовательским.
- Внешний датасет используется как calibration/OOD evidence для reject-layer.
- Качество внешних данных оценивается только через:
  - offline rejection benchmark;
  - MLflow runs;
  - live false-positive tests.

Критерий успеха:
- Внешний subset принимается только если снижает
  `live_negative_false_positive_rate` и не ломает static/dynamic recall.

### H-047: IPN Hand можно использовать как главный внешний датасет для динамики

Статус: `research accepted`

Дата: `2026-06-28`

Источник:
- `https://github.com/GibranBenitez/IPN-hand`
- `https://gibranbenitez.github.io/IPN_Hand/`
- `https://arxiv.org/abs/2005.02134`

Почему это важно:
- IPN Hand решает real-time continuous hand gesture recognition, то есть
  ближе всего к нашей проблеме динамических жестов.
- Датасет содержит RGB-видео `640x480`, `30 fps`, разную дистанцию до камеры,
  13 жестов и non-gesture.
- В нем есть классы, похожие на наши swipe:
  `Throw up`, `Throw down`, `Throw left`, `Throw right`.
- Есть non-gesture и другие динамические действия, которые полезны как
  negative/OOD evidence.

Что берем:
- Идею two-stage online recognition:
  detector `gesture/no-gesture` -> classifier `class`.
- Идею probability queues:
  raw / median / moving average / EWMA.
- Идею накопления class evidence во время active segment.
- Идею sequence-level evaluation через Levenshtein distance.
- Данные:
  - `D0X Non-gesture` -> `negative_external_ipn_dynamic`;
  - unrelated dynamic classes -> dynamic negatives;
  - throw left/up/down/right -> validation/pretraining only, не production
    positives без live-проверки.

Что не берем сейчас:
- Heavy RGB 3D-CNN/ResNeXt pipeline.
- Optical flow как отдельную тяжелую модальность.
- Полную замену наших personalized swipe samples внешними IPN labels.

План:
- Добавить IPN converter:
  raw frames/video + annotations -> MediaPipe landmarks `(frames, 44)`.
- Обучить/сравнить:
  - current dynamic KNN;
  - dynamic KNN + IPN negatives;
  - binary dynamic intent detector;
  - small MLP/ExtraTrees rejector при достаточном числе samples.
- Добавить live metrics:
  - `live_dynamic_intent_precision`;
  - `live_dynamic_intent_recall`;
  - `live_false_dynamic_activation_rate`;
  - `live_sequence_edit_distance`;
  - `live_sequence_accuracy`;
  - `live_dynamic_start_latency_frames`;
  - `live_dynamic_end_latency_frames`.

Полный анализ:
- `docs/experiments/ipn_hand_research_analysis.md`.

### H-048: Dynamic pipeline должен поддерживать произвольные пользовательские жесты, а не только swipes

Статус: `accepted architecture direction`

Дата: `2026-06-28`

Проблема:
- Текущие `swipe_left`, `swipe_up`, `swipe_down` удобны для ранней проверки,
  но финальная система не должна быть набором hardcoded направлений.
- Пользователь должен уметь записать любой динамический жест:
  круг, зигзаг, комбинированное движение, изменение формы руки во время
  движения и т.д.

Вывод:
- Direction rules (`dx < 0`, `dy > 0`) остаются только вспомогательным
  primitive scorer для известных directional gestures.
- Универсальное распознавание должно строиться на последовательности
  landmarks и пользовательском обучении.

Архитектура:
- `MediaPipe landmarks`;
- generic dynamic intent detector;
- sequence normalization;
- user-trained dynamic classifier;
- dynamic open-set reject layer;
- router / command execution.

ML-часть:
- Для малых данных:
  KNN/prototype distance/DTW-like distance/SVM/ExtraTrees.
- При росте данных:
  MLP, TCN, GRU/LSTM, later tiny Transformer.
- IPN Hand используется как external dynamic negative / validation, но не
  заменяет пользовательские positive samples.

Метрики:
- `live_false_dynamic_activation_rate`;
- `live_unknown_dynamic_reject_rate`;
- `live_dynamic_open_set_false_positive_rate`;
- `live_sample_efficiency`;
- `live_sequence_accuracy`;
- `live_confusion_static_vs_dynamic`.

Документ:
- `docs/experiments/custom_dynamic_gesture_strategy.md`.

### H-049: Первый шаг IPN pipeline - конвертер и mapping config

Статус: `implemented`

Дата: `2026-06-28`

Цель:
- Подготовить воспроизводимый способ превратить IPN Hand в формат
  GestureFlow `(frames, 44)`.
- Не менять production-модель.
- Сразу логировать результат в MLflow и документы.

Что добавлено:
- Mapping config:
  `configs/ipn_hand_mapping.json`.
- Converter:
  `scripts/convert_ipn_hand.py`.
- Make target:
  `make ipn-convert`.
- Unit tests:
  `tests/unit/test_convert_ipn_hand.py`.
- Reports:
  - `docs/experiments/ipn_conversion_report.json`;
  - `docs/experiments/ipn_conversion_report.md`.

Mapping:
- `D0X`, pointing/click/open/zoom classes -> `negative_external_ipn_dynamic`.
- `G03-G06 Throw up/down/left/right` -> reference labels, skipped by default,
  чтобы не заменить пользовательские positive swipes внешними данными.

Сверка с планом:
- IPN converter: `done`.
- Mapping config: `done`.
- MLflow conversion report: `done`.
- Production model unchanged: `done`.
- Real conversion of samples: `blocked`, потому что локально еще нет:
  - `data/raw/ipn_hand/frames`;
  - `data/raw/ipn_hand/annotations/ipnall.json`.

Локальный результат:
- `PYTHON=.venv/bin/python make ipn-convert`
  -> status `missing_input`.
- Отчет корректно показывает ожидаемые пути и warning-и.
- MLflow run `ipn-conversion` создан.

Проверки:
- `.venv/bin/python -m py_compile scripts/convert_ipn_hand.py` -> passed.
- `.venv/bin/python -m pytest --no-cov tests/unit/test_convert_ipn_hand.py -q`
  -> `4 passed`.

### H-050: Official IPN annotations можно конвертировать в external dynamic negatives

Статус: `validated`

Дата: `2026-06-28`

Проблема:
- В скачанной папке IPN Hand лежит несколько annotation/list файлов, и не все
  подходят для извлечения сегментов жестов.
- Первый default `annotations/ipnall.json` не совпал с реальным локальным
  download layout.

Правильный файл:
- `Annot_List.txt` - основной файл для текущей конвертации.
- Он содержит `video,label,id,t_start,t_end,frames`, то есть video id, IPN
  class label и границы сегмента по кадрам.
- `Annot_TrainList.txt` и `Annot_TestList.txt` можно использовать позже для
  train/test split.
- `Video_TrainList.txt`, `Video_TestList.txt`, `classIdx.txt`, `metadata.*`
  не дают сами по себе границы сегментов и не подходят как основной источник
  dynamic samples.

Что изменено:
- `scripts/convert_ipn_hand.py` теперь читает official CSV-like `.txt`
  annotations.
- Добавлен auto-discovery для `Annot_List.txt`, поэтому `make ipn-convert`
  работает с текущей структурой download bundle.
- `data/raw/` добавлен в `.gitignore`, чтобы raw archives/frames не попадали
  в репозиторий.

Результат конвертации:
- Input:
  - `data/raw/ipn_hand/frames01.tar`;
  - `data/raw/ipn_hand/drive-download-20260628T185129Z-3-001/Annot_List.txt`.
- Output:
  - `data/external/ipn_hand/negative_external_ipn_dynamic`.
- `segments_found`: `5649`.
- `segments_included`: `4848`.
- `converted_samples`: `200`.
- `skipped_samples`: `28`.
- `detection_rate`: `0.7994`.
- `mlflow_conversion_report`: `met`.
- `production_model_unchanged`: `met`.

Распределение converted samples:
- `negative_external_ipn_dynamic`: `200`.
- IPN labels:
  - `B0A`: `54`;
  - `B0B`: `53`;
  - `D0X`: `19`;
  - `G01`: `10`;
  - `G02`: `11`;
  - `G07`: `11`;
  - `G08`: `11`;
  - `G09`: `10`;
  - `G10`: `10`;
  - `G11`: `11`.

Интерпретация:
- Это не positive gestures для наших команд.
- Это внешние dynamic/OOD negatives: pointing/click/double-click/open/zoom и
  non-gesture segments.
- Следующий эксперимент должен проверить, снижает ли добавление IPN negatives
  ложные срабатывания dynamic модели на случайные движения руки, не ломая
  live accuracy пользовательских swipe/custom gestures.

Проверки:
- `.venv/bin/python -m py_compile scripts/convert_ipn_hand.py` -> passed.
- `.venv/bin/python -m pytest --no-cov tests/unit/test_convert_ipn_hand.py -q`
  -> `5 passed`.

### H-051: IPN external negatives нужно сравнивать только в dynamic scope

Статус: `validated`

Дата: `2026-06-29`

Проблема:
- После конвертации IPN `external-negative-experiments` начал видеть
  `negative_external_ipn_dynamic`.
- Первый прогон показал улучшение и в static, и в dynamic строках, но это
  методологически неверно: IPN Hand - dynamic source, а static должен
  проверяться HaGRID/static negatives.

Изменение:
- `scripts/rejection_method_benchmark.py` получил параметр `include_labels`.
- `scripts/external_negative_dataset_experiments.py` теперь фильтрует external
  negative labels по `domain`:
  - `ipn_hand` -> только `dynamic`;
  - `hagrid` -> только `static`, когда появится локальный источник.
- Live artifacts обучаются с тем же domain-aware фильтром.

Команда:

```bash
PYTHON=.venv/bin/python make external-negative-experiments
```

Результат после фильтра:
- `baseline_internal/static`:
  - `positive_recall=0.9669`;
  - `negative_reject_rate=0.9900`;
  - `negative_false_positive_rate=0.0100`;
  - `coverage=0.5430`.
- `ipn_external/static`:
  - совпадает с baseline, потому что IPN больше не попадает в static scope.
- `baseline_internal/dynamic`:
  - `positive_recall=1.0000`;
  - `negative_reject_rate=1.0000`;
  - `negative_false_positive_rate=0.0000`;
  - `coverage=0.4118`;
  - best method `open_set_policy`.
- `ipn_external/dynamic`:
  - `positive_recall=1.0000`;
  - `negative_reject_rate=1.0000`;
  - `negative_false_positive_rate=0.0000`;
  - `coverage=0.2800`;
  - best method `one_vs_rest_logreg`;
  - sample count `250`, classes `9`.

Интерпретация:
- Offline benchmark не показывает прироста по core dynamic quality, потому что
  baseline уже идеален на текущем offline split.
- IPN все равно полезен как trained negative class для live/OOD проверки:
  `dynamic_classes.json` теперь содержит `negative_external_ipn_dynamic`.
- Решение о пользе IPN нужно принимать по live metrics:
  `false_dynamic_activation_rate`, `sequence_edit_distance`,
  `sequence_accuracy`, latency frames.

Проверки:
- `.venv/bin/python -m py_compile scripts/external_negative_dataset_experiments.py scripts/rejection_method_benchmark.py`
  -> passed.
- `.venv/bin/python -m pytest --no-cov tests/unit/test_external_negative_dataset_experiments.py tests/unit/test_rejection_method_benchmark.py -q`
  -> `5 passed`.

Следующий шаг:
- Запустить live test с `ipn_external` dynamic artifact.
- Затем добавить IPN-style continuous-session metrics:
  `sequence_edit_distance`, `sequence_accuracy`,
  `live_false_dynamic_activation_rate`, `dynamic_start/end_latency_frames`.

### H-052: Live model variants должны переключаться из UI, а не через env

Статус: `implemented`

Дата: `2026-06-29`

Проблема:
- Для live-тестов нужно часто сравнивать:
  - `production`;
  - `baseline_internal`;
  - `ipn_external`;
  - `combined_external`.
- Запускать приложение каждый раз через `DPLM_MODELS_DIR=...` неудобно и
  повышает риск ошибиться при сборе метрик.

Изменение:
- Добавлен controller API:
  - `list_model_variants()`;
  - `apply_model_variant(variant)`;
  - `model_variant`;
  - событие `model_variant_changed`.
- На главной странице в `Quick Settings` добавлен dropdown `Variant`.
- В `Settings -> Пути` добавлен dropdown `Набор моделей` и кнопка применения.
- При смене variant сохраняются:
  - `models_dir`;
  - `model_path`;
  - `classes_path`;
  - `feature_dim_path`.
- Текущий embedded infer сбрасывается, чтобы следующий live inference загрузил
  выбранные model artifacts.
- Path env overrides `DPLM_MODELS_DIR`, `DPLM_MODEL_PATH`,
  `DPLM_CLASSES_PATH`, `DPLM_FEATURE_DIM_PATH` очищаются в текущем процессе,
  чтобы UI-переключатель не конфликтовал с прошлым терминальным запуском.

Live variants:
- `production` -> `models`;
- `baseline_internal` -> `models/experiments/external_negative/baseline_internal`;
- `ipn_external` -> `models/experiments/external_negative/ipn_external`;
- `combined_external` -> `models/experiments/external_negative/combined_external`.

Проверки:
- `.venv/bin/python -m py_compile app/flet_app/controller.py app/flet_app/views/home.py app/flet_app/views/settings.py`
  -> passed.
- `.venv/bin/python -m pytest --no-cov tests/unit/test_flet_controller_commands.py -q`
  -> `41 passed`.

### H-053: Dynamic live errors are split between model choice and event policy

Статус: `testing`

Дата: `2026-06-29`

Проблема:
- Live A/B после добавления `baseline_internal` и `ipn_external` показал, что
  `swipe_up` и `swipe_left` могут давать `10/10`, но `swipe_down` часто
  ошибается в `swipe_up`.
- Пользовательский сценарий объясняет ошибку: после `swipe_down` руку нужно
  вернуть вверх к стартовой позиции, и этот возврат похож на отдельный
  `swipe_up`.
- `random_motion`, `partial_swipe`, `wrong_axis_motion`, `return_motion`
  технически являются negative-классами, но в UI они отображались как обычные
  цели теста, что создавало ощущение ложного распознавания.
- KNN как dynamic baseline всегда выбирает ближайший класс и сам по себе не
  умеет говорить "unknown / reject", поэтому для произвольных жестов нужен
  дополнительный verifier.

Live-замеры пользователя:
- `baseline_internal`:
  - `swipe_up`: `10/10`, accuracy `100%`;
  - `swipe_left`: `10/10`, accuracy `100%`;
  - `swipe_down`: `4/10`, wrong mostly `swipe_up`.
- `ipn_external`:
  - один run `swipe_up` провалился в `swipe_left`;
  - повторный `swipe_up`: `10/10`;
  - `swipe_left`: `10/10`;
  - `swipe_down`: `5/10`.

Изменение:
- Добавлен пользовательский live-evaluation target `no_command`.
  Его смысл: любые случайные движения, partial swipe, wrong axis и return
  motion не должны запускать команду.
- Технические negative-label больше не должны восприниматься как обычные
  пользовательские команды в live UI.
- В dispatch добавлен post-dynamic return guard:
  - короткое suppress-окно после любого dynamic event;
  - более длинное suppress-окно для противоположного направления
    (`swipe_down` -> `swipe_up`, `swipe_left` -> `swipe_right`).
- Negative prediction теперь считается rejection evidence, а не командой.

Что это решает:
- Возврат руки после засчитанного жеста больше не должен превращаться в новый
  command event.
- Negative-сценарии тестируются как false-positive test через `no_command`.
- MLflow продолжает видеть attempt-level данные, но интерпретация становится
  продуктовой: "команда не должна сработать" вместо "распознать fake class".

Что не решает:
- KNN по-прежнему остается baseline-моделью для динамики.
- Если движение действительно похоже на пользовательский жест, nearest-neighbor
  может выбрать ближайший класс. Для этого нужен следующий ML-шаг:
  dynamic sequence verifier.

Следующий ML-шаг:
- Добавить отдельный dynamic method `prototype_dtw`:
  - хранить несколько прототипов на класс;
  - сравнивать sequence shape через DTW / normalized trajectory distance;
  - возвращать gesture только если distance ниже class threshold;
  - иначе возвращать `no_command`.
- Сравнить `knn`, `open_set_policy`, `prototype_dtw`, `one_vs_rest` в MLflow и
  live UI.

### H-054: Dynamic prototype verifier сравнивает `prototype_distance` и `prototype_dtw`

Статус: `implemented`, нужна live validation

Дата: `2026-06-29`

Гипотеза:
- KNN плохо подходит как финальное решение для dynamic gestures, потому что
  всегда выбирает ближайший известный класс.
- Prototype verifier должен лучше решать open-set задачу:
  "похож ли этот сегмент на пользовательский жест достаточно сильно, чтобы
  выполнить команду".

Что реализовано:
- Новый модуль `cv.dynamic_prototype`:
  - `prototype_distance`: нормализованный L2 по canonical sequence;
  - `prototype_dtw`: DTW-distance по последовательности landmarks;
  - per-class thresholds по positive radius;
  - negative guard через synthetic negatives и IPN external negatives;
  - reject reasons: `nearest_negative`, `far_from_prototype`,
    `missing_threshold`.
- Новый экспериментальный скрипт:
  `scripts.dynamic_prototype_experiments`.
- Новый Make target:
  `PYTHON=.venv/bin/python make dynamic-prototype-experiments`.
- Runtime integration:
  - `GestureOnlineInfer` загружает `dynamic_prototypes.json`, если он есть;
  - prototype verifier работает как второй слой после segmentation/motion intent;
  - при reject команда не исполняется;
  - в route/live logs добавлены поля:
    `dynamic_prototype_method`, `dynamic_prototype_distance`,
    `dynamic_prototype_threshold`, `dynamic_prototype_reason`.
- UI model variants:
  - `prototype_distance`;
  - `prototype_dtw`.

Данные эксперимента:
- Train samples: `277`;
- Test samples: `93`;
- External negatives: `true`;
- labels:
  - positives: `swipe_down`, `swipe_left`, `swipe_up`;
  - negatives: `no_gesture_static`, `partial_swipe`, `random_motion`,
    `return_motion`, `wrong_axis_motion`,
    `negative_external_ipn_dynamic`.

Offline comparison:

| Method | Overall | Positive recall | Negative reject | Negative FP | Sequence accuracy | Edit distance |
|---|---:|---:|---:|---:|---:|---:|
| `prototype_distance` | `0.9892` | `0.9444` | `1.0000` | `0.0000` | `0.9444` | `1` |
| `prototype_dtw` | `0.9892` | `0.9444` | `1.0000` | `0.0000` | `0.9444` | `1` |

Интерпретация:
- На текущем offline split оба метода равны по качеству.
- `prototype_distance` выбран текущим default/best, потому что он дешевле по
  вычислениям и проще для live.
- Главное улучшение относительно KNN: negative false positive `0.0000` на
  mixed internal + IPN negative split.
- Оставшийся offline miss: `swipe_left` один раз отвергнут как слишком далёкий,
  но не перепутан с другим классом. Для команд это безопаснее, чем ложное
  выполнение.

Сгенерированные локальные артефакты:
- `models/dynamic_prototypes.json` - лучший method для production/live;
- `models/experiments/dynamic_prototype/prototype_distance/`;
- `models/experiments/dynamic_prototype/prototype_dtw/`;
- `docs/experiments/dynamic_prototype_comparison.md`;
- `docs/experiments/dynamic_prototype_comparison.json`.

Проверки:
- `.venv/bin/python -m py_compile cv/dynamic_prototype.py scripts/dynamic_prototype_experiments.py app/gesture_online_infer.py app/services/recognition_router.py app/flet_app/controller.py`
  -> passed.
- `.venv/bin/python -m pytest --no-cov tests/unit/test_dynamic_prototype.py tests/unit/test_gesture_online_infer_no_hand.py -q`
  -> `23 passed`.
- `.venv/bin/python -m pytest --no-cov tests/unit/test_flet_controller_commands.py -q`
  -> `44 passed`.

Следующий шаг:
- Live A/B:
  - variant `prototype_distance`, expected `no_command`, 20 attempts;
  - variant `prototype_distance`, `swipe_up/down/left`, 20 attempts each;
  - затем то же для `prototype_dtw`, если latency приемлемая.
- Сравнить в MLflow:
  - `live_negative_false_positive_rate`;
  - `live_dynamic_recall`;
  - `live_dynamic_prototype_reason_*`;
  - `live_wrong_dynamic_direction_rate`;
  - `system_runtime_inference_ms_p95_max`.

### H-055: Live A/B prototype verifier показал, что DTW не готов для runtime

Статус: `implemented`, нужна повторная live validation

Дата: `2026-06-29`

Live-замеры пользователя:

| Variant | Target | Result | Вывод |
|---|---:|---:|---|
| `prototype_distance` | `swipe_up` | `8/10` | Работает, misses без wrong direction |
| `prototype_distance` | `swipe_left` | `1/9` и отдельный `0/3` | Главная проблема не verifier, а отсутствие completed dynamic segment (`route=none`) |
| `prototype_distance` | `swipe_down` | `9/10` | Работает, misses без wrong direction |
| `prototype_dtw` | `swipe_down` | `1/4`, wrong `3/4` | Не готов как runtime default |

Что подтвердили логи:
- У `swipe_left` почти все ошибки имеют:
  - `route=none`;
  - `dynamic_segment_frames=0`;
  - пустые `dynamic_prototype_*`.
- Значит `prototype_distance` не отвергал `swipe_left`; сегмент просто не доходил
  до prototype verifier.
- Единственный completed `swipe_left` был принят:
  - `prototype_distance`;
  - distance `0.05045`;
  - threshold `0.05998`;
  - `motion_and_prototype_agree`.
- У `prototype_dtw` появились wrong direction и runtime spikes:
  - `live_wrong_dynamic_direction_rate=0.75` на одном из прогонов;
  - `system_runtime_inference_ms_p95_max` до `44.5 ms`.

Решение:
- `prototype_dtw` оставить как research/offline method, не использовать как live
  default до оптимизации и пересмотра thresholds.
- `prototype_distance` оставить основным verifier-кандидатом.
- Stage-1 dynamic segmenter сделать более recall-oriented:
  - `pre_roll_frames`: `5 -> 3`;
  - `onset_path`: `0.04 -> 0.015`;
  - `onset_displacement`: `0.025 -> 0.012`;
  - `min_active_frames`: `8 -> 5` для online candidate generation.

Почему это безопасно:
- Segmenter теперь должен чаще создавать candidate segment для быстрых
  горизонтальных жестов.
- Precision защищает второй слой:
  - prototype verifier;
  - negative classes;
  - motion/prototype agreement.

Проверка:
- Добавлен unit-test на быстрый горизонтальный swipe с уходом руки из кадра:
  `test_online_segmenter_emits_fast_horizontal_swipe_before_hand_leaves`.

Следующий live шаг:
- Повторить только `prototype_distance`:
  - `swipe_left`: `10` attempts;
  - `swipe_up`: `10` attempts;
  - `swipe_down`: `10` attempts;
  - `no_command/random_motion`: `10` attempts.
- Цель:
  - `swipe_left` должен перейти из `route=none` в `route=dynamic`;
  - `negative_false_positive_rate` должен остаться `0`.

### H-056: New IPN frame packs полезны как external negatives, но требуют стабильного feature extraction

Статус: `blocked_by_mediapipe_runtime`, данные проанализированы

Дата: `2026-06-29`

Гипотеза:
- Новые IPN packs можно использовать для расширения dynamic negative dataset.
- Это должно снизить лишние срабатывания на random/partial/return/wrong-axis
  motions без замены пользовательских positive gestures.

Что сделано:
- Добавлены `frames02.tar`, `frames03.tar`, `frames04.tar`, `frames05.tar`.
- Для `frames04.tar` выполнена итеративная распаковка:
  - extracted files: `158444`;
  - extracted videos: `40`;
  - extracted size: `2.0G`.
- Для `frames04.tar` собран inventory:
  - candidate segments: `1078`;
  - selected test segments: `240`;
  - основные negative labels: `D0X`, `B0A`, `B0B`, `G01`, `G02`, `G07`,
    `G08`, `G09`, `G10`, `G11`.
- Проверен existing compact IPN subset:
  - samples: `200`;
  - `prototype_distance` negative reject rate: `1.0000`;
  - `prototype_distance` false positive rate: `0.0000`;
  - `prototype_dtw` negative reject rate: `1.0000`;
  - `prototype_dtw` false positive rate: `0.0000`.

Что не получилось:
- Headless MediaPipe extraction падает в native runtime:
  `DrishtiMetalHelper / graph_service.h:139 service_ unavailable`.
- Ошибка воспроизводится и на streaming tar conversion, и на распакованных
  кадрах, значит проблема не в tar structure.

Решение:
- Не добавлять пустые/сомнительные samples в обучение.
- Добавить safe MediaPipe preflight в IPN конвертеры.
- Сохранять conversion status и artifacts в MLflow/документы.
- Держать `G03-G06` как `validation_reference`, а не negative training,
  потому что throw up/down/left/right слишком похожи на наши swipe gestures.

Следующий шаг:
- Стабилизировать feature extraction:
  - либо запускать extraction в runtime, где Flet camera уже работает;
  - либо подобрать совместимую MediaPipe Tasks версию;
  - либо вынести batch extraction в Linux/Docker.
- После extraction одного pack:
  - rerun `PYTHON=.venv/bin/python make ipn-external-analysis`;
  - сравнить `baseline_internal` vs `ipn_external_expanded`;
  - включать expanded IPN только в dynamic rejection stage.

### H-057: External negatives должны быть conflict-aware относительно пользовательских жестов

Статус: `implemented`, smoke validated

Дата: `2026-06-29`

Проблема:
- Пользователь может записать любой новый dynamic gesture.
- External negative из IPN не должен навсегда означать "запрещенное движение".
- Если новый пользовательский жест похож на IPN negative, positive sample
  пользователя должен иметь приоритет.

Решение:
- В `scripts.dynamic_prototype_experiments` добавлен
  `filter_conflicting_external_negatives`.
- Перед train/test split строится prototype model только по пользовательским
  positive dynamic gestures.
- Каждый external negative сравнивается с nearest positive prototype.
- Если `distance <= positive_threshold * conflict_margin`, sample исключается
  из negative training для текущей модели и попадает в conflict/reference
  отчет.

Параметры:
- default method: `prototype_distance`;
- default conflict margin: `1.20`;
- override через CLI:
  - `--negative-conflict-method`;
  - `--negative-conflict-margin`;
  - `--disable-negative-conflict-filter`.

MLOps:
- В MLflow и JSON/Markdown отчет теперь пишутся:
  - `negative_external_negative_total`;
  - `negative_safe_external_negative_count`;
  - `negative_conflict_count`;
  - `negative_conflict_rate`;
  - `nearest_positive_labels`;
  - `conflicting_negative_labels`.

Smoke result на текущих данных:
- external negatives: `200`;
- safe external negatives: `200`;
- conflicts removed: `0`;
- conflict rate: `0.0000`;
- dynamic prototype result:
  - overall success: `0.9892`;
  - positive recall: `0.9444`;
  - negative reject rate: `1.0000`;
  - negative false positive rate: `0.0000`.

Вывод:
- Текущий IPN subset не конфликтует с `swipe_up`, `swipe_down`, `swipe_left`.
- При добавлении нового пользовательского жеста похожие external negatives
  будут автоматически исключаться из training scope, а не ломать распознавание.

### H-058: Dynamic UI training должен обновлять prototype/rejection layer автоматически

Статус: `implemented`, smoke validated

Дата: `2026-06-29`

Проблема:
- Кнопка `Обучить dynamic модель` обновляла `dynamic_knn/svm/extra_trees`,
  но не обновляла `models/dynamic_prototypes.json`.
- Из-за этого conflict-aware external negative layer мог отставать от новых
  пользовательских dynamic gestures.

Решение:
- В `AppController.start_training()` добавлен post-training step для
  `training_scope=dynamic`.
- После успешного `cv.train_classifier` запускается:
  `scripts.dynamic_prototype_experiments`.
- Параметры автоматического шага:
  - `--include-external-negatives`;
  - `--external-negative-root data/external`;
  - `--methods prototype_distance`;
  - `--write-production`;
  - `--production-out models/dynamic_prototypes.json`.
- Static training не затрагивается.

Что теперь происходит при записи нового dynamic gesture:
- пользователь записывает sample;
- нажимает `Обучить dynamic модель`;
- обновляется выбранный dynamic classifier;
- затем обновляется prototype verifier;
- external IPN negatives проходят conflict-aware filtering;
- похожие на новый пользовательский жест negatives исключаются из training scope;
- metrics/artifacts уходят в MLflow и markdown/json отчеты.

Smoke result после подключения `frames04`:
- external negatives: `440`;
- safe external negatives: `440`;
- conflicts removed: `0`;
- conflict rate: `0.0000`;
- overall success: `0.9935`;
- positive recall: `0.9444`;
- negative reject rate: `1.0000`;
- negative false positive rate: `0.0000`.

Production artifact:
- `models/dynamic_prototypes.json`;
- method: `prototype_distance`;
- positive labels: `swipe_down`, `swipe_left`, `swipe_up`;
- negative labels:
  `negative_external_ipn_dynamic`, `no_gesture_static`, `partial_swipe`,
  `random_motion`, `return_motion`, `wrong_axis_motion`;
- prototype count: `72`;
- training record count: `457`.

### H-059: UI dynamic training не должен зависеть от выбранного model profile

Статус: `fixed`, regression tested

Дата: `2026-06-29`

Проблема:
- При выбранном профиле `prototype_distance` UI-тренировка dynamic модели
  успешно обновляла `models/dynamic_knn.pkl`, но post-training prototype step
  падал на `SameFileError`.
- Причина: prototype experiment получал одну и ту же папку как
  `--base-models-dir` и как output variant dir:
  `models/experiments/dynamic_prototype/prototype_distance`.
- В результате `dynamic_knn.pkl` копировался в самого себя.

Решение:
- Prototype/rejection post-training теперь отталкивается от реального
  `out_path` свежей dynamic модели, выбранного в UI.
- Для обычного обучения из интерфейса:
  - `--base-models-dir models`;
  - `--production-out models/dynamic_prototypes.json`.
- В копировании базовых dynamic artifacts добавлена защита: same-file copy
  пропускается.

Проверка:
- `py_compile`: OK.
- `pytest tests/unit/test_flet_controller_commands.py tests/unit/test_dynamic_prototype.py`: `51 passed`.
- Smoke run:
  `scripts.dynamic_prototype_experiments --methods prototype_distance ...`
  завершился успешно.

Результат:
- Кнопка `Обучить dynamic модель` снова должна проходить оба этапа:
  classifier + prototype/rejection layer.
- MLflow/Markdown/JSON отчеты обновляются после UI-тренировки без ручного
  запуска второго этапа.

### H-060: Live video lag вызван слишком тяжелым camera/MediaPipe frame budget

Статус: `superseded-live`

Дата: `2026-06-29`

Наблюдение:
- Во время live-тестов видео начало лагать.
- Последние `runtime_performance.jsonl` строки показывали:
  - `detection_ms_avg` примерно `48-111 ms`;
  - `inference_fps_capacity` примерно `9-21 FPS`;
  - целевой `target_fps=30`.
- Значит bottleneck не KNN/prototype, а hand detection + UI frame delivery.

Гипотеза:
- Если уменьшить количество пикселей для MediaPipe и снизить частоту preview
  в Flet, лаг должен снизиться без потери геометрии жеста, потому что
  landmarks нормализованы в координаты `[0..1]`.

Изменение:
- Camera capture запрашивается как `960x540`.
- Inference frame downscale: max width `480`.
- Preview frame cap: `12 FPS`.
- Preview JPEG quality: `62`.
- Runtime performance log теперь пишет:
  - `camera_frame_width/height`;
  - `inference_frame_width/height`;
  - `preview_max_fps`.

Проверка:
- `py_compile`: OK.
- `pytest tests/unit/test_camera_performance.py tests/unit/test_flet_frame_delivery.py`: `5 passed`.
- `pytest tests/unit/test_flet_controller_commands.py`: `46 passed`.

Live validation:
- Перезапустить приложение.
- Включить camera/auto recognition на 20-30 секунд.
- Проверить:
  `tail -n 5 ~/.dplm/logs/runtime_performance.jsonl`.
- Ожидаемый эффект:
  - `inference_frame_width=480`;
  - `preview_max_fps=12`;
  - `detection_ms_avg` должен стать заметно ниже предыдущих `48-111 ms`;
  - видео в Flet должно идти ровнее.

### H-061: 60 FPS smooth-preview нужно тестировать отдельно от ML FPS

Статус: `superseded-live`

Дата: `2026-06-29`

Проблема:
- Пользователь хочет проверить режим `60 FPS`, чтобы видео было визуально
  гладким.
- Но MediaPipe hand detection не может стабильно работать на каждом кадре
  60 FPS, если один inference занимает больше `16.7 ms`.

Решение:
- Разделяем цели:
  - camera/preview target: до `60 FPS`;
  - ML inference cap: `20 FPS`;
  - inference frame: max width `480`.
- В camera loop preview больше не обязан ждать inference каждый кадр.
- На кадрах без inference используется последний `landmarks_json`.
- Runtime performance log теперь пишет:
  - `preview_max_fps`;
  - `inference_max_fps`;
  - `camera_frame_width/height`;
  - `inference_frame_width/height`.

Что проверяем:
- В настройках приложения поставить `FPS=60`.
- Перезапустить приложение.
- Включить `auto` recognition на 20-30 секунд.
- Проверить:
  `tail -n 5 ~/.dplm/logs/runtime_performance.jsonl`.

Ожидаемый результат:
- `target_fps=60`;
- `preview_max_fps=60`;
- `inference_max_fps=20`;
- визуально preview должен быть плавнее;
- качество жестов нужно проверять отдельно live evaluation, потому что ML
  теперь осознанно не запускается на каждом camera frame.

Live result:
- При `preview_max_fps=60` Flet preview может не отображаться вообще.
- Причина вероятно в перегрузке Flet JPEG/base64 update pipeline.
- 60 FPS preview признан неподходящим baseline для contest demo.

### H-062: Для contest demo нужен стабильный ML-first camera mode

Статус: `implemented`, needs live validation

Дата: `2026-06-29`

Наблюдение:
- Главная цель сейчас не идеальное видео, а корректная live-детекция жестов.
- Preview в Flet должен быть стабильным, но не должен блокировать ML.
- Dynamic pipeline должен работать на окне `36` кадров.

Решение:
- `CAMERA_PREVIEW_MAX_FPS=30`.
- `CAMERA_INFERENCE_MAX_FPS=30`.
- `DYNAMIC_RECOGNITION_WINDOW=36`.
- Runtime log пишет:
  - `preview_enabled`;
  - `preview_max_fps`;
  - `inference_max_fps`;
  - `dynamic_window_frames`.
- Добавлен аварийный режим без Flet preview:
  `DPLM_DISABLE_CAMERA_PREVIEW=1`.

Как проверять:
- Обычный режим:
  - в настройках поставить `FPS=60` или `FPS=30`;
  - перезапустить приложение;
  - включить `auto` recognition;
  - ожидать в логах `preview_max_fps=30`, `inference_max_fps=30`,
    `dynamic_window_frames=36`.
- Detection-only режим:
  - запустить приложение с `DPLM_DISABLE_CAMERA_PREVIEW=1`;
  - preview не обновляется;
  - gesture detection и live evaluation продолжают работать.

Критерий успеха:
- Если preview включен: картинка отображается стабильно около `30 FPS`.
- Если preview выключен: распознавание жестов работает без нагрузки Flet UI.

### H-063: Static жесты не должны теряться из-за dynamic warmup и хрупкого finger-count guard

Статус: `implemented`, needs live validation

Дата: `2026-06-29`

Наблюдение:
- После стабилизации dynamic pipeline пользователь сообщил, что static жесты
  перестали детектироваться.
- Offline-проверка текущей `models/knn.pkl` на сохранённых static samples
  показала, что сама ML-модель рабочая:
  - `hend/up/gun/sh3/three` принимаются на своих samples;
  - проблема находится в live-слое вокруг модели.
- В live logs уже встречался rejection:
  `finger_count_mismatch` для `hend`, где модель была уверена в `hend`, но
  heuristic finger-count дал `current=0`.

Гипотеза:
- Static кандидат теряется из-за двух инженерных фильтров:
  - auto-router скрывает static во время dynamic `warming_up`;
  - finger-count guard слишком жёстко отклоняет жесты по нестабильной
    pose-сигнатуре.

Решение:
- Router больше не блокирует static на фазе dynamic `warming_up`.
- Static всё ещё скрывается во время реального dynamic `active/cooldown`, чтобы
  не исполнять static-команду в середине свайпа.
- Finger-count guard применяется только если:
  - сигнатура класса стабильна (`stability >= 0.85`);
  - ожидаемое число non-thumb fingers больше `0`;
  - live-счётчик тоже видит больше `0`.
- Если live finger-count равен `0`, это считается неопределённостью, а не
  доказательством неправильной позы.

Как проверять:
- Включить `auto` recognition.
- Static:
  - `hend`, `gun`, `three`, `up`, `ctrlz`, `sh3` по 10 попыток.
  - Ожидаем route=`static`, selected_reason=`static_fallback`.
- Dynamic:
  - `swipe_up`, `swipe_down`, `swipe_left` по 10 попыток.
  - Ожидаем, что во время активного свайпа route остаётся `dynamic`.
- Negative / near-miss:
  - random hand movement не должен массово превращаться в static.

Метрики:
- `live_static_recall`;
- `live_static_reject_rate`;
- `live_static_rejection_reason_finger_count_mismatch`;
- `live_confusion_static_vs_dynamic`.

### H-064: Static confidence `0.55` означает fallback, а не уверенность модели

Статус: `implemented`, needs live validation

Дата: `2026-06-29`

Наблюдение:
- Пользователь отдельно проверил static model: жесты доходят только до
  confidence `0.55`.
- В коде static-инференса `0.55` является fallback-значением:
  рука найдена, но classifier не дал настоящий prediction.
- Текущий `~/.dplm/config.json` указывал:
  - `models_dir = models/experiments/dynamic_prototype/prototype_distance`;
  - `model_path = .../prototype_distance/knn.pkl`;
  - `classes_path = .../prototype_distance/classes.json`.
- В `prototype_distance` лежат только dynamic artifacts:
  `dynamic_knn.pkl`, `dynamic_classes.json`, `dynamic_prototypes.json`.
  Static `knn.pkl/classes.json/feature_dim.txt` там отсутствуют.

Гипотеза:
- При выборе dynamic prototype variant приложение случайно перенесло static
  model paths в dynamic-only папку.
- Static classifier не загружался, поэтому UI показывал fallback confidence
  `0.55` без реальной static-классификации.

Решение:
- `prototype_distance` и `prototype_dtw` помечены как dynamic-only variants.
- При применении этих variants:
  - `models_dir` остаётся variant-папкой для dynamic artifacts;
  - static `model_path/classes_path/feature_dim_path` остаются production:
    `models/knn.pkl`, `models/classes.json`, `models/feature_dim.txt`.
- Для уже существующего сломанного config добавлен runtime fallback:
  если configured static artifact отсутствует, берём production artifact.

Как проверять:
- Перезапустить приложение.
- В `auto` режиме выбрать `prototype_distance`.
- Static test:
  - `gun`, `hend`, `three`, `up` должны давать не fallback `0.55`, а
    реальные `static_model_label/static_model_confidence` из classifier.
- В логах live evaluation ожидать:
  - `recognition_model_mode=auto`;
  - `route=static` для static gestures;
  - `selected_reason=static_fallback`;
  - `static_decision_source=accepted`.

Критерий успеха:
- Static gestures перестают зависать на `0.55`.
- Dynamic prototype layer продолжает использовать
  `models/experiments/dynamic_prototype/prototype_distance`.

### H-065: Dynamic gestures нужно сравнить с time-series baseline

Статус: `implemented`, needs live validation

Дата: `2026-06-29`

Наблюдение:
- Static gestures снова распознаются, но пользователь сообщил, что dynamic
  gestures в текущем auto pipeline не работают стабильно.
- Текущий `dynamic_stats` сжимает жест в агрегаты: displacement, velocity
  stats, path stats, trajectory summary.
- Для жестов вроде пользовательских свайпов и будущих произвольных движений
  важна форма траектории во времени, а не только суммарные статистики.

Гипотеза:
- Time-series baseline должен лучше подходить для dynamic gestures:
  - жест хранится как последовательность;
  - скорость выполнения меньше влияет после time normalization;
  - модель видит порядок движения, а не только итоговый вектор статистик.

Решение:
- Добавлен feature mode `dynamic_sequence`.
- Он приводит gesture segment к `36` кадрам через canonical time normalization
  и раскладывает последовательность в вектор `36 x raw_dim`.
- Добавлен UI/runtime profile `sequence_knn`.
- Для него используются отдельные artifacts:
  - `dynamic_sequence_knn.pkl`;
  - `dynamic_sequence_classes.json`;
  - `dynamic_sequence_feature_dim.txt`;
  - `dynamic_sequence_feature_mode.txt`;
  - `dynamic_sequence_knn_rejection.json`.
- `sequence_knn` намеренно не подхватывает старый
  `dynamic_prototypes.json`, чтобы проверить сам time-series classifier без
  жёсткого prototype gate.

Обучение:
- Production:
  `models/dynamic_sequence_knn.pkl`.
- Active prototype-distance variant:
  `models/experiments/dynamic_prototype/prototype_distance/dynamic_sequence_knn.pkl`.
- Training scope:
  `no_gesture_static`, `partial_swipe`, `random_motion`, `return_motion`,
  `swipe_down`, `swipe_left`, `swipe_up`, `wrong_axis_motion`.
- Training accuracy: `1.0`.

Как проверять:
- Перезапустить приложение.
- На главной:
  - Variant: текущий `prototype_distance` можно оставить;
  - Mode: `auto`;
  - Dynamic: `sequence_knn`.
- Live evaluation:
  - `swipe_up` 10 попыток;
  - `swipe_left` 10 попыток;
  - `swipe_down` 10 попыток;
  - затем negative gestures по 5-10 попыток.

Метрики:
- `live_dynamic_recall`;
- `live_dynamic_false_positive_rate`;
- `live_dynamic_missed_rate`;
- `live_dynamic_wrong_axis_rate`;
- `live_confusion_static_vs_dynamic`;
- latency/runtime в `runtime_performance.jsonl`.

Критерий успеха:
- Dynamic recall выше, чем у `knn` на `dynamic_stats`.
- False positive на negative motions не растёт критично.
- Камера не лагает сильнее, чем на текущем `knn`.

### H-066: Sequence KNN нужен отдельный open-set verifier

Статус: `implemented`, needs live validation

Дата: `2026-06-29`

Наблюдение:
- Live test показал сильный рост positive recall:
  - `swipe_up`: `20/20`;
  - `swipe_left`: `20/20`.
- При этом negative/почти-жесты стали хуже: `sequence_knn`, как и любой KNN,
  склонен выбирать ближайший positive class даже для незавершенного движения.
- `swipe_down` тяжело тестировать из-за возврата руки вверх: подготовительное
  движение часто становится похожим на `swipe_up`.

Гипотеза:
- Для time-series classifier нужен отдельный reject layer:
  - `sequence_knn` отвечает за positive class;
  - `dynamic_sequence_prototypes.json` отвечает за open-set rejection;
  - motion heuristic не должна перебивать `nearest_negative`.

Решение:
- Обучен отдельный prototype/rejection artifact:
  - `models/dynamic_sequence_prototypes.json`;
  - `models/experiments/dynamic_prototype/prototype_distance/dynamic_sequence_prototypes.json`.
- Offline prototype metrics:
  - overall: `0.9935`;
  - positive recall: `0.9444`;
  - negative reject: `1.0000`;
  - negative false positive: `0.0000`.
- Runtime policy для сохранённого sequence-профиля:
  - `dynamic_sequence_prototypes.json` подключается как отдельный open-set
    verifier;
  - `prototype_rejected` не должен превращаться в accepted gesture только из-за
    направления движения.

Как проверять:
- Перезапустить приложение, чтобы runtime загрузил
  `dynamic_sequence_prototypes.json`.
- Настройки:
  - Mode: `auto`;
  - Dynamic: `sequence_knn`;
  - Variant: `prototype_distance`;
  - threshold: сначала `0.90`, затем при misses проверить `0.80`.
- Live positive:
  - `swipe_up`: 20;
  - `swipe_left`: 20;
  - `swipe_down`: 20, руку после движения лучше убирать из кадра вниз/в сторону.
- Live negative:
  - `partial_swipe`: 10;
  - `wrong_axis_motion`: 10;
  - `return_motion`: 10;
  - `random_motion`: 10.

Критерий успеха:
- Positive recall остается высоким для `swipe_up`/`swipe_left`.
- Negative false positive rate заметно ниже, чем у чистого `sequence_knn`.
- В live logs появляются `dynamic_prototype_method=prototype_distance` и
  rejected attempts с `dynamic_prototype_reason=nearest_negative` или
  `low_confidence`.

Следующая гипотеза:
- Если sequence KNN + prototype verifier даст стабильный reject, сравнить
  модель временных рядов сильнее: LSTM/GRU или lightweight temporal CNN.

### H-067: MLflow scalar charts недостаточны для защиты ML-пайплайна

Статус: `implemented`

Дата: `2026-06-29`

Наблюдение:
- Встроенные графики MLflow для одиночных scalar-метрик выглядят как простые
  горизонтальные карточки. Для демонстрации проекта это плохо показывает ход
  ML-решения: не видно классы, thresholds, negative conflicts и состав
  artifact'ов.

Гипотеза:
- Для конкурсного MLOps-сценария лучше хранить в каждом run не только scalar
  metrics, но и полноценный artifact bundle:
  - HTML-dashboard;
  - SVG-графики;
  - CSV-таблицы;
  - JSON-отчет для воспроизводимости.

Реализация:
- `scripts.dynamic_prototype_experiments` теперь логирует в MLflow
  `dynamic_prototype/index.html`.
- В artifact bundle добавлены:
  - `charts/quality.svg`;
  - `charts/per_class.svg`;
  - `charts/negative_conflict.svg`;
  - `charts/thresholds.svg`;
  - `metrics.csv`;
  - `thresholds.csv`;
  - `dynamic_prototype_report.json`.

Проверка:
- Создан свежий MLflow run:
  `022688808c284af0a6ec16749ac8ba3c`.
- Artifact path:
  `dynamic_prototype/index.html`.
- Unit test:
  `tests/unit/test_dynamic_prototype.py::test_dynamic_prototype_artifact_bundle_contains_dashboard`.

Критерий успеха:
- В MLflow метрики остаются машинно-сравнимыми через scalar charts.
- Красивое объяснение run находится во вкладке `Artifacts`, а не теряется в
  неудобных дефолтных карточках.

### H-068: Sequence MLP как следующая модель для сложных динамических жестов

Статус: `implemented`, needs live validation

Дата: `2026-06-29`

Наблюдение:
- `sequence_knn` хорошо распознает `swipe_up` и `swipe_left` в live, но KNN
  принципиально выбирает ближайший класс даже для неполного или чужого
  движения.
- Для произвольных пользовательских dynamic-жестов нужен baseline, который
  умеет нелинейные временные шаблоны, но не требует тяжелого PyTorch/TensorFlow
  окружения.

Решение:
- Добавлена новая dynamic-модель `sequence_mlp`.
- Признаки: `dynamic_sequence`, то есть canonical sequence из 36 кадров,
  развернутая в один вектор.
- Модель: `StandardScaler + MLPClassifier(hidden_layer_sizes=(128, 64))`.
- Артефакты не перезаписывают `sequence_knn`:
  - `models/dynamic_sequence_mlp.pkl`;
  - `models/dynamic_sequence_mlp_classes.json`;
  - `models/dynamic_sequence_mlp_feature_dim.txt`;
  - `models/dynamic_sequence_mlp_feature_mode.txt`;
  - `models/dynamic_sequence_mlp_rejection.json`;
  - `models/dynamic_sequence_mlp_prototypes.json`.
- UI получил отдельный пункт `sequence_mlp` в обучении и live dynamic profile.
- Prototype/rejection layer обучается отдельно:
  `dynamic_sequence_mlp_prototypes.json`.

Offline результат:
- Training run: `sequence_mlp-dynamic_sequence`.
- Samples: `170`.
- Classes: `8`.
- Feature dim: `1584`.
- Train accuracy: `1.0000`.
- Prototype verifier:
  - overall: `0.9935`;
  - positive recall: `0.9444`;
  - negative reject: `1.0000`;
  - negative false positive: `0.0000`;
  - sequence accuracy: `0.9444`.

Почему не LSTM прямо сейчас:
- В текущем `.venv` нет `torch`, `tensorflow` или `keras`.
- Добавление LSTM сейчас потребовало бы тяжелой зависимости и усложнило бы
  воспроизводимость перед конкурсом.
- `sequence_mlp` закрывает следующий исследовательский шаг: neural baseline на
  временной последовательности без изменения runtime-архитектуры.

Как проверять live:
- Mode: `auto`.
- Dynamic: `sequence_mlp`.
- Reject: `open_set_policy`.
- Threshold: начать с `0.90`, затем сравнить с `0.80`.
- Positive:
  - `swipe_up`: 20;
  - `swipe_left`: 20;
  - `swipe_down`: 20.
- Negative:
  - `partial_swipe`: 10;
  - `wrong_axis_motion`: 10;
  - `return_motion`: 10;
  - `random_motion`: 10.

Критерий успеха:
- `sequence_mlp` не хуже `sequence_knn` по positive recall.
- False positive на negative/почти-жестах ниже или хотя бы не выше.
- Если `sequence_mlp` выигрывает live, сделать его кандидатом production
  dynamic profile; если нет, оставить как research baseline и перейти к
  lightweight GRU/LSTM при добавлении PyTorch.

### H-069: Первый live A/B сигнал в пользу sequence_mlp для динамики

Статус: `validated-positive`, needs complex/negative validation

Дата: `2026-06-29`

Наблюдение:
- После перехода с KNN на `sequence_mlp` dynamic-распознавание стало выглядеть
  как реальная ML-система для временных жестов, а не как nearest-neighbor
  lookup.
- Reject layer при threshold `0.90` стал полезно консервативным: даже не очень
  хороший `swipe_up` не проходит, если confidence ниже порога.

Live результат пользователя:
- `swipe_up`: `20/20`, correct `20`, wrong `0`, missed `0`,
  accuracy `100%`, displayed confidence около `93%`.
- `swipe_left`: `20/20`, correct `19`, wrong `1`, missed `0`,
  accuracy `95%`, displayed confidence около `96%`.

Вывод:
- Для динамики KNN больше не выглядит лучшим production-кандидатом: он полезен
  как baseline, но плохо подходит для произвольных динамических жестов из-за
  nearest-class поведения.
- `sequence_mlp + prototype/reject layer` становится главным кандидатом для
  live dynamic pipeline.
- Порог `0.90` выглядит разумным для команд, потому что система лучше
  пропустит слабый жест, чем выполнит лишнюю команду.

Следующая проверка:
- Записать более сложный dynamic-класс, например `circle_clockwise`,
  `zigzag_right` или `hook_down`.
- Обучить `sequence_mlp` заново.
- Прогнать:
  - новый complex gesture: 20 attempts;
  - старые `swipe_up`/`swipe_left`: по 10 attempts для regression check;
  - negative/почти-жесты: `partial_swipe`, `wrong_axis_motion`,
    `return_motion`, `random_motion` по 10 attempts.

Критерий успеха:
- Complex gesture live recall >= `80%`.
- Старые swipe-классы не проседают ниже `90%`.
- Negative false positive остается низким при threshold `0.90`.

### H-070: First-stage ML intent gate для `static/dynamic/none`

Статус: `implemented-offline`, needs live validation

Дата: `2026-06-29`

Гипотеза:
- Перед конкретной классификацией жеста нужен отдельный ML-слой, который
  решает только намерение: `static`, `dynamic` или `none`.
- Это должно уменьшить ложные срабатывания, когда пользователь просто двигает
  рукой, возвращает руку в стартовую точку или показывает почти-жест.

Что взято из research-подхода:
- Из IPN Hand: динамическое распознавание стоит рассматривать как spotting
  временного события, а не как один frame-level prediction.
- Из HaGRID-like логики: отдельные no-gesture/negative примеры полезны для
  reject/open-set поведения, но не должны становиться пользовательскими
  командами.

Реализация:
- Добавлен `cv/intent_gate.py`: компактные sequence-level признаки для intent.
- Добавлен `scripts/train_intent_gate.py`: обучение `intent_gate_mlp`.
- Runtime router теперь сначала спрашивает gate:
  - `dynamic`: держит static и ждёт/принимает dynamic-сегмент;
  - `static`: разрешает static route;
  - `none`: не отдаёт команду;
  - low confidence / missing model: fallback на старый deterministic router.
- Flet auto-mode передаёт в `GestureRecognitionRouter`
  `models/intent_gate_mlp.pkl`, с fallback из экспериментальной папки моделей в
  production `models/`.
- Live evaluation logs now include `intent_gate_*` route fields.

Данные:
- Internal GestureFlow samples: `291`.
- External negative samples: `440`.
- Total for intent gate: `731`.
- Intent classes:
  - `static`: `121`;
  - `dynamic`: `70`;
  - `none`: `540`.

Offline результат:
- accuracy: `0.9672`;
- macro F1: `0.9480`;
- dynamic precision: `0.9444`;
- dynamic recall: `0.9444`;
- dynamic false positive rate: `0.0061`;
- none recall: `0.9778`.

Артефакты:
- `models/intent_gate_mlp.pkl`;
- `models/intent_gate_metadata.json`;
- `docs/experiments/intent_gate_training.md`;
- `docs/experiments/intent_gate_training.json`;
- `docs/experiments/intent_gate_confusion_matrix.svg`;
- MLflow run: `intent-gate-mlp`, experiment `GestureFlow`,
  artifact path `intent_gate`.

Критерий live-проверки:
- Static gestures: не проседают ниже текущего baseline.
- Dynamic gestures: `swipe_up`, `swipe_left`, complex dynamic gesture остаются
  >= `90%` для стабильных классов и >= `80%` для нового complex-класса.
- Random/no-command/partial/return movements: majority should be rejected as
  `none`, not mapped to a command.
- В MLflow смотреть:
  `live_accuracy`, `live_dynamic_recall`, `live_static_hijack_rate`,
  `live_false_trigger_rate`, `intent_gate_label`,
  `intent_gate_confidence`, `intent_gate_reason`.

### H-071: Positive synthetic augmentation ухудшает live-качество персональных жестов

Статус: `rejected`, removed from production pipeline

Дата: `2026-06-29`

Наблюдение:
- После добавления positive synthetic augmentation пользователь заметил, что
  live-распознавание стало хуже: модель принимает “почти жесты” и неточные
  движения легче, чем нужно.
- Для персональных жестов с маленьким датасетом такие synthetic samples
  расширяют класс слишком агрессивно и размывают границу между настоящим
  жестом и случайным движением.

Решение:
- Новая запись жестов больше не создаёт `aug_sample_*`.
- Базовый dataset loader теперь по умолчанию берёт только реальные
  `sample_*.npy`.
- Старые `aug_sample_*` могут лежать на диске, но обучение, синхронизация БД,
  MLflow summaries и dynamic prototype experiments их не используют.
- Negative/reject samples остаются: они решают другую задачу — научить систему
  молчать, когда команды нет.

Параллельные UX-фиксы:
- Если камера была открыта только для записи жеста, она закрывается после
  завершения/отмены записи и не остаётся активной на главном экране.
- Экран `Жесты` при открытии автоматически синхронизирует `data/gestures` с БД.
- `get_db_gestures()` показывает записанные, но ещё не обученные классы, если у
  них есть реальные samples.

Что проверено:
- `tests/unit/test_flet_controller_commands.py`
- `tests/unit/test_gesture_dataset_files.py`
- `tests/unit/test_gestures_preview.py`
- `tests/unit/test_camera_performance.py`
- `tests/unit/test_flet_frame_delivery.py`

Критерий следующей live-проверки:
- Записать новый complex dynamic жест без augmentation.
- Переобучить `sequence_mlp + intent_gate/prototype`.
- Проверить, что настоящий жест проходит, а partial/random/return движения
  чаще уходят в reject/none.

### H-072: Честный validation split для `sequence_mlp`

Статус: `implemented`, needs retrain/live validation

Дата: `2026-06-29`

Проблема:
- Первый `sequence_mlp` показывал `train_accuracy=1.0000`, но для маленького
  персонального датасета это может означать не качество, а запоминание
  обучающих примеров.
- Для конкурса важно показать, что neural dynamic baseline контролируется не
  только train-метрикой, но и отдельным validation-сигналом во время обучения.

Решение:
- Для `sequence_mlp` включён `early_stopping=True`.
- `validation_fraction=0.20`: 20% train-набора удерживаются внутри sklearn как
  validation split и не используются для обновления весов.
- `n_iter_no_change=30`: обучение останавливается, если validation score долго
  не улучшается.
- `alpha=1e-3` оставлен как L2-регуляризация.
- Если классов или сэмплов слишком мало для stratified validation split,
  обучение не падает: early stopping временно отключается с предупреждением.

MLflow:
- В training run теперь логируются:
  - `sequence_mlp_alpha`;
  - `sequence_mlp_early_stopping`;
  - `sequence_mlp_early_stopping_effective`;
  - `sequence_mlp_validation_fraction`;
  - `sequence_mlp_validation_fraction_effective`;
  - `sequence_mlp_n_iter_no_change`;
  - `random_state`.

Что это даёт:
- Можно сравнивать `sequence_mlp` runs честнее: если live-качество проседает,
  мы видим, была ли модель обучена с validation split и какой patience/alpha
  использовались.
- Это пока не заменяет отдельный offline test split и live evaluation, но
  закрывает первый слой защиты от переобучения.

Следующая проверка:
- Переобучить dynamic model с `sequence_mlp`.
- В MLflow открыть новый run и проверить параметры `sequence_mlp_*`.
- Прогнать live:
  - `swipe_up`: 20;
  - `swipe_left`: 20;
  - новый complex dynamic gesture: 20;
  - `partial_swipe`, `wrong_axis_motion`, `return_motion`, `random_motion`:
    по 10.

Критерий успеха:
- Старые positive dynamic gestures остаются >= `90%`.
- Новый complex gesture >= `80%`.
- Negative/почти-жесты не становятся хуже по false positive rate.

### H-073: `sequence_mlp` зафиксирован как production dynamic profile

Статус: `implemented`, needs UI smoke test

Дата: `2026-06-29`

Наблюдение:
- После live A/B `sequence_mlp + open_set_policy` стал главным кандидатом для
  динамических жестов.
- Старые варианты (`knn`, `svm`, `extra_trees`, `sequence_knn`) полезны как
  исследовательские baselines, но в UI они путают пользователя и повышают риск
  случайно обучить/запустить не production-путь.

Решение:
- Production dynamic profile в runtime теперь только `sequence_mlp`.
- Home UI показывает единственный dynamic profile: `sequence_mlp`.
- Dynamic training UI фиксирует:
  - model type: `sequence_mlp`;
  - feature mode: `dynamic_sequence`;
  - model artifact: `models/dynamic_sequence_mlp.pkl`;
  - metadata artifacts: `models/dynamic_sequence_mlp_*`;
  - verifier artifact: `models/dynamic_sequence_mlp_prototypes.json`.
- Legacy dynamic profile values нормализуются обратно в `sequence_mlp`, чтобы
  старые настройки не уводили runtime в `dynamic_knn.pkl`.

Что это даёт:
- Пользовательский live/testing flow становится однозначным.
- MLflow live run names больше не должны появляться как `...-knn-...` для
  production dynamic-пайплайна.
- Следующие эксперименты с LSTM/GRU можно делать отдельно, не смешивая их с
  production UI.

Следующая проверка:
- Открыть Home и убедиться, что в `Dynamic` доступен только `sequence_mlp`.
- На вкладке Training переобучить dynamic model и проверить лог:
  `model=sequence_mlp`, `feature_mode=dynamic_sequence`,
  `out=models/dynamic_sequence_mlp.pkl`.
- Запустить live evaluation и проверить в MLflow run name:
  `live-<gesture>-auto-sequence_mlp-open_set_policy`.
