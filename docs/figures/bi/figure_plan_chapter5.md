# BI-графики для усиления главы 5

## 1. Состав локального датасета

Файл: `dataset_summary.csv`

Рекомендуемый тип визуализации:
- Tableau: horizontal bar chart.
- Power BI: clustered bar chart.

Поля:
- `Class` -> Rows / Axis.
- `SampleCount` -> Columns / Values.
- `FeatureDim` можно вывести в tooltip.

Подпись:
`Рисунок 5.1 - Распределение обучающих примеров по классам локального датасета`

Зачем нужен:
показывает фактическую базу эксперимента: `CTRLZ = 21`, `Hend = 20`, `New = 20`, `w = 20`, всего 81 читаемый пример. Папки без `sample_*.npy` можно оставить на графике как нулевые классы или отфильтровать через `IncludedInMetrics = yes`.

## 2. Сравнение метрик классификации

Файл: `gesture_metrics_long.csv`

Рекомендуемый тип визуализации:
- Tableau: heatmap (`Square`, `Value` на Color и Label).
- Power BI: Matrix с conditional formatting.

Поля:
- `Variant` -> Rows.
- `Metric` -> Columns.
- `Value` -> Color + Label.

Подпись:
`Рисунок 5.3 - Сравнение вариантов обработки признаков`

Зачем нужен:
показывает не только accuracy, но и precision/recall/F1.

Дополнительный разрез:
файл `gesture_metrics_per_class.csv` позволяет сделать heatmap по отдельным жестам: `Class` -> Rows, `Metric` -> Columns, `Value` -> Color + Label, `Variant` -> Filter.

## 3. Покрытие unit-тестами по группам

Файл: `unit_test_groups.csv`

Рекомендуемый тип визуализации:
- Tableau: horizontal bar chart.
- Power BI: clustered bar chart.

Поля:
- `Group` -> Rows / Axis.
- `Passed` -> Columns / Values.
- `Failed` можно добавить в tooltip.

Подпись:
`Рисунок 5.4 - Распределение пройденных unit-тестов по функциональным группам`

Зачем нужен:
делает главу тестирования сильнее: видно, что проверяются не только ML-метрики, но и БД, команды, политики, контроллер, pointer mode.

Фактический прогон:
`137 passed in 26.01s`

Команда:
`pytest tests/unit --ignore=tests/unit/test_app_controller.py --no-cov`

## 4. Проверка пользовательских сценариев

Файл: `manual_scenarios.csv`

Рекомендуемый тип визуализации:
- Tableau: checklist/table.
- Power BI: table or matrix.

Поля:
- `Scenario`
- `Stage`
- `Priority`
- `ExpectedResult`
- `StatusForDiploma`

Подпись:
`Рисунок 5.5 - Чек-лист пользовательских сценариев проверки GestureBind`

Важно:
перед вставкой в диплом лучше пройти сценарии вручную и заменить `Проверить вручную` на `Пройдено`.

## 4a. Готовность новых жестов

Файл: `gesture_readiness.csv`

Рекомендуемый тип визуализации:
- Tableau: horizontal bar chart.
- Power BI: clustered bar chart.

Поля:
- `Class` -> Rows / Axis.
- `ReadableSampleCount` -> Columns / Values.
- `Status` -> Color / Legend.

Подпись:
`Рисунок 5.x - Готовность жестов к обучению и оценке`

Зачем нужен:
показывает, какие добавленные жесты уже имеют достаточно записанных примеров, а какие пока являются только пустыми папками.

## 5. Сравнение базовой и оптимизированной модели

Файл: `optimization_comparison_long.csv`

Рекомендуемый тип визуализации:
- Tableau: grouped bar chart.
- Power BI: clustered column chart.

Поля:
- `MetricRu` -> Columns / Axis.
- `Value` -> Rows / Values.
- `Model` -> Color / Legend.
- Для времени инференса лучше делать отдельный график или dual-axis, потому что единицы разные.

Подпись:
`Рисунок 5.6 - Сравнение базовой и оптимизированной KNN-модели`

Зачем нужен:
показывает практический эффект оптимизации: accuracy выросла с 78.4% до 96.2%, время снизилось с 15 мс до 8 мс.

## 6. Важность гиперпараметров

Файл: `optimization_parameter_importance.csv`

Рекомендуемый тип визуализации:
- Tableau: bar chart.
- Power BI: bar chart.

Поля:
- `Parameter` -> Rows / Axis.
- `Importance` -> Columns / Values.

Подпись:
`Рисунок 5.7 - Важность гиперпараметров KNN при оптимизации`

Зачем нужен:
показывает, что `n_neighbors` влияет сильнее, чем `metric` и `weights`.

## Минимальный набор для диплома

Если не хочется перегружать текст, вставь 4 рисунка:

1. `Распределение сэмплов по классам`.
2. `Heatmap метрик классификации`.
3. `Unit-тесты по функциональным группам`.
4. `Сравнение базовой и оптимизированной модели`.

Этого достаточно, чтобы глава 5 выглядела как аналитическая, а не только описательная.
