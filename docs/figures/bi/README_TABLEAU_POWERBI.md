# Данные для Tableau или Power BI

## Файлы

- `dataset_summary.csv` - полный список папок жестов, количество сэмплов и признак попадания в метрики.
- `gesture_class_counts.csv` - короткая таблица распределения сэмплов по классам.
- `gesture_readiness.csv` - готовность каждого жеста: есть ли минимум 20 читаемых сэмплов.
- `gesture_metrics_long.csv` - основной файл для heatmap общих метрик.
- `gesture_metrics_per_class.csv` - precision/recall/F1 по каждому жесту.
- `optimization_comparison.csv` - сравнение базовой и оптимизированной модели для раздела 5.5.
- `monitoring_quality_metrics.csv` - KPI по тестам, датасету и текущей модели.

## Обновление CSV после записи жестов

Из корня проекта выполнить:

```bash
python scripts/export_tableau_csvs.py
```

Скрипт читает `data/gestures`, пересчитывает метрики и обновляет CSV в этой папке. Папки жестов без `sample_*.npy` останутся в `dataset_summary.csv` и `gesture_readiness.csv`, но не попадут в классификационные метрики.

Текущий экспорт:

- 81 читаемый сэмпл;
- 4 класса в метриках: `CTRLZ`, `Hend`, `New`, `w`;
- 13 папок жестов всего;
- `models/classes.json` пока содержит 2 класса, поэтому после записи новых жестов модель нужно переобучить отдельно.

## Heatmap метрик в Tableau

1. Открыть Tableau Public/Desktop.
2. `Connect` -> `Text file` -> выбрать `gesture_metrics_long.csv`.
3. Создать новый Sheet.
4. Перетащить `Variant` в `Rows`.
5. Перетащить `Metric` в `Columns`.
6. Перетащить `Value` в `Color`.
7. Перетащить `Value` в `Label`.
8. В `Marks` выбрать тип `Square`.
9. Для `Value` выбрать агрегацию `Average`.
10. В формате числа поставить 3 знака после запятой.
11. В цветовой шкале выбрать последовательную шкалу от светло-желтого/красного к зеленому.
12. Название листа: `Сравнение метрик классификации`.
13. Экспортировать изображение: `Worksheet` -> `Export` -> `Image`.
14. Сохранить как `docs/figures/fig_5_3_tableau_metrics.png`.

## Распределение сэмплов в Tableau

1. Подключить `dataset_summary.csv`.
2. Перетащить `Class` в `Rows`.
3. Перетащить `SampleCount` в `Columns`.
4. Перетащить `IncludedInMetrics` в `Color`.
5. В `Marks` выбрать `Bar`.
6. Отсортировать по `SampleCount` по убыванию.
7. В `Label` добавить `SampleCount`.
8. Для графика только обученных классов поставить фильтр `IncludedInMetrics = yes`.

## Метрики по жестам в Tableau

1. Подключить `gesture_metrics_per_class.csv`.
2. Перетащить `Class` в `Rows`.
3. Перетащить `Metric` в `Columns`.
4. Перетащить `Value` в `Color` и `Label`.
5. В `Filters` выбрать `Variant = Центрирование примера` или сравнивать варианты через `Variant` в `Pages`.

## Готовность жестов в Tableau

1. Подключить `gesture_readiness.csv`.
2. Перетащить `Class` в `Rows`.
3. Перетащить `ReadableSampleCount` в `Columns`.
4. Перетащить `Status` в `Color`.
5. В `Label` добавить `ReadableSampleCount`.

## Heatmap метрик в Power BI

1. Открыть Power BI Desktop.
2. `Get data` -> `Text/CSV` -> выбрать `gesture_metrics_long.csv`.
3. Нажать `Load`.
4. Добавить визуал `Matrix`.
5. В `Rows` поместить `Variant`.
6. В `Columns` поместить `Metric`.
7. В `Values` поместить `Value`, агрегация `Average`.
8. В форматировании значения поставить 3 знака после запятой.
9. Включить `Conditional formatting` -> `Background color` для `Value`.
10. Цветовая шкала: минимум около `0.88`, максимум `1.00`, максимум зеленый.
11. Заголовок визуала: `Сравнение метрик классификации для вариантов обработки признаков`.
12. Экспортировать или скопировать визуал как изображение.
13. Сохранить как `docs/figures/fig_5_3_powerbi_metrics.png`.

## Что вставлять в диплом

Для подписи `Рисунок 5.3 - Сравнение вариантов обработки признаков` лучше вставить heatmap/matrix из Tableau или Power BI. Она показывает четыре метрики сразу: `Accuracy`, `Macro precision`, `Macro recall`, `Macro F1`.

Если BI-инструмента под рукой нет, временный вариант уже создан через matplotlib:

- `docs/figures/fig_5_3_metrics_matrix.png`
- `docs/figures/fig_5_3_metrics_accuracy.png`
