# JMLC Experiments

Эта директория хранит воспроизводимые артефакты ML-части GestureFlow.

Правило: важный ML-вывод не считается готовым, пока рядом нет команды
воспроизведения и сохраненного отчета.

## Структура

Планируемые файлы:

- `dataset_profile.json` / `dataset_profile.md` - паспорт датасета жестов:
  классы, семплы, длины записей, размерности, дисбаланс и проблемы данных.
- `model_comparison.json` / `model_comparison.md` - сравнение признаков и
  моделей на одинаковом validation protocol.
- `threshold_report.json` / `threshold_report.md` - выбор confidence threshold:
  coverage, accepted accuracy, rejected predictions.
- `latency_report.json` / `latency_report.md` - задержки ML и live-pipeline.
- `live_eval.json` / `live_eval.md` - проверка в рабочем приложении: ложные
  срабатывания, успешные команды, ошибки.
- `figures/` - confusion matrix, распределения датасета и графики метрик.

## Минимальный протокол

Каждый отчет должен отвечать на пять вопросов:

1. Какие данные использовались?
2. Как они были предобработаны?
3. Какая модель или настройка проверялась?
4. Какая метрика была основной и почему?
5. Какие ограничения результата остаются?

## Главные метрики

- `macro_f1` - основная offline-метрика для несбалансированных классов.
- `per_class_precision`, `per_class_recall`, `per_class_f1`.
- `confusion_matrix`.
- `false_positive_rate`.
- `recommended_confidence_threshold`.
- `latency_ms`.
- `live_false_triggers_per_10_min`.
- `command_success_rate`.

## Команды

Команды будут добавляться по мере реализации скриптов:

```bash
python -m scripts.jmlc_dataset_profile
python -m scripts.compare_models
python -m scripts.benchmark_latency --mode ml
```

