# Defense Metrics One-Pager

## Главный тезис

GestureBind - не просто классификатор жестов, а полный ML-loop для команд ОС:
record -> train -> offline CV -> live evaluation -> rejection -> command
execution.

## Главные цифры

| Что | Результат | Что сказать |
|---|---:|---|
| Static CV | accuracy `0.8000`, macro F1 `0.7389` | Сложный mixed dataset: positive + negative-like classes |
| Static rejection | neg FP `0.0000`, accepted acc `1.0000` | Для ОС-команд важно уметь не срабатывать |
| Dynamic model | accuracy `0.8209`, macro F1 `0.8136`, dynamic-like F1 `0.9472` | Direction-sensitive gestures отделяются в offline CV |
| Dynamic rejection | open-set policy: neg FP `0.0000`, pos recall `1.0000` | Runtime reject policy защищает от no-command движений |
| Prototype dynamic | overall `0.9873`, sequence acc `0.9565` | Проверил verifier для завершенных motion segments |
| Intent gate | accuracy `0.9672`, macro F1 `0.9480` | Сначала routing: static / dynamic / none |
| Live evaluation | `52` attempts, accuracy `0.731` | Честная webcam-метрика ниже offline из-за distribution shift |
| CNN experiment | validation accuracy `0.7963`, слабые negative F1 | Нейросеть проверена, но не лучше conservative baseline |

## Самая важная интерпретация

Offline CV высокий, но live ниже. Это не "провал модели", а обнаруженный
distribution shift между сохраненными samples и естественным показом жеста.
Поэтому следующий шаг - data quality: active motion segment, direction gates,
straightness, balance dynamic classes, live confirmation.

## Что говорить про safety

Accuracy не достаточна для команд ОС. Я отдельно считаю:

- negative false positive rate;
- accepted accuracy;
- coverage;
- positive recall after reject;
- wrong/missed attempts в live.

Коротко:

> Лучше, чтобы система промолчала, чем случайно выполнила команду.

## Слабые места, которые лучше назвать самому

- `swipe_down` часто путается со `swipe_up` в live.
- Dynamic classes нужно дозаписать и выровнять минимум до 20 real samples.
- CNN пока слабее на negative classes.
- External datasets полезны как OOD negatives, но не заменяют персональные жесты.

## 30-секундный ответ

> Я построил полный ML-loop для gesture-to-command системы. Offline я сравнил
> static/dynamic признаки, ExtraTrees, SVM, KNN, CNN и prototype methods.
> Лучшие цифры: static `0.8000` accuracy / `0.7389` macro F1, dynamic ExtraTrees
> `0.8209` accuracy / `0.8136` macro F1, intent gate `0.9672` accuracy. Но для
> ОС-команд главное - safety: лучшие rejection experiments дают `0.0000`
> negative false positives. Live evaluation честно показал gap: `52` attempts,
> accuracy `0.731`, основная проблема - direction confusion. Следующий шаг -
> улучшить запись dynamic gestures, segmentation и live validation.

## Где детали

Полная версия: [DEFENSE_METRICS_BRIEF.md](DEFENSE_METRICS_BRIEF.md)
