# JMLC Dataset Profile

Generated: `2026-06-24T13:03:39+00:00`

## Summary

| Metric | Value |
|---|---:|
| Class directories | 16 |
| Active classes | 7 |
| Empty classes | 9 |
| Samples | 151 |
| Valid samples | 151 |
| Invalid samples | 0 |
| Active sample min | 20 |
| Active sample max | 30 |
| Imbalance ratio | 1.5 |
| Frame min | 30 |
| Frame max | 60 |
| Frame mean | 35.96 |
| Feature dimensions | 42 |

## Classes

| Class | Samples | Valid | Invalid | Frame min | Frame max | Frame mean | Feature dims | Issues |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `CTRLZ` | 21 | 21 | 0 | 30 | 30 | 30.0 | 42 | - |
| `Hend` | 20 | 20 | 0 | 30 | 30 | 30.0 | 42 | - |
| `Newest` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `Open` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `UP` | 20 | 20 | 0 | 30 | 30 | 30.0 | 42 | - |
| `gun` | 20 | 20 | 0 | 30 | 30 | 30.0 | 42 | - |
| `hand_left` | 30 | 30 | 0 | 60 | 60 | 60.0 | 42 | - |
| `hello` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `palm` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `peace_sign` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `sh3` | 20 | 20 | 0 | 30 | 30 | 30.0 | 42 | - |
| `swipe_right` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `test_gesture` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `three` | 20 | 20 | 0 | 30 | 30 | 30.0 | 42 | - |
| `thumbs_up` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |
| `ываы` | 0 | 0 | 0 | n/a | n/a | n/a | n/a | - |

## Risks

- `low_samples_per_class:CTRLZ,Hend,UP,gun,sh3,three`
- `empty_classes:Newest,Open,hello,palm,peace_sign,swipe_right,test_gesture,thumbs_up,ываы`

## JMLC Interpretation

- Основная offline-метрика должна быть `macro_f1`, потому что пользовательские
  классы могут быть несбалансированы.
- Если есть `mixed_feature_dimensions`, one-hand и two-hand сценарии нужно
  сравнивать отдельно или явно выравнивать признаки.
- Классы с малым числом семплов нельзя использовать как сильное доказательство
  качества модели без дополнительного сбора данных.
