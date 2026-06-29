# Intent Gate Training

- samples: `731`
- feature dim: `25`
- accuracy: `0.9672`
- macro F1: `0.9480`
- records by intent: `{"dynamic": 70, "none": 540, "static": 121}`

## Confusion Matrix

| expected \ predicted | static | dynamic | none |
|---|---:|---:|---:|
| `static` | 28 | 1 | 1 |
| `dynamic` | 0 | 17 | 1 |
| `none` | 3 | 0 | 132 |

## Per-Class Metrics

| intent | precision | recall | false positive rate | support |
|---|---:|---:|---:|---:|
| `static` | 0.9032 | 0.9333 | 0.0196 | 30 |
| `dynamic` | 0.9444 | 0.9444 | 0.0061 | 18 |
| `none` | 0.9851 | 0.9778 | 0.0417 | 135 |
