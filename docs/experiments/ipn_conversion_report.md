# IPN Hand Conversion Report

Generated at: `1782670873.126`

## Summary

- status: `missing_input`
- IPN root: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand`
- frames root: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/frames`
- annotation path: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/annotations/ipnall.json`
- output root: `/Users/remi/Developer/GUAP/DPLM/data/external/ipn_hand`
- segments found/included: `0` / `0`
- converted/skipped: `0` / `0`
- detection rate: `0.0000`

## Expectation Check

| Expectation | Result |
|---|---|
| `mapped_to_negative_labels` | `not_checked` |
| `mlflow_conversion_report` | `met` |
| `outputs_under_data_external` | `not_checked` |
| `production_model_unchanged` | `met` |

## Labels

| Target label | Converted samples |
|---|---:|
| none | 0 |

## Warnings

- frames root not found: /Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/frames
- annotation file not found: /Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/annotations/ipnall.json

## Sample Preview

| Status | Target | IPN label | Video | Frames | Detection | Reason |
|---|---|---|---|---:|---:|---|
| none | none | none | none | 0 | 0.0000 | no samples processed |

## Next Step

If status is `missing_input`, place IPN frames and annotations at the
reported paths or pass `--frames-root` and `--annotation` explicitly.
Then rerun:

```bash
PYTHON=.venv/bin/python make ipn-convert
```
