# IPN Hand Conversion Report

Generated at: `1782675577.697`

## Summary

- status: `ok`
- IPN root: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand`
- frames root: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/frames`
- annotation path: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/drive-download-20260628T185129Z-3-001/Annot_List.txt`
- output root: `/Users/remi/Developer/GUAP/DPLM/data/external/ipn_hand`
- segments found/included: `5649` / `4848`
- converted/skipped: `200` / `28`
- detection rate: `0.7994`

## Expectation Check

| Expectation | Result |
|---|---|
| `mapped_to_negative_labels` | `met` |
| `mlflow_conversion_report` | `met` |
| `outputs_under_data_external` | `met` |
| `production_model_unchanged` | `met` |

## Labels

| Target label | Converted samples |
|---|---:|
| `negative_external_ipn_dynamic` | 200 |

## Sample Preview

| Status | Target | IPN label | Video | Frames | Detection | Reason |
|---|---|---|---|---:|---:|---|
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 0 | 0.0000 | insufficient detected hand frames |
| converted | `negative_external_ipn_dynamic` | `G11` | `1CM1_4_R_#229` | 37 | 0.9737 |  |
| converted | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 59 | 0.9833 |  |
| converted | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 58 | 0.9667 |  |
| converted | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 51 | 0.8500 |  |
| converted | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 53 | 0.8833 |  |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 0 | 0.0000 | insufficient detected hand frames |
| converted | `negative_external_ipn_dynamic` | `G02` | `1CM1_4_R_#229` | 25 | 1.0000 |  |
| converted | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 51 | 0.8500 |  |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 1 | 0.0167 | insufficient detected hand frames |
| converted | `negative_external_ipn_dynamic` | `G08` | `1CM1_4_R_#229` | 35 | 1.0000 |  |
| converted | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 53 | 0.8833 |  |
| converted | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 51 | 0.8500 |  |
| converted | `negative_external_ipn_dynamic` | `G10` | `1CM1_4_R_#229` | 43 | 1.0000 |  |
| converted | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 59 | 0.9833 |  |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 2 | 0.0333 | insufficient detected hand frames |
| converted | `negative_external_ipn_dynamic` | `G09` | `1CM1_4_R_#229` | 29 | 0.8056 |  |
| converted | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 60 | 1.0000 |  |
| converted | `negative_external_ipn_dynamic` | `G07` | `1CM1_4_R_#229` | 45 | 1.0000 |  |
| converted | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 50 | 0.8333 |  |
| converted | `negative_external_ipn_dynamic` | `G01` | `1CM1_4_R_#229` | 31 | 1.0000 |  |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 4 | 0.0667 | insufficient detected hand frames |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#230` | 3 | 0.1250 | insufficient detected hand frames |
| converted | `negative_external_ipn_dynamic` | `G10` | `1CM1_4_R_#230` | 41 | 1.0000 |  |
| converted | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#230` | 48 | 0.8000 |  |
| converted | `negative_external_ipn_dynamic` | `G11` | `1CM1_4_R_#230` | 34 | 1.0000 |  |
| converted | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#230` | 58 | 0.9667 |  |
| converted | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#230` | 53 | 0.8833 |  |
| converted | `negative_external_ipn_dynamic` | `G09` | `1CM1_4_R_#230` | 38 | 0.9268 |  |
| converted | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#230` | 55 | 0.9167 |  |

## Next Step

If status is `missing_input`, place IPN frames under
`data/raw/ipn_hand/frames` and annotations as `Annot_List.txt`
inside `data/raw/ipn_hand`, or pass `--frames-root` and
`--annotation` explicitly.
Then rerun:

```bash
PYTHON=.venv/bin/python make ipn-convert
```
