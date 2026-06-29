# IPN Hand Conversion Report

Generated at: `1782730246.951`

## Summary

- status: `ok`
- IPN root: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand`
- frames root: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/frames04_extracted/frames`
- annotation path: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/drive-download-20260628T185129Z-3-001/Annot_List.txt`
- output root: `/Users/remi/Developer/GUAP/DPLM/data/external/ipn_hand_frames04`
- segments found/included: `5649` / `4848`
- converted/skipped: `240` / `2013`
- detection rate: `0.5243`

## Expectation Check

| Expectation | Result |
|---|---|
| `mapped_to_negative_labels` | `met` |
| `mlflow_conversion_report` | `met` |
| `outputs_under_data_external` | `check` |
| `production_model_unchanged` | `met` |

## Labels

| Target label | Converted samples |
|---|---:|
| `negative_external_ipn_dynamic` | 240 |

## Sample Preview

| Status | Target | IPN label | Video | Frames | Detection | Reason |
|---|---|---|---|---:|---:|---|
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G11` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G02` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G08` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G10` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G09` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G07` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G01` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#229` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `D0X` | `1CM1_4_R_#230` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G10` | `1CM1_4_R_#230` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#230` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G11` | `1CM1_4_R_#230` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#230` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0B` | `1CM1_4_R_#230` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `G09` | `1CM1_4_R_#230` | 0 | 0.0000 | no frame files found for segment |
| skipped | `negative_external_ipn_dynamic` | `B0A` | `1CM1_4_R_#230` | 0 | 0.0000 | no frame files found for segment |

## Next Step

If status is `missing_input`, place IPN frames under
`data/raw/ipn_hand/frames` and annotations as `Annot_List.txt`
inside `data/raw/ipn_hand`, or pass `--frames-root` and
`--annotation` explicitly.
Then rerun:

```bash
PYTHON=.venv/bin/python make ipn-convert
```
