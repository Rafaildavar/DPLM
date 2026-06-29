# IPN Hand Conversion Report

Generated at: `1782725470.799`

## Summary

- status: `mediapipe_unavailable`
- IPN root: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand`
- frames root: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/frames04_extracted/frames`
- annotation path: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/drive-download-20260628T185129Z-3-001/Annot_List.txt`
- output root: `/Users/remi/Developer/GUAP/DPLM/data/external/ipn_hand_frames04`
- segments found/included: `5649` / `4848`
- converted/skipped: `0` / `0`
- detection rate: `0.0000`

## Expectation Check

| Expectation | Result |
|---|---|
| `mapped_to_negative_labels` | `not_checked` |
| `mlflow_conversion_report` | `met` |
| `outputs_under_data_external` | `not_written` |
| `production_model_unchanged` | `met` |

## Labels

| Target label | Converted samples |
|---|---:|
| none | 0 |

## Warnings

- MediaPipe CLI preflight failed: exit=-6;     @        0x11f3fe2d8  mediapipe::api2::TensorsToDetectionsCalculator::Open()
    @        0x11fa8bf70  mediapipe::CalculatorNode::OpenNode()
    @        0x11fa7c06c  mediapipe::internal::SchedulerQueue::OpenCalculatorNode()
    @        0x11fa7bed4  mediapipe::internal::SchedulerQueue::RunNextTask()
    @        0x11ff914a0  mediapipe::ThreadPool::RunWorker()
    @        0x11ff90ea0  mediapipe::ThreadPool::WorkerThread::ThreadBody()
    @        0x1943acc08  _pthread_start
    @        0x1943a7ba8  thread_start
- no IPN samples were converted

## Sample Preview

| Status | Target | IPN label | Video | Frames | Detection | Reason |
|---|---|---|---|---:|---:|---|
| none | none | none | none | 0 | 0.0000 | no samples processed |

## Next Step

If status is `missing_input`, place IPN frames under
`data/raw/ipn_hand/frames` and annotations as `Annot_List.txt`
inside `data/raw/ipn_hand`, or pass `--frames-root` and
`--annotation` explicitly.
Then rerun:

```bash
PYTHON=.venv/bin/python make ipn-convert
```
