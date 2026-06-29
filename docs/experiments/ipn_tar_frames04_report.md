# IPN Tar Conversion Report

Generated at: `1782725064.731`

## Summary

- status: `mediapipe_unavailable`
- tar files: `1`
- videos in archives: `40`
- annotation path: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/drive-download-20260628T185129Z-3-001/Annot_List.txt`
- output root: `/Users/remi/Developer/GUAP/DPLM/data/external/ipn_hand_tar`
- segments found/candidate/selected: `5649` / `1078` / `240`
- converted/skipped: `0` / `0`
- detection rate: `0.0000`

## Labels

| Target label | Converted samples |
|---|---:|
| none | 0 |

## Original IPN Labels

| IPN label | Converted |
|---|---:|
| none | 0 |

## Candidate IPN Labels

| IPN label | Candidate | Selected |
|---|---:|---:|
| `B0A` | 201 | 40 |
| `B0B` | 200 | 40 |
| `D0X` | 397 | 40 |
| `G01` | 40 | 17 |
| `G02` | 40 | 17 |
| `G07` | 40 | 17 |
| `G08` | 40 | 17 |
| `G09` | 40 | 17 |
| `G10` | 40 | 17 |
| `G11` | 40 | 18 |

## Warnings

- MediaPipe CLI preflight failed: exit=-6;     @        0x124a2a2d8  mediapipe::api2::TensorsToDetectionsCalculator::Open()
    @        0x1250b7f70  mediapipe::CalculatorNode::OpenNode()
    @        0x1250a806c  mediapipe::internal::SchedulerQueue::OpenCalculatorNode()
    @        0x1250a7ed4  mediapipe::internal::SchedulerQueue::RunNextTask()
    @        0x1255bd4a0  mediapipe::ThreadPool::RunWorker()
    @        0x1255bcea0  mediapipe::ThreadPool::WorkerThread::ThreadBody()
    @        0x1943acc08  _pthread_start
    @        0x1943a7ba8  thread_start

## Sample Preview

| Status | Target | IPN label | Video | Found | Detected | Detection | Reason |
|---|---|---|---|---:|---:|---:|---|
| none | none | none | none | 0 | 0 | 0.0000 | no samples processed |

## Interpretation

- The script does not extract full archives to disk.
- Only compact `44`-dimensional landmark sequences are saved.
- IPN labels mapped as `validation_reference` are excluded by default to avoid teaching the model to reject swipe-like motions.
