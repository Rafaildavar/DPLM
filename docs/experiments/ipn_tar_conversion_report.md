# IPN Tar Conversion Report

Generated at: `1782726779.575`

## Summary

- status: `mediapipe_unavailable`
- tar files: `4`
- videos in archives: `160`
- annotation path: `/Users/remi/Developer/GUAP/DPLM/data/raw/ipn_hand/drive-download-20260628T185129Z-3-001/Annot_List.txt`
- output root: `/Users/remi/Developer/GUAP/DPLM/data/external/ipn_hand_tar`
- segments found/candidate/selected: `5649` / `3906` / `240`
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
| `B0A` | 805 | 40 |
| `B0B` | 802 | 40 |
| `D0X` | 1179 | 40 |
| `G01` | 160 | 17 |
| `G02` | 160 | 17 |
| `G07` | 160 | 17 |
| `G08` | 160 | 18 |
| `G09` | 160 | 17 |
| `G10` | 160 | 17 |
| `G11` | 160 | 17 |

## Warnings

- MediaPipe CLI preflight failed: exit=-6;     @        0x11f4162d8  mediapipe::api2::TensorsToDetectionsCalculator::Open()
    @        0x11faa3f70  mediapipe::CalculatorNode::OpenNode()
    @        0x11fa9406c  mediapipe::internal::SchedulerQueue::OpenCalculatorNode()
    @        0x11fa93ed4  mediapipe::internal::SchedulerQueue::RunNextTask()
    @        0x11ffa94a0  mediapipe::ThreadPool::RunWorker()
    @        0x11ffa8ea0  mediapipe::ThreadPool::WorkerThread::ThreadBody()
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
