# Dynamic Completion Gate Benchmark

This benchmark truncates every real positive recording by cumulative motion progress at the same fractions used to train the class-conditional completion profiles. Prefixes remain in raw scale-aware coordinates until the gate runs.

## Summary

| Set | Attempts | LSTM accepted | Gate accepted | Final accepted |
|---|---:|---:|---:|---:|
| Full | `60` | `60` (`1.0000`) | `60` (`1.0000`) | `60` (`1.0000`) |
| Prefix | `240` | `214` (`0.8917`) | `6` (`0.0250`) | `6` (`0.0250`) |

Prefix false-accept reduction: `0.9720`.

## Online State-Machine Replay

| Set | Attempts | Correct | Wrong class | No command |
|---|---:|---:|---:|---:|
| Full | `60` | `59` (`0.9833`) | `0` | `1` |
| Prefix | `240` | n/a | `0` | `235` |

Online prefix false-accept rate: `0.0208`.

## Per Class And Fraction

| Class | Fraction | Attempts | LSTM accepted | Gate accepted | Final accepted |
|---|---:|---:|---:|---:|---:|
| `SwipeLeft` | `1.00` | `20` | `20` | `20` | `20` |
| `SwipeLeft` | `0.70` | `20` | `20` | `0` | `0` |
| `SwipeLeft` | `0.55` | `20` | `20` | `0` | `0` |
| `SwipeLeft` | `0.40` | `20` | `20` | `0` | `0` |
| `SwipeLeft` | `0.25` | `20` | `20` | `0` | `0` |
| `diagonal` | `1.00` | `20` | `20` | `20` | `20` |
| `diagonal` | `0.70` | `20` | `20` | `5` | `5` |
| `diagonal` | `0.55` | `20` | `20` | `0` | `0` |
| `diagonal` | `0.40` | `20` | `16` | `0` | `0` |
| `diagonal` | `0.25` | `20` | `5` | `0` | `0` |
| `zoom` | `1.00` | `20` | `20` | `20` | `20` |
| `zoom` | `0.70` | `20` | `20` | `1` | `1` |
| `zoom` | `0.55` | `20` | `16` | `0` | `0` |
| `zoom` | `0.40` | `20` | `17` | `0` | `0` |
| `zoom` | `0.25` | `20` | `20` | `0` | `0` |

## Interpretation

The report measures source recordings from the current personal dataset. Prefixes from one source never cross validation groups during profile calibration. It is an offline safety benchmark, not a replacement for the final live partial-motion matrix.
