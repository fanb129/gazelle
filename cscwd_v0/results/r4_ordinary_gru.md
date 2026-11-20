# R4 compact causal refinement training

- Model: `ordinary_gru`; plan: `r4_compact_v2`.
- Parameters: `347522`; fresh initialization, no test data.
- Selection: `{"status": "selected", "epoch": 1, "candidate_id": "ordinary_gru:epoch_001", "checkpoint": {"path": "/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/selected.pt", "sha256": "a76fcbb48c9880bb887c84f9fcd61da8f71388d8ca0ce9b8e02f86bbeeb0c717", "size_bytes": 1395540}}`.

| Epoch | AUC | L2 | In/Out AP | Accurate stable | State error | Joint gaze response |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.954325 | 0.079602 | 0.960799 | 0.594586 | 0.105289 | 0.610860 |
| 2 | 0.954417 | 0.079503 | 0.960624 | 0.593592 | 0.105289 | 0.606335 |
| 3 | 0.954345 | 0.079565 | 0.960317 | 0.591367 | 0.110026 | 0.601810 |
| 4 | 0.954346 | 0.079371 | 0.959335 | 0.583795 | 0.109237 | 0.601810 |
| 5 | 0.954364 | 0.079356 | 0.953072 | 0.576128 | 0.118593 | 0.601810 |

All epochs and all joint/state metrics are retained in the JSON report.
This is a single-seed development run, not a test or statistical significance result.
