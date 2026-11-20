# R4 compact causal refinement training

- Model: `history_validity`; plan: `r4_compact_v2`.
- Parameters: `372677`; fresh initialization, no test data.
- Selection: `{"status": "selected", "epoch": 1, "candidate_id": "history_validity:epoch_001", "checkpoint": {"path": "/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/selected.pt", "sha256": "6b478e92607efc0215956ec49464cde3b771c7200a4346f48b203da2d1c58ea0", "size_bytes": 1497679}}`.

| Epoch | AUC | L2 | In/Out AP | Accurate stable | State error | Joint gaze response |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.954290 | 0.079562 | 0.960808 | 0.594964 | 0.105114 | 0.610860 |
| 2 | 0.954322 | 0.079507 | 0.960499 | 0.592976 | 0.105991 | 0.606335 |
| 3 | 0.954315 | 0.079572 | 0.959885 | 0.589805 | 0.110289 | 0.601810 |
| 4 | 0.954287 | 0.079583 | 0.958722 | 0.580624 | 0.111868 | 0.597285 |
| 5 | 0.954321 | 0.079476 | 0.953733 | 0.576128 | 0.119207 | 0.601810 |

All epochs and all joint/state metrics are retained in the JSON report.
This is a single-seed development run, not a test or statistical significance result.
