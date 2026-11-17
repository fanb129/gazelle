# R4 compact validation comparison

- Plan: `r4_compact_v2`; single seed, all planned epochs retained.
- Status: development snapshot awaiting review; test configuration is not frozen.

| Model | Epoch | AUC | L2 | AP | Accurate stable | State error | Joint gaze response | Eligible | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| single_frame | 0 | 0.954206 | 0.079605 | 0.960923 | 0.601874 | 0.111254 | 0.610860 | True | False |
| ordinary_gru | 1 | 0.954325 | 0.079602 | 0.960799 | 0.594586 | 0.105289 | 0.610860 | True | True |
| ordinary_gru | 2 | 0.954417 | 0.079503 | 0.960624 | 0.593592 | 0.105289 | 0.606335 | True | False |
| ordinary_gru | 3 | 0.954345 | 0.079565 | 0.960317 | 0.591367 | 0.110026 | 0.601810 | True | False |
| ordinary_gru | 4 | 0.954346 | 0.079371 | 0.959335 | 0.583795 | 0.109237 | 0.601810 | True | False |
| ordinary_gru | 5 | 0.954364 | 0.079356 | 0.953072 | 0.576128 | 0.118593 | 0.601810 | False | False |
| history_validity | 1 | 0.954290 | 0.079562 | 0.960808 | 0.594964 | 0.105114 | 0.610860 | True | True |
| history_validity | 2 | 0.954322 | 0.079507 | 0.960499 | 0.592976 | 0.105991 | 0.606335 | True | False |
| history_validity | 3 | 0.954315 | 0.079572 | 0.959885 | 0.589805 | 0.110289 | 0.601810 | True | False |
| history_validity | 4 | 0.954287 | 0.079583 | 0.958722 | 0.580624 | 0.111868 | 0.597285 | False | False |
| history_validity | 5 | 0.954321 | 0.079476 | 0.953733 | 0.576128 | 0.119207 | 0.601810 | False | False |

- Pareto IDs: `["history_validity:epoch_001", "ordinary_gru:epoch_001", "ordinary_gru:epoch_002", "ordinary_gru:epoch_004", "ordinary_gru:epoch_005", "single_frame:epoch_000"]`.
- Validation preference: `{"candidate_id": "single_frame:epoch_000", "model_id": "single_frame", "epoch": 0, "metrics": {"auc": 0.9542055630729601, "l2": 0.07960456904362263, "inout_ap": 0.9609226821493625, "jitter_mean": 0.014554664754237905, "accurate_stable_ratio": 0.6018742013346585, "state_error_rate": 0.11125405689892108, "state_flip_rate": 0.017669105836159202, "joint_enter_frame_response_rate": 0.7764705882352941, "joint_leave_frame_response_rate": 0.5502958579881657, "joint_gaze_shift_response_rate": 0.6108597285067874, "state_all_response_rate": 0.7597765363128491, "state_enter_frame_response_rate": 0.9470899470899471, "state_leave_frame_response_rate": 0.5502958579881657}, "precision_eligible": true, "precision_checks": {"auc": true, "l2": true, "inout_ap": true}}`.
- Deltas, all state/flip/response metrics and common-event paired delays are in the JSON.
- A selected checkpoint or Pareto membership does not by itself establish method effectiveness.
