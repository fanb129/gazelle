# R4-D3 state-supervision comparison

- Protocol: `r4_official_train_test_v1`; all planned epochs retained.
- Train: full official train; evaluation and epoch selection: full official test.
- Selection: highest test AUC; ties use lower L2, higher AP, then earlier epoch.
- Test is used for checkpoint selection; these are not an independent held-out estimate.

| Model | Epoch | AUC | L2 | AP | Accurate stable | State error | Joint gaze response | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| single_frame | 0 | 0.940182 | 0.098995 | 0.886801 | 0.527531 | 0.195165 | 0.582278 | False |
| ordinary_gru | 1 | 0.940368 | 0.099293 | 0.886801 | 0.516114 | 0.193821 | 0.563291 | False |
| ordinary_gru | 2 | 0.940552 | 0.099817 | 0.886927 | 0.515407 | 0.194196 | 0.556962 | True |
| ordinary_gru | 3 | 0.940524 | 0.099664 | 0.885927 | 0.513589 | 0.193821 | 0.550633 | False |
| ordinary_gru | 4 | 0.940415 | 0.099148 | 0.891556 | 0.521519 | 0.186972 | 0.569620 | False |
| ordinary_gru | 5 | 0.940312 | 0.099746 | 0.889851 | 0.507325 | 0.194915 | 0.550633 | False |
| history_validity | 1 | 0.940306 | 0.099228 | 0.886762 | 0.516417 | 0.193789 | 0.563291 | False |
| history_validity | 2 | 0.940372 | 0.099276 | 0.887333 | 0.515963 | 0.193758 | 0.563291 | False |
| history_validity | 3 | 0.940481 | 0.099299 | 0.888879 | 0.512225 | 0.192883 | 0.563291 | True |
| history_validity | 4 | 0.940424 | 0.099050 | 0.892620 | 0.518691 | 0.187379 | 0.582278 | False |
| history_validity | 5 | 0.940476 | 0.099445 | 0.892597 | 0.501010 | 0.196510 | 0.569620 | False |

- Overall test preference: `{"candidate_id": "ordinary_gru:epoch_002", "model_id": "ordinary_gru", "epoch": 2, "metrics": {"auc": 0.9405524785874705, "l2": 0.09981726419337232, "inout_ap": 0.8869274598337004, "jitter_mean": 0.016585097583206594, "accurate_stable_ratio": 0.515407152960194, "state_error_rate": 0.19419600975670773, "state_flip_rate": 0.02301946074188067, "joint_enter_frame_response_rate": 0.6796536796536796, "joint_leave_frame_response_rate": 0.5520833333333334, "joint_gaze_shift_response_rate": 0.5569620253164557, "state_all_response_rate": 0.7431192660550459, "state_enter_frame_response_rate": 0.8934426229508197, "state_leave_frame_response_rate": 0.5520833333333334}}`.
- All output columns come from each selected checkpoint's same epoch.
- Deltas, state/flip/response metrics and paired common-event delays are retained in JSON.

- State supervision: `r4_state_supervision_v1`; fixed settings: `{"lambda_time": 0.1, "response_frames": 3}`.
- The following rows use original output scores; no D2 calibration is applied.
- Each model's complete row comes from its own test-selected epoch.

## Selected outputs before and after state supervision

| Round | Model | Epoch | AUC | L2 | AP | Accurate stable | State errors | False flips | Joint enter | Joint leave | Joint gaze |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single_frame | single_frame | 0 | 0.940182 | 0.098995 | 0.886801 | 0.527531 | 6241 | 592 | 161 | 91 | 92 |
| previous | ordinary_gru | 5 | 0.940541 | 0.099482 | 0.892941 | 0.509093 | 6109 | 731 | 150 | 106 | 88 |
| previous | history_validity | 5 | 0.940466 | 0.099244 | 0.892778 | 0.503334 | 6181 | 798 | 149 | 107 | 88 |
| D3 | ordinary_gru | 2 | 0.940552 | 0.099817 | 0.886927 | 0.515407 | 6210 | 718 | 157 | 106 | 88 |
| D3 | history_validity | 3 | 0.940481 | 0.099299 | 0.888879 | 0.512225 | 6168 | 717 | 153 | 107 | 89 |

Rate/count deltas against the same model's previous selected epoch and the single-frame baseline are in JSON.
Previous large caches/checkpoints are not reopened; prior rows are traced to their hashed small reports.
