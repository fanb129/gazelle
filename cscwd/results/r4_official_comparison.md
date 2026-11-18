# R4 official train/test comparison

- Protocol: `r4_official_train_test_v1`; all planned epochs retained.
- Train: full official train; evaluation and epoch selection: full official test.
- Selection: highest test AUC; ties use lower L2, higher AP, then earlier epoch.
- Test is used for checkpoint selection; these are not an independent held-out estimate.

| Model | Epoch | AUC | L2 | AP | Accurate stable | State error | Joint gaze response | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| single_frame | 0 | 0.940182 | 0.098995 | 0.886801 | 0.527531 | 0.195165 | 0.582278 | False |
| ordinary_gru | 1 | 0.940317 | 0.099149 | 0.887040 | 0.520964 | 0.193789 | 0.569620 | False |
| ordinary_gru | 2 | 0.940483 | 0.099250 | 0.887553 | 0.518084 | 0.194321 | 0.556962 | False |
| ordinary_gru | 3 | 0.940491 | 0.099153 | 0.888812 | 0.517478 | 0.192476 | 0.556962 | False |
| ordinary_gru | 4 | 0.940526 | 0.099122 | 0.893950 | 0.516367 | 0.186441 | 0.569620 | False |
| ordinary_gru | 5 | 0.940541 | 0.099482 | 0.892941 | 0.509093 | 0.191038 | 0.556962 | True |
| history_validity | 1 | 0.940302 | 0.099207 | 0.887163 | 0.519903 | 0.193852 | 0.569620 | False |
| history_validity | 2 | 0.940384 | 0.099246 | 0.888240 | 0.517781 | 0.193289 | 0.563291 | False |
| history_validity | 3 | 0.940427 | 0.099184 | 0.890105 | 0.516872 | 0.192163 | 0.563291 | False |
| history_validity | 4 | 0.940428 | 0.099000 | 0.894294 | 0.514700 | 0.187473 | 0.575949 | False |
| history_validity | 5 | 0.940466 | 0.099244 | 0.892778 | 0.503334 | 0.193289 | 0.556962 | True |

- Overall test preference: `{"candidate_id": "ordinary_gru:epoch_005", "model_id": "ordinary_gru", "epoch": 5, "metrics": {"auc": 0.9405409348335207, "l2": 0.09948183608586703, "inout_ap": 0.8929411945270439, "jitter_mean": 0.0168164345822856, "accurate_stable_ratio": 0.5090927460092948, "state_error_rate": 0.19103758834198512, "state_flip_rate": 0.02343624763553589, "joint_enter_frame_response_rate": 0.6493506493506493, "joint_leave_frame_response_rate": 0.5520833333333334, "joint_gaze_shift_response_rate": 0.5569620253164557, "state_all_response_rate": 0.7155963302752294, "state_enter_frame_response_rate": 0.8442622950819673, "state_leave_frame_response_rate": 0.5520833333333334}}`.
- All output columns come from each selected checkpoint's same epoch.
- Deltas, state/flip/response metrics and paired common-event delays are retained in JSON.
