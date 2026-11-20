# R4-D3 state-supervised causal refinement

- Model: `ordinary_gru`; protocol: `r4_official_train_test_v1`.
- Supervision: `r4_state_supervision_v1`; lambda_time=0.1; response_frames=3.
- Fresh module initialization; frozen GazeSpot; no D2 output calibration.
- Training: all official train tracks; no validation partition.
- Selection: official test AUC descending, L2 ascending, AP descending, earlier epoch.
- Test is used for checkpoint selection; this is not an independent held-out test estimate.
- Selected: `{"status": "selected", "epoch": 2, "candidate_id": "ordinary_gru:epoch_002", "checkpoint": {"path": "/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/selected.pt", "sha256": "aa07f010532bb34aa2aa298185b37a00aa96ad7221ffdae462d5906c50eb2af0", "size_bytes": 1397396}}`.

| Epoch | Test AUC | Test L2 | Test AP | Accurate stable | State error | Joint gaze response |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.940368 | 0.099293 | 0.886801 | 0.516114 | 0.193821 | 0.563291 |
| 2 | 0.940552 | 0.099817 | 0.886927 | 0.515407 | 0.194196 | 0.556962 |
| 3 | 0.940524 | 0.099664 | 0.885927 | 0.513589 | 0.193821 | 0.550633 |
| 4 | 0.940415 | 0.099148 | 0.891556 | 0.521519 | 0.186972 | 0.569620 |
| 5 | 0.940312 | 0.099746 | 0.889851 | 0.507325 | 0.194915 | 0.550633 |

All metrics in the selected result come from the same epoch.
All epoch checkpoints, complete test predictions and diagnostics remain on the server.
