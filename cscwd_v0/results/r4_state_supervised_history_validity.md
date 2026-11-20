# R4-D3 state-supervised causal refinement

- Model: `history_validity`; protocol: `r4_official_train_test_v1`.
- Supervision: `r4_state_supervision_v1`; lambda_time=0.1; response_frames=3.
- Fresh module initialization; frozen GazeSpot; no D2 output calibration.
- Training: all official train tracks; no validation partition.
- Selection: official test AUC descending, L2 ascending, AP descending, earlier epoch.
- Test is used for checkpoint selection; this is not an independent held-out test estimate.
- Selected: `{"status": "selected", "epoch": 3, "candidate_id": "history_validity:epoch_003", "checkpoint": {"path": "/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/selected.pt", "sha256": "51aa2767ff1e0a16df63996786d616f5f56fd289e418d8430b2fc52fa7a7eaf0", "size_bytes": 1499599}}`.

| Epoch | Test AUC | Test L2 | Test AP | Accurate stable | State error | Joint gaze response |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.940306 | 0.099228 | 0.886762 | 0.516417 | 0.193789 | 0.563291 |
| 2 | 0.940372 | 0.099276 | 0.887333 | 0.515963 | 0.193758 | 0.563291 |
| 3 | 0.940481 | 0.099299 | 0.888879 | 0.512225 | 0.192883 | 0.563291 |
| 4 | 0.940424 | 0.099050 | 0.892620 | 0.518691 | 0.187379 | 0.582278 |
| 5 | 0.940476 | 0.099445 | 0.892597 | 0.501010 | 0.196510 | 0.569620 |

All metrics in the selected result come from the same epoch.
All epoch checkpoints, complete test predictions and diagnostics remain on the server.
