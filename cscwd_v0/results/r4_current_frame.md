# R4-D4 current-frame-only joint refinement

- Model: `current_frame`; protocol: `r4_official_train_test_v1`.
- Control: `r4_current_frame_v1`; fresh state at every frame.
- Original per-frame MSE/BCE; no D3 temporal loss or D2 calibration.
- Same nominal layers as ordinary GRU; structurally unused history weights are documented in parameter_usage.
- Training: all official train tracks; no validation partition.
- Selection: official test AUC descending, L2 ascending, AP descending, earlier epoch.
- Test is used for checkpoint selection; this is not an independent held-out test estimate.
- Selected: `{"status": "selected", "epoch": 5, "candidate_id": "current_frame:epoch_005", "checkpoint": {"path": "/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/selected.pt", "sha256": "93a849123ec80a983143b5cb5c25f400dd00d477a74366cf174feb54c7d9599e", "size_bytes": 1397204}}`.

| Epoch | Test AUC | Test L2 | Test AP | Accurate stable | State error | Joint gaze response |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.940249 | 0.099034 | 0.887221 | 0.519903 | 0.194040 | 0.569620 |
| 2 | 0.940339 | 0.099312 | 0.887878 | 0.518438 | 0.193727 | 0.563291 |
| 3 | 0.940367 | 0.099167 | 0.888965 | 0.516822 | 0.191913 | 0.563291 |
| 4 | 0.940367 | 0.099037 | 0.893718 | 0.517478 | 0.187504 | 0.575949 |
| 5 | 0.940407 | 0.099362 | 0.893067 | 0.507678 | 0.191475 | 0.550633 |

All metrics in the selected result come from the same epoch.
All epoch checkpoints, complete test predictions and diagnostics remain on the server.
