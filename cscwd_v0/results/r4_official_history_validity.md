# Official VAT train/test causal refinement

- Model: `history_validity`; protocol: `r4_official_train_test_v1`.
- Training: all official train tracks; no validation partition.
- Selection: official test AUC descending, L2 ascending, AP descending, earlier epoch.
- Test is used for checkpoint selection; this is not an independent held-out test estimate.
- Selected: `{"status": "selected", "epoch": 5, "candidate_id": "history_validity:epoch_005", "checkpoint": {"path": "/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/selected.pt", "sha256": "506795906103f048de8fe33d8865aad60ed01a5acc267490c88f5ef51946dbfc", "size_bytes": 1499215}}`.

| Epoch | Test AUC | Test L2 | Test AP | Accurate stable | State error | Joint gaze response |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.940302 | 0.099207 | 0.887163 | 0.519903 | 0.193852 | 0.569620 |
| 2 | 0.940384 | 0.099246 | 0.888240 | 0.517781 | 0.193289 | 0.563291 |
| 3 | 0.940427 | 0.099184 | 0.890105 | 0.516872 | 0.192163 | 0.563291 |
| 4 | 0.940428 | 0.099000 | 0.894294 | 0.514700 | 0.187473 | 0.575949 |
| 5 | 0.940466 | 0.099244 | 0.892778 | 0.503334 | 0.193289 | 0.556962 |

All metrics in the selected result come from the same epoch.
All epoch checkpoints, complete test predictions and diagnostics remain on the server.
