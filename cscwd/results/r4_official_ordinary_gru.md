# Official VAT train/test causal refinement

- Model: `ordinary_gru`; protocol: `r4_official_train_test_v1`.
- Training: all official train tracks; no validation partition.
- Selection: official test AUC descending, L2 ascending, AP descending, earlier epoch.
- Test is used for checkpoint selection; this is not an independent held-out test estimate.
- Selected: `{"status": "selected", "epoch": 5, "candidate_id": "ordinary_gru:epoch_005", "checkpoint": {"path": "/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/selected.pt", "sha256": "b1511f1c7bdc9d136956de4f4a42648b7a0ea4ed714b6a18e53685aacb58c96d", "size_bytes": 1397012}}`.

| Epoch | Test AUC | Test L2 | Test AP | Accurate stable | State error | Joint gaze response |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.940317 | 0.099149 | 0.887040 | 0.520964 | 0.193789 | 0.569620 |
| 2 | 0.940483 | 0.099250 | 0.887553 | 0.518084 | 0.194321 | 0.556962 |
| 3 | 0.940491 | 0.099153 | 0.888812 | 0.517478 | 0.192476 | 0.556962 |
| 4 | 0.940526 | 0.099122 | 0.893950 | 0.516367 | 0.186441 | 0.569620 |
| 5 | 0.940541 | 0.099482 | 0.892941 | 0.509093 | 0.191038 | 0.556962 |

All metrics in the selected result come from the same epoch.
All epoch checkpoints, complete test predictions and diagnostics remain on the server.
