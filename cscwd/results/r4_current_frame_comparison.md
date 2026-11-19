# R4-D4 current-frame attribution control

- Control: `r4_current_frame_v1`; original per-frame loss; no D2 calibration or D3 temporal loss.
- History is cleared at every frame; only current observations enter the model.
- Full official test is evaluated every epoch and used for complete-epoch selection.
- Selection: AUC descending, L2 ascending, AP descending, earlier epoch; not an independent held-out estimate.

| Model | Epoch | AUC | L2 | AP | Accurate stable | State error | Joint gaze response | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| single_frame | 0 | 0.940182 | 0.098995 | 0.886801 | 0.527531 | 0.195165 | 0.582278 | False |
| current_frame | 1 | 0.940249 | 0.099034 | 0.887221 | 0.519903 | 0.194040 | 0.569620 | False |
| current_frame | 2 | 0.940339 | 0.099312 | 0.887878 | 0.518438 | 0.193727 | 0.563291 | False |
| current_frame | 3 | 0.940367 | 0.099167 | 0.888965 | 0.516822 | 0.191913 | 0.563291 | False |
| current_frame | 4 | 0.940367 | 0.099037 | 0.893718 | 0.517478 | 0.187504 | 0.575949 | False |
| current_frame | 5 | 0.940407 | 0.099362 | 0.893067 | 0.507678 | 0.191475 | 0.550633 | True |

## Selected outputs

| Model | Epoch | AUC | L2 | AP | Accurate stable windows | State errors | False flips | Joint gaze responses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary_gru | 5 | 0.940541 | 0.099482 | 0.892941 | 10078 | 6109 | 731 | 88 |
| history_validity | 5 | 0.940466 | 0.099244 | 0.892778 | 9964 | 6181 | 798 | 88 |
| current_frame | 5 | 0.940407 | 0.099362 | 0.893067 | 10050 | 6123 | 742 | 87 |

All five matched-epoch comparisons and complete metric/count deltas are retained in JSON.
Stored parameter count matches ordinary GRU, while effective parameter usage differs with zero history.
Prior temporal rows use hashed original reports; old large artifacts are not reopened.
