# R2 causal joint-head smoke training

- Config: `{"model": "ordinary_causal_GRU_joint", "feature_channels": 768, "hidden_dim": 128, "clip_frames": 16, "batch_size": 8, "epochs": 1, "learning_rate": 0.0001, "lambda_state": 0.1, "seed": 0, "optimizer": "AdamW", "weight_decay": 0.01, "gradient_clip_norm": 1.0, "device": "cuda:0", "precision": "float32; no_autocast; tf32_disabled", "history": "own_predicted_outputs; persistent_slots; detach_only_at_TBPTT_chunks", "selection": "none; fixed_smoke_configuration; retain_last_epoch; no_test"}`
- Last epoch: `1`; no checkpoint selection.
- Validation: `{"auc": 0.9543254321198485, "l2": 0.07960169608613302, "inout_ap": 0.9607991920922045, "person_frames": 34201, "localization_frames": 21758}`
- Joint/state temporal results and all epoch records are in the JSON report.
- This is a development smoke run, not a final method or test result.
