# R2 frozen feature probe

- passed: `True`
- contract: `{"layer_index": 11, "channels": 768, "grid": [32, 32], "storage_dtype": "float16", "source_dtype": "torch.float32", "layout": "NCHW", "spatial_pooling": "none", "image_unit": "unique_current_RGB", "learned_projection_cached": false}`
- 原始 R1 heatmap/In/Out float32 缓存保留；特征为显式有损 FP16，未训练时序模块。
- base_prediction_comparison: `{"absolute_tolerance": 1e-05, "heatmap_max_abs_error": 9.834766387939453e-07, "inout_max_abs_error": 4.172325134277344e-07, "argmax_changed_people": 0, "retained_prediction_policy": "original_R1_FP32_arrays_unchanged"}`
- fp16_quantization: `{"max_abs_error": 0.001953125, "rmse": 8.295787608211457e-05, "relative_l2": 0.00020418368920662578, "interpretation": "lossy_feature_storage; not_lossless_FP32_features"}`
- head_grid_coverage: `{"minimum_bbox_area_in_patch_units": 2.5044444444444407, "heads_with_no_patch_center_inside": 0, "pooling_policy": "fractional_bbox_patch_overlap; no_coarse_grid_pooling"}`
- storage: `{"total_unique_images": 58507, "estimated_feature_bytes": 92023554048, "required_bytes_with_metadata_reserve": 92090662912, "available_bytes": 2731964395520, "target_parent": "/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7"}`
- checks: `{"model_all_parameters_frozen": true, "model_in_eval": true, "fp32_heatmaps_agree_with_R1": true, "fp32_inout_agrees_with_R1": true, "argmax_agrees_with_R1": true, "native_grid_kept": true, "fp16_features_finite": true, "disk_space_for_native_cache": true}`
- 下一步：回传本报告，确认原生网格、量化与存储契约后再运行 cache。
