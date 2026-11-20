# R2 frozen feature cache

- passed: `True`
- contract: `{"layer_index": 11, "channels": 768, "grid": [32, 32], "storage_dtype": "float16", "source_dtype": "torch.float32", "layout": "NCHW", "spatial_pooling": "none", "image_unit": "unique_current_RGB", "learned_projection_cached": false}`
- 原始 R1 heatmap/In/Out float32 缓存保留；特征为显式有损 FP16，未训练时序模块。
- cache: `{"path": "/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16", "index_sha256": "f817826cc271b7b257600d2f6db2fac3659f3b74111a7cabb0d84fab359feeb1", "images": 58507, "contract": {"layer_index": 11, "channels": 768, "grid": [32, 32], "storage_dtype": "float16", "source_dtype": "torch.float32", "layout": "NCHW", "spatial_pooling": "none", "image_unit": "unique_current_RGB", "learned_projection_cached": false}, "files": {"scene_features": {"path": "scene_features.npy", "shape": [58507, 768, 32, 32], "dtype": "float16", "sha256": "82145bc436614802c7e4500dbdf39fdddb02ae442c091c5f63ada587df288c9c", "values_sha256": "ecdc096750df8214df5cea3197d2e84ce09a8a34958b5332e83d1dd6198cb828", "size_bytes": 92023554176}}, "checks": {"all_unique_images_cached": true, "fp16_exact_readback": true}, "passed": true}`
- 下一步：普通因果联合头 smoke train；不是最终模型或 test 结果。
