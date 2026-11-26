
## VAT AP head-only / vitb

- report: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/ap/vitb/summary.json`; SHA256: `71c5a5c65a8bc33c071da0929688db0f4cd6958aa6ff5bd6544d94cfd77be063`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/ap/vitb/best_test.pt`; SHA256: `520e7f568bde04abcc8e3ab540fee7b5c059f305b9b3237d37dbbe31b64f06e9`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/ap/vitb/last.pt`; SHA256: `fc4a6a8dec3121d75bae6c2715d8ee046806438d23ee112991b1771941ac3cfe`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/ap/vitb/predictions.jsonl.gz`; SHA256: `4d8a80e5e303297e1912ed014c1b14ebe4c1f67956f5bc1f562dd6bdc0d33fca`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- parent_summary: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json`; SHA256: `c092d90ca2b094e575a94a70c21a61042611a2cd217332d0f10c35f277bb0f99`

## DINOv3 test-selected · gazefollow / b
- report: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/unconditional/gazefollow/vitb/summary.json`; SHA256: `6e027b79a3a8613de06f2a9384d35e800130b7680ff9b35a3d93d1bc5136f4a7`
- protocol: official train/test, no validation; FP32 accuracy; frozen backbone
- initialization: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/final_ablation_seed0/scene_only/gazefollow/vitb/support/final.pt`; SHA256: `0f6ed428b2b1fe4ef4cff349e9353c2ce3bb5e331a7a6537bd2870ecf4de6e0e`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/gazefollow/vitb/sparse_decoder/best_test.pt`; SHA256: `0b6f73c5a6c63a89a38c06c369d59b2aa774ee6510b626e2764c9fad96b4198e`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/gazefollow/vitb/sparse_decoder/last.pt`; SHA256: `9e2027afdeb2c49c03140a638db9be555f73d16529648c68d4516aeddf654210`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/gazefollow/vitb/predictions.jsonl.gz`; SHA256: `3706eaa5e9207e8cad86fcfa9bf1e572bd6b238ce90905dbc301143a35ac7863`

## VAT AP head-only / vitl

- report: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/ap/vitl/summary.json`; SHA256: `a1ff17a050d0d1672ef0e69b27c94a7757da3be11cd848e53581b975fe34ce89`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/ap/vitl/best_test.pt`; SHA256: `1ce3cbe2bf1d0deea78e6d62ccef59426a308348810be9e2878f79a34bb47b87`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/ap/vitl/last.pt`; SHA256: `3be2f10a38c2873f241ce0404f8c0f9af980f118cf03b6462db3371f38313dd4`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/ap/vitl/predictions.jsonl.gz`; SHA256: `42926e0ab47b2b1a0fa03b2ca27b33a26482702acda66a7eab7d085079bf4a8e`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitl16_pretrain.pth`; SHA256: `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`
- parent_summary: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/summary.json`; SHA256: `78b45b94d7fe65cdff17f3bd86f2afd99f7ec6b2add5c3cb83c30d19550f4c9f`
