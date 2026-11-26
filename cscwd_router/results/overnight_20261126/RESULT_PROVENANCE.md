
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
