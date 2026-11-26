
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

## GF Router to VAT · vitb

- report: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/unconditional/vat/vitb/summary.json`; SHA256: `0cf6b2c476453c92fb422ce5c502ad7257cf2233fc7be3a67fe76d7f8b1601a8`
- gf_summary: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/unconditional/gazefollow/vitb/summary.json`; SHA256: `6e027b79a3a8613de06f2a9384d35e800130b7680ff9b35a3d93d1bc5136f4a7`
- gf_manifest: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/unconditional/gazefollow/vitb/manifest.json`; SHA256: `141c6623013ecc2a4a9708c41fd44f91203afe0b1a747c98a151b6e15f1e4d07`
- gf_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/gazefollow/vitb/sparse_decoder/best_test.pt`; SHA256: `0b6f73c5a6c63a89a38c06c369d59b2aa774ee6510b626e2764c9fad96b4198e`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- initial_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/initial.pt`; SHA256: `62df76138f347bc633a64c217d26bf7371a7e2d58f1bdd1cb264ebaf03a462bc`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/best_test.pt`; SHA256: `9cb6bfb7f8f17b4c110199ce4dc53c58d878f15597c690d96e2c32cc63d91ccc`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/last.pt`; SHA256: `f783c41e20014ba758a4f3c39c0dcd734f0e4f29c9a524ebcbd402b6c7ab4dc3`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/predictions.jsonl.gz`; SHA256: `4e90edb8b9a58c188b79c649289e48043a46db39530beb6d8a0bc2d66400619c`
- manifest: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/unconditional/vat/vitb/manifest.json`; SHA256: `1062bd8212825fd5cdcdf21f7d1e3b28f49d5c6bc929646505ff2df277a73b87`
- history: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/unconditional/vat/vitb/history.json`; SHA256: `63cf10db2f24b90c3c50e025316d69aa7218fc114d0eee526540b6e09124a743`
- initialization_report: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/unconditional/vat/vitb/initialization.json`; SHA256: `3d105f39640d7dd2eb8db0661f694c74be5878965603ec32fc0e11ff4fbbb2de`
- initial_test_report: `/home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/unconditional/vat/vitb/initial_test.json`; SHA256: `0be35250ea6a92a1c8c80515f6be6051676499f8dd06d655471e5d7cb1e400a2`
- epoch_0: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/epoch_0.pt`; SHA256: `0703ddf31a1114de30d4f6df8d30418f1c103833b5ad2f34028ee846b5bd33d0`
- epoch_1: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/epoch_1.pt`; SHA256: `4e7d4997d811cd21ad155a079a36d54a68bd9d086c2971f4d546a582ffb3c7fd`
- epoch_2: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/epoch_2.pt`; SHA256: `4fbbba4d055e351b9960d3dd4687997e421498a08165735458d8faf32bbe7ce6`
- epoch_3: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/epoch_3.pt`; SHA256: `090f7b013cb7fa6f108993c2e0d2c859d8e22d85230ec9bb530a4edde507bbec`
- epoch_4: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/epoch_4.pt`; SHA256: `75933b6eb08096c078d1b15a0a51f0b66adad797b822b816dfacba797819997a`
- epoch_5: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/epoch_5.pt`; SHA256: `3e24bb99d2f421597596e4ca77ebaba17283ab59c9385160ed57bc7ff7d2ab9c`
- epoch_6: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/epoch_6.pt`; SHA256: `efeb3c96c37731a22579ca7c54b7ddceb64843ebe31b41a2e43833d58b2be64d`
- epoch_7: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/unconditional/vat/vitb/epoch_7.pt`; SHA256: `ebc831af9ee5930acfa5b848e8aba5f9fa70ffa228bcecf74f7d932b98d8dc23`
