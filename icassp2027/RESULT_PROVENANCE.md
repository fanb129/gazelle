# ICASSP 2027 result provenance

This note records where each number in `main.tex` comes from. It is an internal
writing aid and is not part of the submitted paper.

## Main benchmark and ablation tables

- Source: `2026ACMMM/main.tex` and the corresponding compiled MM manuscript.
- Full GazeSpot configuration: frozen DINOv3 backbone, four uniformly sampled
  feature levels, GGSF, SASA, and the lightweight Transformer decoder.
- GazeSpot ViT-B: GazeFollow `0.958 / 0.101 / 0.043`; VAT
  `0.940 / 0.099 / 0.887` (AUC / L2 / third dataset-specific metric).
- GazeSpot ViT-L: GazeFollow `0.961 / 0.093 / 0.038`; VAT
  `0.944 / 0.092 / 0.912`.
- The four-row component ablation uses DINOv3 ViT-B/16.

## Crowd-density table

- Source: the MM Crowd evaluation in `2026ACMMM/main.tex`.
- Split rule: number of annotated heads in each VAT test frame.
- Frame counts: `<3`: 7,945; `=3`: 3,015; `=4`: 1,628; `>4`: 539.
- These rows compare the DINOv3 ViT-B last-layer baseline with the full
  GazeSpot ViT-B model.

## Head-box perturbation

- Canonical index: `../rebuttal/p0_experiment_commands.md`.
- Machine-readable source:
  `../rebuttal/results/p0/vat_crowd_bbox_noise_test_crowd_ge4.json`.
- Split: VAT `test_crowd_ge4`, 2,167 frames and 9,525 heads.
- Checkpoints recorded by the result JSON:
  - baseline: `train_vat_vitb_v0/.../epoch_7.pt`;
  - GazeSpot: `train_vat_sasa_ggsf/.../epoch_7.pt`.
- Seed: `3106`.
- Jitter: translation and independent width/height scaling at 0%, 5%, 10%,
  and 20%, with normalized boxes clipped to `[0, 1]`.

## GOO-real cross-dataset evaluation

- Source: `../rebuttal/results/p0/gooreal_eval.json`.
- Command record: `../help.md`, using the GazeFollow-trained ViT-B baseline
  (`train_gazefollow_vitb_v0/.../epoch_14.pt`) and GazeSpot
  (`train_gazefollow_sasa_ggsf/.../epoch_14.pt`) checkpoints.
- Dataset: 2,146 GOO-real images and 2,146 heads, evaluated without fine-tuning.
- Baseline: 0.8583 AUC / 0.1997 L2; GazeSpot: 0.8833 / 0.1761.

## Complexity and runtime

- Source: `../rebuttal/results/p0/complexity_latency.json`.
- Device and protocol: RTX 3090, batch size 1, 50 warmup iterations, 200 timed
  iterations, CUDA synchronization enabled, input resolution 512 x 512.
- GazeSpot ViT-B: 3.515M trainable / 89.185M total parameters, 141.08 GFLOPs,
  22.82 ms per image.
- Baseline ViT-B: 2.827M trainable / 88.496M total parameters, 107.22 GFLOPs,
  18.37 ms per image.

## Concurrent ECCV 2026 comparison

- Paper: Jiajie Mi, Xinyu Liu, Mengke Song, and Chenglizhao Chen,
  "Multi-scale Object-aware Gaze Estimation via Geometric Reasoning,"
  ECCV 2026, arXiv:2606.29334.
- Values transcribed from its result table: GazeFollow
  `0.961 / 0.094 / 0.038`; VAT `0.948 / 0.095 / 0.923`; 7.1M trainable
  parameters.
