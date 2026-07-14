#!/usr/bin/env bash
set -euo pipefail

cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P3/screen /home/fb/src/paper/gazelleV1/AAAIResults/logs

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_p3_alchemy.py \
  --phase p3a --candidate r0 --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/r0_continued_baseline \
  --epochs 3 --batch-size 32 --workers 8 --frame-sample-every 6 \
  --validation-fraction 0.1 --validation-seed 3106 \
  --lr 1e-5 --fusion-lr 1e-4 --lr-inout 1e-3 --seed 3106 --device cuda:0

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_p3_alchemy.py \
  --phase p3a --candidate r1 --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/r1_residual_refinement \
  --epochs 3 --batch-size 32 --workers 8 --frame-sample-every 6 \
  --validation-fraction 0.1 --validation-seed 3106 \
  --lr 1e-5 --fusion-lr 1e-4 --lr-inout 1e-3 --seed 3106 --device cuda:0

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_p3_alchemy.py \
  --phase p3a --candidate r2 --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/r2_cross_layer_attention \
  --epochs 3 --batch-size 32 --workers 8 --frame-sample-every 6 \
  --validation-fraction 0.1 --validation-seed 3106 \
  --lr 1e-5 --fusion-lr 1e-4 --lr-inout 1e-3 --seed 3106 --device cuda:0

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_p3_alchemy.py \
  --phase p3a --candidate r3 --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/r3_residual_coord005 \
  --epochs 3 --batch-size 32 --workers 8 --frame-sample-every 6 \
  --validation-fraction 0.1 --validation-seed 3106 \
  --lr 1e-5 --fusion-lr 1e-4 --lr-inout 1e-3 --seed 3106 --device cuda:0

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/select_p3_candidate.py \
  --screen-dir /home/fb/src/paper/gazelleV1/AAAIResults/P3/screen
