#!/usr/bin/env bash
set -euo pipefail

cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P3/confirm /home/fb/src/paper/gazelleV1/AAAIResults/logs

CANDIDATE=$(/home/fb/anaconda3/envs/py310/bin/python -c 'import json; print(json.load(open("/home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/winner.json"))["winner"]["candidate"])')

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_p3_alchemy.py \
  --phase p3b --candidate "${CANDIDATE}" --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --output-dir "/home/fb/src/paper/gazelleV1/AAAIResults/P3/confirm/${CANDIDATE}" \
  --epochs 8 --batch-size 32 --workers 8 --frame-sample-every 6 \
  --validation-fraction 0.1 --validation-seed 3106 \
  --lr 1e-5 --fusion-lr 1e-4 --lr-inout 1e-3 --seed 3106 --device cuda:0

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/evaluate_p3_alchemy.py \
  --phase p3b --candidate "${CANDIDATE}" --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --checkpoint "/home/fb/src/paper/gazelleV1/AAAIResults/P3/confirm/${CANDIDATE}/best.pt" \
  --output-prefix "/home/fb/src/paper/gazelleV1/AAAIResults/P3/confirm/${CANDIDATE}/test" \
  --seed 3106 --device cuda:0
