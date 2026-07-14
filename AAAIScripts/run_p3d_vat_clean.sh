#!/usr/bin/env bash
set -euo pipefail

cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P3/vat_clean /home/fb/src/paper/gazelleV1/AAAIResults/logs

CANDIDATE=$(/home/fb/anaconda3/envs/py310/bin/python -c 'import json; print(json.load(open("/home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/winner.json"))["winner"]["candidate"])')

/home/fb/anaconda3/envs/py310/bin/python -c 'import json; w=json.load(open("/home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/winner.json"))["winner"]["candidate"]; m=json.load(open(f"/home/fb/src/paper/gazelleV1/AAAIResults/P3/gazefollow/{w}/test.report.json"))["metrics"]; assert m["auc"] >= 0.9558 and m["min_l2"] <= 0.0462 and m["avg_l2"] <= 0.1042, f"P3C shows obvious GazeFollow degradation: {m}"'

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_p3_alchemy.py \
  --phase p3d --candidate "${CANDIDATE}" --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint "/home/fb/src/paper/gazelleV1/AAAIResults/P3/gazefollow/${CANDIDATE}/best.pt" \
  --allow-missing-inout \
  --output-dir "/home/fb/src/paper/gazelleV1/AAAIResults/P3/vat_clean/${CANDIDATE}" \
  --epochs 8 --batch-size 32 --workers 8 --frame-sample-every 6 \
  --validation-fraction 0.1 --validation-seed 3106 \
  --lr 1e-5 --fusion-lr 1e-4 --lr-inout 1e-3 --seed 3106 --device cuda:0

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/evaluate_p3_alchemy.py \
  --phase p3d --candidate "${CANDIDATE}" --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --checkpoint "/home/fb/src/paper/gazelleV1/AAAIResults/P3/vat_clean/${CANDIDATE}/best.pt" \
  --output-prefix "/home/fb/src/paper/gazelleV1/AAAIResults/P3/vat_clean/${CANDIDATE}/test" \
  --seed 3106 --device cuda:0
