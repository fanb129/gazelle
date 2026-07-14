#!/usr/bin/env bash
set -euo pipefail

cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P3/gazefollow /home/fb/src/paper/gazelleV1/AAAIResults/logs

CANDIDATE=$(/home/fb/anaconda3/envs/py310/bin/python -c 'import json; print(json.load(open("/home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/winner.json"))["winner"]["candidate"])')

/home/fb/anaconda3/envs/py310/bin/python -c 'import json; w=json.load(open("/home/fb/src/paper/gazelleV1/AAAIResults/P3/screen/winner.json"))["winner"]["candidate"]; m=json.load(open(f"/home/fb/src/paper/gazelleV1/AAAIResults/P3/confirm/{w}/test.report.json"))["metrics"]; passed=sum((m["auc"] >= 0.9420, m["l2"] <= 0.0980, m["inout_ap"] >= 0.9050)); assert passed >= 2, f"P3B failed the frozen 2/3 gate: {m}"'

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_p3_alchemy.py \
  --phase p3c --candidate "${CANDIDATE}" --dataset gazefollow \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --init-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --output-dir "/home/fb/src/paper/gazelleV1/AAAIResults/P3/gazefollow/${CANDIDATE}" \
  --epochs 15 --batch-size 32 --workers 8 \
  --validation-fraction 0.1 --validation-seed 3106 \
  --lr 1e-5 --fusion-lr 1e-4 --seed 3106 --device cuda:0

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/evaluate_p3_alchemy.py \
  --phase p3c --candidate "${CANDIDATE}" --dataset gazefollow \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --checkpoint "/home/fb/src/paper/gazelleV1/AAAIResults/P3/gazefollow/${CANDIDATE}/best.pt" \
  --output-prefix "/home/fb/src/paper/gazelleV1/AAAIResults/P3/gazefollow/${CANDIDATE}/test" \
  --seed 3106 --device cuda:0
