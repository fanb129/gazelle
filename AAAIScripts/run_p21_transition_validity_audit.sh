#!/usr/bin/env bash
set -euo pipefail

if [[ "$(pwd)" != "/home/fb/src/paper/gazelleV1" ]]; then
  echo "Run this script from /home/fb/src/paper/gazelleV1" >&2
  exit 2
fi

mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P21/transition_validity

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/p21_transition_validity_audit.py generate \
  --data-root /newhome/fb/dataset/videoattentiontarget \
  --annotation-json /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --transition-csv /home/fb/src/paper/gazelleV1/AAAIResults/P2/transition_reliability/per_transition.csv \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P21/transition_validity \
  --per-direction 60 \
  --iou-threshold 0.3 \
  --seed 3106

echo "P2.1 contact sheets and audit.csv generated. Complete the review columns before summarize mode."
