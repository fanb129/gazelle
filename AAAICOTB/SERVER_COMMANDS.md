# 服务器运行命令

以下命令默认仓库位于 `/home/fb/src/paper/gazelleV1`，Python 为 `/home/fb/anaconda3/envs/py310/bin/python`，VAT 为 `/newhome/fb/dataset/videoattentiontarget`。路径不同可通过环境变量覆盖。

## 0. 同步后检查

```bash
cd /home/fb/src/paper/gazelleV1

PYTHON_BIN=/home/fb/anaconda3/envs/py310/bin/python
$PYTHON_BIN -c "import torch, torchvision, sklearn; print(torch.__version__, torch.cuda.is_available())"
$PYTHON_BIN -m py_compile AAAICOTB/*.py AAAICOTB/tests/*.py
$PYTHON_BIN -m pytest -q AAAICOTB/tests/test_binding.py
bash -n AAAICOTB/run_pilot.sh
bash -n AAAICOTB/run_full.sh
```

期望：5 个 binding tests 通过，CUDA 为 `True`。

## 1. 一天 pilot（建议先运行）

默认历史 checkpoint 与 `help.md` 一致。先确认：

```bash
cd /home/fb/src/paper/gazelleV1

ls experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt
ls experiments/train_vat_vitb_v0/2026-03-20_22-30-50/epoch_7.pt
ls experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt
ls /newhome/fb/dataset/videoattentiontarget/train_preprocessed.json
ls /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json
```

后台启动：

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p AAAIResults/COTB

setsid nohup env \
  GPU=2 \
  PYTHON_BIN=/home/fb/anaconda3/envs/py310/bin/python \
  DATA_PATH=/newhome/fb/dataset/videoattentiontarget \
  OUTPUT_ROOT=/home/fb/src/paper/gazelleV1/AAAIResults/COTB/pilot \
  bash AAAICOTB/run_pilot.sh \
  > AAAIResults/COTB/pilot.log 2>&1 &

echo $!
```

若 checkpoint 不在默认目录：

```bash
setsid nohup env \
  GPU=0 \
  GF_INIT_CKPT=/absolute/path/to/gazefollow_sasa_ggsf_epoch14.pt \
  V0_VAT_CKPT=/absolute/path/to/v0_vat_epoch7.pt \
  SPOT_VAT_CKPT=/absolute/path/to/gazespot_vat_epoch7.pt \
  bash AAAICOTB/run_pilot.sh \
  > AAAIResults/COTB/pilot.log 2>&1 &
```

监控：

```bash
tail -f AAAIResults/COTB/pilot.log
nvidia-smi
```

完成后首先看：

```bash
/home/fb/anaconda3/envs/py310/bin/python -m json.tool AAAIResults/COTB/pilot/test_annotation_audit.json
/home/fb/anaconda3/envs/py310/bin/python -m json.tool AAAIResults/COTB/pilot/pilot_comparison.json
```

关键字段是：

```text
support_gate.pass
results.far.control_swap_error.value
results.far.delta_swap_error.value
results.far.delta_swap_error.ci95_high
l2_delta_candidate_minus_control
inout_ap_delta_candidate_minus_control
decision.verdict
```

## 2. 只做 4 帧 smoke test

如果想先验证 checkpoint/显存/数据路径，不跑完整 pilot：

```bash
cd /home/fb/src/paper/gazelleV1

CUDA_VISIBLE_DEVICES=0 /home/fb/anaconda3/envs/py310/bin/python -m AAAICOTB.evaluate \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --annotation /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --checkpoint experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --output-dir AAAIResults/COTB/smoke_eval \
  --model-label smoke_gazespot \
  --model-source current \
  --spatial-prior ggsf \
  --fusion sasa \
  --max-frames 4 \
  --batch-size-frames 1 \
  --bootstrap-iterations 0
```

1 个 batch 的训练 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 /home/fb/anaconda3/envs/py310/bin/python -m AAAICOTB.train \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --output-dir AAAIResults/COTB/smoke_train \
  --spatial-prior ggsf \
  --fusion sasa \
  --epochs 1 \
  --batch-size-frames 1 \
  --max-train-batches 1 \
  --max-eval-batches 1 \
  --bind-weight 0.10
```

## 3. 训练 final clean GazeFollow 初始化

pilot 为 GO 后，先得到不带 SASA/GGSF 的 `512 raw_concat` GF checkpoint：

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p AAAIResults/COTB

setsid nohup env CUDA_VISIBLE_DEVICES=0 \
  /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_gazelle_control.py \
  --dataset gazefollow \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --output-dir AAAIResults/COTB/gf_rawconcat \
  --model gazelle_dinov3_vitb16 \
  --spatial-prior none \
  --fusion raw_concat \
  --selected-layers all \
  --epochs 15 \
  --batch-size 60 \
  --workers 8 \
  --seed 3106 \
  > AAAIResults/COTB/gf_rawconcat.log 2>&1 &
```

输出应包含：

```text
AAAIResults/COTB/gf_rawconcat/best.pt
AAAIResults/COTB/gf_rawconcat/run_manifest.json
```

## 4. Weight screen（只用 train-validation）

先以 seed 3106 运行 `λ={0.05,0.10,0.20}`。下面以 `0.05` 为例，另外两个只替换 `--bind-weight` 和输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 /home/fb/anaconda3/envs/py310/bin/python -m AAAICOTB.train \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint AAAIResults/COTB/gf_rawconcat/best.pt \
  --output-dir AAAIResults/COTB/weight_screen/w005 \
  --model gazelle_dinov3_vitb16_inout \
  --spatial-prior none \
  --fusion raw_concat \
  --epochs 8 \
  --batch-size-frames 6 \
  --frame-sample-every 6 \
  --validation-frame-sample-every 6 \
  --validation-seed 9102 \
  --seed 3106 \
  --bind-weight 0.05
```

只比较各目录 `history.json` 中 validation joint score；禁止用 test 选择 `λ`。

## 5. 三随机种子 full confirmation

把验证集选出的权重传给 `BIND_WEIGHT`。例如 `0.10`：

```bash
cd /home/fb/src/paper/gazelleV1

setsid nohup env \
  GPU=0 \
  BIND_WEIGHT=0.10 \
  GF_RAW_CKPT=/home/fb/src/paper/gazelleV1/AAAIResults/COTB/gf_rawconcat/best.pt \
  OUTPUT_ROOT=/home/fb/src/paper/gazelleV1/AAAIResults/COTB/full \
  bash AAAICOTB/run_full.sh \
  > AAAIResults/COTB/full.log 2>&1 &
```

最终看：

```bash
tail -f AAAIResults/COTB/full.log
/home/fb/anaconda3/envs/py310/bin/python -m json.tool AAAIResults/COTB/full/three_seed_summary.json
```

## 6. 需要反馈回来的最小结果

请把以下文件同步回来即可继续分析，无需传 checkpoint：

```text
AAAIResults/COTB/pilot/train_annotation_audit.json
AAAIResults/COTB/pilot/test_annotation_audit.json
AAAIResults/COTB/pilot/pretrained_v0/summary.json
AAAIResults/COTB/pilot/pretrained_gazespot/summary.json
AAAIResults/COTB/pilot/grouped_control/history.json
AAAIResults/COTB/pilot/cotb_w010/history.json
AAAIResults/COTB/pilot/grouped_control_eval/summary.json
AAAIResults/COTB/pilot/cotb_w010_eval/summary.json
AAAIResults/COTB/pilot/pilot_comparison.json
AAAIResults/COTB/pilot.log
```
