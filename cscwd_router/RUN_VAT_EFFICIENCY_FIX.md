# VAT-B 效率采样修复与 GPU 1 重启

2026-11-24。修复提交 `29c6607` 已 push，并经 Git 在服务器部署到独立 worktree：

`/home/fb/src/paper/gazelleV1_efficiency_vat_fix`

原目录保持 `4162da7`，GPU0 的 GF 效率与 GPU2 的 VAT-L 训练继续使用原源码。原目录的更新安排在 GF 效率全部结束后，避免同一矩阵混入两版源码 SHA。独立目录共享原 `checkpoints`、`dinov2`、`dinov3`，没有复制大权重。

## 原因与验证

VAT `train_preprocessed.json` 顶层是序列，帧数据在 `frames`，人物数据在每帧 `heads`；GF 顶层直接是图像记录。原效率入口将两者都按 GF 解析，因而在进入计时前触发 `KeyError: 'heads'`。修复改为按 source summary 的 dataset 显式选择官方 `load_data_gazefollow` / `load_data_vat(sample_rate=1)`，其余模型和计时逻辑不变。

先前合成效率测试没有覆盖 VAT 的序列结构；现在补齐嵌套标注的采样与队列/恢复回归，70/70 CPU 通过。服务器只读预检成功展开 475 序列、58,507 帧，8 个真实 RGB/GT 框匹配官方 loader，B1 是 B8 的样本前缀，四个正式参数请求均通过。GF B8 的采样 metadata 与正在运行的旧版本逐值一致，原目录 HEAD 与三个核心源码 SHA 未变。没有在此检查中执行 GPU forward、训练或计时。

## 直接启动 GPU 1

目录与依赖链接已经创建，直接执行：

```bash
cd /home/fb/src/paper/gazelleV1_efficiency_vat_fix
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_efficiency_vat_fix/cscwd_router/code/scripts/run_router_efficiency.py \
  --summaries /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --official-checkpoint-dir /home/fb/src/paper/gazelleV1/checkpoints \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124_vatfix \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --batches 1 8 --precision fp16 --warmup 50 \
  --latency-iters 200 --throughput-iters 500 --repeats 5 --seed 0 \
  --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/efficiency_vat_b_gpu1_vatfix.log 2>&1 < /dev/null &
```

仍测当前已接受的 GF→VAT-B checkpoint，official/K100/K50、FP16、B1/B8 正反序，所有计时参数与原计划相同。

```bash
tail -f /home/fb/src/paper/gazelleV1/logs/cscwd_router/efficiency_vat_b_gpu1_vatfix.log
```

## 输出与审查

小报告写回原仓库 `cscwd_router/results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/`，含四个 cell JSON 与 summary JSON/MD；实际完成后从原仓库选择性提交。旧失败日志与 GF 已生成单元保留。真实预检与本机测试证据见 `results/e3_vat_schema_fix_checks.json`。
