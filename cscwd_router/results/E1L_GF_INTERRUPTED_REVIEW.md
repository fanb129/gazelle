# E1-L：GF-L 中断后收口与结果审查

日期：2026-11-23。服务器结果提交 `459651a8a6485a2a2ab2b5ccb418ae742817e7a8` 已 push 并 pull 本地。用户决定不补最后一轮；本次只读核验、导出已有最佳权重并归档，没有新增训练、GPU forward 或测速。

GF-L 已保存的结果可接续效率测试和完整 Router→VAT-L 迁移，当前不必先新增 GF-L dense 训练。按实际 **support 3 + decoder 4 epochs** 收口；原计划 3+5 保留在 manifest，decoder 第 5 轮途中关机，未写成完整 epoch。

## 1. 单一选中模型的 GF official test

- AUC：**0.960007099**。
- Avg L2：**0.094061417**。
- Min L2：**0.037721492**。
- FP32、K50、4,782 个查询，完整官方 train/test，无 val，base seed 0。
- 按既定 test Min L2 最小、AUC 最大、更早 epoch，选 **decoder 第 2 轮（epoch_1）**，三项数值来自同一轮。

| Decoder epoch（1-based） | Train keep ratio | Test AUC ↑ | Test Avg L2 ↓ | Test Min L2 ↓ | 选择 |
|---|---:|---:|---:|---:|---|
| 1 | 1.00 | 0.948091 | 0.109770 | 0.053190 |  |
| 2 | 0.75 | 0.960007 | 0.094061 | 0.037721 | selected |
| 3 | 0.50 | 0.960557 | 0.093636 | 0.038111 |  |
| 4 | 0.50 | 0.960678 | 0.093370 | 0.037724 |  |

第 4 轮 AUC/Avg L2 最好，但 Min L2 `0.037723783` 略高于第 2 轮 `0.037721492`，差 `0.000002291`，因此不改选择规则、不拼接不同轮指标。当前正式候选、后续效率和迁移来源统一为第 2 轮。

[实验清单](../EXPERIMENT_PLAN.md#6-公开文献参照不是本轮结果) 已登记 GazeLLE-L 文献指标 `0.958/0.099/0.041`；本次相对差 `+0.002007/-0.004939/-0.003279`，Avg/Min L2 相对降低 `4.99%/8.00%`。该行是 reported reference，同口径官方 L checkpoint 尚未重新推理；不将它写成本轮实测基线或显著性结论。当前仍没有正式速度结果。

## 2. 本次如何收口

原 trainer 在每轮 train/test 后原子保存 `last.pt`，其中同时保存当前模型、最佳模型、history、optimizer 和 scheduler。decoder 已提交 epoch 0–3，日志显示进入 epoch 4 后中断；最近持久化的恢复点是第 4 轮末。

从该 `last.pt` 的 `best_model_state` 导出标准 compact `best_test.pt`，没有再执行训练或 test。导出后 52 个 tensors 与内嵌 best 逐值一致，原 `last.pt` 的 SHA 保持不变。

- best checkpoint：`/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitl/sparse_decoder/best_test.pt`。
- best SHA256：`0e3c848ce62731c53762696488e4911e86272dd4824a2db6b69a2f0e9ee5bebf`。
- resume checkpoint：`/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitl/sparse_decoder/last.pt`。
- resume SHA256：`589417842fffe946a17eb99dc7ad8b8b5da8141a471b2c6496fda99548c916ca`。

独立收口报告为 [interrupted_summary.json](testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json)，明确 `completed=false`、`closed_by_user=true`、`checkpoint_export_complete=true`、actual epochs 3+4 / planned 3+5。没有生成冒充完整训练的 `summary.json`，也没有改原 manifest 的 5 轮计划。

指标直接来自已完整跑过的第 2 轮 test 记录；没有新增独立复评，也没有 `predictions.jsonl.gz`，本轮不声称做过逐查询指标重聚合。后续评测入口可补同一模型的逐查询导出，不以此改变所选 epoch。

## 3. 工件与训练核验

CPU 检查见 [e1l_interrupted_server_checks.json](e1l_interrupted_server_checks.json)：

- manifest fingerprint 与两个 checkpoint context 一致；18 个源码文件、DINOv3 原生源码树和 train/test 标注 SHA 均匹配。
- support 3 轮，decoder 4 轮；每个完整 epoch 113,458 个 train 人次、4,782 个 test 查询。已提交 optimizer steps 为 support **85,095**、decoder **14,184**，scheduler 为 3/4，与 history 一致。第 5 轮未保存的中间状态不计入这些持久化更新数。
- support 结束时 40 个初始 decoder tensors 与原 initializer 逐值相同；decoder 阶段 12 个 Router tensors 始终与 support 相同。所选模型中 39 个 decoder tensors 相对 support 有改变。
- backbone 可训练参数数为 0、checkpoint 不含 backbone tensors；保存的 task tensors 均有限，support final 与 support last 一致。
- 6 个 PT/PTH 的路径/SHA、2 份标注 SHA 均核对；导出的 52 个 best tensors 与选择记录一致，原恢复 checkpoint 未改。
- Git 回传后 4 份原始 JSON 与 18 个本地相关源码 SHA 再次匹配；本地 3 条未推送提交和用户目录迁移保留。大工件仍在服务器，未入 Git。

## 4. 本轮结构与初始化标签

当前运行已由 manifest 固定为冻结 DINOv3-L/16、512 输入、raw-concat `[5,11,17,23]`、block 11 后路由、K50、无 SASA/GGSF，仅 RGB/真实头框/单帧。

历史初始化为 `experiments/train_gazefollow_vitl/2026-01-30_13-14-15/epoch_14.pt`，四路投影兼容，但精确历史层号未闭合，继续按 [L 初始化取舍](L_INITIALIZER_DECISION.md) 标为 warm-start。本次新训练与权重有效，不把它改写成来源已完全认证的同配置 dense 基线；当前结果支持先使用该 Router 模型接续推理与迁移。

## 5. 下一步

先适配 DINOv3 正式推理/效率入口，使用上方同一 `best_test.pt`，与官方同规模 GazeLLE 在空闲 RTX 3090、B1/B8 正反序、统一精度下比较完整 forward；不由 K50 或训练参数量推断实际加速。

VAT-L 采用同一 GF-L Router + heatmap decoder 完整迁移，backbone 冻结，仅新建 in/out。现有 transfer 入口要求 `completed=true` 的 GF summary，不能直接把本收口 JSON 传给原命令；下一实现须显式接入“用户收口、实际 history 完整、best 已核验”的来源，保留中断标记和 SHA 校验，再提供完整命令由用户手动跑，不靠修改布尔值绕过检查。当前没有启动 VAT-L 或测速。
