# E2 · VAT seed0 B/L 结果审查

日期：2026-11-23。结论：GPU 1 的 ViT-B、ViT-L 队列均正常完成，结果与预定训练/选模协议一致；两组 AUC 尚未达到同规模 GazeLLE 的文献参照值，效率尚未测量。

## 1. 结果回传

- 训练代码提交：`5006bc43e3c507206fa53557c8171dafb9a83dac`，RTX 3090，`CUDA_VISIBLE_DEVICES=1`，PyTorch `2.6.0+cu124`。
- 日志存在 `QUEUE_COMPLETE dataset=vat sizes=b,l`；检查时没有对应 Router 队列/训练进程。这里只确认 VAT 任务结束，不以显卡当前利用率判定显卡空闲。
- 已从服务器提交并 push `0fc284430297d38244f4583c4adae17a486a0bf7`，本机已 fast-forward pull。该提交仅包含 8 份 VAT JSON 与两组 VAT provenance；旧 T4/ConvLSTM/GF 文件留在服务器原位置，没有混入提交。
- 原始结果位于 [vitb/summary.json](testselected_k050_seed0/vat/vitb/summary.json)、[vitl/summary.json](testselected_k050_seed0/vat/vitl/summary.json)；各自同目录含 manifest 与两个阶段 history。PT 与压缩逐查询明细留服务器，路径/SHA 见 [RESULT_PROVENANCE.md](RESULT_PROVENANCE.md)。

## 2. 最终逐帧 test

| 模型 | AUC ↑ | L2 ↓ | In-out AP ↑ | 所选 decoder epoch（从 1 开始） | GT point coverage |
|---|---:|---:|---:|---:|---:|
| DINOv3 ViT-B + raw-concat + Router K50 | 0.913548 | 0.110971 | 0.901738 | 4 | 93.7594% |
| DINOv3 ViT-L + raw-concat + Router K50 | 0.923034 | 0.097970 | 0.916599 | 2 | 94.9213% |

均为 base seed 0、FP32、同一个选定 checkpoint 的完整指标。最终 rate-1 test 包含 31,978 个人物查询，其中 20,655 个画内；AUC/L2 在画内查询求均值，AP 使用全部查询。官方 train/test 没有切 val，训练 rate 6，共 22,641 个人物查询/epoch。

## 3. 选模与训练轨迹

两组实际完成 support 3 epochs 与 sparse decoder/inout 5 epochs。support 只更新 Router，decoder 阶段只更新 decoder/inout，backbone 始终冻结。decoder 训练预算为 K100→K75→K50→K50→K50；所有逐轮 test 都固定 K50、rate 6，共 5,461 个查询，其中 3,530 个画内。

| decoder epoch | B test AUC | B test L2 | L test AUC | L test L2 |
|---|---:|---:|---:|---:|
| 1 | 0.829191 | 0.246212 | 0.891156 | 0.102757 |
| 2 | 0.901370 | 0.119520 | 0.922349 | **0.099612** |
| 3 | 0.911748 | 0.113655 | 0.928828 | 0.102202 |
| 4 | 0.913401 | **0.110842** | 0.930844 | 0.101970 |
| 5 | 0.916997 | 0.111974 | 0.931478 | 0.102541 |

按预定 test L2 最小规则，B 选择第 4 轮，L 选择第 2 轮；再用这两个模型做 rate-1 test。L 的选中模型出自 K75 训练轮，但评测和最终报告都是 K50。后续轮 AUC 更高而 L2 更差，不能将其 AUC 与较早轮的 L2/AP 拼成一行，也没有根据 rate-1 结果重新选择 epoch。

## 4. 工件和指标核验

服务器只读 CPU 检查记录在 [e2_vat_seed0_server_checks.json](e2_vat_seed0_server_checks.json)：

- 14 个大工件的实际 SHA 全部匹配：两份初始化、两份 backbone、八份新 PT、两份压缩逐查询明细。
- 两组各 19 项源码/源码树摘要、标注 SHA、context fingerprint 均匹配；回传的 8 份原始 JSON 在本机再次核对 SHA。
- 3+5 epoch 顺序、逐轮样本数、更新次数和参数冻结记录匹配。support 阶段 decoder tensors 与初始化逐值相同；decoder 阶段 Router tensors 与 support 产物逐值相同。compact checkpoint 不含 backbone tensors。
- `best_test.pt` 的 epoch、selection 与 history 一致，其 state 与 `last.pt` 内嵌 best state 逐值相同，模型 tensors 全部有限。
- 两组各 31,978 条导出记录的索引、路径、head index、in/out 和 gaze 坐标与完整 test 标注逐条匹配。重聚合 AUC/L2/AP 与 summary 差值均为 **0.0**。

这是工件与导出数值核验，没有重新执行模型 forward；逐查询 AUC/L2 使用保存的指标重聚合，没有从未保存的热图重新计算。未运行 GPU 评测或测速。

## 5. 对论文目标的含义

以下 GazeLLE 值沿用 [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md) 已登记的文献参照，不是本轮在服务器重新评测的官方 checkpoint：

| 规模 | GazeLLE reported AUC/L2/AP | 本轮 K50 AUC/L2/AP | K50 − reported AUC/L2/AP |
|---|---|---|---|
| B | 0.933 / 0.107 / 0.897 | 0.913548 / 0.110971 / 0.901738 | −0.019452 / +0.003971 / +0.004738 |
| L | 0.937 / 0.103 / 0.903 | 0.923034 / 0.097970 / 0.916599 | −0.013966 / −0.005030 / +0.013599 |

B 的 AP 较高，AUC 和 L2 较差；L 的 L2/AP 较好，AUC 较差。当前不能写“VAT 全面优于 GazeLLE”，也没有相对 GazeLLE 更快的实测证据。L 的三项指标也尚未达到计划中 Mi et al. 的文献参照 `0.948 / 0.095 / 0.923`。

本轮 full-train/test-selected 结果与旧 source-video holdout 分开保留。没有新同表示 dense 对照，尚不能把差距全部归因于 Router，也不能用 GT point coverage 推断定位质量。

下一步先收齐并审查 GF，再对实际选中 checkpoint 完成 E3：官方 GazeLLE B/L 同口径推理与同硬件测速。可对现有 checkpoint 补 K75/K100 推理，观察预算响应；不新增 dense 训练，不将单 checkpoint 改 K 的结果冒充各预算重训的最优值。
