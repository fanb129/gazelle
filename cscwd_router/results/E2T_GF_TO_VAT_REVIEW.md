# E2-T：完整 GF-B Router→VAT-B 结果审查

日期：2026-11-23。服务器原始结果提交 `170ace87b6e0f80062d85fddf80b55439085e688` 已 push 并 pull 本机。5 份原始 JSON 位于 [vat_from_gfb_router_k050_joint_seed0/](vat_from_gfb_router_k050_joint_seed0/)，CPU 工件与逐查询复核见 [e2t_gfb_to_vatb_server_checks.json](e2t_gfb_to_vatb_server_checks.json)。本次仅检查已有结果，没有新增训练、模型 forward 或 GPU 测速。

完整 GF Router/decoder 迁移明显恢复 VAT 定位；当前最佳是第 1 轮、Router 仍为 GF 权重的模型。后 7 轮联合适配没有改善预定选模指标。AP 低于旧 VAT-B，应优先保住定位，再适配分类头。

## 1. 最终逐帧指标：同一 checkpoint 的完整一行

均为 VAT rate-1 test、FP32、K50，31,978 人次、20,655 画内，按 rate-6 test L2 选定后仅复评。

| 实验 | AUC ↑ | L2 ↓ | In/out AP ↑ | 选择 |
|---|---:|---:|---:|---|
| 旧 VAT-B Router（含低 LR 续训后保留的旧 best） | 0.913548 | 0.110971 | 0.901738 | 原 decoder 第 4 轮 |
| 完整 GF-B Router/decoder→VAT-B | 0.940033 | 0.100131 | 0.883183 | 迁移第 1 轮 |
| 本次 − 旧模型 | +0.026485 | -0.010840 | -0.018555 | 单一模型，未拼指标 |

L2 相对降低 9.77%。这不是旧 VAT decoder 单纯续训的结果：初始化改为完整继承 GF 已训练 Router 和热图 decoder，仅新增 VAT in/out 分支。

[实验清单](../EXPERIMENT_PLAN.md#6-公开文献参照不是本轮结果) 已登记的 GazeLLE-B 文献参照为 `0.933/0.107/0.897`。当前 AUC/L2 高于/低于该参照，AP 仍低 `0.013817`，不能写成 VAT 三指标全面超过；同口径官方 VAT checkpoint 尚未重新推理。文献指标与本轮 `test_selected` 的来源分别标注，速度仍待同硬件实测。

## 2. 真实执行内容与选模

- 继承已验收 GF-B 第 5 轮 K50 checkpoint：`experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/sparse_decoder/best_test.pt`。52 个共享 tensors 与 GF 原文件逐值一致，仅 5 个 in/out tensors 用 seed 0 新建。
- DINOv3-B/16、512 输入、四层 raw-concat `[2,5,8,11]`、block 5 路由、K50；backbone 全程冻结，仅 RGB/真实头框/单帧。
- 实际完成 8 epochs。首轮 Router 固定，更新 decoder/inout；后 7 轮 coverage loss 更新 Router，同时 BCE 更新 decoder/inout。初始 LR `1e-4/1e-5/1e-3`，8 轮 cosine，AdamW weight decay 0，batch 16、累积 2，seed 0。
- 每轮 train 22,641 人次、708 次 optimizer steps；decoder/inout 累计 5,664 次，Router 实际更新 4,956 次。第 1 轮 Router 变化 tensor 数为 0，后续每轮 12 个 Router tensors 均有改变；decoder/inout 分别有 39/5 个 tensors 改变。
- 全量 train/test，无 val。逐轮 rate-6 test 共 5,461 人次、3,530 画内，按 L2 最小、AUC 最大、更早 epoch 选 **epoch_0（第 1 轮）**；它同时是 8 轮中 AUC 最高的一轮。best PT 与 epoch_0 state 逐值一致，last 与 epoch_7 一致。

| Epoch（1-based） | Router | AUC ↑ | L2 ↓ | AP ↑ | GT point coverage | Hard Gaussian coverage | Soft coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | fixed | 0.939900 | 0.100475 | 0.884416 | 0.962606 | 0.869189 | 0.359506 |
| 2 | joint | 0.938741 | 0.103659 | 0.885456 | 0.966856 | 0.847113 | 0.405644 |
| 3 | joint | 0.938988 | 0.104556 | 0.888502 | 0.966856 | 0.843054 | 0.402427 |
| 4 | joint | 0.937622 | 0.103947 | 0.890810 | 0.967705 | 0.842093 | 0.411786 |
| 5 | joint | 0.938278 | 0.104288 | 0.889190 | 0.964873 | 0.837521 | 0.406528 |
| 6 | joint | 0.938173 | 0.103844 | 0.890482 | 0.967139 | 0.839836 | 0.408873 |
| 7 | joint | 0.938132 | 0.103708 | 0.890648 | 0.966572 | 0.839597 | 0.408854 |
| 8 | joint | 0.938104 | 0.103765 | 0.890500 | 0.967139 | 0.837403 | 0.403559 |

上述各轮均为 rate-6，不能用它们替换最终 rate-1 行中的某个指标。最高 AP 的第 4 轮没有单独 rate-1 复评，不能将其 AP 拼给第 1 轮。

迁移初始化、尚未做任何 VAT 更新的诊断为 AUC/L2/AP `0.937055/0.100489/0.599645`（rate-6，in/out 随机，不参与选模）。第 1 轮定位为 `0.939900/0.100475`，表明大部分定位恢复在完整 GF 初始化时已经存在；VAT 第一轮主要让新 in/out 分支适应任务，但不能据此拆分 Router 与 decoder 的独立因果贡献。

旧 VAT dense 初始化已实际测过的 rate-6 AUC/L2/AP 为 `0.939097/0.103012/0.893611`；本次选中模型同 rate-6 为 `0.939900/0.100475/0.884416`。完整迁移已恢复该参考的定位水平，AP 仍较低。旧 dense 与本次迁移不构成同起点、同训练预算的 Router 消融。

## 3. 为什么后续联合适配没有选中

从第 1 轮到第 8 轮，soft coverage `0.359506→0.403559`、GT point coverage `0.962606→0.967139`，但 hard Gaussian mass coverage `0.869189→0.837403`、L2 `0.100475→0.103765`。连续 support 的 coverage 改善并未对应到 hard Top-K 后完整 gaze 热图的定位改善；只看 GT 点是否进入选中 patch 也不足以解释结果。

这支持下一轮保护当前定位分支，不能证明退化仅由 Router 更新引起，因为 decoder 与 in/out token 也同时更新。联合阶段额外加入 Router loss，不能直接比较首轮与后续的 total loss 数值来判断整体收敛。

AP 从首轮 `0.884416` 到第 4 轮 `0.890810` 有提升，但仍低于旧 VAT 模型。新 in/out 仅在第一轮学到当前选中状态，而旧 VAT 初始化已经有分类训练；适配不足是一个可检验解释，不认定为唯一原因。

## 4. 结果核对

服务器只读 CPU 检查通过：20 个已登记工件路径/SHA（14 个 PT/PTH/gzip）；源码、标注、manifest fingerprint 与报告一致。8 轮冻结状态、实际参数变化和 optimizer 内部 step 与 history 一致。逐帧导出 31,978 条记录逐条匹配原 test 的路径、人物索引、画内标记和 gaze 标注；AUC/L2/AP 重聚合与 summary 的差值全部为 0。

本地 pull 后 5 份原始报告 SHA、20 个相关源码文件 SHA 再次匹配服务器检查记录。PT/gzip 未下载或入 Git，大工件路径/SHA 保存在 [RESULT_PROVENANCE.md](RESULT_PROVENANCE.md)。服务器只提交当前 5 份 JSON 与对应 provenance block，未提交其他未完成/无关结果；本地 rebase 保留 L 来源文档与用户原有目录迁移。

## 5. 下一步（E2-H，尚未实现或运行）

保留当前 `best_test.pt`，下一项优先从它适配 **仅 `inout_head` MLP**，冻结 backbone、Router、完整 heatmap decoder 与 `inout_token`，先改善 AP。`inout_head` 作用于共享 transformer 的输出，不回写热图；冻结共享路径能把分类适配与定位更新分开，实施后仍应实际核对逐查询热图一致性。

不能直接把现有入口的 `decoder-lr=0`、`router-lr=0` 当作上述实验：当前 `inout` 参数组还包含会参与共享 self-attention 的 `inout_token`，它改变时可能改变热图。下一任务需独立控制参数范围、保存初始化/冻结证据、沿用 train/test 与单 checkpoint 选模，再由用户手动跑服务器。此处没有发布尚不存在的运行命令。

当前 B 迁移已经恢复定位，不以这组结果为由切换 GazeSpot 或新训两个 L dense；L 来源问题继续独立处理。正式效率在空闲 GPU 上测最终选中的同一 checkpoint，不用 K50 patch 比例代替加速比。
