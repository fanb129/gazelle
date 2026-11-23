# GF-B 与 VAT 低 LR 续训审查

日期：2026-11-23。服务器结果提交 `8f2ae9cddd27799a3b2923754e47d21d5d07e962` 已 push，并已在本机 fast-forward pull。GF-B、VAT 续训 B/L 均完成；检查时 GF-L 仍处于 K75 decoder 训练段，继续等待，不修改运行任务。

## 1. GF-B：K50 定位精度改善，效率尚未测量

| GF official test，FP32 | AUC ↑ | Avg L2 ↓ | Min L2 ↓ |
|---|---:|---:|---:|
| 官方 GazeLLE-B/14，已有服务器报告 | 0.956141 | 0.104136 | 0.045332 |
| DINOv3-B/16 raw-concat + Router K50，seed 0 | 0.956236 | 0.101550 | 0.043420 |
| Router − GazeLLE | +0.000095 | −0.002586 | −0.001912 |

原始文件：[GF-B summary](testselected_k050_seed0/gazefollow/vitb/summary.json)、[history](testselected_k050_seed0/gazefollow/vitb/sparse_decoder_history.json)、[官方 GazeLLE 报告](t4_gf_fulltrain_seed3106/accuracy/dense_official.json)。

GF-B 实际完成 support 3 + decoder 5 epochs，base seed 0，完整官方 train，无 val。每轮 test 按 Min L2 最小选定 **第 5 轮**；三项指标都来自该模型。第 4 轮的 AUC 更高，但不将它与第 5 轮的 L2 拼接。

两行均为 4,782 个查询、相同 test 标注 SHA、FP32。官方旧报告使用 image-with-all-heads 入口，当前 Router 使用 per-person 入口；该测试集 4,782 张图均为单头查询，因此查询集合相同。官方 dense 工件的 decoder tensors 已与原官方 `gazelle_dinov2_vitb14.pt` 逐值核对一致；这次回传已有报告，没有重新跑官方模型。

相对该官方报告，Avg L2 降低 **2.4830%**，Min L2 降低 **4.2178%**。AUC 差仅 `0.000095`，写为基本持平；未做显著性判断。新模型与基线在 backbone、输入 512/448、多层/单层和选模来源上有差异，这一行是完整系统比较，不能将定位改善全部归因于 Router。没有当前 checkpoint 的延迟/吞吐结果，尚不能写精度与效率双优。

## 2. VAT：额外 8 轮没有改善 L2 选模结果

以下为最终 rate-1 test，各 31,978 个人物查询，20,655 个画内：

| K50 模型 | AUC ↑ | L2 ↓ | In-out AP ↑ | 最终选中模型 |
|---|---:|---:|---:|---|
| VAT-B 续训前 | 0.913548 | 0.110971 | 0.901738 | 原第 4 轮 |
| VAT-B 续训后 | 0.913548 | 0.110971 | 0.901738 | 仍为原第 4 轮 |
| VAT-L 续训前 | 0.923034 | 0.097970 | 0.916599 | 原第 2 轮 |
| VAT-L 续训后 | 0.923034 | 0.097970 | 0.916599 | 仍为原第 2 轮 |

原始文件：[B summary](vat_continue_k050_lr1e5_seed0/vitb/summary.json)、[L summary](vat_continue_k050_lr1e5_seed0/vitl/summary.json)。

两组均从原第 5 轮末继续，固定 K50，decoder/in-out LR 为 `1e-5/1e-4`，新 cosine 8 epochs。加上原始 5 轮，共做了 13 个 Router decoder 适配 epoch，其中 11 轮训练预算为 K50。Router/backbone 冻结。

最终数字相同是预定的旧 best 保留规则生效，续训实际执行：两组各更新 **5,664 次**，各有 **44 个 decoder tensors** 改变，最大绝对变化 B `0.0272925571`、L `0.0269123502`。新导出的 best tensors 与旧 best 逐值相同，新 last 则有实际更新。

| rate-6 选模指标 | VAT-B | VAT-L |
|---|---:|---:|
| 旧 best L2 | 0.110842 | 0.099612 |
| 新 8 轮最低 L2 | 0.111874 | 0.102485 |
| 新 8 轮最高 AUC | 0.919339 | 0.932664 |

新 AUC 最高轮的完整 rate-6 指标分别为：B 第 10 轮 `0.919339/0.113698/0.900398`；L 第 11 轮 `0.932664/0.103061/0.915474`。这些是稀疏采样 test 的候选模型指标，不是最终逐帧成绩；不能将其较高 AUC 与旧 best 的 L2/AP 混拼。按既定 L2 主选择指标，本次续训没有产生应替换旧模型的新 checkpoint。

## 3. 原始初始化的 dense/K100 诊断

两组 probe 都使用原始 VAT dense decoder，在相同 rate-6 test（5,461 个查询、3,530 画内）上运行 native dense 和 routed K100。两条路径的 AUC/L2/AP 差值均为 **0.0**，精度路径检查通过；这里没有宣称 backbone 激活逐值等价。

| 同为 rate-6 test | AUC ↑ | L2 ↓ | AP ↑ |
|---|---:|---:|---:|
| B 原始 dense 初始化 / routed K100 | 0.939097 | 0.103012 | 0.893611 |
| B 当前选中 K50 | 0.913401 | 0.110842 | 0.900630 |
| L 原始 dense 初始化 / routed K100 | 0.944592 | 0.094232 | 0.916532 |
| L 当前选中 K50 | 0.922349 | 0.099612 | 0.916863 |

来源：[B probes](vat_continue_k050_lr1e5_seed0/vitb/probes.json)、[L probes](vat_continue_k050_lr1e5_seed0/vitl/probes.json)。本次回传的 probe 由用户先前运行的续训脚本生成，采用当前精度路径，优先于此前旧训练日志的近似参照；其中 L probe 与旧日志略有差异，不混用。

相比各自旧 dense 初始化，选中 K50 的 rate-6 AUC 分别降低 `0.025697/0.022243`，L2 增加 `0.007830/0.005380`。差距在相同采样口径下仍然存在。该比较包含稀疏化和 decoder 适配变化，不是同预算 matched dense 因果消融；当前证据不支持把它归为遗漏 GF decoder 迁移或 K100 路径失效。

## 4. 工件核验

详细记录：[e1_gfb_e2c_server_checks.json](e1_gfb_e2c_server_checks.json)。只读 CPU 检查通过：

- 29 个工件/标注路径 SHA 匹配，其中 21 个为 PT/PTH/gzip；官方 decoder 的独立 SHA 与 T1 登记一致。
- GF-B 与两组 VAT 的源码/原生源码树、context fingerprint、标注 SHA、checkpoint metadata、选模与导出 best 一致。
- GF-B 的 3+5 epoch、逐轮样本数和更新数正确；support 阶段 decoder 不变，decoder 阶段 Router 不变。
- VAT 的新增 epoch 6–13、学习率、每轮 708 次更新和 K50 正确；新 last 确有参数变化，Router 固定，旧 best 保留符合 L2/AUC/较早 epoch 排序。
- GF 的 4,782 条记录与两组 VAT 各 31,978 条记录，身份/坐标/in-out 均与真实 test 标注逐条匹配，指标重聚合与 summary 差值均为 0。
- 本机对 13 份回传 JSON 再次核对 SHA。大文件留服务器，仅路径/SHA 经 Git 回传。

本轮没有新增训练或 GPU model forward/timing；只核验用户已运行的产物。只回传已完成的 GF-B、VAT B/L 续训和官方 GF 精度报告；GF-L 与旧 T4 其他产物未混入本次提交，不把这次基线回传视为 T4 总验收。

## 5. 分析与下一步

GF-B 说明当前 K50 完整方法已有可用的定位精度；VAT 的当前训练路线仍不满足目标。本次低 LR 续训的结果支持停止原样延长固定 Router 的 decoder 训练，不能再用“只差几轮”作为提升依据。这不等价于排除了从一开始低 LR 训练、GF Router 迁移或其他路由学习方式。

当前 Router 在 VAT 续训中固定，因此额外 8 轮不会改变选区。最终 GT point coverage 为 B `93.76%`、L `94.92%`，Gaussian target mass coverage 为 `75.62%/80.21%`；GF-B 对应约 `94.58%/87.19%`。这些观察提示还应检查选区与回填表示的适配，目标点保留率本身不足以保证定位质量；它们没有独立分离各因素的因果作用。

下一步优先做三件事，不再新增 dense 训练：

1. 对现有 VAT 选中 checkpoint 测 K50/K75/K100 预算响应，完整报告每个预算的同模型指标；这是推理分析，不充当各预算重训的最优 Pareto。
2. GF-B 已完成，可实现并运行 **GF Router + heatmap decoder → VAT**，新增 in/out head 在 VAT 学习。此前只有 dense decoder 的 GF→VAT 迁移，Router 是在较小 VAT 数据上从头学；GF support 每轮 113,458 查询，VAT 为 22,641，迁移整个已适配稀疏表示的模型有明确动机，但效果仍待真实实验。
3. GF-L 继续等待；GF-B 可先做 E3 官方 GazeLLE 同硬件、同 checkpoint 的 B1/B8 正反序测速，尽早验证论文效率主张。
