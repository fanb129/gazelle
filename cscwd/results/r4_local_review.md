# R4 第一轮真实结果本地验收

日期：2026-11-18。结果提交：`f1d41e4`；运行源码提交：`7459abc`；计划：`r4_compact_v2`。

首轮训练/验证/选模链路验收通过。回传小报告、源码/计划 SHA256、固定预算、覆盖、loss/指标分母、全部候选精度条件、排序/Pareto 与 manifest 共 235 项本地一致性核对通过。两模型均新初始化、seed 0、五个 epoch，共享初始化一致。完整 NPY/PT/逐事件数据留服务器；本地核对小报告及其算式，没有重新计算全量预测或逐事件延迟。机器报告保留原样。

以下均为从 VAT **official train 划出的 validation**，共 206 tracks / 34,201 人物帧，定位计分 21,758 帧；新增模块的开发 train 为 827 tracks / 98,362 人物帧。冻结的原 GazeSpot checkpoint 曾在 official train 上训练，当前划分是新增模块的开发留出。

| Validation 指标 | 原始 ViT-B 单帧 | 普通 GRU epoch 1 | 新机制 epoch 1 |
|---|---:|---:|---:|
| AUC ↑ | 0.954205563 | 0.954325432 | 0.954289944 |
| L2 ↓ | 0.079604569 | 0.079601696 | 0.079562421 |
| In/Out AP ↑ | 0.960922682 | 0.960799192 | 0.960808102 |
| 准确稳定 ↑ | 60.1874% | 59.4586% | 59.4964% |
| 状态错误 ↓ | 11.1254% | 10.5289% | 10.5114% |
| Jitter ↓ | 0.014554665 | 0.014595220 | 0.014696182 |
| joint gaze_shift 响应 ↑ | 61.0860% | 61.0860% | 61.0860% |
| joint enter 响应 ↑ | 77.6471% | 74.7059% | 74.7059% |
| leave 响应 ↑ | 55.0296% | 57.3964% | 57.3964% |

新机制相对同分区单帧：状态错误 3,805→3,595（少 210 帧，下降 0.6140 个百分点），准确稳定 12,717→12,571（少 146/21,129 个窗口，下降 0.6910 个百分点）。joint gaze_shift 仍为 135/221；joint enter 132/170→127/170，state enter 179/189→174/189；leave 93/169→97/169。状态翻转 594→621，Jitter 略升。

新机制与同预算 GRU 的差距很小：多 8 个准确稳定窗口，少 6 个状态错误帧；joint gaze_shift、joint enter、leave 的响应数相同。当前结果显示状态纠偏的折中，尚未支持历史有效性机制已带来明显额外收益。包含单帧的总体 validation 首选仍为 `single_frame:epoch_000`。

全部五轮的准确稳定占比持续下降，训练 total loss 继续下降。普通 GRU epoch 5、新机制 epoch 4/5 的 AP 低于单帧 AP−0.002，不满足预定精度条件。此轮不支持通过简单延长训练取得目标收益；泛化或目标失配是后续诊断方向，不据这些汇总数字断言具体成因。

按训练前固定规则，两种模型均选 **epoch 1**，`selected.pt` 的 SHA256 与各自 `epoch_001.pt` 一致：

- 新机制：`/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/selected.pt`；SHA256 `6b478e92607efc0215956ec49464cde3b771c7200a4346f48b203da2d1c58ea0`。
- 普通 GRU：`/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/selected.pt`；SHA256 `a76fcbb48c9880bb887c84f9fcd61da8f71388d8ca0ce9b8e02f86bbeeb0c717`。

论文的 validation 分析可报告上述 checkpoint 的全部指标，并注明分区；正式 VAT test 主表须使用 R5 统一 test 评测结果。固定一个 checkpoint 后报告其所有指标，不按列混用 epoch。当前开发快照保持 `test_configuration_frozen=false`，本次不直接进入 R5。

ViT-L 比较：本地 `icassp2027/RESULT_PROVENANCE.md` 与主表记载的 GazeSpot ViT-L 为 VAT test **0.944 / 0.092 / 0.912**。R4 是另一数据分区；原始 ViT-B validation 已为 **0.954206 / 0.079605 / 0.960923**，因此不能把数值高于 ViT-L test 当作本轮改进或超过 ViT-L 的证据。真正的本轮增益以同一 validation 上单帧/GRU/新机制的差值判断。

下一步先用现有 epoch-1 缓存分解 heatmap 与 In/Out 对准确稳定的贡献，组合原始/修正坐标与原始/修正状态，明确标为缓存诊断而非重训消融；同时汇总已有 validity/refresh 诊断。当前预测稳定窗口仅 16,603→16,598，而画内状态错帧 603→807、画外错帧 3,202→2,788，优先查清状态阈值影响。若状态收益主要来自分数偏置，按既定精简计划补开发 train 拟合的严格单调状态校准，validation 只评估；AP 排序保持不变。根据诊断再决定一次有针对性的训练调整，后续再补关键机制消融和 R5。
