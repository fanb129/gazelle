# R4 官方 train/test 训练与逐轮选模

版本：`r4_official_train_test_v1`。日期：2026-11-18。用户明确指定仅使用官方 train/test，并确认按 AUC 优先选择最佳 epoch。本计划是后续运行的当前协议；`r4_compact_v2` 及其 validation 结果保留为历史开发实验。

## 数据与模型

| 用途 | 官方分区 | tracks | 人物帧 | 唯一图像 |
|---|---|---:|---:|---:|
| 新增模块训练 | 完整 train | 1,033 | 132,563 | 58,507 |
| 每轮评测与 checkpoint 选择 | 完整 test | 298 | 31,978 | 13,127 |

计数来自已验收 T1/T2/R1 缓存报告；不再为新增模块保留 validation。旧 validation 的 34,201 人物帧重新纳入官方 train 训练。原始 CSV 轨迹、当前真实头框与因果状态规则不变。

复用原 GazeSpot ViT-B VAT `epoch_7.pt`，冻结完整骨干、GGSF/SASA 与原 decoder，仅训练新增模块。输入为当前 RGB 的冻结空间特征、当前真实头框、原始 heatmap/In/Out 以及自身过去预测；GT 只进入当前训练损失或离线评测。纯 RGB，不使用未来帧、深度或检测框。

完整 train 特征与原预测直接复用：`r2_feature_cache.json` 和 `r1_train_single_frame_cache.json`。test 原预测复用 `t2_single_frame_cache.json`；补充同一原生 `768×32×32` / FP16 test 特征缓存。模型、原预测、源 index 与特征 contract 用 SHA256/覆盖核对。test 特征提取不训练参数，也不以 test 标签构造模型输入。

## 训练与选模

首轮重训普通 GRU 和新机制，各 seed 0、五个 epoch，新增参数从头初始化，不续训旧 validation checkpoint。两模型共用：

| 参数 | 值 |
|---|---|
| epochs / seed | 5 / 0 |
| optimizer / learning rate / weight decay | AdamW / 0.0001 / 0.01 |
| lambda_state / gradient_clip_norm | 0.1 / 1.0 |
| hidden_dim / clip_frames / batch_size | 128 / 16 / 8 |
| correspondence temperature | 0.1 |
| 精度 | 新模块 FP32，关闭 AMP/TF32 |
| 状态 | 同人物连续段跨 chunk 保留，TBPTT 仅 detach；跨人/缺口重置 |

每完成一个 epoch，立即从空状态开始递推完整 test 轨迹，保存该轮全部标准/joint/state 指标、预测、历史控制诊断和 checkpoint；评测不反传。逐轮更新 `training_history.json`，保留所有 `epoch_XXX.pt`。

训练结束后按以下字典序，从实际完成的五轮中选一轮：

1. test AUC 更高。
2. AUC 完全相同，test L2 更低。
3. 前两项完全相同，test In/Out AP 更高。
4. 仍完全相同，较早 epoch；跨模型完全同分时按 `model_id` 字典序。

保留全精度数值比较，显示时舍入不制造并列。不使用旧 precision eligibility 门槛；准确稳定、状态错误与分型响应完整报告。最佳整轮复制为 `selected.pt`，最后一轮复制为 `last.pt`。每种方法独立按同一规则选 epoch，论文每行对应一个 checkpoint 的整套指标。

这是 **test-selected checkpoint reporting**：test 同时用于逐轮评测与 epoch 选择。报告和论文方法部分明确说明这一协议，不将其描述为未参与模型选择的独立测试估计。旧 validation 指标保留原标签，不改名为 test，不与新数据量的训练直接混作同次运行。

## 执行与交付

本地完成代码与合成检查，服务器由用户手动运行。唯一后台入口见 `implementation_plan.md` §11，物理 GPU 1。入口依次补 test 特征、完整 train 训练普通 GRU、完整 train 训练新机制、生成同源比较。预算在运行前固定，不改变骨干或现有损失。

- 小报告：`r4_official_test_features.json/.md`、`r4_official_ordinary_gru.json/.md`、`r4_official_history_validity.json/.md`、`r4_official_comparison.json/.md`、`r4_official_selection.json` 与 `RESULT_PROVENANCE.md`。
- 大文件：新的 test 特征缓存，以及 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/` 内全部 checkpoint、逐轮 test 预测、诊断、逐事件明细；不进 git，记录实际路径/SHA256。
- 首轮结果回传后，比较同一官方 test 上单帧/GRU/新机制，判断是否存在定位、状态或时序收益，再安排最少机制对照与论文。旧 R5 的单独“冻结后首次 test”流程并入本轮逐 epoch test 报告；后续只汇总已实际跑过的结果。
