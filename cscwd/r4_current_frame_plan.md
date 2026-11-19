# R4-D4：当前帧无历史联合修正对照

版本：`r4_current_frame_v1`。2026-11-19，在真实运行前固定。数据/测试与选轮沿用 `r4_official_train_test_v1`。

## 目的与模型

训练一个只利用当前帧的联合修正头，判定原首轮GRU/HV的AP等收益是否需要历史。使用原首轮逐帧目标和预算，与原首轮比较；D3及D2分别保留，不混入本轮。

`CurrentFrameRefiner`继承原 `CausalJointRefiner`，保留相同当前视觉特征、raw heatmap/InOut、真实头框、feature projection、GRUCell和双残差头。每次step无条件以空状态调用原单步计算，忽略任何传入state/reset，并返回 `state=None`：hidden、previous heatmap和previous InOut始终为零。它没有跨帧信息，也没有history validity或目标模板。

整个GazeSpot ViT-B（backbone及原decoder）冻结。新模块从头初始化，seed 0；共享初始化SHA必须与原GRU一致。只使用当前RGB的冻结特征与真实框。GT只用于训练loss与离线评测，不进入模型输入。

## 参数公平性

保留原层/参数名称及shape，记录总参数量相同；**不声称有效参数作用完全相同**。

- `gru.weight_hh`乘以全零hidden，不提供历史信息。
- `gru.weight_ih`中对应过去热图与过去状态分数的列始终接收零输入。
- GRU bias仍可能影响当前输出，不能把全部recurrent bias算作无效。
- 结构上零输入权重即使没有任务梯度，也可能因AdamW weight decay变化。参数使用报告不是FLOPs或实际梯度非零元素的测量。

训练报告保存 `parameter_usage`，明确这些结构差异。D4用于隔离是否使用历史，不把它等同于所有有效参数严格匹配的任意MLP。

## 固定预算与损失

`L = L_heatmap_MSE + 0.1 * L_state_BCE`，复用原 `joint_loss`：定位只含有效GT画内帧，状态含全部有效人物帧；不加D3稳定/转换loss，不应用D2输出校准。

固定 `epochs=5, learning_rate=0.0001, lambda_state=0.1, hidden_dim=128, clip_frames=16, batch_size=8, seed=0, weight_decay=0.01, gradient_clip_norm=1.0, temperature=0.1`，AdamW、FP32、TF32关闭。temperature仅保留原预算字段，D4没有correspondence温度运算。

复用原iterator和相同的每轮连续段顺序，但每帧模型状态均为空，chunk只作为批处理单位。训练一次只跑这一个配置，不续训旧checkpoint。

## 数据、评测与选轮

- 完整官方train：1,033 tracks / 132,563人物帧 / 58,507图像。
- 每epoch完整官方test：298 tracks / 31,978人物帧 / 13,127图像。
- 复用既有train/test原预测、原生FP16特征，不重新提取图像特征；不使用R1 validation分区。
- 保存全部五轮checkpoint、预测与完整标准/joint/state指标；按test AUC更高→L2更低→AP更高→较早epoch选整轮，保存selected/last。明确test用于选轮，不称独立留出估计。
- 输入报告、缓存、旧代码和原实验均只读；模型/指标实现直接import复用。

## 比较内容与判读

比较含单帧、D4全部五轮与选中整行、原首轮GRU/HV选中整行，并保留D4与两者**每个相同epoch**的差值，避免把不同选轮造成的变化当成历史贡献。

新报告核对与原首轮的来源、特征、初始化、预算和原始基线；本轮selected缓存/事件明细完整核验，D4与单帧的响应事件按轨迹/原帧号配对。旧轮主要追溯已验收小报告，不要求重读旧大NPY/PT。

D4若与原GRU/HV相当或更好，说明对应收益在本配置下不需要历史；GRU领先D4而HV不领先GRU，说明应重点重构历史选择。只有核心稳定/响应同时得到支持时，才继续扩展当前机制。单seed结果按实际范围解释，不根据结果临时改目标、epoch规则或拼列。

## 输入与产物

输入小报告：`r1_train_single_frame_cache.json`、`t2_single_frame_cache.json`、`r2_feature_cache.json`、`r4_official_test_features.json`；参照为 `r4_official_comparison.json`。

新实验根目录：`/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/`，保存 `training_config.json`、`training_history.json`、`selection.json`、五轮 `.pt`、`selected.pt`、`last.pt`和五轮完整test预测/明细；不生成历史控制诊断。

小报告为 `results/r4_current_frame.json/.md`、`r4_current_frame_comparison.json/.md`、`r4_current_frame_selection.json`与provenance。阶段日志为 `r4_current_frame_train.log`、`r4_current_frame_comparison.log`，外层为 `r4_current_frame_round.log`。

唯一服务器手动入口为 `code/scripts/run_current_frame_round.sh`，完整GPU 1命令见实现计划§15.1。大文件留服务器，实际路径/SHA由入口记录；回传小报告后本地验收。没有尚未实测的D4性能数字。
