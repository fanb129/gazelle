# R4-D3：稳定与变化响应的状态时序监督

版本：`r4_state_supervision_v1`。2026-11-18，在真实训练前固定。数据/逐epoch测试与整轮选择沿用 `r4_official_train_test_v1`。

## 本轮问题与设置

D2的train全局状态偏置只为新机制补回100个准确稳定窗口，gaze_shift仍88/158；校准后仍低于单帧与普通GRU。当前模型仅由逐帧heatmap MSE与状态BCE训练。本轮保持架构与输入，加入明确的状态时序监督，检验其能否改善稳定/响应折中。

普通GRU与历史有效性模型使用相同loss、相同数据和预算，新增模块均从头随机初始化，不从原epoch 5续训。整个GazeSpot ViT-B及原decoder继续冻结。只使用当前/过去RGB、当前真实头框和模型自身预测历史。GT状态只构造loss，不进入模型输入、门控、预测历史或reset；不引入未来帧。

## 固定loss

令 `p_t` 为模型输出的画内概率，`y_t` 为当前GT状态，固定：

\[
L=L_{heatmap}+0.1\,[L_{BCE}+0.1\,(L_{stable}+L_{transition})].
\]

CLI记为 `lambda_state=0.1, lambda_time=0.1, response_frames=3`。只运行这一组，不根据test扫描权重或阈值。

- `L_heatmap` 与 `L_BCE` 沿用原实现：全部有效GT画内帧的像素MSE均值、全部有效人物帧的未加权状态BCE均值。
- `L_stable`：同人物、原始帧相邻且 `y_t == y_(t-1)` 时，计算 `abs(p_t - p_(t-1))`，按符合条件的pair独立取均值。真实状态翻转的pair不加该项；双方GT均为画内的注视位置变化仍要求画内概率一致。
- `L_transition`：只有观察到同连续段内的真实GT状态翻转时，开启当前帧及后续最多两帧的窗口，对所选帧的**当前状态**BCE独立取均值。再次翻转则重新开启窗口；人物/连续段首帧不假设发生翻转。窗口只由当前及过去GT决定，不提前读取未来标签。
- 没有符合条件的pair/帧时，该项为可微零。transition帧仍参与普通BCE，因此额外得到状态变化后的响应监督。两个新增项各自在每个训练chunk内归一化，再按上述公式合成；不按类别或视频另调权重。

保留逐帧监督以维持当前帧正确性；稳定项只针对GT状态不变的pair，转换项在真实变化后加强监督。本轮不改变heatmap训练目标或推理结构，也不预写标准指标保持不变。

## 跨chunk和梯度

继续使用原persistent-slot连续轨迹iterator与TBPTT：`clip_frames=16, batch_size=8`。模型state与loss carry分开。

loss carry只存每个slot的上一有效预测分数、GT状态、可用标记和转换窗口剩余帧数。同人物跨chunk保留，边界上一预测分数detach；chunk内部相邻预测都可获得梯度。跨人物/原帧缺口依据既有metadata reset清空，padding不参与、不桥接。窗口在chunk边界内延续，不能因截断漏计转换帧或稳定pair。

每epoch记录原heatmap/state损失，新增 `stable_sum/stable_pairs/stable_loss`、`transition_sum/transition_frames/transition_loss`，及按固定权重重建的组件总损失。epoch各项均按全epoch各自分母汇总；它是池化组件摘要，不是逐chunk优化目标的简单平均。稳定pair/转换帧的总数另按train轨迹metadata计算，核对每轮没有漏计或重复。

## 预算、测试与比较

- 两模型各五个epoch、seed 0、AdamW、learning_rate 0.0001、weight_decay 0.01、gradient_clip_norm 1.0、hidden_dim 128、temperature 0.1。FP32训练/前向，TF32关闭。共享部分的初始化、每轮连续段顺序与原首轮保持。
- 完整官方train：1,033 tracks / 132,563人物帧 / 58,507图像；每轮完整test：298 / 31,978 / 13,127。不使用validation分区。
- 所有epoch结果和checkpoint全保留；以test AUC更高→L2更低→AP更高→较早epoch选择一整轮，selected/last分别保存。test用于选择epoch，在报告中明确，不称独立留出估计。
- 原指标协议和 `score > 0.5` 阈值保持。本轮不应用D2校准，不将新旧不同epoch/分支的最好列拼成一行。
- 比较包含单帧、本轮十个候选、每模型选轮、本轮事件配对；另与已验收原首轮各自选中的epoch 5作同模型差值比较。原前轮小报告只读，D2校准结果不混入。
- 效果看AUC/L2/AP、准确稳定、状态错帧/错误翻转、joint/state分型响应与共同事件配对，不因共同响应者平均延迟下降就忽略失响应事件。GT位移事件仍为代理标签。

本轮检验时序训练目标，不单凭改善宣称validity机制成立。先比较新机制与普通GRU；有实际时序信号后补最少无历史头/读门对照。

## 输入和产物

复用已完成的官方train/test预测与原生FP16特征：`r1_train_single_frame_cache.json`、`t2_single_frame_cache.json`、`r2_feature_cache.json`、`r4_official_test_features.json`。原首轮比较为 `r4_official_comparison.json`。启动核对来源与缓存SHA，不重新提特征。

唯一服务器启动入口为 `code/scripts/run_state_supervised_round.sh`，完整GPU 1命令见 `implementation_plan.md` §14.1。

- 新实验目录：`/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/{ordinary_gru,history_validity}/`。
- 每模型保存 `training_config.json`、`training_history.json`、`selection.json`、`epoch_001.pt`–`epoch_005.pt`、`selected.pt`、`last.pt`、逐轮完整 `test_epoch_001/`–`test_epoch_005/`；新机制每轮包含同次前向的历史诊断。
- 小报告：`results/r4_state_supervised_ordinary_gru.json/.md`、`r4_state_supervised_history_validity.json/.md`、`r4_state_supervised_comparison.json/.md`、`r4_state_supervised_selection.json` 与provenance。
- 训练、比较日志按上述小报告同名写 `.log`，外层为 `r4_state_supervised_round.log`。
- 新训练保留通用task/protocol以复用已有官方评测/选模，另在config、报告、checkpoint中写 `supervision_version=r4_state_supervision_v1` 区分训练目标。旧源码、原训练计划、预测与结果不覆盖。

本地只做合成检查；真实训练与评测由用户在服务器手动执行。大文件只留服务器，报告和 `RESULT_PROVENANCE.md` 记录实际路径/SHA；经Git回传小结果后验收。
