# R3 历史视觉对应与有效性控制

2026-11-17。R3 本地实现与检查完成；本次没有新增 VAT 性能结果。正式训练按 [r4_compact_v2](../r4_compact_plan.md) 进行。

## 实现与输入

模型为 `cscwd/code/stag/history_model.py` 的 `HistoryValidityRefiner`，继承 R2 的 `CausalJointRefiner`，复用投影、GRU 和双输出残差头。旧 `gazelle`、R2 模型、加载器与训练入口均保持原样。

输入为当前冻结场景特征 `F_t`、当前单帧热图 `h_t`、当前单帧画内分数 `s_t`、当前真实头框，以及同一人物过去预测形成的状态。特征复用 R2 的 **768×32×32 / FP16** 缓存；进入新增模块后为 FP32。匹配温度固定为 **0.1**，不是已调优的实验结果。

`step(F_t,h_t,s_t,bbox,state,reset)` 返回热图、画内分数和新状态，兼容既有训练/验证函数。`step_with_diagnostics(...)` 多返回一个诊断字典。两接口均不接收 GT 点、GT 状态、目标身份或未来帧。

## 逐帧计算

1. **当前证据**：保留 R2 的场景、分数面积头框、当前 raw 热图区域三个视觉 token，以及当前 raw 热图和状态分数。当前路径不随历史门关闭。
2. **历史对应**：过去模板 `target` 与当前原始冻结特征逐 patch 做 cosine matching，经过温度 0.1 的 softmax 得到当前对应分布。模板和人物摘要存于原始冻结特征空间，避免 TBPTT 更新投影后，旧记忆与新特征使用不同的表示坐标系。
3. **连续控制**：小型 MLP 读取当前人物 token、当前预测区域 token、对应区域 token，加六项摘要：匹配峰值、归一化对应熵、对应分布与当前预测区域一致性、人物外观变化、当前 raw 分数、上次修正分数。输出 `validity=g` 和 `refresh=u`。
4. **历史读入**：`g` 同时控制旧 hidden、对应 token、当前对应空间证据和旧状态分数。GRU 的历史种子为 `g*(hidden_prev+matched_token)`；输入保留全部当前 token，并使用当前网格中的对应证据摘要，不混合旧图像坐标。两个输出都由该 hidden 修正。
5. **空间修正**：热图在 R2 空间残差之外增加 `correspondence_gain*g*(cosine-mean(cosine))`，再上采样到 64×64 并修正原始 logits。空间证据有界且中心化，softmax 分布用于区域读出。In/Out 分支修正原始连续分数的 logits。
6. **记忆写入**：将本帧修正热图非负化、缩放至特征网格并按空间和归一化，再加权读出当前原始特征。令 `retention=g*(1-u)`，则 `target_next=retention*target_prev+(1-retention)*predicted_target_current`。`g=0` 时旧预测路径全部切断，模板由当前预测重新建立；`g` 高且 `u` 低时保留模板，`u` 高时刷新模板。位置更新由每帧重新建立的视觉对应完成。

`g/u` 是主任务损失训练出的连续控制量，不是有真值监督的保留/更新/释放三分类，也不把高 In/Out 分数或高相似度直接视为“历史正确”。原热图不是校准概率；非负归一化只定义视觉区域读出。全零区域权重读出零向量。

热图残差头、状态残差头和 `correspondence_gain` 均零初始化，初始输出与 R2 相同，为 clamp 后的单帧预测。普通 GRU 与新模型使用相同 seed 构造时，共享模块的初始权重逐项一致。

## 状态与边界

| 字段 | 内容 |
|---|---|
| `hidden` | 同人物的循环状态 `[B,128]` |
| `heatmap` | 本帧修正热图，保留为自身预测状态与重载记录；旧坐标不直接用于新帧融合 |
| `inout` | 本帧修正的连续画内分数 |
| `target` | 过去预测区域形成的冻结空间模板 `[B,768]` |
| `person` | 当前真实头框内冻结空间摘要 `[B,768]` |
| `available` | 是否已有同人物连续历史，仅表示历史存在 |

首帧和元数据 reset 时全部状态清零；该帧 `g=0`，随后 `available=true`。这个标记不由 GT 或 In/Out 阈值决定。跨 chunk 使用既有 `detach_state` 保留所有字段的值并截断梯度；跨人/原帧缺口重置。padding 的损失和输出记录由既有 valid mask 排除，新连续段使用 metadata reset。

## 诊断与最小消融

每帧返回 `history_validity / refresh / write_strength / match_peak / match_entropy / agreement / person_change / history_available`，以及 native grid 的 `correspondence_map`、`correspondence_xy`。对应坐标是**当前 patch 中心**的归一化 `(x,y)`，与标准定位 argmax 的 `/64` 坐标规则分列；没有历史时对应 map/xy 为零，应结合 `history_available` 读取。

`cscwd/code/stag/history_diagnostics.py` 的 `HistoryDiagnosticWriter` 保存 `controls.npy[N,8]`、`correspondence_xy.npy[N,2]` 和 `source_global_indices.npy[N]`；index 含列名、轨迹、原始帧号、紧凑/原全局切片与 SHA256。按 valid 排除 padding，完整且每帧只写一次后发布 index。大文件留服务器，R4 调用时传入来源、模型和 checkpoint metadata。

首项消融使用 `history_mode="always_keep"`：将**历史有效性读门**固定为历史存在标记，保留视觉对应、模板 refresh 和 GRU 内部门。它不等于关闭所有记忆更新，也不表示永不遗忘；在 R4 按同预算重新训练。

## 参数与本地检查

在实际构造的 `feature_channels=768 / hidden_dim=128` 下，普通 GRU **347,522 参数**，R3 **372,677 参数**，新增 **25,155 参数**。固定 seed 的共享权重一致性检查和 native 768×32×32 FP16 输入的两步前向均通过。

**24 项新增检查通过**：模型/机制 16 项，合成训练/完整验证/检查点/诊断落盘集成 8 项；另有 R2 模型 10 项、连续数据 8 项回归通过。合成检查验证移动区域对应、受控保留/刷新/释放、零有效性切断历史、prefix 因果性、全状态跨 chunk 等价、逐人重置/隔离、GT 变更不改前向或 reset，以及双头、GRU、投影、gate 和对应增益的梯度。完整命令与来源见 [r3_local_checks.md](r3_local_checks.md) 和对应 JSON。

这些检查说明结构可表达并可训练；是否能学会识别真实历史失效，由 R4 真实训练与验证回答。R3 不追加服务器 smoke 作业。下一任务实现精简 R4 的训练、逐 epoch 报告和 checkpoint 选择入口，提供两种模型的完整服务器 nohup 命令。
