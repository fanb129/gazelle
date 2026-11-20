# R4 精简训练与比较计划

版本：`r4_compact_v2`，2026-11-17。用户已授权压缩 R4，先取得真实训练信号，再补与核心主张直接相关的证据。以下训练次数、参数和选择规则均为计划，尚未运行。

本计划替代 R1 中完整版 R4 的后续执行预算与选模规则。`development_protocol.py`、R1 已导出的 `selection_config.json`、R1/R2 机器报告和原验收记录保留原样；R2 的一个 epoch smoke 仍按当时协议解释，不用本次修订追认已有配置。

## 1. 第一轮：两种模型、同一预算

比较既有冻结单帧基线、普通因果 GRU 与 R3 新机制。只训练后两种模型；复用 R1 原预测、R2 原生 FP16 特征缓存和既定开发组划分。两种模型均从新模块随机初始化开始，使用 seed 0 的相同初始化规则，不从 R2 的 `last.pt` 续训；共享的输入/输出结构按同一规则初始化。记录实际参数量及结构差异。

| 参数 | 两种可训练模型共用的计划值 |
|---|---|
| 配置数 / seed | 每种 1 组 / `0` |
| epochs | 固定 `5`，每个 epoch 完整验证，不早停 |
| learning rate / optimizer | `0.0001` / AdamW |
| lambda_state | `0.1` |
| hidden_dim / clip_frames / batch_size | `128 / 16 / 8` |
| weight_decay / gradient_clip_norm | `0.01 / 1.0` |
| 输入 | 同一冻结 ViT-B 空间特征、原始 heatmap/In/Out、当前真实头框；历史为自身过去预测 |
| 损失 | GT 画内有效帧 heatmap MSE + `0.1` × 全部有效帧状态 BCE |
| 精度与状态衔接 | FP32 新模块；跨 chunk 仅 detach，同人物连续段保留状态，跨人/缺口重置 |

训练只用开发 train 标签。validation 每个 epoch 都按完整连续轨迹从空状态递推；test 不参与结构、阈值、checkpoint 或配置选择。固定输出第 1–5 个 epoch 的完整验证指标，不把已有一个 epoch 的 GRU smoke 与新方法的五个 epoch 直接作为公平主比较。

## 2. checkpoint 选择与折中报告

保留单帧为参考，按相同 validation 记录计算：

- `AUC >= baseline_AUC - 0.002`。
- `L2 <= baseline_L2 + 0.002`。
- `In/Out AP >= baseline_AP - 0.002`。

同时满足三项的 epoch 为 `precision_eligible`。取消 R1 中“每一种状态响应率都不得低于单帧”的硬门槛；enter、leave、gaze_shift、状态错误与翻转、Jitter 和准确稳定性仍完整报告，以明确展示代价。

对每种模型的 eligible epochs 使用以下字典序选 checkpoint：

1. 准确稳定占比高者优先。
2. joint `gaze_shift` 响应率高者优先。
3. In/Out AP 高者优先。
4. L2 低者优先。
5. AUC 高者优先。
6. 较早 epoch 优先；跨模型完全同分时使用 `model_id` 字典序。

每种模型保存选中 checkpoint 及 `last.pt`。若没有 eligible epoch，则记录 `no_precision_eligible_checkpoint`，保留单帧部署参考，不从 test 挑一个替代配置。

报告所有 epoch，而非只保留胜出行。另计算候选与单帧参考的 Pareto 集：AUC、AP、准确稳定占比、joint gaze_shift 响应越大越好，L2、全帧状态错误率越小越好；其他指标仍逐列列出。只有其他候选在这些轴上全部不差且至少一项更好时，才将该候选记为被支配。单帧没有被超过、或新机制没有超过普通 GRU 时，直接保留该结果。

第一轮用于判断改善的方向及是否值得补机制实验，不把一个 seed 的小幅差值称为稳定复现或统计显著。若新机制未提供值得保留的定位/状态/时序折中，先检查训练与机制，不展开整套消融或网格搜索。

## 3. 有信号后的最少补充

首项消融固定为：**关闭历史有效性读门，保留当前视觉输入、历史目标对应和模板刷新，并重新训练**。R3 对应 `history_mode="always_keep"`，仅将 validity 固定为历史存在标记，GRU 内部门和 learned refresh 保留，不称为关闭所有记忆控制。使用第一轮同一配置、seed 0、五个 epoch 和同一选择规则；不在训练好的新机制上临时关闭模块后，把分布变化当作完整训练对照。该实验回答有效性读门是否有额外作用。

根据实际要写的贡献，再补直接相关的一项对照：

- 主要收益位于阈值状态错误率或进出响应：补开发 train 拟合的全局严格单调状态校准，validation 仅评估。分类仍用既定阈值，连续分数 AP 与排序保持不变；拟合方式在运行前保存配置。
- 要声称“历史本身带来收益”：补同样训练、同当前视觉输入和双输出的无历史修正头，使用相同预算与选择规则。

第一轮不展开多种滤波器、四参数门控网格、完整 `heatmap_only / state_only / joint` 矩阵、多项结构消融、多 seed 或 bootstrap。需要复现时优先按相同配置重复新机制和最强对照，先固定 seed 清单，再运行并全报；仅完成单 seed 时如实报告这一范围。论文主张以实际完成的对照为边界。

## 4. 交付与进入 test

关键小报告继续使用 `results/r4_validation_comparison.json/.md` 和 `results/r4_selection_manifest.json`，新增记录 `plan_version=r4_compact_v2`、实际跑过的模型/epoch/seed、所有指标、precision eligibility、排序理由及 Pareto 集。原 R1 manifest 的 SHA256 继续作为来源，当前计划另存版本与 SHA256，不修改历史文件。

选定代码、模型、checkpoint、特征契约、阈值和解释范围后进入 R5。大文件仍留服务器，记录路径/SHA256；本地只写代码，用户手动执行服务器作业。具体训练脚本实现后提供完整 GPU/Python/参数/日志/输出路径的 nohup 命令；本文件没有虚构尚未完成的训练结果或耗时。
