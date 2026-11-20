# GazeSpot + ConvLSTM：直接优化VAT主指标

2026-11-20。按用户新指令重置主线：在GazeSpot上增加时序模块，直接提高VAT的AUC、L2、AP，保持官方train/test。取消尚未实现的D7冻结诊断，不继续候选选择门路线。以下为本次讨论的推荐方案，代码实现为下一任务。

## 目标和架构

复用现有GazeSpot-B VAT checkpoint，冻结全部原模型参数，只训练新增时序适配器。使用GGSF/SASA和人物条件Transformer之后、原heatmap_head/inout_head之前的任务特征，不再使用原末层DINO场景特征加手工摘要作为主接口。

原模型每人物每帧提供空间特征 `X_t [256,32,32]` 和InOut token `z_t [256]`。新模块为一个共享、单向ConvLSTM，hidden_channels=64：

```text
u_t = Conv1x1_256_to_64(X_t) + broadcast(Linear_256_to_64(z_t))
(H_t, C_t) = ConvLSTM(u_t, H_previous, C_previous)
X'_t = X_t + Conv1x1_64_to_256(H_t)
z'_t = z_t + Linear_64_to_256(global_average_pool(H_t))
heatmap_t = original_heatmap_head(X'_t)
inout_t = original_inout_head(z'_t)
```

两个残差输出投影零初始化，使初始输出对应原GazeSpot。原heads参数冻结且处于eval，训练时保留经heads回传到新增模块的梯度。新模块直接产生可解码的增强特征，不在两张固定预测图之间做选择，原InOut分数也不再锁死。

ConvLSTM保留空间网格，记住同人物过去帧的任务表征。人物起点/原帧号缺口重置，跨chunk保留状态并detach；只使用当前和过去帧，单向，纯RGB和真实头部框。

## 训练和评测

- 官方完整train训练，每epoch完整官方test；不划validation。
- 固定首轮配置：seed0、10epochs、clip16、batch4、hidden64、AdamW lr0.0001/weight_decay0.01、gradient_clip_norm1.0。
- 损失只用 `heatmap_MSE + 0.1 * inout_BCE`；MSE仅GT画内，BCE全部有效人物帧，不新增oracle/候选效用/时序辅助loss。
- 保存每轮完整指标和checkpoint，按既定AUC更高→L2更低→AP更高→较早epoch选整轮，selected/last同时保留。test参与选轮的约定保持。
- 首轮直接比较原GazeSpot-B和GazeSpot-B+ConvLSTM的AUC/L2/AP。已有grid/grid-EMA可保留参考，时序稳定指标降为附加观察，不再作为主线任务。

## 实现顺序

1. 在 `cscwd/code/` 新增独立wrapper，提取并缓存上述两种冻结任务特征，使用FP32保留原head输入；旧gazelle包只import不修改。
2. 实现ConvLSTM适配器与直接双输出训练，复用已有轨迹和原主指标函数。
3. 一个完整服务器手动入口完成所需特征缓存与10轮训练/逐轮test，使用物理GPU1；实现后提供全部参数、日志/输出路径的nohup命令。

不先做额外诊断作业或多个候选机制。原D1–D6结果保留。新代码仍只放 `cscwd/code/`，小结果放 `cscwd/results/`，大特征/PT留服务器并记录provenance。本次没有新增训练或性能数字。
