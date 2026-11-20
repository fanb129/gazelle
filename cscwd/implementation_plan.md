# GazeSpot + ConvLSTM：实现计划

2026-11-20。当前主线唯一规格。旧工作整体归档至 `cscwd_v0/`，不再执行旧诊断或候选选择路线。

## 1. 目标

以GazeSpot-B VAT checkpoint为底座，增加严格因果的空间时序模块，直接优化VAT AUC、L2、AP。ConvLSTM先作为时序基线；有效后再针对具体不足设计新模块，不把ConvLSTM本身列为创新点。

## 2. 固定架构

使用GazeSpot人物条件Transformer之后、原heatmap/InOut预测头之前的任务特征：空间特征 `X_t [256,32,32]` 和InOut token `z_t [256]`。

```text
u_t = Conv1x1_256_to_64(X_t) + broadcast(Linear_256_to_64(z_t))
(H_t, C_t) = ConvLSTM(u_t, H_previous, C_previous)
X'_t = X_t + Conv1x1_64_to_256(H_t)
z'_t = z_t + Linear_64_to_256(global_average_pool(H_t))
heatmap_t = original_heatmap_head(X'_t)
inout_t = original_inout_head(z'_t)
```

原GazeSpot全部参数冻结，仅训练新增模块。两个残差输出投影零初始化，从原checkpoint的输出开始学习。原heads保持eval，训练时允许梯度经其输入回到新增模块。任务特征缓存使用FP32。

历史只来自同人物的当前/过去帧；单向ConvLSTM，人物切换/帧号缺口重置，跨chunk保留状态并detach。纯RGB和真实头部框；GT只用于loss/计分。

## 3. 训练与选轮

- 官方完整train训练，每epoch完整test，无validation。
- 首轮只跑一个配置：seed0、10epochs、clip16、batch4、hidden64、AdamW lr0.0001/weight_decay0.01、gradient_clip_norm1.0。
- 损失 `heatmap_MSE + 0.1 * inout_BCE`，定位仅GT画内，状态全部有效人物帧。
- 保留每epoch完整指标与checkpoint，按AUC更高→L2更低→AP更高→较早epoch选一整轮；保存selected和last，不拼列。test参与checkpoint选择的约定保持。
- 首轮主比较为原GazeSpot-B与新增ConvLSTM。主表AUC/L2/AP，时序稳定性仅作附加观察。

## 4. 实现与运行

下一任务依次完成：任务特征提取/缓存→ConvLSTM适配器→完整训练与逐epoch test入口。完成后提供一个服务器手动nohup入口，串起缓存和10轮训练，不另加诊断作业。

| 项目 | 约定 |
|---|---|
| 新代码 | `cscwd/code/{data_prep,stag,scripts}/` |
| 原底座 | `gazelle/`只import不改 |
| 新论文 | `cscwd/latex/` |
| 新小结果 | `cscwd/results/` |
| 服务器 | `fb@3090.lab` |
| 服务器仓库 | `/home/fb/src/paper/gazelleV1` |
| Python | `/home/fb/anaconda3/envs/py310/bin/python` |
| GPU | 物理GPU1，外层CUDA_VISIBLE_DEVICES=1，内部cuda:0 |
| VAT数据根 | `/newhome/fb/dataset/videoattentiontarget` |
| GazeSpot-B checkpoint | `/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt` |

既有服务器数据/缓存/checkpoint不随本地归档移动。新代码独立实现，数据只读；新实验输出使用新目录。每个服务器脚本实现后给出Python/GPU/全部参数/日志/输出路径，用户手动执行；不假设已运行。大NPY/PT不进Git，来源写provenance；不填未实测数字。

## 5. 进展

- **V1-0 / 2026-11-20 / 目录重建完成**：原 `cscwd/` 全量移动为 `cscwd_v0/`，365个原文件/链接（含隐藏/忽略项）逐项保持内容；旧根进度另存 `cscwd_v0/PROGRESS_SNAPSHOT.md`。新目录仅含本方案、目录骨架与新结果provenance入口。没有新训练/评测数字。
- **下一步V1-1**：实现上述任务特征缓存、ConvLSTM基线与完整服务器训练入口，完成本地必要检查后给用户GPU1命令。
