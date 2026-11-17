# 数据集与评价指标速查

整理日期：2026-11-10。用于配合[逐篇论文解读](literature_review.md)，明确实验输入输出、标注和指标之间的对应关系。

## 1. 三种任务不要混淆

| 任务 | 典型推理输入 | 典型输出 | 常见指标 | 本目录对应论文 |
|---|---|---|---|---|
| 第三人称注视目标估计 / gaze following | 场景RGB、指定人物的头部框；部分方法还用头部裁剪、深度 | 该人物的二维目标热图/点；画内外概率 | 热图AUC、归一化L2、In/Out AP | P01—P07的主要任务 |
| 人物与注视联合检测 | 场景RGB或视频，不以真实头部框作为同样的输入条件 | 人物头部检测及其关联目标/状态 | 检测与匹配后定义的注视指标 | P04；须核对检测和匹配协议 |
| 第一人称注视预测 | 佩戴式相机视频、历史预测 | 佩戴者在当前图像中的注视热图 | Adaptive F1、Precision、Recall等 | P08 |

人脸/眼部图像到三维注视方向的 gaze estimation 又是另一任务，通常使用角度误差。不能将其角度误差与 VAT 的二维归一化距离直接比较。

## 2. 数据集、模型输入与标签

### GazeFollow

- 类型：静态图像；约122k图像、130k人物实例；官方测试含4,782个人物注视实例，每例约10份目标标注。
- 推理输入：场景RGB及被查询人物的头部框；裁剪、位置掩码可由此派生。
- 标签：训练用目标点及扩展标注中的画内外信息；测试多标注用于刻画人类分歧。
- 输出：热图及二维预测点；标准测试主表通常不报告 VAT 式 In/Out AP。
- 指标：AUC、Avg L2、Min L2。
- 用途：单帧训练/预训练；不支持仅凭该数据集证明跨帧建模有效。
- 依据：[Gaze-LLE第4节](https://openaccess.thecvf.com/content/CVPR2025/papers/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.pdf)、[PDP第4节](https://openaccess.thecvf.com/content/WACV2023/papers/Miao_Patch-Level_Gaze_Distribution_Prediction_for_Gaze_Following_WACV_2023_paper.pdf)。

### VideoAttentionTarget / VAT

- 类型：第三人称视频；约71k视频帧、164,541个人物帧标注；包含约109.6k画内和55.0k画外实例。
- 计数单位：文献还报告1,331条人物头部轨迹。不同文献把sequence/clip/track混用，不能把“轨迹数”当成“独立源视频数”。
- 推理输入：RGB帧与指定人物框；视频模型还需时序排列及稳定人物身份。
- 标签：头部框、单个二维注视点或画外标记；同一帧可以标注多个人物。
- 输出：每人每帧热图/目标点和画内外概率。
- 指标：只在有效画内样本上统计定位AUC/L2；画内外AP覆盖相应有效状态样本。
- 局限：原始常用标注没有可直接等同于完整“目标遮挡/真实注视切换”真值的字段；不能随意命名困难子集。
- 依据：[原始论文及官方代码](https://github.com/ejcgt/attention-target-detection)、[PDP第4.1节](https://openaccess.thecvf.com/content/WACV2023/papers/Miao_Patch-Level_Gaze_Distribution_Prediction_for_Gaze_Following_WACV_2023_paper.pdf)。

本项目之前只读检查服务器时，可用预处理数据为训练58,507帧/132,563个人物帧、测试13,127帧/31,978个人物帧。这里是本项目本地版本计数，不覆盖文献中的数据集计数。已有预处理JSON没有稳定track_id，应从原始人物轨迹文件恢复，不能按heads列表位置匹配身份。

### ChildPlay

- 类型：儿童和成人自然交互视频；95个源视频、401段片段、120,549帧、257,928个人物帧标注。
- 标签：头部框、目标坐标及七类状态：inside-frame、outside-frame、gaze-shift、occluded、eyes-closed、uncertain、not-annotated。
- 推理输入：RGB、指定人物信息；标签是监督/评价信息，不是测试时额外输入。
- 标准输出：人物目标热图和画内外概率。使用该数据集不等于模型必须输出七分类状态。
- 指标：AUC、Dist、In/Out AP，以及P.Head/后续定义的PLAH；可分别评估儿童与成人。
- 评价边界：先确认各状态下坐标是否有效；gaze-shift、遮挡等状态不自动具有可靠落点。occluded不能未经核验当作“被注视物体受遮挡”。
- [原论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Tafasca_ChildPlay_A_New_Benchmark_for_Understanding_Childrens_Gaze_Behaviour_ICCV_2023_paper.pdf)、[作者数据发布](https://zenodo.org/records/8252535)。

### VideoCoAtt

- 类型：第三人称共同注意视频。
- 标签：共同注意目标框及相关人物；标注目的与逐人VAT目标点不同。
- 传统实验输出：当前帧是否存在共同注视、共同目标位置。
- 传统指标：共同注视判定指标A/Accuracy及到目标框中心的像素L2；不是热图AUC。
- MTGS扩展：将共同注意转为人物对关系，使用SA AP。旧帧级A与新人物对AP不可混比。
- [CVPR 2020第5.4节](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chong_Detecting_Attended_Visual_Targets_in_Video_CVPR_2020_paper.pdf)、[MTGS第4.3节](https://www.proceedings.com/content/079/079017-0500open.pdf)。

### UCO-LAEO

- 类型：第三人称相互注视数据。
- 标签：人物头部及人物对是否互相注视。
- 原始任务输出：人物对LAEO分数；主要评价互相注视识别。
- MTGS扩展：通过统一标注还形成目标点及相关社会注视标签；使用LAEO AP或其联合协议下的LAEO F1，必须注明是哪种设置。
- [MTGS第4.1—4.3节及表3](https://www.proceedings.com/content/079/079017-0500open.pdf)。

### VSGaze

- 类型：统一、扩展VAT、ChildPlay、VideoCoAtt、UCO-LAEO的标注资源，不是独立采集的第五套视频。
- 标注：补全并核验头部轨迹；统一逐人目标点、画内外与人物对LAH/LAEO/SA标签。
- 输入：多帧场景和多人的头部轨迹/裁剪。
- 输出：每人热图与状态、人物对关系分数。
- 指标：Dist、In/Out AP、LAH F1、LAEO F1、SA AP。
- 公平比较：联合训练后的VSGaze结果、其组成数据集结果、单数据集微调结果是不同协议，必须分开。
- [MTGS正文与补充材料](https://www.proceedings.com/content/079/079017-0500open.pdf)。

### GOO-Real

- 类型：零售环境的真实图像；当前项目使用的跨域测试有2,146张图像/人物查询。
- 推理输入：RGB及被查询人物信息。
- 标签：目标位置及相关物体信息；参与者被指示看向物体，与第三方“猜测人物在看哪里”的标注来源不同。
- 在Gaze-LLE中的任务：仅用GF训练后，直接评估跨域定位。
- 指标：AUC、L2；不把该设置写成视频时序评价或画内外AP。
- [Gaze-LLE表5及讨论](https://openaccess.thecvf.com/content/CVPR2025/papers/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.pdf)。

### EGTEA Gaze+ / Ego4D / EgoExo4D

- 类型：第一人称注视数据，在本文作为ARGaze的方法迁移参考。
- EGTEA Gaze+：第一人称烹饪视频；ARGaze使用官方划分。
- Ego4D：ARGaze使用其中带注视标注的子集，而不是整个数据集。
- EgoExo4D：ARGaze使用第一人称子集，将注视信息投影为二维坐标；另构建参与者/地点/任务域外划分。
- 推理输入：佩戴者视角视频及模型预测历史；没有第三人称被查询人物头部框。
- 输出：当前二维热图；不是VAT式每个外部人物的画内外状态。
- 指标：Adaptive F1、Precision、Recall，另评价流式效率。
- 依据：[ARGaze第4.1节及附录A/B](https://arxiv.org/html/2602.05132v1)。

## 3. 指标定义与使用条件

### 热图 AUC

将热图位置视作分类对象，真实目标位置/区域为正样本，预测响应为置信分数，计算ROC曲线下面积。

GF可使用多个标注点构成正样本图；VAT/ChildPlay通常围绕单个标注点构造容忍区域。正样本图大小、热图缩放、阈值等实现差异会影响数值。因此同名AUC并不自动代表同协议，引用其他论文的重新计算结果时应保留脚注。

AUC不是画内外准确率，也不评价概率是否校准。不能因为AUC高或热图尖锐就声称“不确定性估计准确”。

### 二维定位距离 L2

对预测点与真实点计算欧氏距离。常见第三人称协议分别把横、纵坐标除以图像宽、高，映射到单位方形：

`L2 = sqrt((x_pred/W - x_gt/W)^2 + (y_pred/H - y_gt/H)^2)`。

热图预测点通常取argmax。VideoCoAtt旧协议的像素距离不做同样归一化，不能与VAT的0.1量级距离直接比较。不同论文若用不同resize或归一化，应以实际评估实现为准。

### GazeFollow Avg L2 与 Min L2

- Min L2：预测点到多个真实标注点中的最近距离。
- Gaze-LLE正文明确将Avg L2描述为“预测点到平均标注坐标的距离”。
- “到平均坐标的距离”与“到各标注距离的均值”通常不相等，不能在实现时自行替换。
- 若其他论文只写Average distance，应核对其代码，不能仅凭列名认定两者一致。

### In/Out AP

评价人物注视目标是否在画面内的二分类排序能力。常用实现以in-frame=1、预测画内概率为分数；部分表格标题写Out-of-frame AP，引用时仍需核对其正类定义。

计算应覆盖有效画内和画外样本，不能只在画内样本上计算。AP不是Accuracy，正类翻转会改变AP，不能用同一个数简单互换解释。

### P.Head 与 PLAH

- ChildPlay最初的P.Head：判断预测是否落入一个人物头部框，并与真实“看头部”状态比较，计算precision。
- Sharingan采用更严格定义：预测点与真实点需要落在同一个头部框。
- 二者依赖可用头部检测/标注，定义不同的数值不宜直接排名。

### LAH F1、LAEO F1 与 SA AP

依据MTGS协议：

- LAH：有向“人物i看向人物j”。正确预测不仅要超过阈值，还要识别正确目标人物。
- LAEO：两个人相互注视，涉及双向关系与论文指定的目标选择规则。
- F1：`2 × Precision × Recall / (Precision + Recall)`。LAH/LAEO各有自己的样本和正例定义。
- SA AP：人物对是否共享注意目标的平均精度；与传统“整帧是否存在共同注视”不同。
- 模型直接解码关系与从热图后处理推导关系须分别报告。

### ARGaze Adaptive F1、Precision、Recall

评价第一人称空间注视预测，不是人物画内外分类，也不是社会注视F1。原文说明跟随GLC协议，但已核验的指标描述没有完整给出所有阈值和汇总实现细节。后续若复现，应先核对二值化方法、GT目标区域和帧级/整体聚合方式；本文不补造未明确的定义。

### 效率

至少记录硬件、输入分辨率、batch size、人物数、时序长度、精度、计时同步及缓存范围。应区分骨干推理、时序增量、解码和整个在线系统。

训练时缓存特征能节省实验成本，不代表部署时省去了每帧的特征提取。数据加载或CPU-GPU传输是否包含在延迟内也应写清。

## 4. 新实验的比较约束

1. 同一张主表固定人物框来源：真实框、外部检测框、端到端预测框分开。
2. 固定输入模态：纯RGB与额外深度/姿态/分割分开。
3. 标明原论文版、后续DINO版与本项目复现；不拼接不同变体的最好指标。
4. 标明单帧、窗口式视频、严格因果流式；使用未来帧的设置单独处理。
5. 按源视频划分训练与验证，不能把相邻帧随机分到不同集合。
6. 区分原始数据集、VSGaze扩展标注及不同微调协议。
7. 稳定、切换、画内外转换、合成视觉退化分别定义；合成测试明确写“合成”，不替代完整测试集。
8. 用真实标签定义评价子集是允许的，但测试时门控、记忆重置及写入不能读取这些标签。
9. 抖动降低不自动代表预测更准，应结合真实轨迹及切换响应分析，防止静止预测取得虚假的“稳定性”。

## 5. 当前实现可复用点

- Gaze-LLE/GazeSpot单帧热图、画内外分数及冻结特征可作为时序模块输入。
- R1 将从 VAT train 固定源视频分组清单；该留出仅约束新增时序模块。现有冻结 VAT checkpoint 此前已见过 VAT train，所有对照共享这一底座条件。
- T1 已恢复原始人物轨迹；逐帧 heads 顺序不是人物身份。T2 test 预测缓存不能替代 R2 所需的冻结视觉特征缓存。
- GF与GOO-Real可以支持单帧或跨域定位分析，不能代替真实视频评价。
- ChildPlay的状态标注有参考价值，但不应在确认视频可获取性、标签定义和有效坐标之前纳入必须完成的实验。

这些是实验设计建议，尚不是新模型已经验证的结论。

## 6. 2026-11-17 联合修正方案的指标约定

当前方案见 [temporal_joint_refinement.md](temporal_joint_refinement.md)，执行与验收以 [implementation_plan.md](implementation_plan.md) 的 R1–R6 为准。本节区分已实现口径与规划扩展，不新增实测数字。

### 已有规则与新状态分支

- 原规则 STAG 只改 heatmap，连续 In/Out 分数完全不变，因此 AP 与任何仅由原始状态分数得到的分类结果严格不变。
- T4 的 `leave_frame` 窗口在下一代理事件前截断，GT 为画外；成功只取决于状态，因此原规则的该类响应率、删失与延迟不变。
- `enter_frame / gaze_shift` 的 joint 成功还要求当前定位正确，所以即使 In/Out 不变，也可随 heatmap 改变。
- 新联合模块输出新的连续状态分数，具有改变 AP 和状态转换响应的能力；尚无新方案实测收益。改变分类阈值不改变 AP；统一的严格单调分数变换保持排序，也不改变 AP。

### R1 已实现、等待真实结果的扩展

| 指标 | R1 需要固定的定义 | 必须同时解释的量 |
|---|---|---|
| 状态错误翻转率 | 同 track、原帧号连续、相邻 GT 状态相同的帧对；分子为预测状态发生翻转的帧对数，分母为全部合格帧对 | 画内/画外分型、状态错误率与真实转换响应；恒定错误预测不能仅靠低翻转率获益 |
| 状态错误率 | 全部有效人物帧上按固定阈值作状态分类，与当前 GT 比较 | AP 使用连续分数，独立报告 |
| 状态转换响应 | 在 GT 画内外翻转后寻找持续状态正确的区间，只计状态、不加定位条件 | 与 T4 的 joint 响应分列；沿用或显式版本化连续帧数、窗口与删失规则 |
| 运动/目标转移诊断 | 训练组中按固定规则抽样，分别记录同目标运动、相机变化、可辨识换目标、误跳与无法判定 | 原 VAT 不提供完整目标 ID，人工诊断不自动替代全量真实切换标签 |

AUC/L2 继续只用共同 GT 画内样本，不能按预测 In/Out 丢弃困难帧来提高定位指标。Jitter 和准确稳定占比沿用 T4 版本；新增状态指标不回写旧结果。

实现入口：`code/stag/state_metrics.py` 的 `state_temporal_v1`。阈值 `score>0.5`，连续正确 3 帧，最多观察 30 帧；独立状态响应仅在下一 GT 状态事件/gap/track 结束前截断，不被 gaze 点位移截断。R1 新入口同时导出原 joint 指标和独立 state 指标；全画外分区的定位/AP 记 `null` 并记录原因。选模预算和容忍度见 `code/stag/development_protocol.py`，均为预声明配置，不是已跑增益。

按源视频选择 train/validation，报告候选预算、精度容忍度及匹配/选模规则。普通因果头与新机制共享冻结视觉输入；额外加入同样训练的当前帧修正头。One Euro 只改坐标时，原热图 AUC 和原状态 AP 的来源注明为未修改，不能将坐标改善称为热图改善。

现有 test 已用于先导观察；新方案在开发集选定结构、阈值和 checkpoint 后再统一报告。统计以源视频为配对/重采样单位，实际 seed 和完整候选表一起保存；不同训练运行不与旧规则结果混算。
