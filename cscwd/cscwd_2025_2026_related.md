# CSCWD 2025–2026 · 注视/共享注意力方向相关论文梳理

> 抓取时间 2026-11-12。覆盖第 28 届（CSCWD 2025，DOI 段 `10.1109/cscwd64889.2025.*`）与
> 第 29 届（CSCWD 2026，DOI 段 `10.1109/cscwd68734.2026.*`）。
> 数据源：OpenAlex（source S7407087873）逐年拉取，2025 年 435 篇、2026 年 746 篇，共 1181 篇；
> 关键词命中再用 Crossref 按 container-title=Computer Supported Cooperative Work in Design 反查交叉核对。
> 结论基于标题+摘要（及 Crossref 元数据），**未逐篇读全文 PDF**，引用前需核原文。

## 0. 一句话结论

两届 CSCWD 里没有第三人称注视跟随 / 联合注意（joint attention, shared gaze, mutual gaze, LAEO, VFOA）方向的论文。
只有 2 篇真正以 eye/gaze 为对象，1 篇在会议场景里用手工规则检测 eye contact。
这是指定两届会议及既有标题/摘要检索范围内的结论，用于寻找场景锚点。venue 内同题较少不证明全领域方法创新；当前方案应与视频 gaze following、自适应滤波和目标记忆等先行工作对照，见 [temporal_joint_refinement.md](temporal_joint_refinement.md)。

## 1. 直接相关（2 篇，必读必引）

| # | 论文 | 年份 | 与本题关系 |
|---|---|---|---|
| A1 | DNE-Net: An Improved Dual-View Gaze Estimation Model for Interaction in Virtual Environment | 2026 | 唯一一篇 gaze estimation。双目近眼红外 + 双视角特征融合 + attention 卷积块，MPIIGaze 角误差 3.95°，自采数据集 3.45°；用户研究称注视交互把任务时间降 16.9%、提升 VR 协同表现。**第一人称/头戴式**，不是第三人称画面内定位，正好可以当"现有 gaze 工作偏近眼、面向单人"的靶子。 |
| A2 | Eye-Tracking-Based Modeling of Emotional and Engagement States in Human–Computer Interaction | 2026 | 低维眼动特征 + Circumplex 情感模型 + 优化 KNN，联合建模情绪与投入度（engagement）。VR 自采数据。可作为"眼动信号→用户状态"这条线的 CSCWD 内部依据，但它是传感器序列分类，不做空间注视定位。 |

链接：

- A1 https://doi.org/10.1109/cscwd68734.2026.11582113
- A2 https://doi.org/10.1109/cscwd68734.2026.11581589

## 2. 概念上最近：协同场景的注视/群体行为（related work 的重点）

| # | 论文 | 年份 | 与本题关系 |
|---|---|---|---|
| B1 | Recognition of Group Actions from Individual Actions in Brainstorming Sessions | 2025 | **最贴近"共享注意力"概念**。远程会议里由单人动作推断群体动作，明确实现了 Eye Contact（两人互看）动作检测，再靠启发式规则上推讨论/协作/孤立等群体状态。纯规则+基础检测，没有学习式注视定位，也没有热图。是我们"多人协同共享注意力" motivation 的最佳 CSCWD 内部引用。 |
| B2 | TriFER: Multimodal Affective Understanding for Collaborative Work and Interaction | 2026 | 面向协同工作的三模态细粒度情绪识别，CMU-MOSEI。同属"协作场景理解用户状态"，可对齐 CSCWD 的关切；跟注视无直接竞争。 |
| B3 | Embodied Aesthetics: A Collaborative VR Learning Environment with Motion Capture-Driven Natural Interaction | 2026 | 多用户 VR 教学系统，动捕+眼动，评 usability/immersion/social presence。**系统级论文也能中**，说明 CSCWD 接受"用眼动做协同体验"的系统贡献——可作系统级贡献的锚点参照。 |
| B4 | Bilateral Virtual Companions: The Impact of Virtual Humans' Movement and Voice Realism on User Perception and Experience in Multi-user VR Cinemas | 2025 | 多用户 VR 社会临场感/亲密度实验（声音真实感显著影响 social presence）。可借它的 CSCWD 式评测指标写法（immersion / social presence / intimacy 量表）。 |
| B5 | WS-ZSGAR: A Weakly Supervised Framework for Zero-Shot Group Activity Recognition | 2026 | 视频级弱监督群体活动识别，Volleyball/NBA。多人时序理解的 CSCWD 内部代表。 |
| B6 | Dense Student Behavior Recognition Algorithm for Classroom Surveillance Based on Improved Real-Time Detection Transformer | 2025 | 教室监控下的密集学生行为检测（LAF-DETR，遮挡+多尺度）。是"多人、持续、部署导向"的场景证据，可支撑"协同感知要能常开"的论证。 |

链接：

- B1 https://doi.org/10.1109/cscwd64889.2025.11033359
- B2 https://doi.org/10.1109/cscwd68734.2026.11582540
- B3 https://doi.org/10.1109/cscwd68734.2026.11582743
- B4 https://doi.org/10.1109/cscwd64889.2025.11033578
- B5 https://doi.org/10.1109/cscwd68734.2026.11582139
- B6 https://doi.org/10.1109/cscwd64889.2025.11033264

## 3. 方法与场景可借鉴（非注视任务）

| # | 论文 | 年份 | 可借鉴点 |
|---|---|---|---|
| C1 | An Adaptive Attention-Aware Method for Occluded Multi-Pedestrian Tracking | 2025 | 多人视频里的遮挡处理（全局-局部遮挡系数 + 时空注意力 + 匹配代价调整）。对应 STAG 里的"人物消失/重现"与 track 恢复。 |
| C2 | GesFSFF: Dynamic Gesture Recognition via Feature Distribution Smoothing and Hybrid Feature Fusion | 2026 | 纯视频时序建模在 CSCWD 能中（NVGesture/Briareo SOTA）：帧重参数化池化 + 全局局部融合。说明"时序+视频"本身不是 CSCWD 的拒稿理由，缺的只是协同落点。 |
| C3 | Multimodal Interaction-Driven User Intent Perception Model and Application | 2025 | 语音/手势/**眼动**三通道融合做交互意图识别，沉浸式环境，7 人用户研究。可引作"眼动作为协同交互通道"的 CSCWD 先例。 |
| C4 | SD-HOSNet: Hand-Object Segmentation with Single and Dual-Hand Interaction Relation Modules | 2026 | 第一人称手-物交互分割，EgoHOS。"看什么"与"抓什么"在同域，可作为第一人称输出空间的对照。 |
| C5 | Fast Few-Shot Key Gesture Spotting Tool with Fewer Sensors and Frames | 2025 | VR 头显自身传感器做少样本手势触发：轻量化+可部署的写法可参考。 |
| C6 | Multi-Features Fusion Scene Classification Model Based on Body Skeleton and Facial Keypoints Recognition Over Large-Scale Crowd Videos | 2025 | 大规模人群视频里融合骨架/面部关键点做场景分类，多人特征聚合思路。 |

链接：

- C1 https://doi.org/10.1109/cscwd64889.2025.11033422
- C2 https://doi.org/10.1109/cscwd68734.2026.11582520
- C3 https://doi.org/10.1109/cscwd64889.2025.11033301
- C4 https://doi.org/10.1109/cscwd68734.2026.11582443
- C5 https://doi.org/10.1109/cscwd64889.2025.11033542
- C6 https://doi.org/10.1109/cscwd64889.2025.11033655

## 4. 不算相关、但容易误命中的（排除理由）

- Stereo Gaussian Fusion for Naked-eye 3D Display（2026）：裸眼 3D，非注视。
- Glance and Gaze（2025）：只借"gaze"作词，3D 形状分类。
- Trust the Eye or the Crowd?（2026）：MLLM 从众性。"eye"是比喻。
- EYEs（2026）：对抗样本检测框架缩写。
- Precise Segmentation of Eye Muscle Area in Sheep（2025）：兽医学分割。

## 5. 对 CSCWD 投稿的判断

1. **检索范围内未命中直接同题**：既有 1181 篇标题/摘要扫描未找到 joint/shared attention、gaze following 或第三视角视线目标定位；该检索结论不外推为整个领域没有相关工作。
2. **风险**：没有同题论文意味着没有本地审稿人池，6 页里必须用一整段把"多人协同为什么需要细粒度共享注视"讲成立体（DNE-Net 的第一人称局限 + 会议场景仍靠人工规则 这两条可以当靶子）。
3. **可用的锚点**：B1（会议 eye contact）+ B3（眼动 VR 系统）+ C2（视频时序）说明会议中存在协同场景视觉感知的相关应用。不能据这几篇概括全领域手段都粗粒度、手工或离线；方法现状另按文献原文说明。
4. 上述论文只读到摘要层面；若要把数字写进 related work，需先取全文核对。

## 6. 未覆盖与待补

- 2024 届（CSCWD 2024，`10.1109/cscwd61419.2024.*`）未纳入，如需三年窗口可再拉一次。
- 摘要缺失条目（如 SynaLumi）未做正文核对。
- 原始数据：`_cscwd_2025_2026.json`（OpenAlex works + 还原摘要）。
