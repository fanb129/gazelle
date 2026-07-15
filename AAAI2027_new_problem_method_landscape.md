# Gazelle AAAI 2027 新问题与方法方向：从平均精度竞争转向可选择、可校准的可靠 gaze following

## 摘要

本报告从零开始检索并核验了 2023–2026 年 gaze following、video attention target、multi-person/social gaze、open-world/OOD、semantic conditioning、temporal reliability 与 foundation model adaptation 相关工作。在不读取 Gazelle 旧实验的第一阶段，文献显示：multi-person decoding、social gaze、object/semantic target、VLM cue、promptable subject selection、frozen VFM decoder、LoRA/MoE adaptation 等格子已在 2024–2026 年迅速拥挤；相反，主流系统仍默认对每个样本给出 gaze point/heatmap 和 in/out 判断，却很少回答“模型何时不可信、拒答后风险能否下降、这种可靠性在 domain shift 下是否保持”。独立阶段得到两个最强空白：selective gaze following 与 event-aware temporal reliability。第二阶段才读取仓库 P0/P1a/P2/P3：P0 否定 GGSF 与 person-adaptive SASA 叙事，P1a 否定 person-conditioned hierarchy router，P3 的 fusion refinement 全部未胜 baseline；更关键的是，P2 已发现 transition degradation，却因有效切换样本不足而按预注册门槛 No-Go。综合文献 novelty、现有数据、单卡 RTX 3090 约束和仓库负结果，本报告推荐 **Shift-Aware Selective Gaze Following**：在不改变现有 gaze predictor 的前提下，联合估计 localization/visibility risk，并允许 abstention；用 risk–coverage、failure detection、calibration 与 worst-domain risk 取代只看平均 AUC/L2/AP 的评测。该方向无需新人工标注，首先可用现有 VAT 预测和一次无训练推理在一天内证伪。

## 1. 研究问题

本调研冻结三个研究问题：

- **RQ1：** 2023–2026 的 gaze following / VAT 文献中，哪些“能力—评测”单元已被填满，哪些仍能构成明确而非模块拼装式的新 problem setting？
- **RQ2：** 哪些缺口无需新数据或人工标注，可由 GazeFollow、VAT 及已有公开扩展数据支撑，并能在单卡 RTX 3090 上建立可信 baseline、metric 和 falsifiable pilot？
- **RQ3：** 在把仓库 P0/P1a/P2/P3 作为排除项后，哪个方向仍有 AAAI 级别的 problem + method + evaluation 贡献强度？

核心判断是：**2026 年最稀缺的已不是更强的 feature fusion，而是对 gaze prediction 风险本身的建模与评测。** 这一判断不是从旧 Gazelle 叙事推导，而是先由独立文献空格得到，再被仓库负结果加强。

## 2. 调研方法

### 2.1 检索视角

调研采用五个相互独立的检索视角：

1. gaze following / VAT 的主流 architecture、dataset 与 metric；
2. multi-person、joint/shared attention、social gaze 与 head-target association；
3. open-world、OOD、cross-dataset generalization、uncertainty、calibration 与 failure detection；
4. semantic/object-aware、open-vocabulary、VLM cue 与 promptable gaze；
5. temporal modeling、gaze shift、streaming/causal inference 与 foundation model adaptation。

每个视角均先做宽检索，再沿 dataset、method 名称和引用链做窄检索。优先使用 CVF、NeurIPS、ECCV、AAAI、OpenReview、arXiv 和作者项目页。纳入条件是：标题、作者、年份和本文引用的主张至少可由论文摘要或正文核验。无法确认的工作不进入结论。

### 2.2 时间范围与边界

重点覆盖 2023–2026；为说明 task 定义，保留少量早期基线，如 VAT 2020。这里的“空白”严格解释为“本次系统检索未发现直接工作”，不声称绝对不存在。appearance-based 3D gaze estimation、egocentric gaze prediction 和 scanpath prediction 仅作为邻域证据，不与 third-person scene-level gaze following 混为同一任务。

### 2.3 先文献、后仓库

候选方向先在不读取 P0/P1a/P2/P3 的条件下形成。之后才读取：

- `AAAIScripts/AAAI2027_DIAGNOSTIC_AND_PILOT_PLAN.md`；
- `AAAIResults/P0/`、`AAAIResults/P1a/`、`AAAIResults/P2/`、`AAAIResults/P21/`；
- `AAAIResults/P3/screen/leaderboard.csv` 与各 run manifest。

这保证仓库旧叙事只作为排除项和经验约束，而不是 idea 生成起点。

## 3. 文献分类：已经拥挤的方向与仍为空的评价维度

### 3.1 从 point heatmap 到 distribution、object 与 semantic target

VAT 将静态 gaze following 扩展到视频和 out-of-frame 判断，奠定了逐帧 AUC/L2/AP 评测范式 [1]。PDP 不再把 in/out 当作完全独立的二分类，而是把画内 patches 与 outside token 合成 patch-level distribution，显式讨论单标注和 annotation variance [2]。与此同时，Object-aware Gaze Target Detection 联合输出 gaze point、gazed object 类别与位置 [4]，Semantic Gaze Target Detection 进一步在 GazeFollow/GazeHOI 上建立 localization + recognition benchmark [7]，GazeSeg/PixelGaze 类工作又把输出推进到 pixel mask [21]。

这些工作共同说明：**“把 point 改成 distribution / object / semantic mask”本身已不足以构成新主线。** PDP 处理 ambiguity，却不评价 prediction risk；semantic/object-aware 方法提高 target 可解释性，却仍默认总要输出答案。真正未被覆盖的是：当 distribution 多峰、head evidence 缺失或 semantic cue 冲突时，模型是否应拒答，以及拒答是否真的降低错误。

### 3.2 Multi-person 与 social gaze 已从并行推理推进到联合关系预测

Sharingan 用 controlled person token 实现单次 scene encoding 下的 multi-person gaze prediction [5]；GazeHTA 显式评价 head-target instance association [19]；MTGS 则联合预测所有人物的 gaze target、LAH、LAEO 与 shared attention，并构建 VSGaze [6]。更近的统一 VLM/social gaze 工作继续覆盖多人物语义和关系推理 [16]。

因此，**“多人一次前向”“人物 token”“shared scene”“social gaze graph”均不是空格。** 仓库 P0 的 crowded/X-TCR 现象也不能自动转化为新 problem：错误落在其他标注目标附近，可能同时由 head size、target separation、普通 localization error 或 semantic saliency 导致。新的 multi-person idea 若没有独立的关系标签或 counterfactual query protocol，很容易与 Sharingan、GazeHTA、MTGS 重合。

### 3.3 Semantic conditioning 与 promptable gaze 在 2026 年已高度拥挤

2024 年的 VLM cue 工作已用 BLIP-2 等模型抽取 pose、person-object interaction 与 social cue，并观察到 prompt sensitivity 和一定 cross-domain improvement [8]。2026 年的 GazeAnywhere 更进一步定义 Promptable Gaze Target Estimation：用文本或视觉 concept 指定“看谁的 gaze”，同时预测 subject box、in/out 与 gaze heatmap，并用 120K human-in-the-loop concept annotations 训练 [12]。TextGaze 用 LVLM 生成 textual scene cues，OmniGF/GazeVLM 类工作则把 localization、semantic 与 social reasoning 合并 [15][16][22]。

这意味着 generic “Gazelle + text/VLM”或“用语义提示提升 gaze”已是高碰撞方向。仍可能有价值的空格不是再加文本，而是：

- prompt 是否只完成 subject grounding，还是泄露 target prior；
- referent-preserving paraphrase 是否导致 gaze 输出漂移；
- prompt、head cue 与 scene semantics 冲突时，模型是否知道自己不可信。

### 3.4 Foundation model adaptation 已从 frozen decoder 走向 LoRA、MoE 与语义纠偏

Gaze-LLE 证明 frozen DINOv2 + lightweight gaze decoder 可在多个 benchmark 上取得强结果，且只需约 2% 的可训练参数；论文报告训练可在 RTX 4090 上约 1.5 小时内完成 [10]。之后，GazeMoE 用 frozen foundation feature 和 mixture-of-experts 做 cue routing [14]；2026 年的 gaze reasoning 工作直接指出 VFM 会依赖 semantically salient objects，并提出 head-conditioned local LoRA 与 out-of-cone penalty，专门改善 non-salient target [13]；TextGaze 又加入 LVLM textual cue [15]。

所以以下方向应视为碰撞项：generic DINO/CLIP adapter、head-conditioned LoRA、foundation + MoE、单纯 saliency correction、VLM cue fusion。单卡 3090 适合 frozen backbone + 小头，但 **可算不等于新颖**。新方法必须改变问题定义或评价目标，而不能只更换 adapter。

### 3.5 OOD 已有 cross-dataset accuracy，但 gaze-following reliability 仍未成为独立 setting

ChildPlay 直接证明成人主导 gaze 数据对儿童 gaze，尤其 looking-at-face 行为，存在显著 domain gap [3]。Gaze-LLE 报告 GazeFollow 到 VAT、ChildPlay、GOO-Real 的 cross-dataset performance [10]；GazeAnywhere 也在 private child OOD set 上报告 L2/AP [12]。然而这些工作主要回答“平均准确率掉多少”，没有系统回答：

- 失败能否由模型自己的 confidence 检出；
- 在固定 coverage 下，abstention 是否降低 localization/in-out risk；
- confidence/calibration 是否跨 adult、child、retail、video domain 保持；
- OOD score 是否只是识别 dataset identity，而非预测真正的错误。

邻域 appearance-based gaze estimation 已开始严格区分 error–uncertainty correlation 与 calibration，并研究 post-hoc domain-shift correction [17]；Conformal Risk Control 等通用方法也提供风险控制 baseline [18]。但这些不能被写成 gaze following 已解决：third-person gaze following 同时包含 spatial heatmap、in/out、multi-annotation ambiguity 与 person query，输出结构不同。

### 3.6 Temporal architecture 已有，temporal reliability metric 仍不足，但本仓库已有负证据

VAT 2020 和 MTGS 2024 都使用 temporal information [1][6]。MTGS 的 appendix 明确指出：gaze shift 在 ChildPlay 中少于 10%，常规平均 metric 难以反映 temporal benefit，并把新 temporal metric 列为 future work [6]。其结果还显示 temporal model 不一定优于 static model。这为 stable jitter、transition lag、in/out event delay 等指标提供了直接文献动机。

但该空格对 Gazelle 并非全新起点：仓库 P2 已经在 131 个 VAT sequences、31,680 matched transitions/模型上做 transition-conditioned reliability audit。四个模型均出现显著 switch degradation，但 in→out/out→in 只有 238/251 个样本，低于预注册的 500/方向门槛；P2.1 的 120 个 contact-sheet 人工核验尚未完成。因此 literature gap 存在，**但当前 Gazelle 数据证据按项目自己的规则仍是 No-Go**。不能绕过这一负结果重新把 temporal module 包装成首选。

## 4. 文献—问题 taxonomy

| 问题单元 | 代表工作 | 2026 状态 | 对 Gazelle 的含义 |
|---|---|---|---|
| 单人 point/heatmap + in/out | VAT、PDP | 成熟 | 只优化 AUC/L2/AP 不够新 |
| object/semantic/pixel target | Object-aware、Semantic Gaze、GazeSeg | 拥挤 | 不建议只加 semantic head/mask |
| multi-person/head-target association | Sharingan、GazeHTA | 已覆盖 | P0 的 crowded 不能直接变成 binding novelty |
| social gaze / shared attention | MTGS、OmniGF | 已覆盖 | graph consistency 单独贡献弱 |
| VLM cue / promptable subject | VLM cue、GazeAnywhere、TextGaze | 快速拥挤 | 只能研究 leakage、robustness 或 reliability |
| frozen VFM / LoRA / MoE | Gaze-LLE、GazeMoE、HCLoRA | 高碰撞 | generic adapter No-Go |
| cross-dataset/OOD accuracy | ChildPlay、Gaze-LLE、GazeAnywhere | 已有平均精度 | **failure detection / calibration / abstention 为空格** |
| temporal modeling | VAT、MTGS | 已有 architecture | event metric 有 gap，但仓库 P2 样本门槛失败 |
| causal/streaming third-person gaze | 主要只在邻域 egocentric 工作出现 | 检索未发现完整 protocol | 有 novelty，但需新证据或更强视频数据 |

## 5. 仓库 P0/P1a/P2/P3：排除项与经验约束

### 5.1 P0：旧 crowded、GGSF、SASA 机制叙事被否定

P0 的关键结果不是某个模型平均分略高，而是原机制解释站不住：

- GGSF mask mean 为 0.9482、spatial entropy 近 1、inter-person L1 仅 0.0093，几乎是 identity；
- SASA 平均权重明显偏向 L11，但 inter-person L1 仅 0.0009，近似任务级固定深度配方，而不是 person-adaptive routing；
- GazeSpot 相比 baseline 只有 AUC 的 paired CI 未跨 0，L2、X-TCR、association margin 不稳定，in/out AP 反而下降；
- people count 与 L2 退化有独立关联，但 X-TCR 同时受 head size、target distance、target separation 混杂，不能写成已证实的 inter-person binding failure。

约束：不要复活 GGSF，不要把 SASA 改名继续包装，不要把 crowded 当成已被证明的新 problem。

### 5.2 P1a：人物条件 router 学到了动态性，但没有任务收益

P1a 的 person-conditioned residual router 确实把 inter-person layer-weight L1 从 0 提到 0.01407，说明“动态性可被学到”；但 AUC、L2、AP、X-TCR、margin 的 paired CI 全部跨 0，且不同分桶收益方向相反。按冻结判据为 No-Go。

约束：以后不能以“模块学到了非平凡权重”代替任务贡献；任何新 head/router 必须先有独立 failure target 和明确 risk metric。

### 5.3 P2：transition degradation 存在，但样本与有效性不足

P2 对四个模型都观察到 switch−stable Brier、absolute error 和 out→in L2 的显著退化；切换帧 AP 约 0.49，呈 boundary-lag-like pattern。但模型本身是逐帧静态模型，现象也可能来自视觉证据滞后、VAT 标签约定、遮挡/出画或 annotation noise。由于 238/251 低于 500/方向门槛，严格判定 No-Go；P2.1 仍 awaiting manual review。

约束：temporal reliability 只能列为条件候选，不能成为当前第一名；除非先通过 P2.1 或引入足量、可信的外部视频事件数据。

### 5.4 P3：更复杂 fusion refinement 未胜 continued baseline

VAT train-derived validation 的 P3 screening 中：

- continued baseline：AUC 0.93316、L2 0.12682、AP 0.95516；
- residual refinement：三项均退化；
- cross-layer attention：三项均退化；
- residual + coordinate loss：三项均退化。

winner 是 epoch-0 continued baseline。约束：不要继续以 fusion complexity 为搜索轴；优先做 post-hoc、model-agnostic、可证伪的新 setting。

## 6. 候选 idea

### Idea 1：Shift-Aware Selective Gaze Following（推荐）

**Problem setting。** 给定图像/视频帧与 queried person，模型不再被要求永远输出一个看似确定的 gaze point，而是输出：

1. gaze heatmap / point；
2. in/out probability；
3. localization/visibility risk；
4. 在风险超过阈值时 abstain，或输出一个空间 prediction set。

核心问题是：在固定 coverage 下，模型能否稳定降低真实 gaze error；source-domain calibration 能否迁移到 video、child、retail 等 shift。

**与已有工作的实质差异。** PDP 预测 distribution 但不做 selective risk；Gaze-LLE/GazeAnywhere 报告 cross-dataset accuracy 但不评价 failure detection；appearance-based gaze uncertainty 不是 third-person target heatmap + in/out setting。该 idea 改变输出契约和评价目标，而非增加 backbone 模块。

**数据与标注。** 不需要新人工标注。GazeFollow 多标注 test 可评估 ambiguity；VAT 支持 localization + in/out；ChildPlay/GOO-Real 可作真实 domain shift。第一阶段甚至可只复用仓库已有 VAT labels/predictions。

**方法雏形。** 先做 model-agnostic risk decomposition：

- visibility risk：in/out logit margin、Brier-calibrated score；
- localization ambiguity：heatmap entropy、peak width、separated top-mode margin；
- representation novelty：frozen DINO feature 到 source feature bank 的距离；
- input evidence quality：head size/visibility 的 learned embedding，但不能用 GT target metadata；
- video 扩展可加入 motion-compensated inconsistency，但不是首版核心。

用 source validation split 训练很小的 calibrator/risk head，或先用纯 post-hoc score。Conformal risk control 只作为 ID exchangeability 下的 baseline；**不能宣称其保证自动延伸到 OOD**。

**Baselines。** max heatmap peak、entropy、mode margin、in/out margin、temperature scaling、isotonic regression、MC dropout、deep ensemble、feature kNN、selective regression、split conformal/CRC，以及 random/oracle rejection。

**Metrics。** failure AUROC/AUPRC、AURC、risk@50/70/90% coverage、coverage@target-risk、ECE/Brier、prediction-set coverage/area、worst-domain risk、worst-head-size/crowd-bin coverage，并保留 AUC/L2/AP。

**3090 可行性。** 很高。pilot 无训练；完整方法可冻结 Gazelle/Gaze-LLE，只训练百万级以下 risk head。无需重跑 36 小时级 GazeFollow backbone。

**AAAI 强度。** 高，但有条件：必须形成“平均精度相近的模型具有显著不同 selective risk”“标准 confidence 在 shift 下失效”“gaze-specific risk decomposition 跨域改善 AURC”三层证据。只有 entropy + temperature scaling 不够。

**Fatal flaw。** dataset/annotation-protocol shift 可能被当成视觉 OOD；风险头也可能只学会拒绝 small head 或 crowded 样本，造成 coverage bias。必须做 source-only calibration、per-domain separate reporting、worst-group coverage 和不含 GT-derived metadata 的严格输入审计。

### Idea 2：Counterfactual Target-Visibility Consistency

**Problem setting。** 对同一 head-query image 生成 label-preserving 或 label-changing paired views：

- 保留 target 的 crop/resize 下，heatmap 应随坐标变换 equivariant；
- target 被裁出但 head 保留时，in/out 应从 in 变 out，且不应在画内 hallucinate；
- 无关 salient distractor 被抑制/增强时，预测应保持稳定。

**实质差异。** PDP 有 outside token，2026 gaze reasoning 处理 non-salient target，但都没有系统的 paired counterfactual transition protocol。该方向评价的是因果/变换一致性，不只是自然 test set 平均误差。

**数据与标注。** GazeFollow/VAT 的 head box、gaze point 和 in/out 足以自动构造，不需人工标注。自然 VAT in/out transition 可作为 synthetic crop 的外部 sanity check。

**Baselines/metrics。** 普通 crop augmentation、PDP、Gaze-LLE、HCLoRA/OOC；paired transition accuracy、equivariance error、hallucination mass、counterfactual calibration error、原图 accuracy retention。

**3090 可行性。** 很高；benchmark 与 pilot 只需推理，方法可用一致性 fine-tuning 小头。

**AAAI 强度。** 中高。若能证明现有 SOTA 在合法 intervention 下系统失败，并用自然 transition 验证，可能形成新 robustness setting。

**Fatal flaw。** synthetic crop 可能引入 boundary/scale shortcut；“裁出 target”不完全等价于自然 out-of-frame；改变 saliency 也可能合法改变人类 attention interpretation。若没有严格的 invariance 定义和自然数据验证，容易被认为是 augmentation paper。

### Idea 3：Leakage-Free Promptable Gaze

**Problem setting。** 对 GazeAnywhere/Gaze-Co 类 promptable gaze，分离 subject grounding 与 gaze reasoning：referent-preserving paraphrase 不应改变 gaze；同图 subject prompt swap 应正确切换人物；删除 action/pose 后若人物仍唯一，gaze 不应大幅退化；轻微错误属性应触发 uncertainty 而不是 confident wrong answer。

**实质差异。** GazeAnywhere 已证明 promptable subject selection，因此不能再声称首个 semantic-conditioned gaze。新贡献只能是 prompt leakage/ambiguity protocol + semantic bottleneck + selective grounding。

**数据与标注。** raw GazeFollow/VAT 不含文本，单独不支持；公开 Gaze-Co 来自 GazeFollow/VAT/ChildPlay，通常无需新增人工 gaze 标注。paraphrase/swap 可程序生成，但需确保 referent 唯一。

**Baselines/metrics。** GazeAnywhere、OVD+Gaze-LLE、CLIP grounding；subject AP、`L2 | correctly grounded`、paraphrase consistency、swap accuracy、action-removal gap、prompt-risk calibration。

**3090 可行性。** 中等。不能复现 4×H100 的 DINOv3-L 全规模训练，但 frozen CLIP-B/L + 小 semantic bottleneck 可做。

**AAAI 强度。** 中等。只有发现 headline gain 明显依赖 target-leaking prompt，且新方法在 subject-only prompt 上保持 gaze accuracy，才足够强。

**Fatal flaw。** GazeAnywhere 已做 prompt-component ablation；appearance-only 本身较强。reviewer 也可能把它视为普通 prompt robustness，而不是 gaze 核心问题。

### Idea 4：Causal Event-Aware Video Gaze Following（条件候选）

**Problem setting。** 在时刻 t 只能用当前与过去帧，输出 gaze heatmap/in-out；分别评价 stable phase jitter、target-trajectory transition lag、old-target persistence 与 in/out event delay。

**实质差异。** VAT/MTGS 问“temporal feature 是否提高逐帧分数”；本 idea 问“系统是否稳定且不过度迟滞”，并强制 causal/streaming protocol。

**数据与标注。** VAT 理论上可自动派生；GazeFollow 只能做静态预训练；ChildPlay/VSGaze 可补充 gaze-shift 验证。无需新 gaze 标注，但事件有效性可能需要人工 QC。

**Baselines/metrics。** framewise Gazelle/Gaze-LLE、EMA、median、Kalman、Chong temporal、MTGS、causal-mask MTGS、future smoother oracle；residual jitter、ShiftDelay、transition integral error、in/out event F1/delay、accuracy–stability Pareto。

**3090 可行性。** 高；冻结 backbone，预缓存 feature，只训 2–5M causal adapter。

**AAAI 强度。** 文献上高，但对本项目当前仅为条件候选。

**Fatal flaw。** 仓库 P2 已按预注册门槛 No-Go：238/251 个切换不足，P2.1 未验证 annotation/track/scene-cut validity。继续做方法会违反项目自己的证据纪律。除非 P2.1 通过或换用足量外部视频数据，否则不应主推。

## 7. Novelty / feasibility / fatal flaw 对比

| 排名 | Idea | Novelty | 无新人工标注 | Raw GF/VAT 支持 | 3090 | 可信 baseline/metric | AAAI 潜力 | 主要 fatal flaw |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Shift-Aware Selective Gaze Following | 4.5/5 | 是 | 强 | 强 | 强 | 高 | OOD 下 calibration/coverage 不保证；可能只学 dataset 或 group bias |
| 2 | Counterfactual Target-Visibility Consistency | 4/5 | 是 | 强 | 强 | 强 | 中高 | synthetic artifact 与 invariance 合法性 |
| 3 | Leakage-Free Promptable Gaze | 3.5/5 | 通常是 | 需 Gaze-Co | 中 | 强 | 中 | 已有 component ablation；可能只是 prompt robustness |
| 4 | Causal Event-Aware Video Gaze | 文献 4.5/5；项目 2/5 | gaze 标注无需新增，但需 QC | VAT 样本不足 | 强 | 强 | 条件式高 | P2 预注册 No-Go，transition validity 未确认 |

不推荐作为独立主线的方向：generic multi-person router、social graph consistency、open-vocabulary target classification、foundation adapter/LoRA/MoE、static multi-layer fusion。它们要么已有直接强工作，要么被 P0/P1a/P3 否定。

## 8. 推荐第一名：Shift-Aware Selective Gaze Following

### 8.1 推荐理由

1. **与已有工作边界最清楚。** 现有论文有 distribution、cross-domain accuracy、appearance-gaze uncertainty，但未形成 scene-level gaze following 的 selective risk setting。
2. **与仓库负结果正交。** 不需要复活 GGSF/SASA/router，也不依赖 P2 稀少 transition；P3 越证明 fusion 难以带来收益，post-hoc/model-agnostic reliability 越合理。
3. **无需新人工标注。** 失败标签可由已有 gaze point/in-out GT 自动计算；多个公开 domain 已存在。
4. **单卡友好。** 第一个关键结论不需训练；完整方法只需 frozen feature + 小 risk head。
5. **容易证伪。** 如果 confidence 无法排序错误、拒答不能降低风险或跨域排序反转，就应立即 No-Go。

### 8.2 建议的论文贡献结构

若 pilot 通过，可把 AAAI 贡献组织为：

1. 定义 Selective Gaze Following：联合 spatial localization、visibility 与 abstention/prediction set；
2. 建立 model-agnostic reliability protocol：failure detection、risk–coverage、calibration、worst-domain/group coverage；
3. 提出 gaze-specific decomposed risk estimator，而不是通用 entropy calibration；
4. 在 GazeFollow、VAT、ChildPlay、GOO-Real 上证明平均 AUC/L2/AP 与可靠性排序并不等价。

注意：如果只有第 1、2 点而没有方法或强 empirical inversion，AAAI 强度可能不足；如果只有第 3 点而没有新 protocol，则会像普通 calibration transfer。

## 9. 最多一天、无需大规模训练的 Go/No-Go pilot

### 9.1 目标

只回答一个问题：**现有 Gazelle 输出中是否存在无需改 backbone 就能可靠排序的“会错样本”，且选择性拒答是否改善 localization，而不仅是过滤 out-of-frame？**

### 9.2 输入与计算

- 直接复用 `AAAIResults/P0/vat_spot_full.records.csv`、P1a 两组 records 和对应 VAT sequence split；已有 31,978 queries/模型。
- 做一次无训练 inference dump，额外保存每个 heatmap 的：最大 peak、normalized entropy、effective area、空间分离后的 top-1/top-2 mode margin；同时保留 in/out logit margin。
- 不启动 GazeFollow/VAT 训练；如已有 checkpoint/data，一张 3090 只做推理。post-hoc logistic/isotonic 只在 VAT train-derived validation sequences 拟合，test 完全隔离。

### 9.3 Failure 定义

预先冻结两个 endpoint，避免只挑好看的阈值：

- visibility failure：in/out classification 错误；
- localization failure：GT in-frame 且 normalized L2 > 0.15。

联合 failure 为二者 OR，但必须同时单独报告 in-frame localization，防止所有 gain 只来自 out-of-frame。

### 9.4 Baselines

1. random rejection；
2. in/out margin；
3. heatmap peak；
4. heatmap entropy；
5. top-mode margin；
6. 简单线性/logistic risk score；
7. isotonic calibration；
8. oracle error ranking，仅作为上界。

第一天不做 MC-dropout、ensemble 或新 neural head，以免把 pilot 变成训练实验。

### 9.5 指标与统计

- failure AUROC/AUPRC；
- AURC 与 excess-AURC（相对 random）；
- 50%、70%、90% coverage 下的 in-frame L2 risk；
- visibility ECE/Brier；
- crowd、head-size、target-distance bins 的 coverage 与 risk；
- 2,000 次 sequence-cluster bootstrap 95% CI；
- 在至少三种已有 Gazelle variants 上复现，避免只对单 checkpoint 有效。

### 9.6 Go 判据

以下同时满足才 Go：

1. 对 **in-frame localization failure**，最佳非 oracle score 的 AUROC 95% CI 下界 > 0.60；
2. 70% coverage 时，in-frame L2 相对 random rejection 降低至少 20%，bootstrap CI 不跨 0；
3. 改善在至少三个模型、以及至少两个 head-size/crowd bins 中同方向；
4. 联合 score 显著优于仅用 in/out margin，证明不是换名重做 out-of-frame classification；
5. risk–coverage 曲线基本单调，没有靠拒绝单一小群体制造总体收益。

### 9.7 No-Go 判据

任一项出现即 No-Go 或降级：

- 所有 confidence 对 localization failure 的 AUROC 接近 0.5；
- 选择性收益只来自 out-of-frame error；
- score 在不同模型或 head-size/crowd bins 中方向反转；
- 70% coverage 的改善低于 10%，或 CI 大幅跨 0；
- 简单 confidence 完全足够且没有 gaze-specific failure decomposition 的空间，此时最多是一篇 benchmark note，不足以支撑 AAAI 方法论文。

### 9.8 Pilot 后的分流

- **Go：** 再做 GazeFollow source calibration → VAT/ChildPlay/GOO-Real cross-domain evaluation，并设计百万参数以下 risk head。
- **No-Go：** 转向 Idea 2 的 counterfactual visibility pilot；不要回到 router/fusion 搜索。
- **Temporal 复活条件：** 只有 P2.1 人工核验通过，或获得足量外部 transition data，才重新考虑 Idea 4。

## 10. 自我对抗审查与限制

1. “未发现 selective gaze following”是检索结论，不是绝对不存在证明；投稿前应对 2026 下半年新论文再做一次 targeted update。
2. GazeFollow/VAT 的 annotation protocol 不同，cross-domain reliability 可能混合视觉 shift 与标签 shift；报告必须分开解释。
3. selective prediction 的价值取决于应用是否允许 abstention。论文需要给出 human-in-the-loop 或 downstream decision 场景，而不能假设拒答天然有用。
4. conformal guarantee 依赖 exchangeability；OOD 实验只能报告 empirical coverage，除非引入额外假设或 target calibration data。
5. 当前推荐来自文献空格与已有负结果；真正决定是否投入 AAAI 的应是第 9 节 pilot，而不是这份报告本身。

## 11. 对研究问题的回答

**RQ1：** 2023–2026 已填满 multi-person、social gaze、semantic/object target、promptable subject、frozen VFM、LoRA/MoE 等多数 architecture 格子；仍明显不足的是 prediction risk、abstention、calibration 与 event-level temporal reliability。后者对 Gazelle 已被 P2 的样本门槛限制。

**RQ2：** Selective Gaze Following 最符合无新标注、raw GazeFollow/VAT 支持、3090 可完成、baseline/metric/pilot 可证伪的要求。Counterfactual visibility 次之；promptable leakage 需 Gaze-Co；causal temporal 当前受 P2 约束。

**RQ3：** 在排除 P0/P1a/P2/P3 后，最有 AAAI 潜力的是 Shift-Aware Selective Gaze Following。它必须以新输出契约、可靠性 protocol、gaze-specific risk decomposition 和跨域系统实验形成完整贡献；如果一天 pilot 不能证明错误可排序和风险可降低，应立即 No-Go。

## 参考文献

[1] Eunji Chong, Yongxin Wang, Nataniel Ruiz, James M. Rehg, “[Detecting Attended Visual Targets in Video](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Detecting_Attended_Visual_Targets_in_Video_CVPR_2020_paper.html),” CVPR, 2020.

[2] Qiaomu Miao, Minh Hoai, Dimitris Samaras, “[Patch-Level Gaze Distribution Prediction for Gaze Following](https://openaccess.thecvf.com/content/WACV2023/html/Miao_Patch-Level_Gaze_Distribution_Prediction_for_Gaze_Following_WACV_2023_paper.html),” WACV, 2023.

[3] Samy Tafasca, Anshul Gupta, Jean-Marc Odobez, “[ChildPlay: A New Benchmark for Understanding Children’s Gaze Behaviour](https://openaccess.thecvf.com/content/ICCV2023/html/Tafasca_ChildPlay_A_New_Benchmark_for_Understanding_Childrens_Gaze_Behaviour_ICCV_2023_paper.html),” ICCV, 2023.

[4] Francesco Tonini, Nicola Dall’Asen, Cigdem Beyan, Elisa Ricci, “[Object-aware Gaze Target Detection](https://openaccess.thecvf.com/content/ICCV2023/html/Tonini_Object-aware_Gaze_Target_Detection_ICCV_2023_paper.html),” ICCV, 2023.

[5] Samy Tafasca, Anshul Gupta, Jean-Marc Odobez, “[Sharingan: A Transformer Architecture for Multi-Person Gaze Following](https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html),” CVPR, 2024.

[6] Anshul Gupta, Samy Tafasca, Arya Farkhondeh, Pierre Vuillecard, Jean-Marc Odobez, “[MTGS: A Novel Framework for Multi-Person Temporal Gaze Following and Social Gaze Prediction](https://papers.nips.cc/paper_files/paper/2024/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html),” NeurIPS, 2024.

[7] Samy Tafasca, Anshul Gupta, Victor Bros, Jean-Marc Odobez, “[Toward Semantic Gaze Target Detection](https://papers.nips.cc/paper/2024/hash/dbeb7e621d4a554069a6a775da0f7273-Abstract-Conference.html),” NeurIPS, 2024.

[8] Anshul Gupta, Pierre Vuillecard, Arya Farkhondeh, Jean-Marc Odobez, “[Exploring the Zero-Shot Capabilities of Vision-Language Models for Improving Gaze Following](https://openaccess.thecvf.com/content/CVPR2024W/GAZE/html/Gupta_Exploring_the_Zero-Shot_Capabilities_of_Vision-Language_Models_for_Improving_Gaze_CVPRW_2024_paper.html),” CVPRW, 2024.

[9] Qiaomu Miao, Alexandros Graikos, Jingwei Zhang, Sounak Mondal, Minh Hoai, Dimitris Samaras, “[Diffusion-Refined VQA Annotations for Semi-Supervised Gaze Following](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5621_ECCV_2024_paper.php),” ECCV, 2024.

[10] Fiona Ryan et al., “[Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders](https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html),” CVPR, 2025.

[11] Qiaomu Miao et al., “[Multi-view Gaze Target Estimation](https://openaccess.thecvf.com/content/ICCV2025/papers/Miao_Multi-view_Gaze_Target_Estimation_ICCV_2025_paper.pdf),” ICCV, 2025.

[12] Xu Cao et al., “[Gaze Target Estimation Anywhere with Concepts](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_Gaze_Target_Estimation_Anywhere_with_Concepts_CVPR_2026_paper.pdf),” CVPR, 2026.

[13] Shijing Wang et al., “[Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following](https://arxiv.org/abs/2605.22607),” arXiv:2605.22607, 2026.

[14] Zhuangzhuang Dai et al., “[GazeMoE: Perception of Gaze Target with Mixture-of-Experts](https://arxiv.org/abs/2603.06256),” arXiv:2603.06256, 2026.

[15] Junhui She et al., “[TextGaze: Prompting Gaze Target Estimation with Textual Scene Cues](https://arxiv.org/abs/2607.10130),” arXiv:2607.10130, 2026.

[16] Qiaomu Miao et al., “[OmniGF](https://arxiv.org/abs/2605.26399),” arXiv:2605.26399, 2026.

[17] Zheng et al., “[Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_Enhancing_Accuracy_of_Uncertainty_Estimation_in_Appearance-based_Gaze_Tracking_with_CVPR_2026_paper.pdf),” CVPR, 2026.

[18] Angelopoulos et al., “[Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html),” ICLR, 2024.

[19] Zhi-Yi Lin, Jouh Yeong Chew, Jan van Gemert, Xucong Zhang, “[GazeHTA: End-to-end Gaze Target Detection with Head-Target Association](https://arxiv.org/abs/2404.10718),” arXiv:2404.10718, 2024.

[20] Yuqi Hou et al., “[Multi-Modal Gaze Following in Conversational Scenarios](https://openaccess.thecvf.com/content/WACV2024/html/Hou_Multi-Modal_Gaze_Following_in_Conversational_Scenarios_WACV_2024_paper.html),” WACV, 2024.

[21] “[Towards Pixel-Level Prediction for Gaze Following](https://arxiv.org/abs/2412.00309),” arXiv:2412.00309, 2024/2025.

[22] Mathew et al., “[GazeVLM: A Vision-Language Model for Multi-Task Gaze Understanding](https://openaccess.thecvf.com/content/CVPR2026W/AI4RWC/html/Mathew_GazeVLM_A_Vision-Language_Model_for_Multi-Task_Gaze_Understanding_CVPRW_2026_paper.html),” CVPRW, 2026.

[23] Jia Li et al., “[ARGaze: Autoregressive Transformers for Online Egocentric Gaze Estimation](https://arxiv.org/abs/2602.05132),” arXiv:2602.05132, 2026.
