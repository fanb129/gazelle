# ICASSP 2027 当前稿完整审核与修改建议

## 一、总体结论

当前版本相较 ACM MM 2026 投稿稿有明显进步：叙事更克制，已把“物理视锥”降级为“头部条件空间门控”；加入了 GOO-real OOD、头框噪声、总参数量和延迟；也补入了并行工作的 Mi 等人方法，避免继续声称所有指标均为 SOTA。5 页结构符合模板允许的“4 页正文 + 1 页纯参考文献”，正文没有越界，字体均已嵌入。

但以当前状态直接投稿风险仍然较高。最严重的问题不是英文措辞，而是**核心基线与 GazeSpot 使用了不同输入分辨率**：现有实验记录显示基线为 `448×448`，GazeSpot 为 `512×512`。这会同时污染模块消融、拥挤场景、GOO-real、头框噪声和复杂度比较，使“提升来自多层融合与门控”的核心因果结论不够成立。此外，表 1 有一处可核实的文献数值错误，方法与实现细节也存在若干不一致。

建议判定：**有竞争力，但尚未达到可放心提交状态；先解决 P0 问题再润色。**

## 二、P0：投稿前必须修正

### P0-1：统一输入分辨率，重新建立公平基线

论文写道“图像均缩放到 `512×512`”，并写道“在 `512×512` 下，GazeSpot 为 141.1 GFLOPs/22.8 ms，对应基线为 107.2 GFLOPs/18.4 ms”。但复杂度脚本和结果 JSON 明确记录：

- 基线：`448×448`；
- GazeSpot：`512×512`。

这不是只影响时延表述的小问题。当前 OOD、VAT Crowd、头框扰动和最后一层基线均使用 `model_v0` 检查点，而该模型工厂固定为 `448×448`；GazeSpot 工厂固定为 `512×512`。更高输入分辨率会增加 token 网格密度，本身就可能改善点定位，因此不能把全部增益归因于 Multi-level、SASA 或 GGSF。

**推荐修复方案：**

1. 在同一个 `512×512` 代码路径中训练/评估所有消融：
   - Last-layer only + no GGSF；
   - Four-layer raw concat；
   - Four-layer + SASA；
   - Four-layer + SASA + GGSF。
2. 用新的同分辨率 last-layer baseline 重跑：GazeFollow、VAT、VAT 各密度桶、GOO-real、bbox jitter、参数/FLOPs/延迟。
3. 复杂度至少报告两组：
   - **同分辨率 512 公平比较**，隔离模块开销；
   - 如需保留原生配置，再单列 `448 baseline vs. 512 GazeSpot`，明确这是端到端配置比较，不是模块开销比较。
4. 如果算力不足，次优方案是把 GazeSpot 全部改为 `448×448` 并重跑；不要继续混用分辨率却声称是受控消融。

### P0-2：修正表 1 的 Gaze-LLE ViT-B AUC

当前表 1 将 Gaze-LLE（ViT-B）的 GazeFollow AUC 写为 `0.958`。本地保存的 Gaze-LLE 原论文表格为 `0.956 / 0.104 / 0.045`，ACM MM 原稿也写的是 `0.956`。因此应改为 `0.956`，并重新检查下划线标记。Gaze-LLE ViT-L 的 `0.958 / 0.099 / 0.041` 才是正确值。

### P0-3：修正训练细节与真实代码不一致

当前稿笼统写成“训练使用 Adam 和余弦学习率衰减”。实际代码中：

- GazeFollow 使用 Adam + CosineAnnealingLR；
- VAT 使用两个学习率参数组：in/out 分支 `1e-3`，其余网络 `1e-5`，且当前训练脚本没有余弦调度器；
- VAT 的热图 BCE 只在 in-frame 样本上计算。

建议将实现段压缩但准确地写为：

> GazeFollow is trained for 15 epochs with Adam and cosine decay. VAT is fine-tuned for 8 epochs using Adam with learning rates $10^{-3}$ for the in/out head and $10^{-5}$ for the remaining trainable modules. The heatmap loss is applied only to in-frame VAT samples.

同时补充 batch size、所选四个层的索引、checkpoint 选择准则和随机种子。篇幅不足时可删减泛泛解释，不应删复现关键参数。

### P0-4：使方法描述与实现保持一致

当前概述称“线性投影先对齐各层通道维度”，但实现是：四层 DINO 特征保持原通道数，SASA 对它们打分并加权，拼接后只经过一次 `1×1` 投影。图 2 也把 Linear 画在融合之后。应删除“Linear projections align their channel dimensions”或改为：

> The selected DINOv3 layers share the same channel dimension and patch grid; after SASA weighting and concatenation, a single $1\times1$ projection maps the representation to the decoder dimension.

另外，GGSF 实现包含 `BatchNorm2d + ReLU + Dropout + Conv + Sigmoid`，正文只提到归一化和 ReLU。建议明确 dropout，至少在实现细节中给出其值 `0.1`。

### P0-5：收紧对 GGSF 的机制表述

当前版本已明确 GGSF 不是三维视锥，这是正确方向。但还需进一步避免“抑制无关区域”“区分合理目标”这样的强机制暗示。GGSF 的输入只有 $[x-c_x,y-c_y,w,h]$，不包含头部外观、姿态、眼睛朝向或场景内容；对于相同头框几何，它会生成相同的空间门控。因此它本质上学习的是**数据驱动的头部相对空间偏置**，不能知道某个人具体朝哪个方向看。

建议把主要表述改为：

> a learned, head-relative spatial reweighting prior that reduces the influence of some distant or competing regions

而不是暗示它能够独立判别哪些区域真正“无关”。这会让论文更可信，也与 rebuttal 中的降级定位一致。

## 三、P1：显著提高录用概率的修改

### P1-1：把 rebuttal 中最有价值的对照实验放回正文

rebuttal 已包含 raw concat、equal weight、SASA、fixed Gaussian 等对照，但当前 ICASSP 稿只保留组件开关消融。这仍然无法充分回答“是否只是任意多尺度融合/任意坐标条件化”的质疑。

正文第 4 页底部仍有较大空白，建议优先加入一个紧凑控制表：

- Fusion：raw concat / equal weight / SASA；
- Spatial prior：none / fixed Gaussian / CoordConv / GGSF。

如果只能放三行，优先放 `equal weight`、`fixed Gaussian`、`GGSF`，因为它们最直接支撑两个创新点。所有对照必须使用相同输入分辨率、训练预算和种子。

### P1-2：报告多次运行或置信区间

主基准上的增益较小，例如 GazeFollow 平均 $L_2$ 从 0.094 到 0.093，VAT AUC 也低于 Mi 等人。没有多种子结果时，评审容易把小幅差异视为随机波动。

最低成本方案：对 `same-resolution last-layer baseline` 与完整 GazeSpot 在 VAT `heads >= 4` 上运行 3 个种子，报告 mean ± std；如资源允许，再对 GazeFollow 和 VAT 全集重复。至少应为核心拥挤场景结论提供统计稳定性。

### P1-3：完整呈现头框噪声实验，而不是只写 20% 端点

当前正文只写“最多 20% 扰动后为 0.923/0.124”。建议给出 0/5/10/20% 曲线或四行小表，并明确：

- 平移与宽高缩放如何采样；
- 是否裁剪到 `[0,1]`；
- 随机种子；
- 帧数与头部查询数；
- AUC、$L_2$ 和 In/Out AP；
- 基线也使用完全相同的扰动。

目前 10% 噪声下 $L_2$ 反而略有改善，这很可能是单次随机扰动噪声；增加多种子均值后再做“不过度敏感”的结论更稳妥。

### P1-4：增强 Crowd 子集的可复现性

目前只给出按标注头部数量分桶和帧数，尚未兑现 rebuttal 中“发布构建脚本、frame/head IDs、bucket definitions、confound summaries”的承诺。建议至少在正文或公开仓库中提供：

- 固定的 VAT test split 来源；
- 去重单位是 frame 还是 head query；
- 每桶 frame 数与 head-query 数；
- 指标按 frame 还是按 head query 聚合；
- in-frame/out-of-frame 比例；
- 生成 JSON/ID 列表的脚本路径与 commit hash。

摘要中的“at least four people”也应改成“frames with at least four annotated heads”，因为分桶依据是标注头部，而不是完整的人数检测结果。

### P1-5：量化 DINOv3 特征论证，或进一步弱化它

当前稿已写明 PCA 不能作为因果证据，这是良好修正；但图 3(b) 仍占用了宝贵版面，而 rebuttal 中已有 token cosine、boundary separability 和 FG/BG contrast 等量化结果。建议二选一：

1. 加入一行紧凑量化表，支撑“最后一层局部区分性更弱”；或
2. 删除 DINOv2/DINOv3 PCA 小图，把空间留给更强的控制实验，并把动机改成更一般的“last-layer-only may be suboptimal”。

不建议重新强化“DINOv3 过平滑”这一大命题，因为现有任务性能差异仍然很小，且容易引来额外举证要求。

### P1-6：公平报告参数量

当前 caption 虽解释 Params 为可训练参数，但列名仍是“Params”，容易重复触发上一轮质疑。建议直接改为 `Trainable Params`。正文同时报告：

- trainable params；
- total params including the frozen VFM；
- 外部 YOLO/SAM 的成本是否计入 Mi 等人方法。

对于不同论文的参数数字，应标明“reported by the original papers”，避免给人统一复算过的印象。

## 四、P2：写作、图表与版式

### P2-1：图内文字过小，存在模板合规风险

ICASSP 模板要求包括图注在内全文不小于 9 pt。当前图 2 和图 3 内部标签在最终单页尺寸下明显小于正文，部分仅约 4–6 pt；图 2 还嵌入了 Calibri、Microsoft YaHei、Cambria Math 等字体，存在一个全角右括号 `）`，视觉风格不统一。

建议重绘或大幅简化图 2：

- 删除低价值标签和装饰性虚线框；
- 只保留三段主流程及 GGSF/SASA 的关键运算；
- 全部文字使用与正文一致的字体，最终尺寸至少 8–9 pt；
- 把详细层结构移到代码或补充材料。

图 3 中每幅样例太小，几乎无法在打印尺寸下判断热图质量。建议只保留 2–3 个最清楚的样例，或删除 PCA 子图后放大 crowd visualization。

### P2-2：英文细节修正

- `uses neither depth and pose inputs nor ...` 改为 `uses neither depth nor pose inputs, and requires no ...`。
- `with at least four people` 改为 `with at least four annotated heads`。
- SASA 权重实际上同时受图像内容和目标头框门控影响，`image-dependent weights` 可改为 `query-dependent weights` 或 `image- and head-conditioned weights`。
- `the in/out prediction` 建议统一为 `in-/out-of-frame prediction`。
- `Softmax` 在普通行文中可写为 `softmax`。
- GazeFollow 不应表述为“contains only in-frame annotations”而忽略数据集语义差异；更稳妥写法是“the standard GazeFollow protocol evaluates in-frame fixation annotations and does not use the in/out loss”。

### P2-3：进一步压缩贡献表述

当前贡献句是一个长句，且“evaluations that isolate...”在分辨率问题解决前不成立。建议改成三项并列、可验证贡献：

1. head-relative multiplicative gating before multi-level aggregation；
2. query-dependent layer weighting over four frozen VFM levels；
3. same-resolution evaluation on standard, dense-crowd, OOD, and noisy-box settings。

### P2-4：利用第 4 页空白提升证据密度

第 4 页底部仍有约五分之一页面空白，而正文证据又恰好缺少控制实验与复现细节。建议不要仅通过放大段间距填充，而应加入：

- 一个 5–7 行的 strong-control 表；
- 一段可复现设置；
- 或一张 bbox noise 小曲线。

第五页只含参考文献、两栏基本平衡，符合模板要求；字体已全部嵌入，页面尺寸为 US Letter，未发现越界或缺失引用警告。

## 五、对 ACM MM rebuttal 承诺的落实情况

| Rebuttal 承诺 | 当前稿状态 | 结论 |
|---|---|---|
| 弱化 physical frustum / gaze cone | 已明确“not a calibrated 3-D frustum” | 已落实 |
| GOO-real OOD | 已加入表 2 | 已落实，但受 448/512 混杂影响 |
| Head-box noise | 只写 20% 端点 | 部分落实 |
| Latency/FPS/FLOPs | 已写参数、FLOPs、延迟 | 部分落实；分辨率表述错误，FPS 未入正文 |
| Trainable/total params 区分 | 正文已写两者，表头仍为 Params | 基本落实 |
| Gaussian/CoordConv/FPN/Equal-weight controls | 当前正文未呈现 | 未落实 |
| 定量 feature analysis | 仅保留 PCA 与谨慎措辞 | 未落实 |
| 多种子/置信区间 | 未报告 | 未落实 |
| Crowd subset 脚本/IDs/confounds | 只给规则与帧数 | 未落实 |
| ChildPlay | 未评估，但加入了 GOO-real | 可接受，宜在局限中诚实说明 |

## 六、建议的修改顺序

1. 先完成同分辨率基线和四行消融，确认核心增益仍成立。
2. 基于新检查点重跑 Crowd、GOO-real、bbox noise 和复杂度。
3. 修正 Gaze-LLE 数字、训练细节、损失定义和方法实现描述。
4. 从 rebuttal 中挑选最强控制实验放入第 4 页空白处。
5. 加入三种子结果或置信区间。
6. 重绘图 2、放大图 3，并统一字体。
7. 最后再做英文精修和 PDF eXpress/模板合规检查。

## 七、提交前最低检查清单

- [ ] 所有基线和消融使用相同输入分辨率、数据划分、训练预算与评估代码。
- [ ] 表 1 的 Gaze-LLE ViT-B AUC 改为 0.956，并重新检查粗体/下划线。
- [ ] 论文、代码、JSON provenance 中的分辨率和时延一致。
- [ ] VAT 的双学习率、无 cosine scheduler、in-frame-only heatmap loss 写准确。
- [ ] GGSF/SASA 结构描述与 `gazelle/model.py` 一致。
- [ ] Crowd 子集同时给出 frame 数、head-query 数和聚合方式。
- [ ] 至少一个核心比较给出 3 seeds 的 mean ± std。
- [ ] 图内字体达到可读尺寸，不再含全角标点和多套字体。
- [ ] 第 5 页只含参考文献；所有字体嵌入；无 overfull、未定义引用或页码。
