# W1：三份投稿材料的复核与复用决定

日期：2026-11-22。完整读取 MM 9 页、rebuttal 1 页、ICASSP 5 页，共 15 页；渲染检查 MM 第 2 页、rebuttal 全页和 ICASSP 第 2 页，并核对 MM 图 2 的候选源素材。本文是内部写作记录。

## 1. 读取对象与定位

| 文件 | 本轮核对内容 | 对 CSCWD 的结论 |
|---|---|---|
| `rebuttal/2026ACMMM-fanb-final-3.pdf` | 第 1–2 页摘要/动机/贡献；第 2 页图 2；第 3–5 页方法与图 3；第 6–8 页实验、消融、特征分析和结论；第 9 页参考文献 | 已使用 DINOv3、多层特征、GGSF、SASA；“routing”指层间加权，不是实际 token 稀疏执行 |
| `rebuttal/rebuttal.pdf` | 全页：头部条件门控澄清、融合 controls、复杂度、特征统计及 AP 分析 | 已将头部条件化和自适应多层 routing 作为中心贡献；不能把它们仅换名后再次申报 |
| `icassp2027/Fan.pdf` | 第 1 页摘要/贡献；第 2–3 页 HCSG/SASA/decoder；第 4 页表 4、分析与结论；第 5 页参考文献 | 多层访问本身被认为解释大部分收益；raw concat 也在消融中；必须承认复用 |

三份 PDF 与 19 份旧实验 JSON 的路径和 SHA256 见 `../results/WRITING_EVIDENCE_20261122.json`。本轮没有改动原论文 PDF、ICASSP 稿件或旧 `gazelle/` 代码。

## 2. 三稿已经覆盖的技术内容

MM 的标题与主要动机是拥挤场景的表示和空间约束。摘要、贡献和结论均使用多层几何/语义互补、GGSF 和 SASA；第 8 页消融单列多层 raw concat。其第 7 页将可训练参数少与轻量效率相联系，但冻结 backbone 的前向仍需执行。

rebuttal 首段明确把标准头框转为 aggregation 之前的 head-conditioned gating，再用 adaptive multi-layer routing 平衡几何与语义。它还承认最终 gaze direction 需要图像证据，并弱化“精确物理 frustum”的解释。CSCWD 的头框输入不能被描述成已经测得的方向。

rebuttal 的效率表述为其 GazeSpot 与相应 reproduced baseline 的成本比较；不能转用为新 Router 的速度。其 token cosine、boundary separability、FG/BG contrast 属于旧 dense-feature 分析，也不是 sparse route 能保持定位的证据。不同旧实验的配置与分辨率需要回溯原报告；本轮不把它们用于 backbone 优劣的因果结论。

ICASSP 标题直接包含 multi-level fusion，贡献句包含 adaptive multi-level fusion；第 4 页明确写到访问四层解释大部分增益。其第 2 页还已描述同图多人物共享 scene features。因此，DINOv3、多层访问、头部条件化、多人共享场景编码，都不能单独承担 CSCWD 新颖性。

## 3. CSCWD 的可审查差异

| 方面 | ICASSP | CSCWD 候选方法 |
|---|---|---|
| 计算位置 | 完整编码后调制特征和融合层级 | 编码器中途决定 suffix 的位置集合 |
| 输出决策 | soft feature gate / layer weights | fixed-budget token indices |
| 是否跳过 encoder patch 计算 | 未提出 | 选中位置以外不执行后续深层 patch 更新 |
| decoder 网格 | dense 多层特征 | dense early features + selected deep features + intermediate fill |
| 多人机制 | 共享 dense 场景编码 | 聚合多查询 support，在共同预算下共享 sparse suffix |
| 主实证 | 定位与鲁棒性 | 同表示 dense 对照、实际效率、条件化及预算控制 |

当前旧 Router 的 `raw_concat`、关闭 SASA/GGSF 与这条边界一致。是否形成足够独立的新研究还要由最终方法、实验和实际投稿稿件判断；本轮确定实质差异，不把措辞调整当作消除重叠的保证。

## 4. MM 图 2 的具体复用决定

- **识别**：MM PDF 第 2 页图 2；五列样例，三行依次为输入、DINOv2 last-layer PCA、DINOv3 last-layer PCA。
- **候选源文件**：`icassp2027/2026ACMMM/Figure/dinov2VSdinov3_compressed.pdf`。与 MM 页面渲染图目视对应；源文件 SHA 在证据 JSON 中登记。
- **保留**：原始示例、绘图材料和旧稿定位。
- **当前主文决定**：不直接复用其“DINOv3 过平滑 → 多层补救”的标题和论证。其内容不包含浅层方向、support、选择集合或稀疏后输出，不能替代新 Router 的方法证据。
- **新图任务**：制作查询条件改变 support 的图、真实 token 选择和完整执行图。使用真实推理结果时登记 checkpoint 与样本；本轮没有制造模型输出。
- **辅助使用**：如最终需要展示旧 PCA，仅用作注明来源与投影设置的辅助特征观察；是否复用具体图像仍按实际权利和素材条件处理。已拒稿与在审稿覆盖范围分别核对。

## 5. 行文与证据落点

主线、五段中英文 Introduction 工作稿、贡献边界、图表安排和新旧训练协议见 `PAPER_STORY.md`。旧 GF accuracy、旧 split-clean efficiency 和 VAT holdout 明确分开记录：它们不是同一个 checkpoint、同一数据划分或同一选模规则的联合主表。

现有证据支持 GF 中相对自身 dense 的精度–效率权衡；尚未证明相对官方 GazeLLE 的统一测速优势，也未证明多层提高剪枝耐受性或 VAT official-test 精度保持。最终写作根据完成的对照填写具体结果。

## 6. 本轮使用的外部一手来源

- [GazeLLE 官方实现与多人用法](https://github.com/fkryan/gazelle)：冻结 DINOv2 + 轻量 decoder；同图场景编码一次。
- [DINOv3 原始技术报告](https://arxiv.org/abs/2508.10104)：作为第三方预训练编码器的引用，不推出其 gaze 定位必优于 DINOv2。
- [DynamicViT 原论文](https://proceedings.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html)：通用 token sparsification 已有工作，新增主张须限定并做对照。
- [Dynamic Spatial Sparsification](https://arxiv.org/abs/2207.01580)：dense prediction 的快/慢路径与结构化网格保持也有先例，回填本身不单列为首次贡献。
- [ICASSP 2027 editorial policies](https://2027.ieeeicassp.org/about/editorial-policies/) 与 [SPS policy and procedures](https://signalprocessingsociety.org/volunteers/policy-and-procedures)：不得同时提交实质重叠工作；引用/披露不构成实质重复投稿的豁免。
- [官方 GF 训练脚本](https://github.com/fkryan/gazelle/blob/main/scripts/train_gazefollow.py) 与 [官方 VAT 训练脚本](https://github.com/fkryan/gazelle/blob/main/scripts/train_vat.py)：逐 epoch test 的实际行为与最终逐帧 VAT 评测要求。

本轮尝试读取 ACM 的版权/权责页面未成功，因此没有据此给出“所有旧图均可无条件复用”的权利结论。CSCWD 2027 的具体重叠稿件与素材要求仍以会方实际规定为准。
