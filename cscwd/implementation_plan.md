# CSCWD 时序方向 · 实现规格与进展

> 本文档是给 codex 的实现规格 + 进展记录。codex 开工前读，每完成一个任务写回"进展记录"。
> 关联：论文结构 `paper_blueprint.md`，venue 扫描 `cscwd_2025_2026_related.md`。冲突时以本文档为准。
> 当前版本：2026-11-17。方案解释见 [temporal_joint_refinement.md](temporal_joint_refinement.md)。T1–T6 与 R1 已验收；R1 来源/分组与 12 例人工诊断见 `results/r1_local_verification.json`、`results/r1_manual_review.json/.md`。当前下一步 R2，R2–R6 尚未开始；§8 为已完成的 R1 运行记录。

## 0. 目标（一句话）

在冻结 GazeSpot ViT-B VAT 单帧模型之上，训练一个**严格因果、联合输出 heatmap 与 In/Out 分数的轻量时序模块**，结合当前人物/场景视觉证据判断历史注视证据是否仍然有效，改善定位、状态判断、准确稳定性与切换响应之间的折中。

当前规则 STAG（报告中的 `stag`）保留为先导基线；新方案暂用实现标识 `temporal_joint_refiner`，不预先承诺方法名称或性能增益。主研究问题是历史证据的保留、位置更新与释放，不将 EMA、因果推理、冻结骨干或视觉模板本身列为新贡献。

## 1. 技术方案

### 1.1 单帧底座（复用，不改）
- `gazelle/model.py` 的 `GazeLLE`（GazeSpot）：冻结 DINOv3 + GGSF + SASA + decoder，输出 heatmap + inout。
- 用现有 GazeSpot VAT checkpoint 跑前向，**不重训 backbone**。用户于 2026-11-12 确认两套 `epoch_7.pt`，并选择更快的 ViT-B；T2 主缓存及后续单帧/STAG 对照统一使用 ViT-B：
  - ViT-L：`/home/fb/src/paper/gazelleV1/experiments/train_vat_vitl_sasa_ggsf/2026-03-25_14-35-38/epoch_7.pt`。
  - ViT-B：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`。

### 1.2 track 恢复
VAT 原始数据按轨迹组织：`annotations/{split}/{video}/{clip}/` 下每人一个 CSV，文件内即逐帧轨迹。
旧 `data_prep/preprocess_vat.py` 用 `merge_dfs` 拍平，丢了 person_id。T1 已用 `code/data_prep/preprocess_vat_tracks.py` 恢复轨迹。
T1 保留原始 CSV 帧序，头部框使用当前帧原始真实框并裁至图像边界；旧脚本的 11 帧居中框平滑不带入因果时序协议。

### 1.3 已完成的规则 STAG（先导基线）
对每个 track 做因果时序融合。公式（只用 ≤ t 的信息）：

```
每帧 t：
  h_t  = 单帧 heatmap (64×64)
  s_t  = 单帧 inout 概率
  b_t  = head bbox（归一化）

历史：
  H_{t-1} = 平滑 heatmap

门控信号（切换感知，三个可得分量）：
  d_head = ||center(b_t) - center(b_{t-1})||        # 注视者头动了
  trans  = 1[ (s_t>0.5) != (s_{t-1}>0.5) ]          # 画内↔画外翻转
  d_heat = ||argmax(h_t) - argmax(H_{t-1})||        # 预测点跳变

门控权重：
  α_t = sigmoid(w0 + w1·d_head + w2·trans + w3·d_heat)
  # α 高 = 信任当前帧（快速响应切换）；α 低 = 信任历史（平滑抑制抖动）

平滑（heatmap 级 EMA）：
  H_t = α_t · h_t + (1-α_t) · H_{t-1}
  H_0 = h_0

评测点：p̂_t = argmax(H_t)
```

- 严格因果：一切输入只依赖 ≤ t。
- 已跑第一版使用预先固定的 `[-2,20,2,10]`，只修改 heatmap，In/Out 分数原样保留。四参数校准在 R4 中作为对照，不再作为主方法的唯一推进路线。

### 1.4 当前主方案：视觉证据驱动的因果联合修正（待实现）

- **输入**：当前 RGB、当前真实头框、原始单帧 heatmap/In/Out，以及同人物的过去预测和冻结视觉特征；不使用未来 RGB、未来特征或未来预测。
- **表征**：由冻结场景空间特征获得人物区域表示、当前候选区域证据，以及历史预测注视区域的视觉表示；历史区域由模型预测产生。头框中心位移不当作头朝向，In/Out 概率不当作定位可信度。
- **普通基线先行**：先实现同输入的普通因果 GRU 联合头，建立可训练主路径。新增的可学习投影、时序层和两个修正头放在 `cscwd/code/stag/`；原模型及其 decoder 保持冻结。
- **任务机制**：在普通联合头基础上，利用历史目标区域与当前场景的视觉对应、当前人物证据及预测分布，学习历史有效性，控制历史的保留、更新或释放。视觉对应支持移动目标的位置更新；不能仅用旧坐标直接平均来替代这一判断。
- **输出**：每个人每帧的修正 heatmap 与修正 In/Out 连续分数。新状态分支具有改变 AP 的能力；实际是否提高由 R4/R5 结果确定。定位评测继续以 GT 画内掩码为准，不按预测状态删去定位样本。
- **训练**：仅训练新增模块。主损失为当前帧画内 heatmap 监督与全部有效人物帧的 In/Out 分类监督；首版选用热图 MSE 与状态 BCE，不为新增机制预设无标注依据的目标身份/真实切换监督。训练和推理的历史状态均来自模型；不以 GT 热图或状态填充历史。连续片段从空状态开始，截断反传只截断梯度，缺口/跨人物重置。
- **证据边界**：VAT 原始点位移事件仍称 `GT_proxy`，不是目标身份切换真值。普通门控、目标模板与时序记忆已有先例；新增机制的价值通过同输入普通因果头、自适应滤波和机制消融来检验。

### 1.5 指标目标与选模原则

| 指标 | 新方案目标 | 评价要求 |
|---|---|---|
| AUC ↑ / L2 ↓ | 提高或保持定位质量 | 共同 GT 画内样本；不靠预测 In/Out 过滤困难帧 |
| In/Out AP ↑ | 改善画内外分数排序 | 全部有效人物帧；调整判定阈值本身不改变 AP |
| Jitter ↓ / 准确稳定占比 ↑ | 减少错误跳变、持续正确跟随 | 同报 GT 位移及运动片段误差，不能以固定预测获得假稳定 |
| 分型响应率 ↑ / 延迟 ↓ | 减少历史额外引入的滞后和失响应 | 分别报告 `enter_frame / leave_frame / gaze_shift`；配对延迟与删失计数一起报告 |
| 状态错误翻转率 ↓ | 减少虚假的画内外翻转 | R1 固定定义，只在连续且 GT 状态保持的帧对上计分，并同报状态错误率/转换响应 |

R1 固定新增状态指标、候选预算和 validation 选模规则；不填未跑的增益目标数字。R4 按预先固定的定位/状态精度容忍度筛选配置，再比较准确稳定性和分型响应的 Pareto 折中。全部候选及选择依据保留，不在已查看的 test 结果上选参数。

## 2. 项目结构（本次代码/论文全在 cscwd/ 下，不与旧代码混）

```
cscwd/
├── code/                      # 本次时序方向代码（全部新代码放这里）
│   ├── data_prep/             # T1 track 恢复
│   │   └── preprocess_vat_tracks.py
│   ├── stag/                  # 已有基线/指标；R 阶段在这里新增联合时序模块
│   │   ├── __init__.py
│   │   ├── cache.py           # T2 NPY 存储与离线 track 读取
│   │   ├── ema.py             # T5 固定 α 因果对照
│   │   ├── case_selection.py  # T6 确定性案例选择
│   │   ├── plotting.py        # T6 原始热图/轨迹/门控诊断绘图
│   │   ├── gating.py          # 平滑 + 切换门控
│   │   └── metrics.py         # 时序指标
│   └── scripts/               # 入口脚本
│       ├── cache_single_frame.py   # T2 单帧预测缓存
│       ├── run_stag.py             # T3 STAG 推理
│       ├── eval_temporal.py        # T4 时序评测
│       ├── run_ema_ablation.py     # T5 固定 EMA 批量对照
│       └── visualize_temporal.py   # T6 可视化
├── latex/                     # 论文（main.tex + refs.bib + fig/）
├── results/                   # 关键小结果（JSON/MD），git 传回
│   └── RESULT_PROVENANCE.md   # 每个数字→来源运行
├── temporal_joint_refinement.md # 当前方案、动机、训练与证据边界
└── *.md                       # 蓝图/规格/文献
```

- 复用旧代码只 `import` 根目录 `gazelle`（只读，不改）；任何**新代码文件**都落在 `cscwd/code/`。方案与计划 Markdown 放 `cscwd/`。
- 大文件（heatmap 缓存 `.npy`、checkpoint `.pt`）**不进 git**，留服务器；`cscwd/.gitignore` 已忽略。

## 3. 实现任务清单

### 3.1 T1–T6：已完成的先导阶段

以下保留当时的实现规格，完成情况与真实结果见 §6。涉及“后续/可选”的旧描述按 §3.2 的新计划执行；不重新从 T1 开始，也不把旧规则 STAG 当成已训练的新方案。

#### T1 track 恢复
- 新增 `cscwd/code/data_prep/preprocess_vat_tracks.py`（不改旧脚本），产出 `train_preprocessed_tracks.json` / `test_preprocessed_tracks.json`。
- 结构：`sequence → tracks → frames`，每个 track 内按帧序，每帧含 `path / bbox_norm / gazex_norm / gazey_norm / inout / person_id`。
- person_id 来自"每人一个 CSV"这一原生组织（CSV 内顺序即帧序，无需额外匹配）。
- track 另含 `track_id`（`split/video/clip/person_id`）和 `source_csv`；frame 另含原图文件名中的 `frame_index`，保留缺口，不插帧。`gazex_norm / gazey_norm` 沿用单元素列表格式。
- **验收**：脚本能在服务器数据上跑通；track 数与 person CSV 数一致；帧数与原拍平 JSON 一致（总数核对）。

#### T2 单帧预测缓存（服务器）
- 用 GazeSpot VAT checkpoint，沿新 track JSON 跑前向，逐 track 逐帧缓存 `heatmap`（64×64）与 `inout` 概率。
- 缓存为 numpy/parquet，按 track 组织，可离线读取。
- 实现为 float32 `heatmap.npy` / `inout.npy` / `argmax_xy.npy`，沿 `sequence → track → frame` 连续存储；服务器 `index.json` 保存每条 track 的 `[start, stop)`、原始帧记录和文件 SHA256。`stag.cache.load_cache` / `iter_cached_tracks` 只依赖 NumPy，离线按 track 取出预测。
- 使用原模型 transform、GGSF + SASA，全部参数冻结、`eval` + `inference_mode`，不使用 autocast/TF32；模型仅接收当前 RGB 和真实头部框。argmax 坐标按原 `vat_l2` 的 `column/64, row/64` 定义。
- **验收**：缓存覆盖 test 全部 track；单帧 argmax 点 + 单帧 heatmap 可由缓存完全重建（不重新前向）。

#### T3 STAG 门控（后处理级，先跑通）
- 纯 numpy 实现 §1.3 公式，输入 T2 缓存，输出平滑后 heatmap / 点。
- 第一版手调 `w`（建议初始：w1、w3 为大正数触发快速响应，w2 中等，w0 为负偏向平滑）。
- 首轮固定权重 `[w0,w1,w2,w3]=[-2,20,2,10]`，不在 test 上搜索参数。头部中心与热图 argmax 均用归一化坐标计算位移；每条 track 独立初始化，原始帧号不连续时从当前帧重置。
- 仅平滑热图，`inout` 概率保持单帧原值（用于门控，也用于 AP），所以本版不改变 AP。门控只接收预测热图/概率、真实头部框及帧号；真实注视点和画内外标签仅在共用标准指标函数中参与评测。
- `stag/metrics.py` 在 T3 提供标准三指标以满足验收；两侧调用同一个 `evaluate_predictions`，AUC/L2 仅统计画内人物帧，AP 覆盖全部人物帧，逐人物帧汇总。T4 再加入时序指标与独立评测入口。
- **验收**：在 track 上跑通；单帧 vs STAG 的 L2/AUC/AP 有完整对比数字。
- **备注**：若手调门控有正增益，再考虑可学习门控（轻量 MLP，在 track 上拟合），作为消融。

#### T4 评测脚本 + 时序指标
- 标准三件套：AUC↑ / 归一化 L2↓ / In-Out AP↑（VAT test，真实框，纯 RGB）。
- 新增时序指标（**贡献，须诚实定义**）：
  - 帧间抖动 Jitter↓：相邻帧同 track 预测点位移均值/分布。
  - 切换响应延迟↓：切换事件（inout 翻转 / gaze 位移超阈值）后预测重新稳定跟上的帧数。
  - 稳定占比↑：预测连续稳定的帧比例。
- **诚实红线**：抖动降低必须同报**真实注视点位移**作参照（证明不是预测僵死变钝，而是跟上真实轨迹）。
- **口径对齐**：单帧 baseline 与 STAG 必须在**同一个 track 级评测脚本**上重跑，旧拍平数字只作参照、不进时序对比表。

##### T4 时序指标的实现定义（本项目定义，首轮固定）

- 坐标均为归一化 `(x,y)`；相邻仅指同 track 的原始 `frame_index` 差为 1，不跨人物或缺口。阈值在两侧完全相同，不调整 T3 权重。
- **Jitter**：在连续且两帧真实画内的帧对上，统计预测点位移的 mean/median/p90/p95；同时在**完全相同帧对**上报告真实注视点位移分布。按人物帧对汇总，不等权平均 track。
- **稳定占比**：连续 **3 帧**、全部真实画内的窗口，以结束帧为计数单位；若窗口内每个相邻预测点位移均 `≤0.02` 则计为稳定。同口径报告 GT 稳定占比。另报**准确且稳定占比**，要求稳定窗口内每帧预测为画内且到当帧 GT 距离 `≤0.10`。
- **切换代理事件**：连续帧真实 inout 翻转，或两帧均真实画内且 GT 位移 `>0.15`；分别记为 `enter_frame / leave_frame / gaze_shift`。这是由标签定义的代理事件，不另造人工切换标注。
- **切换响应**：从事件帧 `t`（含 `t`）起，寻找第一段连续 **3 帧**预测状态正确且画内定位误差 `≤0.10` 的区间；其起始帧减 `t` 为延迟，可为 0。记录的是成功区间的起始延迟，确认该区间需观察完整 3 帧。
- **观察窗口**：最多 **30 帧**（含事件帧），在下一 GT 代理事件、原始帧号缺口或 track 结束前截断。窗口不足 3 帧记为 `insufficient_followup`；足长但未响应记为 `censored`。按事件类型同报总数、可评估数、响应率、已响应条件延迟分布、未响应数和短窗口数；另在两侧共同响应的同一事件集合上报告延迟差，保留各侧独有响应数。
- `stag/metrics.py` 的 `evaluate_temporal` 只计分已保存预测，标签不进入门控；`scripts/eval_temporal.py` 共用标准/时序评测函数重跑两侧。小 JSON 保存汇总与完整协议，逐 track/事件明细留服务器 JSON 并记录路径/SHA256；无样本统计写 `null`。

#### T5 消融
| 变体 | 平滑 | 门控 | 回答 |
|---|---|---|---|
| 单帧底座 | ✗ | ✗ | baseline |
| 纯 EMA（固定 α） | ✓ | ✗ | 平滑单独贡献 |
| 规则门控 STAG | ✓ | ✓(规则) | 切换感知贡献 |
| 可学习门控 STAG | ✓ | ✓(可学习) | 学习门控是否更优 |

- T4 回传后的首轮 T5：先比较单帧、固定 α EMA、当前规则 STAG。固定 EMA 包含 `α=sigmoid(-2)`（去掉三个门控信号，与当前偏置一致），再预先固定 `α=0.25/0.5/0.75`，全部报告，不从 test 挑选最好一行作为结论。
- 对照仍用当前 ViT-B checkpoint、同一 T2 缓存和 T4 协议，比较精度—稳定性—响应的折中，尤其看相近平滑程度下门控是否比普通 EMA 更快响应。若后续选择参数，使用按源视频划分的 train/validation；test 用于最终报告。
- 可学习门控后置：先确认规则门控相对普通 EMA 的额外收益，再决定学习化；当前不直接铺开 MLP。
- 实现入口 `code/scripts/run_ema_ablation.py` 先核验 T2/T3 缓存和 T4 来源链，原样继承 T4 阈值与协议，重现单帧/STAG 标准和时序汇总后运行四个固定 EMA。新预测缓存、alpha/reset 和逐轨迹/事件明细留服务器，JSON/Markdown 小报告完整列出六种方法；配对延迟分别比较 EMA−单帧、EMA−STAG，独有响应字段为 `reference_only / candidate_only`。

#### T6 定性可视化
- 拥挤多人协同画面（VAT crowd 子集）"谁在看谁/什么"时序可视化。
- 切换场景：一人从看 A 转到看 B，并列 GT、单帧、STAG 和固定 EMA 的预测响应，按实际输出呈现切换速度与滞后。
- 根据 T4 实际结果，同时展示稳定收益段、切换滞后段与失败段；并列 RGB/真实框、GT 轨迹、单帧/STAG 轨迹、热图与 alpha。检查旧峰残留、热图融合和门控响应，定性图的结论以实际片段为准。
- T5 后的选样计划：① STAG 能响应、强 EMA 不能响应；② 单帧能响应、STAG 未响应或滞后；③ STAG 准确且稳定、单帧抖动。失败组将全部 5 个单帧独有 gaze_shift 事件列入候选清单，逐事件记录来源与结果；代表片段使用确定性排序选取。另补充真实标注头数 `≥4` 的拥挤场景，不用检测框。
- 主要并列方法为单帧、STAG、EMA `0.25/0.5`；六种方法的完整折中仍以 T5 表为准。导出轨迹、逐帧定位误差、热图、原始预测 inout、STAG alpha 和三个门控信号，逐轨迹/事件明细从服务器已生成的 `ablation_details.json` 读取。
- T6 用于解释已测到的响应补偿、精度代价与失败机制；完成诊断后再决定一项具体门控改动。当前不立即扩展 MLP 或在 test 上调权重，后续参数选择转到按源视频划分的 train/validation。
- 实现类别为 `stag_over_ema / single_frame_only / stag_lag / stable_gain / crowded`。前两类分别与 T5 的 STAG 相对 EMA `0.25` 独有响应数、单帧相对 STAG 独有响应数核对；`single_frame_only` 全部渲染，其他组按 track_id/原帧号取前 2 例。稳定收益取符合 T4 半径/连续帧阈值且 GT 稳定、STAG 准确稳定、单帧抖动的首个连续命中 episode 窗口；拥挤候选按每 clip 最早满足真实标注人数 ≥4 的帧选择。无候选组保留 0，不替换或放宽条件。
- 上下文默认前 10/后 20 帧，裁至同一 track 的连续原帧号片段。默认 5 个快照优先保留事件帧及前/后 1 帧，再取上下文端点；离线图展示事件后帧仅用于诊断，预测仍为原始因果缓存。事件 status/延迟与实际计分窗口一并标出，GT 画外时不画落点、不计算定位误差。
- 每例保存四方法 RGB/真实框/原始热图图、6 面板时间线和逐帧 JSON。热图统一原始 `[0,1]` 色标，显示透明度 `0.38`；不混入 GT、不修改预测。完整图默认 140 dpi 留服务器，小预览 70 dpi；预览默认覆盖各可用类别的首例，最多 5 例（每例两图），写 `results/t6_previews/` 随小报告回传。完整候选清单与全部失响应案例不受预览配额截断。

### 3.2 R1–R6：当前执行计划（R1 已验收，R2–R6 待开始）

每次只推进当前一个任务：本地实现与合成检查 → 用户在服务器手动运行 → 小结果经 git 回传 → 完成验收并更新本文档/`PROGRESS.md` → 下一任务。R1 真实结果与人工审核已完成；R2–R6 的文件名仍为规划产物。需要服务器运行的入口在相应任务实现完成后，给出 GPU、Python、全部参数、日志和输出路径齐全的 nohup 命令。

#### R1 数据分组、单帧诊断与评测协议

- **依赖**：复用 T1 轨迹和已确定的 ViT-B checkpoint；不依赖新的时序模型。
- **工作**：以 VAT 原始 train 为开发数据，按标注 `video` 字段及其可追溯源视频身份分组，固定 seed、组清单与划分比例；同源 clip、人物、相邻窗口不跨 train/validation。记录分组字段实际含义及源视频对应关系，不将 track 数当独立视频数。
- **预测准备**：复用已有 T2 缓存脚本生成 train 单帧预测，使用无增强、冻结/eval 的当前帧前向；缓存只生成一次，再按已固定清单取 train/validation。旧 `t5_calibration_train_cache` 只有实际文件和来源核对后才可复用，当前不标记为已运行。
- **诊断**：在训练组中按固定抽样规则查看稳定段和代理事件，记录“同目标运动 / 相机变化或切镜 / 可辨识的换目标 / 模型误跳 / 无法判定”及原始帧号。主依据为 RGB、GT 与单帧预测；不从已看过的 test 片段选择新信号。人工诊断不自动成为全量真实切换标签，也不用于填充模型历史。
- **协议**：保留 T4 定义；增加连续 GT 状态相同帧对上的预测状态翻转率、全帧状态错误率，以及独立于定位条件的画内外状态转换响应。阈值、分母、短窗/删失处理和版本在训练前固定；原有 joint 响应另列，不改写 T4 旧列。
- **选模计划**：根据开发组分布固定候选范围、精度容忍度和 Pareto 比较规则，写成配置；同时说明按源视频汇总/配对及后续不确定性估计的单位。validation 用于新增模块选模；冻结 VAT 底座此前已在 VAT train 训练，不能称整个流水线未见过这些数据。
- **规划产出**：`results/r1_split_and_diagnosis.json/.md`（划分参数、组名、计数、诊断摘要、协议与选择规则）；完整清单/逐帧诊断留服务器并记路径、SHA256。主路径代码放 `code/data_prep/` 和 `code/scripts/`。
- **验收**：source-video 组交集为空；全量 train 缓存计数对齐 T1 的 **1,033 tracks / 132,563 人物帧**；抽样可复现且来源可追溯；新增指标定义和选模配置已固定。R1 完成后才进入 R2。
- **已实现配置**：默认 validation 按视频组取 `0.2`，seed `20261117`，复用只读 `gazelle.data_splits.grouped_holdout_split` 的稳定 SHA256 划分；默认组是 annotation video 目录，报告注明不是已核实的独立录制 ID，可用显式映射合并同源目录。机器报告检查组/人物/图像隔离，回传后再确认分组语义。
- **诊断抽样**：四类 `gaze_shift_proxy / state_transition / prediction_jump / stable_reference`，每类最多 3 例、每类每 clip 最多 1 例，按 seed+case identity 的 SHA256 排序，不按收益大小排序。上下文前 10/后 20 帧仅供离线观察，裁至同人物连续帧；图含最多 5 个快照和当前头部放大图。标签初始为空、`review_status=unreviewed`，不自动将代理事件填成真实换目标。
- **状态协议**：`state_temporal_v1` 用 `score>0.5` 分类；错误翻转只计 GT 状态保持的连续帧对；状态转换窗口只在下一 GT 状态转换、gap、track 结束或 30 帧上限截断，不因画内 GT 点跳变截断；连续正确 3 帧的起始延迟计分，与 T4 joint 协议分列。
- **选模配置**：`code/stag/development_protocol.py` 固定首轮预算与规则并随结果导出 `selection_config.json`。训练筛选 seed 为 0，确认 seed 为 `0/1/2`；每个可训练变体最多 4 组学习率/状态损失权重组合。精度容忍度为 AUC/AP 绝对下降至多 `0.002`、L2 增加至多 `0.002`，且可评估的 joint gaze_shift/state 响应率不低于单帧；合格配置先按准确稳定占比排序，完整折中全报，无合格项保留单帧参考。以上是预声明的配置值，不是实测收益。新增训练尚未执行。
- **状态区分**：脚本顶层 `passed` 仅表示来源/覆盖/分组与导出机器检查；`stage_status=awaiting_local_review`。真实结果回传后查看全部选中训练案例，另记人工诊断及分组语义，再验收 R1，不据机器成功直接进入 R2。
- **2026-11-17 验收**：结果提交 `d6adba4`，来源/覆盖/协议/计数与 12 张预览核对通过；全部选中案例已记录人工判断，保留无法判定项。接受 40 个 annotation-video/program 组的 32/8 划分，完整统计与边界见 `results/r1_manual_review.md`。原机器报告不回写，人工记录将 R1 标为已验收；下一步 R2。

#### R2 冻结视觉缓存与普通因果联合头

- **依赖**：R1 清单、预测缓存和指标协议。
- **缓存**：在当前帧原模型前向中提取冻结空间特征，通过独立 wrapper/import 接入，不修改 `gazelle`。先用少量 train 图像确认特征层、空间网格、dtype、存储量与读取形状，再将契约写入配置和缓存 index。场景特征按唯一图像保存，track/person 记录引用图像索引，避免为每人重复存全部 DINO 层；全部可学习投影在新模块内完成，不缓存随训练变化的表示。
- **兼容**：原始 heatmap/In/Out 仍保留 float32；与相同设置下 T2 路径核对。新增视觉缓存不等同于旧预测缓存，单独记录 checkpoint、transform、输入和代码 SHA256；所有新方法共享相同视觉输入、缓存精度与初始化来源。
- **普通模型**：先实现最小因果 GRU 联合头，接收当前人物区域/场景表示、原始预测及过去状态，输出 heatmap 和 In/Out 连续分数。保持底座与原 decoder 冻结，采用当前帧 heatmap MSE（仅 GT 画内）和状态 BCE（全部有效帧）。损失权重、hidden size、序列长度与优化参数完整保存为配置，由 R4 在 R1 预算内选定。
- **历史**：以模型预测构造区域与状态，初始为空；训练片段按时间顺序展开，截断反传时只 detach 状态。跨人/缺口重置；validation 按完整连续 track 从空状态递推，不以 GT 或未来帧 warm start。
- **规划产出**：`results/r2_feature_cache.json/.md`（特征契约、覆盖与冻结前向核对）；联合头本地实现/合成检查记录；普通联合头的首次训练配置。缓存 NPY 与完整 index 留服务器，尚不生成新 test 性能结论。
- **验收**：train/validation 可按清单读取；prefix 在未来帧变化时保持不变；历史不含 GT；两个输出及损失掩码正确；可学习参数仅属于新模块，缓存特征与底座无梯度；在服务器完成最小训练/验证主路径并回传实际日志摘要。此 smoke run 不选最终配置。

#### R3 历史有效性机制

- **依赖**：R2 普通联合头及同源缓存；以 R1 诊断中的可辨识现象确定首个最小机制。
- **工作**：由过去模型预测热图加权读取历史目标视觉表示，与当前空间特征建立对应；当前头框区域表示提供人物证据。结合对应质量、当前预测分布及人物变化，估计历史是否仍有效，并控制记忆更新/释放；历史目标的位置更新与采用新目标的判断分别保留诊断量。
- **输出**：仍为 heatmap 与 In/Out 分数；两个分支都能利用当前证据与选择性历史。保存历史有效性、对应位置和更新强度，供后续解释。推理不使用 GT 点、状态、目标 ID 或事件标签；原始 heatmap 不是自动校准的概率分布，区域读出使用显式定义的非负归一化权重。
- **监督**：首版使用 R2 相同的当前帧主损失；不把“保留/更新/释放”直接包装成已有三分类真值。若 R1 支持新增辅助监督，先在配置中写清标签构造与仅依赖当前/过去标签的规则，再纳入 R4 对照。
- **公平性**：普通因果头与新机制共享可得输入和训练数据，记录参数/计算差异。去掉“选择性历史”时仍保留视觉输入，避免把增加输入量的收益归因于门控。
- **规划产出**：`results/r3_mechanism_design.md`（首版结构、字段、训练配置和与普通头的差异），本地因果/梯度/轨迹状态检查记录；机制图与代码只描述已实现结构。
- **验收**：保留/位置更新/释放路径能在合成序列上分别触发并可解释；两个分支输出有效；符合 R2 的冻结、prefix 与重置要求；结构和候选配置固定后进入 R4。此项不以合成检查证明真实切换识别能力。

#### R4 新模块训练、validation 对照与消融

- **依赖**：R1–R3 产物。所有训练仅使用开发 train 组标签，validation 用于配置选择；只优化新增模块。
- **对照**：同一冻结 ViT-B 单帧底座、全部 T5 固定 EMA、原规则 STAG、开发集校准的四参数门控、One Euro 坐标滤波、速度自适应热图 EMA、普通因果 GRU 联合头和新机制。坐标滤波只提供坐标/状态类结果，原始热图 AUC 原样注明，不能声称坐标滤波改善了热图；分数保持原值的对照明确 AP 不变。
- **R1 后补充的计划对照（未运行）**：仅用开发 train 标签拟合一项全局严格单调的状态分数校准，固定在校准后分数上使用既定分类阈值。该对照保持 AP 排序，用于区分画外阈值偏置与真正的时序/视觉收益；与原始状态分数都报告。R4 实现前将拟合方式写入新版本实验配置，不修改 R1 已存档的 selection manifest、预算、阈值或结果。
- **消融**：普通当前帧修正头（去历史，同样训练两个输出）、heatmap-only、state-only、joint；在同输入联合头中分别去掉视觉对应、选择性历史。分开回答“多了训练/输入”“用了历史”“历史有效性机制”和“联合状态输出”的贡献，避免只与未训练的单帧底座比较。
- **训练记录**：保存 split/缓存/代码 SHA256、完整参数、seed、各损失、validation 每项指标、checkpoint 与选模依据。关键结论按预先固定的多个训练 seed 检查重复性；seed 数和候选预算先写入 R1/R4 配置，不根据结果追加到某一方法。完整日志/checkpoint 留服务器。
- **比较**：按 R1 的精度容忍度与选择规则保留折中配置，比较相近 AUC/L2/AP 或相近稳定性下的响应。分型响应率、共同响应事件的配对延迟、两侧独有响应数同时报告；使用源视频为单位的配对汇总/重采样，不能把相邻帧当独立样本夸大证据。
- **规划产出**：`results/r4_validation_comparison.json/.md`，含全部已跑候选/消融、实际 seed 和选定配置；`results/r4_selection_manifest.json`，固定代码版本、特征契约、checkpoint SHA256、阈值及选模规则。完整超参搜索明细留服务器并记录来源。
- **验收**：普通联合头和强简单对照齐全；热图与状态两条路径的效应可分辨；选定配置及结论都来自 validation。若主要收益只落在 AP/状态响应，则按这一实际结果收缩贡献；不预写所有指标同时提升。

#### R5 冻结配置后的统一 test 报告

- **依赖**：R4 selection manifest 已固定。按同一特征契约生成 test 视觉缓存，不能为了 test 调整网格、阈值或模块结构。
- **工作**：统一评测冻结配置及所需对照，报告 AUC/L2/AP、Jitter/同 mask GT 位移、准确稳定占比、三类 joint 响应，以及独立状态错误翻转/转换响应。保持旧 T4 协议的列可直接追溯，新增协议另标版本。
- **统计与案例**：使用 R1/R4 已固定的组级配对规则；确定性选择收益/失败案例，展示原始预测、普通联合头、新机制与强滤波对照。区分目标运动、换目标和状态变化，无法判定的片段保留该标签。
- **规划产出**：`results/r5_test_comparison.json/.md`、小预览与 provenance。checkpoint、预测/特征 NPY、完整候选/逐帧明细只记服务器路径与 SHA256，不覆盖 T1–T6 报告。
- **验收**：test 配置与 R4 manifest 一致；覆盖与分母对齐；连续分数 AP 不依赖分类阈值；两输出和分型行为均有完整结果。已有 test 曾用于 T1–T6 探索的事实写入实验说明，不称从未查看过的全新测试集。报告后不再围绕 test 调参。

#### R6 论文落稿与协作证据

- **依赖**：R5 的真实结论。按 `paper_blueprint.md` 写入 `cscwd/latex/`，方法与贡献只对应已实现、已跑通的结果。
- **工作**：写清当前/过去信息边界、冻结条件、联合训练、强对照、代理事件与评测口径；One Euro、Chong/PDP/MTGS、ARGaze/EyeTAG 等先行方法用于说明差异，不宣称首次使用时序、门控或目标模板。
- **协作证据**：先核实 VideoCoAtt/VSGaze 等资源是否提供可用标注与同条件人物轨迹，再决定一个有标注的共同关注/交流事件验证；实际运行时单独建立协议和 provenance。仅有 VAT 时，标题与主结论聚焦逐人在线注视目标估计，把协作写为应用动机，不声称已证明协作收益。
- **规划产出**：论文源稿、引用和实际图表；`results/r6_paper_evidence.md` 逐条连接贡献、真实运行与图表。存在下游实验时另存其小报告；未跑的实验不进入结果表。
- **验收**：每个数字和结论可追溯到真实结果；文献报告值与本项目实测分表；主配置统一 ViT-B；摘要按既有约定不放具体数值。

## 4. 数据、路径与 git 工作流

### 路径（服务器，以实际为准）
- 数据集：`fb@3090.lab:/newhome/fb/dataset/videoattentiontarget`
- Python：`/home/fb/anaconda3/envs/py310/bin/python`（help.md 用过；若服务器默认 python 即可，则直接用 python）
- 服务器代码仓库路径：`/home/fb/src/paper/gazelleV1`（用户于 2026-11-12 确认）
- 本机无数据；本机只写代码。

### git 工作流（代码/结果双向传递）
1. 本地改 `cscwd/code/` → commit → push。
2. 服务器 `git pull`。
3. 服务器 nohup 跑（命令由 codex 给出，**用户手动执行**）。
4. 关键小结果（JSON/MD）写进 `cscwd/results/` → 服务器 commit → push。
5. 本地 `git pull` 拿到结果；大文件不传，只在 `RESULT_PROVENANCE.md` 记路径 + SHA。

### nohup 命令模板（codex 给出时套用）
```
CUDA_VISIBLE_DEVICES=<gpu> setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/<script>.py --<args> > <run>.log 2>&1 &
```
- 每个脚本给出：GPU 号、完整参数、日志路径、输出结果路径。
- 结果输出统一写 `cscwd/results/`（或服务器可读路径，小 JSON 再 git 回传）。

## 5. 关键约定

- 严格因果，只用过去帧，不用未来帧。
- 复用 GazeSpot checkpoint，不重训 backbone。
- 纯 RGB、真实框，不进深度/未来帧/检测框设置。
- 所有数字只写真实跑过的；不编数字、不编对照。
- 新代码全放 `cscwd/code/`，旧 `gazelle` 只 import 不改。
- 结果回写本文档"进展记录"区 + 根目录 `PROGRESS.md`。

## 6. 进展记录（codex 逐项写回）

> 每完成一个任务，在此记：任务号 / 完成内容 / 产出的文件与数字 / 下一步。不要写"风险清单"，只写推进状态。

- venue 扫描完成（`cscwd_2025_2026_related.md`）：既有标题/摘要检索范围内未发现直接同题论文，B1/B3/C2 为场景锚点；这不是全领域创新性的证据，方法先行工作按当前方案文档定位。
- **2026-11-12 · T1 本地实现完成**：新增 `code/data_prep/preprocess_vat_tracks.py`，从原生 `.txt/.csv` 人物文件恢复 `sequence → tracks → frames`；保存人物身份、当前帧真实框、注视标签和原始帧号。沿用旧脚本混合正负注视坐标的修正规则，框不作跨帧平滑。
- **T1 产出与数字**：新增 `code/data_prep/test_preprocess_vat_tracks.py`；本地合成数据回归 **6 项通过**，覆盖人物身份/帧序/归一化、未来行变化不影响历史帧、图像帧与人物帧计数、同总数错配、逆序拒绝、CLI 双 split 输出及失败退出。命令：`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/data_prep -p 'test_*.py' -v`。这些是本地测试，未生成 VAT 实验数字。
- **T1 服务器产出**：完整 JSON 写入数据集下 `cscwd_tracks/{train,test}_preprocessed_tracks.json`；小报告写 `results/t1_track_recovery.json`，含轨迹数、唯一图像帧数、人物帧数、逐帧人数核对、输入/输出 SHA256 与运行命令。核对全部通过后，脚本把实际数字及大文件路径/SHA256 追加至 `results/RESULT_PROVENANCE.md`。完整 track JSON 已加入 `.gitignore`。
- **2026-11-12 · T1 结果回传规则修正**：根目录 `*.json` 会忽略小报告；已在 `cscwd/.gitignore` 放行 `results/` 直属 JSON，完整 track JSON、缓存目录和大文件继续忽略。服务器本次可用 `git add -f cscwd/results/t1_track_recovery.json` 加入指定报告后回传。
- **2026-11-12 · T1 服务器验收完成**：已拉取结果提交 `17699cc`，报告 `results/t1_track_recovery.json` 的 train/test 各项核对与总 `passed` 全为 `true`，脚本 SHA256 与本地一致。train：**1,033 tracks / 58,507 图像帧 / 132,563 人物帧**；test：**298 tracks / 13,127 图像帧 / 31,978 人物帧**。原始人物 CSV 数、CSV 行数、旧 JSON 总数及逐帧人数均一致。原始帧号缺口 train 2 处、test 0 处。大文件路径与 SHA256 已记入 `results/RESULT_PROVENANCE.md`。用户授权验收通过后继续，进入 T2。
- **2026-11-12 · T2 本地实现完成**：新增 `code/scripts/cache_single_frame.py`、`code/stag/__init__.py`、`code/stag/cache.py` 与 `code/scripts/test_cache_single_frame.py`；对 T1 全量 track JSON 按原顺序推理，保存原始 float32 预测、对应帧索引及 argmax。NPY 重新打开后按原始值流 SHA256 验证逐值一致，并重算 argmax 核对；完整覆盖并读回通过后才写出缓存索引、小报告和 provenance。
- **T2 本地数字**：**9 项合成回归通过**，覆盖真实 tensor 推理下的冻结/eval 状态、仅当前帧输入、标签不参与预测、未来帧变化与 batch 边界下的历史输出一致性、跨轨迹 batch 与缺口映射、float32 无损恢复、错误 checkpoint/不完整缓存/有损精度拒绝、纯 NumPy 离线读取。命令：`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/scripts -p 'test_*.py' -v`。尚未运行服务器单帧前向，不填模型性能数字。
- **2026-11-12 · T2 底座选择**：按用户速度优先的指示，主缓存切换为 ViT-B `train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；后续单帧 baseline 与 STAG 共用此缓存和 checkpoint。ViT-L 路径保留为已确认配置；未报告两者速度对比数字。
- **2026-11-12 · T2 服务器验收完成**：已拉取结果提交 `24cf720`。`results/t2_single_frame_cache.json` 确认模型为指定 ViT-B GazeSpot VAT `epoch_7.pt`；覆盖 **298 tracks / 13,127 图像帧 / 31,978 人物帧**，原始 float32 heatmap、inout、argmax 全部精确读回，argmax 从缓存热图重建通过。输入 JSON SHA256 与 T1 一致，脚本/缓存实现 SHA256 与 T2 提交 `8e239ba` 一致。报告记录 NVIDIA GeForce RTX 3090，整项缓存作业耗时 **731.430930521 秒**（包含模型加载、推理、写盘与校验）。缓存路径和各 NPY SHA256 已写入 provenance。
- **2026-11-12 · T3 本地实现**：新增 `code/stag/gating.py`、`code/stag/metrics.py`、`code/scripts/run_stag.py` 及配套合成测试。门控纯 NumPy、严格因果；首轮固定 `[-2,20,2,10]`，只平滑热图并保持画内概率。入口核验 T2 源缓存 SHA256，逐 track 生成 STAG heatmap/argmax，并保存 `alpha/signals/reset` 诊断数组；单帧与 STAG 在同一轨迹记录、同一标准指标函数中评测，输出两侧 AUC/L2/AP 及差值。共用缓存写入器增加可选 `task` 标记，默认仍为 T2。
- **T3 本地验证完成**：**19 项 T3 合成测试通过**（门控 9、标准指标 6、CLI/缓存集成 4），覆盖手算公式、严格因果 prefix、预测状态翻转、原始帧号缺口及跨人物重置、旧 VAT 指标函数逐例对齐、读回/来源哈希与完整双侧报告；共用写入器调整后 **9 项 T2 缓存回归通过**。Python 3.10 语法、CLI help、nohup 命令语法与结果忽略规则检查通过。测试仅使用临时合成数据，模型标准指标等待服务器运行。
- **T3 测试命令**：`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/stag -p 'test_*.py' -v`（15 项）；`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_run_stag.py -v`（4 项）；T2 回归：`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_cache_single_frame.py -v`（9 项）。
- **2026-11-12 · T3 服务器验收完成**：已拉取结果提交 `eb28855`，T2 来源报告/index/输入/检查点 SHA256、T3 实现 SHA256、全量覆盖与逐帧对齐通过；两侧均为 **31,978 人物帧 / 20,655 画内定位帧**。单帧 **AUC 0.9401816577151993 / L2 0.09899483221303321 / AP 0.8868010398787617**；初始 STAG **AUC 0.9368598071467544 / L2 0.09919693983541644 / AP 0.8868010398787617**。差值（STAG−单帧）为 **AUC −0.0033218505684449173 / L2 +0.0002021076223832352 / AP 0**。初始规则定位未改善，继续 T4 量化时序行为，保持当前权重。
- **2026-11-12 · T4 本地实现**：`code/stag/metrics.py` 新增固定协议的 jitter/同 mask GT 位移、稳定/准确稳定占比、切换代理事件响应与删失计数；新增 `code/scripts/eval_temporal.py`，核验两侧完整缓存 SHA256 与 T2→T3 来源链，在同轨迹上重算标准三指标并核对 T3 报告，输出两侧时序汇总、共同响应事件延迟差和服务器逐轨迹/事件明细。
- **T4 本地验证完成**：**15 项 T4 合成测试通过**（时序指标 11、CLI/缓存集成 4），包括同 mask GT 位移、窗口分母、准确稳定、缺口/跨人物隔离、代理事件延迟/未响应/短窗、共同响应事件延迟差、无事件 `null` 和来源/明细 SHA256；**10 项回归通过**（标准指标 6、T3 运行入口 4）。Python 3.10 语法、CLI help、服务器命令语法和结果忽略规则检查通过。
- **T4 测试命令**：`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/stag -p 'test_*metrics.py' -v`（11 项新增 + 6 项标准回归）；`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_eval_temporal.py -v`（4 项新增）；`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_run_stag.py -v`（4 项 T3 回归）。本地只使用临时合成数据。
- **2026-11-12 · T4 服务器结果验收完成**：已拉取 `7f15cd7`，报告来源链、代码 SHA256、全量 298 tracks/31,978 人物帧、T3 标准指标重现、帧对/窗口/代理事件对齐和计数关系全部核对通过。时序统计包含 **20,225 对画内相邻帧 / 19,796 个稳定窗口 / 688 个代理事件**，其中 581 个事件可评估、107 个窗口不足 3 帧。逐轨迹/事件明细仍在服务器，路径/SHA256 见报告。
- **T4 真实结果**：单帧→STAG 的 Jitter mean **0.016646765511375713→0.012135128302046163**（降低 **27.10%**），同帧对 GT 位移 mean **0.0059218882811410915**；稳定占比 **77.3439%→86.6741%**，准确且稳定占比 **52.7531%→58.1532%**（增加 **5.40 个百分点**）。全部事件响应率 **59.2083%→58.5198%**；共同响应的 337 个事件平均延迟差（STAG−单帧）为 **+0.13353115727002968 帧**。
- **T4 切换分型与判断**：`gaze_shift` 共 199 个代理事件、158 个可评估，响应率 **58.2278%→55.0633%**；共同响应的 87 个事件 STAG 平均慢 **0.27586206896551724 帧**，单帧独有响应 5 个、STAG 独有响应 0 个。`leave_frame` 结果一致，符合画内概率不变的实现。当前结论为**有稳定性收益，定位略降，切换响应未改善**；尚无普通 EMA 对照，不能将稳定收益归因为门控。两侧全部已响应事件集合不同，不用各自条件均值近似相同来宣称响应加快。
- **2026-11-12 · T5 固定 EMA 对照本地实现**：新增 `code/stag/ema.py`、`code/scripts/run_ema_ablation.py` 和配套合成测试。EMA 严格因果、每条 track 独立初始化、缺口从当前帧重置，画内概率保持原值。四个系数为 `sigmoid(-2)` 与预先固定的 `0.25/0.5/0.75`；六种方法全报，不做 test 选优或训练。入口核验 T2/T3/T4 来源链并重现 T4 参考结果，复用原标准/时序指标函数，保存各 EMA 缓存及诊断数据，并输出每个 EMA 相对单帧和 STAG 的同事件响应比较。
- **T5 本地验证完成**：**13 项合成测试通过**（EMA 核心 9、完整 T2→T3→T4→T5 集成 4），覆盖手算递推、prefix 因果性、缺口/跨人重置、输入不变、`α=0/1`、偏置匹配 EMA 与零门控项 STAG 逐值一致、六种方法全报、T4 参考指标重现、完整缓存/诊断/明细 SHA256、配对响应计数与方向、源缓存不变、来源篡改拒绝和无 Torch 离线运行。Python 3.10 语法、CLI help、服务器命令语法及结果忽略规则检查通过；未改旧模型或评测协议，未填真实 EMA 数字。
- **T5 测试命令**：`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/stag -p test_ema.py -v`（9 项）；`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_run_ema_ablation.py -v`（4 项）。
- **2026-11-13 · T5 服务器结果验收完成**：已拉取结果提交 `3de06cd`，检查六种预先声明方法、T2/T3/T4 来源报告 SHA256、实现 SHA256、T4 协议与参考指标重现、全量覆盖、画内概率文件一致、事件计数和直接配对分区关系均通过；Markdown 与 JSON 完全一致。六种方法均覆盖 **298 tracks / 31,978 人物帧 / 20,655 画内定位帧**，使用同一 **20,225 帧对 / 19,796 窗口 / 158 个可评估 gaze_shift**。完整 EMA 缓存、alpha/reset 及明细仍留服务器，路径/SHA256 见报告和 provenance。
- **T5 真实对照结果**：STAG 的准确稳定占比为 **58.1532%**，六种配置中最高。与 Jitter 接近的 EMA `0.5`（**0.012135 vs 0.011699**）相比，STAG 准确稳定占比高 **2.4348 个百分点**、L2 更低（**0.099197 vs 0.099635**），AUC 更低（**0.936860 vs 0.939690**），gaze_shift 响应率相同（**55.0633%**）；直接共同响应的 **84 个事件**上 STAG 平均快 **0.15476190476190477 帧**，双方各有 3 个独有响应。
- **T5 强平滑对照**：EMA `0.25` 与 STAG 的原始稳定占比接近（**86.6640% vs 86.6741%**）；STAG 准确稳定占比高 **0.8436 个百分点**、gaze_shift 响应率高 **4.4304 个百分点**，直接共同响应的 **80 个事件**平均快 **0.5625 帧**，新增 7 个响应，但 Jitter 与 AUC 不及该 EMA。相对偏置匹配 EMA，STAG 在 **73 个共同切换事件**平均快 **1.4931506849315068 帧**，新增 14 个响应，同时平滑更弱。
- **T5 判断**：当前 STAG 有部分自适应折中收益，未被一个固定 EMA 在上述主要指标上全面支配，但也没有全面优于普通 EMA。EMA `0.75` 的实测 **AUC 0.940120 / L2 0.098941 / Jitter 0.014229 / gaze_shift 响应率 56.9620%**，是强简单对照；仍保留为预先固定的实测配置，不据此选定 test 最优模型。STAG 相对单帧的切换缺口仍存在（87 个共同事件慢 **0.27586206896551724 帧**、少响应 5 个），不把“优于强 EMA”写成“优于单帧”。
- **2026-11-13 · T6 本地实现完成**：新增 `code/stag/case_selection.py`、`code/stag/plotting.py`、`code/scripts/visualize_temporal.py` 及三个配套测试文件。入口核验 T5 全部源报告、六种方法缓存和逐事件明细 SHA256，按已声明的五类规则选样，保留全部单帧独有切换失败案例；四方法展示使用冻结缓存，输出候选 manifest、逐帧数据、原始热图/真实框图、门控时间线及小预览。
- **T6 本地数字与验证**：**15 项合成测试通过**（选样 8、绘图 4、完整来源链/真实 CLI 集成 3），覆盖固定排序/全失败保留/缺口/拥挤计数、事件前后快照、GT 画外 `null`、计分窗口、缓存不变、图片/manifest/源文件 SHA256、预览配额和无 Torch 离线执行。已用临时合成 RGB 检查完整图及小预览，图例、色标、原帧号、事件状态/延迟与窗口标注清楚。Python 3.10 语法、CLI help、服务器命令与结果忽略规则检查通过。真实 VAT 案例尚由服务器产出。
- **T6 测试命令**：`/private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/stag -p test_case_selection.py -v`（8 项）；`MPLCONFIGDIR=/private/tmp/gazelle-t6-mpl /private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/stag -p test_plotting.py -v`（4 项）；`MPLCONFIGDIR=/private/tmp/gazelle-t6-mpl /private/tmp/gazelle-t1-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_visualize_temporal.py -v`（3 项）。合成视觉检查图留 `/private/tmp/gazelle-t6-render/`，不作真实实验材料。
- **2026-11-13 · T6 服务器与本地预览验收完成**：用户已同步结果提交 `9e0fdd0`。来源 T5 报告/明细/六种 cache index、T6 实现 SHA256、全量覆盖和选择计数一致；已核验本地 **10 张预览 PNG** 的 SHA256/大小。五类候选数为 **7/5/11/726/16**（门控收益/单帧独有响应/STAG 滞后/稳定收益 episode/拥挤 clip），完整输出 **13 例**，5 个单帧独有切换失败全部保留。已实际查看五类各一例的帧图与时间线；其余完整图和逐帧 JSON 保留服务器，未声称已逐帧检查全部 13 例。
- **T6 真实观察**：CBS `2487_2818/s01` 的 `2517` 事件中单帧和 STAG 延迟均为 **0**，两个 EMA 在 **3 帧**计分窗口内未形成成功区间，显示门控可避免固定平滑的首帧拖延。CBS `12137_12258/s03` 的 `12144` 事件中单帧 **+8 帧**、STAG **+12 帧**、EMA `0.25` **+13 帧**、EMA `0.5` **+12 帧**，显示初版仍有额外历史惯性。Downton `13066_13217/s02` 的 `13190` 事件中单帧 **+6 帧**，STAG 有短暂低误差段但未连续成功 3 帧而记为 `censored`；该标签不表示从未预测到正确位置。
- **T6 机制判断**：所看切换/滞后片段中 `d_head` 很小、预测 inout 不翻转，alpha 主要受 `d_heat` 驱动。大跳变触发本身不区分可信纠正与噪声回跳，短暂正确观测也会被历史混合延后或缩短；这一现象支持先校准历史权重和响应强度。旧/新热图峰幅值竞争与全部失败例是否同因，仍需源缓存/逐帧数值核验，当前不写成已证实的总体结论。
- **旧下一步（2026-11-13 记录，已由 R 阶段替代）**：曾计划复用 `cache_single_frame.py` 生成 train 缓存，并在视频分组 train/validation 上校准四参数门控。该 train 缓存尚未回传验收，四参数校准转为 R4 对照；保留此记录及原命令作为历史。
- **2026-11-17 · R0 方案与计划落地完成**：新增 `temporal_joint_refinement.md`，将主问题确定为“当前视觉证据下的历史有效性判断”，新增模块联合修正 heatmap 与 In/Out；完整 ViT-B GazeSpot 和原 decoder 冻结。原规则 STAG 与 T1–T6 结果保留为先导基线；AP 旧版不变、新版可学习，实际收益留待训练/评测。
- **R0 产出与数字**：同步本文档 §0/§1/§3.2、`paper_blueprint.md`、`README.md`、`codex_prompt.md`、`datasets_and_metrics.md` 与根 `PROGRESS.md`，补充文献入口；写入 R1–R6 的依赖、输出和验收。本次只更新文档，没有新增模型结果、训练运行或服务器作业，已有结果文件不变。
- **R0 时的下一步（已落实为下述 R1 本地实现）**：固定 source-video 分组、开发集单帧缓存与诊断协议、状态指标和 validation 选模规则；真实结果回传后再做 R1 验收。
- **2026-11-17 · R1 本地实现完成**：新增 `code/data_prep/development_split.py`，复用原只读分组 helper；新增 `code/stag/state_metrics.py`、`development_diagnostics.py`、`development_protocol.py` 与入口 `code/scripts/prepare_development.py`。入口验证 T1 train JSON/覆盖、缓存文件 SHA256、既定 ViT-B checkpoint/backbone/protocol，输出开发分组、标准/joint/state 基线指标、选模配置和训练组诊断图；不训练模型、不读取 test 预测或指标。T2 小报告仅用于校验共享底座身份。
- **R1 产出与本地数字**：**35 项新增合成测试通过**（分组 10、状态指标 11、诊断 8、CLI/来源链集成 6），**17 项原标准/时序指标回归通过**。已实际查看合成 RGB/头部诊断图。测试覆盖标签不影响分组、跨人/缺口隔离、状态响应独立于定位、global slices 分区计分、人工标签未完成、缓存只读、来源不符拒绝及 JSON 空统计。原 `gazelle` 与 T4 指标实现未改；没有新增真实 VAT 数字。
- **R1 本地命令**：`/private/tmp/gazelle-r1-venv/bin/python -m unittest discover -s cscwd/code/data_prep -p test_development_split.py -v`；同解释器执行 `-m unittest discover -s cscwd/code/stag -p test_state_metrics.py -v`（11）、`-p test_development_diagnostics.py -v`（8）、`-m unittest discover -s cscwd/code/scripts -p test_prepare_development.py -v`（6）。原指标回归使用 `-m unittest discover -s cscwd/code/stag -p 'test_*metrics.py' -v`，其中另含已计入新增的 11 项状态测试，不重复计为新增。
- **R1 本地实现时的下一步（已完成下述回传验收）**：按 §8 生成 train 原始预测与 R1 准备报告，回传小 JSON/MD、全部选中小预览及 provenance，再确认分组和逐例人工诊断。
- **2026-11-17 · R1 结果拉取与验收完成**：快进拉取 `d6adba4`，本地原有 provenance 补充已恢复。新增 `results/r1_local_verification.json`，记录来源小报告/9 个代码文件/12 张 PNG 的 SHA256、协议、组划分和计数关系的本地核对；服务器 NPY/index/manifest/逐事件数据未在本地重算。总覆盖 **1,033 tracks / 132,563 人物帧 / 58,507 图像帧**，train/validation 为 **32/8 组、827/206 tracks、98,362/34,201 人物帧**；稳定 seed 重建验证组一致。
- **R1 真实数字与诊断**：validation 单帧 **AUC 0.954206 / L2 0.079605 / AP 0.960923**；准确稳定 **60.19%**，joint gaze_shift **135/221（61.09%）**。状态画外/画内错误 **25.73%/2.77%**，独立进入/离开响应 **179/189（94.71%） / 93/169（55.03%）**；state 与 joint enter 的可评估分母不同。新增 `results/r1_manual_review.json/.md`，全部 12 例已看快照，保留无法判定标签；固定抽样不当作真实事件全量标注。原机器报告不修改。
- **当前下一步（R2，未开始）**：按 R1 manifest 确定冻结视觉特征契约，先核对小头部区域在特征网格中的覆盖，再实现普通因果联合头与最小训练/验证。R1 可接受，无需重跑；本次审核不提前实现或训练 R2。R4 已补记单调状态校准强对照，尚未运行。

## 7. 先导阶段服务器命令存档

T1–T6 的命令对应已完成运行，供复现使用。末尾 train 缓存命令是旧方案尚未验收的准备命令；当前按 R1 重新明确报告命名、划分来源与执行入口，不将本节视为新阶段立即运行的指令。

### T1 服务器手动运行

在服务器仓库拉取本分支后执行。此步骤为 CPU 预处理，命令按模板标明 GPU 0。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/data_prep/preprocess_vat_tracks.py \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_tracks \
  --legacy-dir /newhome/fb/dataset/videoattentiontarget \
  --splits train test \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.log 2>&1 &
```

日志最后输出 `T1 verification passed` 且报告顶层 `passed` 为 `true` 后回传：

```bash
git add cscwd/results/t1_track_recovery.json cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record T1 VAT track recovery results"
git push origin codex/coverage-aware-router
```

### T2 服务器手动运行（ViT-B 主缓存）

本轮使用用户确认的 ViT-B GazeSpot VAT `epoch_7.pt`，GPU 0，batch size 8。原骨干工厂从仓库根目录的 `checkpoints/dinov3_vitb16_pretrain.pth` 加载冻结预训练骨干；脚本同时记录它与 VAT checkpoint 的 SHA256。缓存目录为新目录，一次完整运行生成一个缓存。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/cache_single_frame.py \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --track-json /newhome/fb/dataset/videoattentiontarget/cscwd_tracks/test_preprocessed_tracks.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --model gazelle_dinov3_vitb16_inout \
  --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda:0 \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.log 2>&1 &
```

日志最后输出 `T2 verification passed` 后回传小报告；完整 NPY 和缓存索引留服务器，文件路径/SHA256 自动追加进 provenance。

```bash
git add cscwd/results/t2_single_frame_cache.json cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record T2 ViT-B single-frame cache results"
git push origin codex/coverage-aware-router
```

### T3 服务器手动运行（初始规则门控 + 同口径标准指标）

本步骤只读 T2 缓存，使用 CPU 做纯 NumPy 门控和标准指标计算；按模板标明 GPU 0。权重顺序为 `bias / d_head / trans / d_heat`，首轮固定 `-2 / 20 / 2 / 10`。输出为新目录，完整 NPY 和索引留服务器。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/run_stag.py \
  --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test \
  --cache-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json \
  --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test \
  --weights -2 20 2 10 \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.log 2>&1 &
```

日志最后输出 `T3 run completed` 后，回传小报告与自动追加的指标/大文件路径/SHA256。报告 `passed` 表示缓存与评测流程完成，指标变化见 `metrics.delta_stag_minus_single_frame`。

```bash
git add cscwd/results/t3_stag_comparison.json cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record T3 initial STAG comparison results"
git push origin codex/coverage-aware-router
```

### T4 服务器手动运行（统一标准与时序评测）

只读已有 T2/T3 缓存，CPU 评测，按模板标明 GPU 0；原始 RGB、模型和 STAG 门控无需重跑。所有阈值在下列命令中显式固定。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/eval_temporal.py \
  --single-frame-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test \
  --single-frame-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json \
  --stag-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test \
  --stag-report /home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json \
  --details-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/temporal_rule_v1_test \
  --shift-threshold 0.15 \
  --response-radius 0.10 \
  --stable-threshold 0.02 \
  --hold-frames 3 \
  --max-response-frames 30 \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/t4_temporal_comparison.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/t4_temporal_comparison.log 2>&1 &
```

日志最后输出 `T4 evaluation completed` 后回传小报告；逐轨迹/事件明细 `temporal_details.json` 留在服务器 `--details-dir`，路径与 SHA256 自动追加到 provenance。

```bash
git add cscwd/results/t4_temporal_comparison.json cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record T4 paired temporal evaluation results"
git push origin codex/coverage-aware-router
```

### T5 服务器手动运行（四个固定 EMA + 单帧/STAG 共六种方法）

CPU 后处理与评测，按模板标明 GPU 0；不重新前向模型。`sigmoid(-2)` 由已核验 STAG 报告中的偏置计算，另三个系数显式传入。T4 的所有指标阈值从源报告原样继承，命令不重新设置阈值。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/run_ema_ablation.py \
  --single-frame-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test \
  --single-frame-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json \
  --stag-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test \
  --stag-report /home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json \
  --temporal-report /home/fb/src/paper/gazelleV1/cscwd/results/t4_temporal_comparison.json \
  --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test \
  --ema-alphas 0.25 0.5 0.75 \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.json \
  --summary /home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.md \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.log 2>&1 &
```

日志最后输出 `T5 ablation completed: 6 methods` 后回传三个小结果；四套完整缓存和 `ablation_details.json` 留服务器 `--output-dir`，路径与 SHA256 自动写入 provenance。

```bash
git add cscwd/results/t5_ema_ablation.json cscwd/results/t5_ema_ablation.md cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record T5 fixed EMA ablation results"
git push origin codex/coverage-aware-router
```

### T6 服务器手动运行（确定性选样与案例图）

CPU 绘图，按模板标明 GPU 0；只读已有六种方法缓存，不重新前向或平滑。所有缓存/明细路径从已回传的 T5 报告读取。完整输出目录和预览目录均为新目录。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/visualize_temporal.py \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --ablation-report /home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.json \
  --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics \
  --preview-dir /home/fb/src/paper/gazelleV1/cscwd/results/t6_previews \
  --cases-per-group 2 \
  --preview-cases 5 \
  --context-before 10 \
  --context-after 20 \
  --min-heads 4 \
  --snapshots 5 \
  --dpi 140 \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/t6_visualization.json \
  --summary /home/fb/src/paper/gazelleV1/cscwd/results/t6_visualization.md \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/t6_visualization.log 2>&1 &
```

日志最后输出 `T6 visualization completed` 后回传小报告和低分辨率 PNG。完整 `selection_manifest.json`、`cases/*/case_data.json` 与全分辨率图留服务器，实际路径/SHA256 自动追加到 provenance。

```bash
git add cscwd/results/t6_visualization.json cscwd/results/t6_visualization.md cscwd/results/t6_previews cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record T6 temporal diagnostic cases and previews"
git push origin codex/coverage-aware-router
```

### 旧 T5 校准准备命令：train 冻结预测缓存（未执行验收，非当前默认入口）

复用已经在 test 跑通的 T2 脚本，仅对训练轨迹做无增强、冻结模型的前向，不训练 backbone 或 decoder。train 的轨迹/人物帧计数以 T1 真实报告为验收参照。先生成完整 train 缓存，再做视频组划分，避免重复前向。

```bash
cd /home/fb/src/paper/gazelleV1
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/cache_single_frame.py \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --track-json /newhome/fb/dataset/videoattentiontarget/cscwd_tracks/train_preprocessed_tracks.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --model gazelle_dinov3_vitb16_inout \
  --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/train \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda:0 \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/t5_calibration_train_cache.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/t5_calibration_train_cache.log 2>&1 &
```

输出 `T2 verification passed` 后回传小报告（复用脚本的成功标记仍为 T2），大文件路径/SHA256 自动记录。

```bash
git add cscwd/results/t5_calibration_train_cache.json cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record frozen train cache for temporal gate calibration"
git push origin codex/coverage-aware-router
```

## 8. R1 服务器手动运行（已完成，保留复现命令）

两步顺序执行：第一步日志出现 `T2 verification passed` 且 `r1_train_single_frame_cache.json` 中 `passed=true` 后，再执行第二步。两步都只做冻结前向或离线准备，不训练时序模块。输出使用 R1 新目录，不覆盖 T1–T6 或旧准备缓存。

### 8.1 GPU 0：生成 train 原始单帧预测

沿用已跑通的 T2 脚本，因此成功标记和报告 `task` 仍为 T2，报告文件名标识 R1 用途。输入覆盖以 T1 实际 train 结果为准。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/cache_single_frame.py \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --track-json /newhome/fb/dataset/videoattentiontarget/cscwd_tracks/train_preprocessed_tracks.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --model gazelle_dinov3_vitb16_inout \
  --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda:0 \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.log 2>&1 &
```

### 8.2 CPU：分组、基线指标与训练组诊断

按模板标明 GPU 0，脚本实际不导入模型、不使用 GPU。默认无显式 source-video map，按 annotation video 目录整组划分；小报告列出组名与映射语义，回传后确认。读取 T2 test 小报告只为核对 checkpoint/backbone/protocol 身份，不读取其 test 预测或性能。

```bash
cd /home/fb/src/paper/gazelleV1

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/prepare_development.py \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions \
  --cache-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json \
  --recovery-report /home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.json \
  --reference-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json \
  --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_development_v1 \
  --preview-dir /home/fb/src/paper/gazelleV1/cscwd/results/r1_previews \
  --validation-fraction 0.2 \
  --seed 20261117 \
  --cases-per-category 3 \
  --preview-cases 12 \
  --context-before 10 \
  --context-after 20 \
  --snapshots 5 \
  --report /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json \
  --summary /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.md \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.log 2>&1 &
```

成功标记：`R1 preparation completed; stage=awaiting_local_review`。报告 `passed=true` 仅表示机器准备通过，不能替代人工诊断或 R1 整体验收。

服务器保留 `split_manifest.json`、`selection_config.json`、`diagnosis_manifest.json`、`baseline_details.json`、`cases/*/case_data.json` 和完整图；路径、SHA256 与预声明配置写入小报告/provenance。小预览默认覆盖全部选中例，少于配额时按实际数量输出。

### 8.3 关键结果回传

第二步成功后，在服务器提交：

```bash
git add cscwd/results/r1_train_single_frame_cache.json \
  cscwd/results/r1_split_and_diagnosis.json \
  cscwd/results/r1_split_and_diagnosis.md \
  cscwd/results/r1_previews \
  cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record R1 development split and diagnostic preparation"
git push origin codex/coverage-aware-router
```

本地拉取后核对 SHA256、覆盖/分组、各项指标和图，再另记 `results/r1_manual_review.md`：逐例记录人工 labels/notes、无法判定项及分组语义的接受或映射调整。保留原机器报告和未审阅模板的来源；不回写成伪造的自动真值。人工诊断与分组核对完成后，才把 R1 写为验收完成。
