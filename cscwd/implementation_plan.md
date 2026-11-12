# CSCWD 时序方向 · 实现规格与进展

> 本文档是给 codex 的实现规格 + 进展记录。codex 开工前读，每完成一个任务写回"进展记录"。
> 关联：论文结构 `paper_blueprint.md`，venue 扫描 `cscwd_2025_2026_related.md`。冲突时以本文档为准。

## 0. 目标（一句话）

在冻结 DINOv3 单帧 gaze 估计（GazeSpot）之上，加一个**严格因果**的时序门控 STAG（Shift-aware Temporal Gating），利用 VAT 连续帧的 head 轨迹 + 历史预测，实现**时序稳定 + 注视切换感知**。

## 1. 技术方案

### 1.1 单帧底座（复用，不改）
- `gazelle/model.py` 的 `GazeLLE`（GazeSpot）：冻结 DINOv3 + GGSF + SASA + decoder，输出 heatmap + inout。
- 用现有 GazeSpot VAT checkpoint 跑前向，**不重训 backbone**。用户于 2026-11-12 确认两套 `epoch_7.pt`，并选择更快的 ViT-B；T2 主缓存及后续单帧/STAG 对照统一使用 ViT-B：
  - ViT-L：`/home/fb/src/paper/gazelleV1/experiments/train_vat_vitl_sasa_ggsf/2026-03-25_14-35-38/epoch_7.pt`。
  - ViT-B：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`。

### 1.2 track 恢复
VAT 原始数据按轨迹组织：`annotations/{split}/{video}/{clip}/` 下每人一个 CSV，文件内即逐帧轨迹。
现有 `data_prep/preprocess_vat.py` 用 `merge_dfs` 拍平，丢了 person_id。重写以保留轨迹。
T1 保留原始 CSV 帧序，头部框使用当前帧原始真实框并裁至图像边界；旧脚本的 11 帧居中框平滑不带入因果时序协议。

### 1.3 STAG 时序门控（平滑 + 切换）
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
- 第一版 `w` 手调或简单拟合（在 track 上网格/最小二乘），跑通拿数字；之后可选升级为可学习门控（见 T3 备注）。

## 2. 项目结构（本次代码/论文全在 cscwd/ 下，不与旧代码混）

```
cscwd/
├── code/                      # 本次时序方向代码（全部新代码放这里）
│   ├── data_prep/             # T1 track 恢复
│   │   └── preprocess_vat_tracks.py
│   ├── stag/                  # STAG 门控核心
│   │   ├── __init__.py
│   │   ├── cache.py           # T2 NPY 存储与离线 track 读取
│   │   ├── gating.py          # 平滑 + 切换门控
│   │   └── metrics.py         # 时序指标
│   └── scripts/               # 入口脚本
│       ├── cache_single_frame.py   # T2 单帧预测缓存
│       ├── run_stag.py             # T3 STAG 推理
│       ├── eval_temporal.py        # T4 时序评测
│       └── visualize_temporal.py   # T6 可视化
├── latex/                     # 论文（main.tex + refs.bib + fig/）
├── results/                   # 关键小结果（JSON/MD），git 传回
│   └── RESULT_PROVENANCE.md   # 每个数字→来源运行
└── *.md                       # 方案/蓝图/规格/文献
```

- 复用旧代码只 `import` 根目录 `gazelle`（只读，不改）；任何**新文件**都落在 `cscwd/code/`。
- 大文件（heatmap 缓存 `.npy`、checkpoint `.pt`）**不进 git**，留服务器；`cscwd/.gitignore` 已忽略。

## 3. 实现任务清单（按依赖顺序，逐项勾选）

### T1 track 恢复
- 新增 `cscwd/code/data_prep/preprocess_vat_tracks.py`（不改旧脚本），产出 `train_preprocessed_tracks.json` / `test_preprocessed_tracks.json`。
- 结构：`sequence → tracks → frames`，每个 track 内按帧序，每帧含 `path / bbox_norm / gazex_norm / gazey_norm / inout / person_id`。
- person_id 来自"每人一个 CSV"这一原生组织（CSV 内顺序即帧序，无需额外匹配）。
- track 另含 `track_id`（`split/video/clip/person_id`）和 `source_csv`；frame 另含原图文件名中的 `frame_index`，保留缺口，不插帧。`gazex_norm / gazey_norm` 沿用单元素列表格式。
- **验收**：脚本能在服务器数据上跑通；track 数与 person CSV 数一致；帧数与原拍平 JSON 一致（总数核对）。

### T2 单帧预测缓存（服务器）
- 用 GazeSpot VAT checkpoint，沿新 track JSON 跑前向，逐 track 逐帧缓存 `heatmap`（64×64）与 `inout` 概率。
- 缓存为 numpy/parquet，按 track 组织，可离线读取。
- 实现为 float32 `heatmap.npy` / `inout.npy` / `argmax_xy.npy`，沿 `sequence → track → frame` 连续存储；服务器 `index.json` 保存每条 track 的 `[start, stop)`、原始帧记录和文件 SHA256。`stag.cache.load_cache` / `iter_cached_tracks` 只依赖 NumPy，离线按 track 取出预测。
- 使用原模型 transform、GGSF + SASA，全部参数冻结、`eval` + `inference_mode`，不使用 autocast/TF32；模型仅接收当前 RGB 和真实头部框。argmax 坐标按原 `vat_l2` 的 `column/64, row/64` 定义。
- **验收**：缓存覆盖 test 全部 track；单帧 argmax 点 + 单帧 heatmap 可由缓存完全重建（不重新前向）。

### T3 STAG 门控（后处理级，先跑通）
- 纯 numpy 实现 §1.3 公式，输入 T2 缓存，输出平滑后 heatmap / 点。
- 第一版手调 `w`（建议初始：w1、w3 为大正数触发快速响应，w2 中等，w0 为负偏向平滑）。
- 首轮固定权重 `[w0,w1,w2,w3]=[-2,20,2,10]`，不在 test 上搜索参数。头部中心与热图 argmax 均用归一化坐标计算位移；每条 track 独立初始化，原始帧号不连续时从当前帧重置。
- 仅平滑热图，`inout` 概率保持单帧原值（用于门控，也用于 AP），所以本版不改变 AP。门控只接收预测热图/概率、真实头部框及帧号；真实注视点和画内外标签仅在共用标准指标函数中参与评测。
- `stag/metrics.py` 在 T3 提供标准三指标以满足验收；两侧调用同一个 `evaluate_predictions`，AUC/L2 仅统计画内人物帧，AP 覆盖全部人物帧，逐人物帧汇总。T4 再加入时序指标与独立评测入口。
- **验收**：在 track 上跑通；单帧 vs STAG 的 L2/AUC/AP 有完整对比数字。
- **备注**：若手调门控有正增益，再考虑可学习门控（轻量 MLP，在 track 上拟合），作为消融。

### T4 评测脚本 + 时序指标
- 标准三件套：AUC↑ / 归一化 L2↓ / In-Out AP↑（VAT test，真实框，纯 RGB）。
- 新增时序指标（**贡献，须诚实定义**）：
  - 帧间抖动 Jitter↓：相邻帧同 track 预测点位移均值/分布。
  - 切换响应延迟↓：切换事件（inout 翻转 / gaze 位移超阈值）后预测重新稳定跟上的帧数。
  - 稳定占比↑：预测连续稳定的帧比例。
- **诚实红线**：抖动降低必须同报**真实注视点位移**作参照（证明不是预测僵死变钝，而是跟上真实轨迹）。
- **口径对齐**：单帧 baseline 与 STAG 必须在**同一个 track 级评测脚本**上重跑，旧拍平数字只作参照、不进时序对比表。

#### T4 时序指标的实现定义（本项目定义，首轮固定）

- 坐标均为归一化 `(x,y)`；相邻仅指同 track 的原始 `frame_index` 差为 1，不跨人物或缺口。阈值在两侧完全相同，不调整 T3 权重。
- **Jitter**：在连续且两帧真实画内的帧对上，统计预测点位移的 mean/median/p90/p95；同时在**完全相同帧对**上报告真实注视点位移分布。按人物帧对汇总，不等权平均 track。
- **稳定占比**：连续 **3 帧**、全部真实画内的窗口，以结束帧为计数单位；若窗口内每个相邻预测点位移均 `≤0.02` 则计为稳定。同口径报告 GT 稳定占比。另报**准确且稳定占比**，要求稳定窗口内每帧预测为画内且到当帧 GT 距离 `≤0.10`。
- **切换代理事件**：连续帧真实 inout 翻转，或两帧均真实画内且 GT 位移 `>0.15`；分别记为 `enter_frame / leave_frame / gaze_shift`。这是由标签定义的代理事件，不另造人工切换标注。
- **切换响应**：从事件帧 `t`（含 `t`）起，寻找第一段连续 **3 帧**预测状态正确且画内定位误差 `≤0.10` 的区间；其起始帧减 `t` 为延迟，可为 0。记录的是成功区间的起始延迟，确认该区间需观察完整 3 帧。
- **观察窗口**：最多 **30 帧**（含事件帧），在下一 GT 代理事件、原始帧号缺口或 track 结束前截断。窗口不足 3 帧记为 `insufficient_followup`；足长但未响应记为 `censored`。按事件类型同报总数、可评估数、响应率、已响应条件延迟分布、未响应数和短窗口数；另在两侧共同响应的同一事件集合上报告延迟差，保留各侧独有响应数。
- `stag/metrics.py` 的 `evaluate_temporal` 只计分已保存预测，标签不进入门控；`scripts/eval_temporal.py` 共用标准/时序评测函数重跑两侧。小 JSON 保存汇总与完整协议，逐 track/事件明细留服务器 JSON 并记录路径/SHA256；无样本统计写 `null`。

### T5 消融
| 变体 | 平滑 | 门控 | 回答 |
|---|---|---|---|
| 单帧底座 | ✗ | ✗ | baseline |
| 纯 EMA（固定 α） | ✓ | ✗ | 平滑单独贡献 |
| 规则门控 STAG | ✓ | ✓(规则) | 切换感知贡献 |
| 可学习门控 STAG | ✓ | ✓(可学习) | 学习门控是否更优 |

### T6 定性可视化
- 拥挤多人协同画面（VAT crowd 子集）"谁在看谁/什么"时序可视化。
- 切换场景：一人从看 A 转到看 B，单帧抖动/滞后 vs STAG 平滑且快速跟上。

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

- 已确认：venue 扫描完成（`cscwd_2025_2026_related.md`）——CSCWD 2025/2026 无 gaze following / shared attention 论文，motivation 需自证，锚点 B1/B3/C2。
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
- **下一步**：用户手动运行下列 T4 命令，回传 `results/t4_temporal_comparison.json` 与 provenance。完整明细留服务器；拿到真实时序数字后进入 T5 消融。

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
