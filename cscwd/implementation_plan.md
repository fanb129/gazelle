# CSCWD 时序方向 · 实现规格与进展

> 本文档是给 codex 的实现规格 + 进展记录。codex 开工前读，每完成一个任务写回"进展记录"。
> 关联：论文结构 `paper_blueprint.md`，venue 扫描 `cscwd_2025_2026_related.md`。冲突时以本文档为准。

## 0. 目标（一句话）

在冻结 DINOv3 单帧 gaze 估计（GazeSpot）之上，加一个**严格因果**的时序门控 STAG（Shift-aware Temporal Gating），利用 VAT 连续帧的 head 轨迹 + 历史预测，实现**时序稳定 + 注视切换感知**。

## 1. 技术方案

### 1.1 单帧底座（复用，不改）
- `gazelle/model.py` 的 `GazeLLE`（GazeSpot）：冻结 DINOv3 + GGSF + SASA + decoder，输出 heatmap + inout。
- 用现有 GazeSpot VAT checkpoint 跑前向，**不重训 backbone**。checkpoint 路径/epoch 向用户确认。

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
- **验收**：缓存覆盖 test 全部 track；单帧 argmax 点 + 单帧 heatmap 可由缓存完全重建（不重新前向）。

### T3 STAG 门控（后处理级，先跑通）
- 纯 numpy 实现 §1.3 公式，输入 T2 缓存，输出平滑后 heatmap / 点。
- 第一版手调 `w`（建议初始：w1、w3 为大正数触发快速响应，w2 中等，w0 为负偏向平滑）。
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
- **下一步**：用户手动运行下列 T1 命令，将小报告与 provenance 经 git 传回；以实际报告完成服务器验收。当前停在 T1，用户确认后再进 T2。

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
