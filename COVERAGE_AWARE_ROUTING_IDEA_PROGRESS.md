# Coverage-Aware Spatial Routing：Idea 与全链路进度总账

> 最后更新：2026-08-11  
> 工作分支：`codex/coverage-aware-router`  
> GazeFollow formal full-train 运行提交：`efa15f731561988545d97e2d5b5012c15c73bf3d`  
> 详细实现、历史分析与服务器命令见
> [COVERAGE_AWARE_ROUTING_IMPLEMENTATION.md](COVERAGE_AWARE_ROUTING_IMPLEMENTATION.md)。

本文档是当前 idea 的**唯一高层进度总账**，用于回答四个问题：

1. 当前到底在解决什么问题，方法的核心思想是什么；
2. 工程、实验、论文和投稿全链路分别完成到了哪里；
3. 每个结论由哪个原始结果文件支撑；
4. 下一步应该做什么，以及做到什么程度才算完成。

本文档不替代训练命令文档，也不把所有试错日志重新复制一遍。若本文档中的转述
与链接的 JSON/JSONL 冲突，**以原始结果文件为准**。

---

## 0. 状态与证据规则

### 0.1 状态标记

| 标记 | 含义 |
|---|---|
| ✅ 已完成 | 已达到本阶段预先定义的目的，结果和 provenance 可核对 |
| 🟡 部分完成 | 已有有效证据，但还缺关键对照、统计或写作产物 |
| ⚠️ 失败/暂停 | 实验有效地暴露了限制，当前 recipe 不再继续消耗算力 |
| ⬜ 未开始 | 尚无可复用结果或正式产物 |
| 🔒 已冻结 | 结果已经进入最终报告口径，不再用 official test 调参 |

### 0.2 证据等级

| 等级 | 可以怎么用 | 当前例子 |
|---|---|---|
| A：正式主证据 | 可进入论文主表或核心结论 | full-train/no-eval 后的一次 official test；正式正反序效率矩阵 |
| B：正式 validation 证据 | 可用于选 recipe、消融、因果对照和稳健性 | split-clean holdout、三 routing seed、VAT source-video holdout |
| C：工程/诊断证据 | 可解释设计演进，不能替代最终论文数字 | 早期 official-test-per-epoch、初始化 control、stage profile |
| D：负结果 | 用于停止错误路线或形成 limitation | joint training 崩溃、VAT K50/K75 accuracy gate 失败 |

### 0.3 数据真实性原则

- 所有表格尽量链接到产生数字的具体 `summary.json`、`history.jsonl`、评测 JSON
  或 benchmark JSON/MD。
- 派生的差值和 `mean ± std` 必须列出原始 seed 文件；计算口径写在表旁。
- 正式 official-test 产物由
  [pipeline_complete.json](experiments/coverage_router/gf_fulltrain_official_test_seed3106/pipeline_complete.json)
  记录 checkpoint 和结果 SHA256。
- 本地同步有意不包含 `.pt`；checkpoint 的角色和哈希由服务器 formal validator
  与 `pipeline_complete.json` 闭环，本机不声称重新执行过原始推理。
- 早期每轮读取 official test 的 run 只保留为诊断，不能与最终 formal 结果混用。

### 0.4 “完成”必须经过的闸门

本文档不因为“代码文件已经存在”就把一项工作标成论文完成。每个阶段依次经过：

```text
已定义 -> 已实现 -> 本地验证 -> 服务器运行 -> 结果审计
       -> claim 成立或负面闭环 -> 写入论文 -> 可复现发布
```

例如 VAT 当前已经完成“服务器运行、结果审计和负面闭环”，但没有完成“正向跨
数据集 claim”；正式效率已经完成结果审计，但尚未整理进论文正文。负面结果也可以
完成一个决策阶段，不需要永远显示为“进行中”。

当前 `experiments/` 被 `.gitignore` 忽略，正式 JSON/MD 虽已同步到本机，但不会
自动进入 Git 历史。投稿前必须建立一个受版本控制、只含小型 JSON/MD 和 SHA 的
immutable evidence bundle；否则代码提交无法单独恢复本文档引用的结果。

---

## 1. 一页结论

### 1.1 当前 idea 的一句话

现有 gaze target estimation 模型通常先让整张图经过同样深、同样昂贵的视觉
推理，之后才利用“谁在看”的信息。本文先在 DINOv3 中层根据人物头部、场景和
几何预测一个高召回 gaze support，再只让深层 blocks 处理该 support 内的 token，
从而把深层计算分配变成 **gaze-conditioned**，而不是对整图一视同仁。

### 1.2 当前主结果

在预先固定、训练期间不读取测试集的 GazeFollow formal pipeline 中，K50 将
DINOv3 blocks 6–11 的 patch token 从 `1024` 减到 `512`。相对 matched dense：

| 指标 | learned K50 | matched dense | K50 - dense |
|---|---:|---:|---:|
| AUC ↑ | `0.956462` | `0.957264` | `-0.000802` |
| Avg L2 ↓ | `0.104375` | `0.102592` | `+0.001783` |
| Min L2 ↓ | `0.045044` | `0.043748` | `+0.001296` |
| exact-point coverage ↑ | `94.123%` | `100%` | `-5.877 pp` |
| hard heatmap coverage ↑ | `86.300%` | `100%` | `-13.700 pp` |

原始证据：
[comparison.json](experiments/coverage_router/gf_fulltrain_official_test_seed3106/comparison.json)、
[learned K50 result](experiments/coverage_router/gf_fulltrain_official_test_seed3106/learned_k50_image_b16_fp32.json)、
[matched dense result](experiments/coverage_router/gf_fulltrain_official_test_seed3106/matched_dense_image_b16_fp32.json)。

独立的 RTX 3090 正反序效率矩阵显示：batch 1 延迟持平；batch 4/8 相对原始
dense 的端到端 speedup 分别为 `1.061x / 1.149x`，但没有显存收益。原始证据见
[B1 forward](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b1_forward.json)、
[B1 reverse](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b1_reverse.json)、
[B4 forward](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b4_forward.json)、
[B4 reverse](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b4_reverse.json)、
[B8 forward](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b8_forward.json)、
[B8 reverse](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b8_reverse.json)。

### 1.3 当前总判断

> GazeFollow 上的核心假设已经得到支持：learned gaze-aware Router 能把后半段
> 深层推理集中到一半空间，以小幅精度代价换取批量推理加速。

当前可以写“小幅精度—效率 trade-off”，不能写“完全无损”“整网计算减半”
或“2x 加速”。VAT 尚未保持 dense 精度，因此跨数据集普适性没有成立。

### 1.4 当前完成度

| 范围 | 估计完成度 | 解释 |
|---|---:|---|
| 工程主闭环 | `约 90%` | 路由、真实 sparse suffix、训练/评测、正式协议、benchmark 已完成 |
| GazeFollow 核心证据 | `约 80%` | 主结果、因果对照、三 seed、效率已完成，关键组件消融未完成 |
| 完整论文实验包 | `约 60–65%` | 缺 Router 输入、预算、回填、路由位置、多人效率、可视化和统计 |
| 论文正文与投稿材料 | `约 10–15%` | 已有故事和方法草稿，没有形成完整论文稿 |

上述百分比是工作量和证据完整度估计，不是方法成功概率。

---

## 2. Idea：问题、方法和论文故事

### 2.1 要解决的问题

当前许多 gaze-target 方法存在共同的计算结构：

```text
整张图 -> 完整视觉 backbone 深层推理 -> 再加入人物 bbox/head 信息 -> gaze decoder
```

这意味着 backbone 在决定“哪些区域值得继续深挖”时，并不知道当前查询的是谁。
即使 gaze target 只可能落在少量空间区域，所有 patch 仍经过同样昂贵的深层
blocks。只在 decoder 末端乘一个 mask 不会减少 DINOv3 的主要计算。

本文的问题不是针对某一篇 object-aware 论文，而是针对这一类结构性错配：

> 能否先预测一个高召回、人物条件化、允许多峰和不确定性的 gaze support，再把
> 深层视觉计算分配给这个 support，同时尽量保持 gaze 定位精度？

当前 motivation 草稿和完整公式来源：
[实现文档第 1–3 节](COVERAGE_AWARE_ROUTING_IMPLEMENTATION.md)。

### 2.2 这里的 object 和 mask 分别是什么

- `bbox` 标出的是**正在看的人**，不是被注视的物体。
- gaze target 是该人在图中真正注视的位置或对象。
- Router 的 mask 不是物体检测框、SAM 分割、固定 FOV cone，也不是最终 heatmap。
- mask 表示“哪些候选空间值得继续进行昂贵深层推理”。它可以是不规则、多峰的。

### 2.3 方法流程

```text
输入图像
  -> DINOv3 blocks 0–5：全部 1024 patches，dense
  -> Router：scene feature + head appearance + bbox-relative geometry
  -> 每人 free-form support
  -> 同图多人 max-union
  -> Top-K（在固定 K 预算内优先 head footprint 与 8 个 escape anchors）
  -> 同步 gather patch token 与原始 RoPE 位置
  -> DINOv3 blocks 6–11：只处理 K 个 patch + special tokens
  -> sparse token scatter 回 32x32
     未选位置使用 block-5 early-exit feature
  -> 原 GazeLLE multi-layer decoder
  -> gaze heatmap / in-out score
```

实现证据：

- [gazelle/routing/router.py](gazelle/routing/router.py)：support、max-union、Top-K；
- [gazelle/routing/backbone.py](gazelle/routing/backbone.py)：prefix/suffix、RoPE gather、scatter；
- [gazelle/routing/model.py](gazelle/routing/model.py)：support 与真实 sparse 路径；
- [gazelle/routing/losses.py](gazelle/routing/losses.py)：coverage/budget/entropy；
- [scripts/train_coverage_router.py](scripts/train_coverage_router.py)：训练与 checkpoint；
- [scripts/eval_coverage_router.py](scripts/eval_coverage_router.py)：正式评测。

### 2.4 当前成功 recipe

```text
full-train dense decoder：15 epochs，K100
  -> support：3 epochs，K25，只训练 Router
  -> learned K50：5 epochs，K100 -> K75 -> K50 -> K50 -> K50
     Router、DINO 全冻结，只适配 decoder

同一个 dense final
  -> matched dense：5 epochs，K100，只适配 decoder
```

正式 recipe 的 manifest 与 history：

- dense：
  [manifest](experiments/coverage_router/gf_fulltrain_dense_seed3106/run_manifest.json)、
  [history](experiments/coverage_router/gf_fulltrain_dense_seed3106/history.jsonl)、
  [summary](experiments/coverage_router/gf_fulltrain_dense_seed3106/summary.json)；
- support：
  [manifest](experiments/coverage_router/gf_fulltrain_support_k025_seed3106/run_manifest.json)、
  [history](experiments/coverage_router/gf_fulltrain_support_k025_seed3106/history.jsonl)、
  [summary](experiments/coverage_router/gf_fulltrain_support_k025_seed3106/summary.json)；
- K50：
  [manifest](experiments/coverage_router/gf_fulltrain_sparse_fixedrouter_decoder_k050_seed3106/run_manifest.json)、
  [history](experiments/coverage_router/gf_fulltrain_sparse_fixedrouter_decoder_k050_seed3106/history.jsonl)、
  [summary](experiments/coverage_router/gf_fulltrain_sparse_fixedrouter_decoder_k050_seed3106/summary.json)；
- matched dense：
  [manifest](experiments/coverage_router/gf_fulltrain_dense_decoder_continue_seed3106/run_manifest.json)、
  [history](experiments/coverage_router/gf_fulltrain_dense_decoder_continue_seed3106/history.jsonl)、
  [summary](experiments/coverage_router/gf_fulltrain_dense_decoder_continue_seed3106/summary.json)。

### 2.5 候选 contribution

当前最稳妥的 contribution 结构是：

1. **问题重述**：指出 gaze query 与 backbone 深层计算分配之间的结构性错配；
2. **Coverage-aware single-mask routing**：以人物条件化的 free-form support
   代替固定角度 FOV，多人时用一个 max-union mask，不增加第二条 DINO 分支；
3. **真实 backbone 稀疏化**：路由发生在 DINOv3 内部，patch/RoPE 同步 gather，
   未选位置用浅层特征回填，而不是只给 decoder 乘 mask；
4. **Accuracy-efficiency evidence**：正式 GazeFollow 结果、learned-router 因果对照、
   三 seed 稳健性和 B1/B4/B8 实测效率。

其中第 2 点还缺 Router 输入和回填消融；在这些实验完成前，不能把所有组件都写成
已被实验证明的贡献。

### 2.6 明确不能声称的内容

- 不能说 K50 完全无损或与 dense 统计等价；
- 不能把“后六层 patch 减半”写成“整网 FLOPs 减半”；
- 不能声称 batch-1 实时加速或节省显存；
- 不能声称成功联合微调了 Router、decoder 与 DINO suffix；成功 recipe 中
  Router 和 DINO 在 K50 adaptation 时均冻结；
- 不能用 GazeFollow official test 证明多人 union，因为 4,782 张测试图全是
  single-head；
- 不能声称 VAT 上已经成立；K50/K75 均未通过 accuracy gate。

---

## 3. 全链路阶段总览

| ID | 阶段 | 完成定义 | 当前状态 | 下一动作 |
|---|---|---|---|---|
| R0 | 问题定义 | 从多篇方法抽象出共同结构问题 | 🟡 部分完成 | 将现有 motivation 绑定到系统文献证据，不只讲一个对手 |
| R1 | 最新相关工作调研 | 完整 gaze target / efficient ViT / dynamic token 文献表 | ⬜ 未完成 | 新建独立 literature MD，逐篇方法、指标、缺点和可借鉴点 |
| R2 | 假设与故事 | 明确问题、机制、可证伪实验和诚实边界 | ✅ 已完成 | 用消融验证 scene/head/geometry 与 early-fill |
| M0 | 方法设计 | single mask、union、Top-K、sparse suffix、scatter 闭环 | ✅ 已完成 | 冻结 K50 主结构，消融不读取 official test |
| E0 | 工程实现 | forward/backward/checkpoint/eval/benchmark 可运行 | ✅ 已完成 | 增加 `router_input_mode` 和 fill-mode 消融接口 |
| E1 | 结构正确性 | K100 等价、RoPE 正确、固定预算、无 NaN/OOM | ✅ 已完成 | 只在接口变化后补回归测试 |
| D0 | 数据与评测协议 | image-disjoint holdout、formal full-train、official one-shot | ✅ 已完成 | 保持 frozen，不退回 legacy test-per-epoch |
| X0 | Support 可行性 | learned support 显著优于未训练/随机 | ✅ 已完成 | 输入消融解释 support 来自什么信息 |
| X1 | 真实 sparse 可训练性 | sparse suffix 后 gaze 精度可恢复 | ✅ 已完成 | K25/K75 形成预算 Pareto |
| X2 | 错误 joint recipe | 判断是否应继续 joint training | ⚠️ 已失败并停止 | 不再复跑同一 recipe |
| X3 | Learned Router 因果对照 | 同 K、同 decoder 预算下优于未训练 Router | ✅ 已完成 | 组件消融补齐机制证据 |
| X4 | Matched dense 公平对照 | dense 与 K50 decoder 更新预算一致 | ✅ 已完成 | 已进入最终比较口径 |
| X5 | 多 seed 稳健性 | 至少三次 routing-stage 重复 | 🟡 部分完成 | 已排除幸运 Router seed；不是完整 end-to-end 三 seed |
| X6 | 真实效率 | B1/B4/B8、正反序、dense/K100/K50 | ✅ 已完成 | K25/K75 只有在预算消融后再测 |
| X7 | 多人 union | grouped-image 精度与真实耗时都成立 | 🟡 部分完成 | K100 等价与 grouped loader 已验证；缺 union-vs-repeat latency |
| X8 | 跨数据集 VAT | source-video holdout 与最终 official 均通过 | ⚠️ holdout 失败并暂停 | 只允许新 context-preserving mask，不再扫 K87.5 |
| X9 | GazeFollow formal refit/test | full train 无 eval，固定 checkpoint 各 test 一次 | 🔒 已完成并冻结 | 不再使用 official test 调参或选模 |
| X10 | 关键消融 | 输入、预算、回填、路由位置至少完成主表 | ⬜ 未完成 | 第一项实现 Router 输入消融 |
| X11 | 定性与统计 | support 可视化、失败案例、paired CI/bootstrap | ⬜ 未完成 | 在 split-clean validation 保存逐样本输出 |
| P0 | 论文定位与标题 | 一句话定位、标题和 claim 边界冻结 | 🟡 部分完成 | 根据消融决定突出 routing 还是 context preservation |
| P1 | Introduction | 问题、现有不足、洞察、贡献形成完整初稿 | ⬜ 未开始 | 文献总表完成后写 |
| P2 | Related Work | gaze target、object/FOV、efficient ViT 三组组织 | ⬜ 未开始 | 先完成 R1 |
| P3 | Method | 公式、训练、复杂度和多人 union 成稿 | 🟡 部分完成 | 从实现文档改写为论文语言并补消融接口 |
| P4 | Experiments | protocol、主表、效率、消融、稳健性、限制 | 🟡 部分完成 | 主表已有；缺 X10/X11 与外部 SOTA 表 |
| P5 | 图表 | 方法总图、Pareto、support 图、失败案例 | ⬜ 未开始 | 消融和逐样本输出完成后制作 |
| P6 | Abstract/Conclusion | 与最终证据严格一致 | ⬜ 未开始 | 全部主实验冻结后写 |
| P7 | Supplement/Reproducibility | 命令、配置、哈希、额外表、代码说明 | 🟡 部分完成 | 现有 runner 很完整，需整理为匿名发布包 |
| S0 | 投稿准备 |  venue、模板、页数、匿名、checklist、supp | ⬜ 未开始 | 主实验与论文初稿完成后决定 |

---

## 4. 实验阶段与证据账本

### 4.1 P0 工程 smoke：真实 sparse 路径可以运行

**状态：✅ 工程目标完成；证据等级 C。**

早期 smoke 已完成小批量 forward/backward/eval/checkpoint 路径，K25 实际 keep
为 `0.25`，8 个 eval 样本的 exact-point coverage 为 `0.8082`。该小样本数字
不代表模型性能，只证明路径能跑通：

- [sparse_smoke/history.jsonl](experiments/coverage_router/sparse_smoke/history.jsonl)
- [sparse_smoke/summary.json](experiments/coverage_router/sparse_smoke/summary.json)

当前结构测试覆盖 Router、Top-K、RoPE gather、scatter、K100 等价、训练协议和
formal evaluator：

- [tests/test_coverage_router.py](tests/test_coverage_router.py)
- [tests/test_routed_backbone_utils.py](tests/test_routed_backbone_utils.py)
- [tests/test_coverage_router_training_protocol.py](tests/test_coverage_router_training_protocol.py)
- [tests/test_formal_coverage_router_eval.py](tests/test_formal_coverage_router_eval.py)
- [tests/test_validate_coverage_router_refit.py](tests/test_validate_coverage_router_refit.py)

### 4.2 Support pilot：Router 确实学会了“保留哪里”

**状态：✅ 核心假设通过；正式证据等级 B。**

split-clean GazeFollow holdout 中，K25 learned support 的 exact-point/hard coverage
为 `83.99% / 70.03%`；同一个 dense checkpoint 中未训练 Router 的 K25 control
只有 `15.36% / 15.92%`。这说明提升不是固定 head/escape priority 自然带来的。

| K25 support | exact-point coverage | hard coverage | 证据 |
|---|---:|---:|---|
| learned Router | `0.839926` | `0.700287` | [support summary](experiments/coverage_router/gf_splitclean_support_k025_seed3106/summary.json) |
| untrained Router | `0.153629` | `0.159233` | [untrained K25 eval](experiments/coverage_router/gf_splitclean_dense_seed3106/eval_holdout_untrained_router_k025_b16_fp32.json) |

早期 legacy support 的 official-test coverage 也显示相同趋势，但因为旧协议每轮读取
test，只保留为诊断：
[legacy support summary](experiments/coverage_router/gf_support_k025_seed3106/summary.json)。

### 4.3 直接 joint training：失败并形成训练策略结论

**状态：⚠️ 有效负结果；证据等级 D。**

从第一个 epoch 起同时更新 Router、decoder 和 DINO suffix 的 K25 15-epoch run
没有稳定恢复。AUC 从 epoch 0 的 `0.8583` 一度降至 `0.5435`，epoch 14 仅
`0.5946`；Avg L2 最终为 `0.4888`。这否定的是该 joint recipe，不是否定
coverage-aware routing 本身。

- [joint history](experiments/coverage_router/gf_sparse_b5_k025_seed3106/history.jsonl)
- [joint manifest](experiments/coverage_router/gf_sparse_b5_k025_seed3106/run_manifest.json)
- [joint summary](experiments/coverage_router/gf_sparse_b5_k025_seed3106/summary.json)

由此固定了后续 staged recipe：先学 support，再冻结 Router/DINO，只让 decoder
逐步适应 K100→K75→K50。

### 4.4 固定 Router 的 K50/K25：证明 sparse 表示可以被 decoder 适应

**状态：✅ 工程 proof-of-concept 完成；证据等级 C。**

早期 legacy 结果用于决定 K50 是主 Pareto 点：

| 旧协议结果 | AUC ↑ | Avg L2 ↓ | Min L2 ↓ | 证据 |
|---|---:|---:|---:|---|
| fixed-router K50 | `0.952305` | `0.110029` | `0.050069` | [summary](experiments/coverage_router/gf_sparse_fixedrouter_decoder_k050_seed3106/summary.json) |
| fixed-router K25（首个真实 K25 epoch 3） | `0.948858` | `0.113609` | `0.052081` | [history](experiments/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106/history.jsonl) |
| K50 suffix 微调诊断 | `0.954059` | `0.106171` | `0.046624` | [summary](experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/summary.json) |

这些 run 在 protocol cleanup 之前，不进入最终论文主表。它们的价值是证明：失败
主要来自突然改变特征分布和 joint optimization，而不是“只要删 token 就必然
无法定位”。

初始化 control 进一步确认 K100 路径能恢复 dense 结果，而未经适配的 K25/K50
性能下降明显：
[K100](experiments/coverage_router/gf_sparse_init_controls/k100.json)、
[K50](experiments/coverage_router/gf_sparse_init_controls/k050.json)、
[K25](experiments/coverage_router/gf_sparse_init_controls/k025.json)。

### 4.5 Split-clean 数据与选模协议

**状态：✅ 完成；证据等级 B。**

GazeFollow train 按 normalized image path 做稳定 SHA256 分组，固定 seed 3106：

- train：`105,954` records、`102,132` in-frame persons；
- validation：`11,773` records、`10,746` images、`11,326` in-frame persons；
- assignment fingerprint：
  `817d9f2200ebc9881f6d2318cdb60d7d8d7af394f98f0e3461a6d8bd2c26b5e3`。

证据：
[dense data_split](experiments/coverage_router/gf_splitclean_dense_seed3106/data_split.json)、
[dense manifest](experiments/coverage_router/gf_splitclean_dense_seed3106/run_manifest.json)、
[explicit holdout eval](experiments/coverage_router/gf_splitclean_dense_seed3106/eval_holdout_b16_fp32.json)。

这套 holdout 用于 recipe、消融和 seed 分析；official test 不再承担选模职责。

### 4.6 Learned Router 因果对照

**状态：✅ 完成；证据等级 B。**

两个模型都保留 50% patch、只训练相同 decoder、使用相同五轮 curriculum；唯一
关键差别是 Router 是否经过 support 训练。

| K50 holdout | AUC ↑ | Avg L2 ↓ | exact point ↑ | hard coverage ↑ |
|---|---:|---:|---:|---:|
| learned Router | `0.963612` | `0.113830` | `0.951263` | `0.890519` |
| untrained Router | `0.958509` | `0.125419` | `0.535582` | `0.534865` |
| learned - untrained | `+0.005102` | `-0.011589` | `+0.415681` | `+0.355654` |

证据：
[learned summary](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/summary.json)、
[untrained summary](experiments/coverage_router/gf_splitclean_sparse_untrainedrouter_decoder_k050_seed3106/summary.json)。

这建立了“gaze-aware ranking 有因果贡献”，但还没有解释 scene/head/geometry
三个输入各自贡献多少；这正是当前第一个待做消融。

### 4.7 Matched dense：公平比较额外 decoder 训练预算

**状态：✅ 完成；证据等级 B。**

K50 和 dense 都从同一个 dense epoch-14 checkpoint 出发，之后都只更新 decoder
五轮。selected epoch 4 的结果：

| holdout | AUC ↑ | Avg L2 ↓ | K50 - dense |
|---|---:|---:|---:|
| learned K50 | `0.963612` | `0.113830` | — |
| matched dense | `0.964819` | `0.113860` | `-0.001208 AUC / -0.000030 L2` |

证据：
[K50 summary](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/summary.json)、
[matched dense summary](experiments/coverage_router/gf_splitclean_dense_decoder_continue_seed3106/summary.json)。

因此旧实验中偶尔出现的 K50 L2 优势不能被解释为模型天然更准；最稳妥结论是
L2 基本持平、AUC 有小幅代价。

### 4.8 Routing-stage 三 seed 稳健性

**状态：🟡 已排除幸运 Router seed；证据等级 B。**

三次运行共享同一个 seed-3106 dense 起点和同一数据 split，只重新初始化并训练
Router/后续 adaptation。因此它是 routing-stage 三 seed，不是完整 end-to-end
三 seed。

| seed | K50 AUC | dense AUC | delta AUC | K50 Avg L2 | dense Avg L2 | delta L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 3106 | `0.963612` | `0.964819` | `-0.001208` | `0.113830` | `0.113860` | `-0.000030` |
| 3107 | `0.963424` | `0.964701` | `-0.001277` | `0.114433` | `0.113704` | `+0.000729` |
| 3108 | `0.963651` | `0.964624` | `-0.000973` | `0.114407` | `0.113811` | `+0.000596` |

配对 `mean ± sample std`：

- AUC：`-0.001153 ± 0.000159`；
- Avg L2：`+0.000432 ± 0.000405`；
- K50 exact-point coverage：`0.9486 ± 0.0023`；
- K50 hard coverage：`0.8907 ± 0.0029`。

原始 seed 证据：

- K50：
  [3106](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/summary.json)、
  [3107](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3107/summary.json)、
  [3108](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3108/summary.json)；
- matched dense：
  [3106](experiments/coverage_router/gf_splitclean_dense_decoder_continue_seed3106/summary.json)、
  [3107](experiments/coverage_router/gf_splitclean_dense_decoder_continue_seed3107/summary.json)、
  [3108](experiments/coverage_router/gf_splitclean_dense_decoder_continue_seed3108/summary.json)；
- support：
  [3106](experiments/coverage_router/gf_splitclean_support_k025_seed3106/summary.json)、
  [3107](experiments/coverage_router/gf_splitclean_support_k025_seed3107/summary.json)、
  [3108](experiments/coverage_router/gf_splitclean_support_k025_seed3108/summary.json)。

### 4.9 Query unit 与多人 union

**状态：🟡 数据路径和精度口径通过，真实多人效率未完成。**

修正 batch/precision 后，K100 的 person-unit 与 image-unit AUC/L2 逐值一致，证明
grouped loader 没有 flatten/indexing 错误：
[K100 corrected comparison](experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/query_unit_k100_b4_amp.json)。

split-clean validation 中有 `428` 张 multi-head 图、`1008` 个 in-frame people；
完整分布见
[query group audit](experiments/coverage_router/gazefollow_train_query_groups_seed3106.json)。
K50 image-union 的工程检查通过，并把 backbone 输入从 1008 person 次降到 428
image 次：
[multi-head K50 comparison](experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/query_unit_multi_holdout_k050_b1_fp32.json)。
K100 grouped/person 等价 control 也通过：
[multi-head K100 comparison](experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/query_unit_multi_holdout_k100_b1_fp32.json)。

这些结果使用较早 checkpoint，证明的是数据路径和共享 union 可运行，不是最终
多人速度结论。还缺在同一正式 checkpoint 上比较 per-image union 与 per-person
重复 suffix 的 latency/throughput。

当前可用于论文的多人**精度诊断**来自 split-clean seed-3106 的 selected K50 与
matched dense，而不是上述旧 query-unit checkpoint：

| holdout stratum | K50 AUC | dense AUC | delta AUC | K50 Avg L2 | dense Avg L2 | delta L2 |
|---|---:|---:|---:|---:|---:|---:|
| single-head | `0.964120` | `0.965278` | `-0.001158` | `0.113266` | `0.113515` | `-0.000248` |
| multi-head | `0.958413` | `0.960128` | `-0.001714` | `0.119595` | `0.117390` | `+0.002205` |
| two-head | `0.957838` | `0.959745` | `-0.001906` | `0.118366` | `0.116782` | `+0.001585` |
| three-plus | `0.959484` | `0.960842` | `-0.001357` | `0.121884` | `0.118522` | `+0.003362` |

原始字段位于
[K50 summary `.selection.metrics.head_count_strata`](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/summary.json)
和
[matched dense summary](experiments/coverage_router/gf_splitclean_dense_decoder_continue_seed3106/summary.json)。
结果没有多人容量崩溃，但 multi-head 的 L2 代价比 single-head 更明显。

### 4.10 正式效率矩阵

**状态：✅ K50 批量效率 gate 通过；证据等级 A。**

协议固定 RTX 3090、AMP FP16、512×512、每图一人、50 warmup、`200×5`
latency 和 `500×5` throughput，并同时做正序/反序。下表使用两种顺序内比例的
几何平均：

| batch | K50 vs dense latency | speedup | throughput | peak memory | 结论 |
|---:|---:|---:|---:|---:|---|
| 1 | `-0.15%` | `1.002x` | `+0.25%` | `+1.05%` | 延迟持平 |
| 4 | `-5.72%` | `1.061x` | `+5.49%` | `+1.75%` | Pass |
| 8 | `-12.93%` | `1.149x` | `+14.56%` | `+2.79%` | Strong Pass |

原始结果：

- B1：
  [forward JSON](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b1_forward.json)、
  [reverse JSON](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b1_reverse.json)；
- B4：
  [forward JSON](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b4_forward.json)、
  [reverse JSON](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b4_reverse.json)；
- B8：
  [forward JSON](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b8_forward.json)、
  [reverse JSON](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/benchmark_formal_b8_reverse.json)。

K100 是相同 Router/routed 路径但不删 patch 的 control。K50 相对 K100 在 B4/B8
快 `15.85% / 19.24%`，证明 token pruning 本身有效；B1 固定开销抵消收益。

显存没有下降，因此论文只能报告批量 latency/throughput 改善，不能声称 memory
saving。早期 B1 benchmark 受 autocast cache、运行顺序和 GPU 占用影响，已被本
矩阵取代，不进入论文证据。

该矩阵使用 split-clean K50 checkpoint 测执行图，不是 full-train official 最终
checkpoint 本身；两者结构、token 数与推理路径相同，因此可以支撑部署效率，但
论文必须把 accuracy checkpoint 与 efficiency checkpoint/protocol 分开写清。

### 4.11 VAT 跨数据集诊断

**状态：⚠️ Support gate 通过，accuracy gate 失败并暂停；证据等级 B/D。**

VAT 使用 source-video-disjoint holdout：40 个源视频中 36 个训练、4 个验证，
避免相邻 clip/frame 跨 split。selected tuple：

| VAT holdout | AUC ↑ | L2 ↓ | In-out AP ↑ | exact point ↑ | hard coverage ↑ |
|---|---:|---:|---:|---:|---:|
| dense base | `0.948849` | `0.100565` | `0.920883` | `1.000000` | `1.000000` |
| support K25 | `0.948849` | `0.100565` | `0.920883` | `0.864264` | `0.618282` |
| learned K50 | `0.925232` | `0.108179` | `0.940644` | `0.963961` | `0.837562` |
| learned K75 | `0.936652` | `0.107026` | `0.941260` | `0.990232` | `0.947077` |
| matched dense | `0.946727` | `0.103769` | `0.938995` | `1.000000` | `1.000000` |

证据：
[dense](experiments/coverage_router/vat_splitclean_dense_seed3106/summary.json)、
[support](experiments/coverage_router/vat_splitclean_support_k025_seed3106/summary.json)、
[K50](experiments/coverage_router/vat_splitclean_sparse_fixedrouter_decoder_k050_seed3106/summary.json)、
[K75](experiments/coverage_router/vat_splitclean_sparse_fixedrouter_decoder_k075_seed3106/summary.json)、
[matched dense](experiments/coverage_router/vat_splitclean_dense_decoder_continue_seed3106/summary.json)。

K75 相对 K50 回收了部分精度，说明删除过多关系上下文确实是原因之一；但 K75
相对 matched dense 仍为 `-0.010075 AUC / +0.003258 L2`，没有通过预设 gate。
继续 K87.5 只删除 12.5% suffix token，效率意义很弱，因此当前停止扫 K。

若以后重开 VAT，只允许在固定 K50 总预算内加入 context-preserving 单 mask，
例如 target halo 或少量 global anchors；先在同一 holdout 验证，不读取 VAT
official test。

### 4.12 GazeFollow full-train/no-eval 与一次 official test

**状态：🔒 已完成并冻结；证据等级 A。**

正式 pipeline 的训练链为 `dense15 -> support3 -> {K50-5, matched-dense-5}`。
四阶段均为 `gazefollow_full_train_no_eval_v1`，训练期间不构造 eval loader，不生成
best checkpoint，只使用固定最后一轮 `final.pt`。

协议与完成证据：

- [pipeline_started.json](experiments/coverage_router/gf_fulltrain_official_test_seed3106/pipeline_started.json)
- [pipeline_complete.json](experiments/coverage_router/gf_fulltrain_official_test_seed3106/pipeline_complete.json)
- [dense data split](experiments/coverage_router/gf_fulltrain_dense_seed3106/data_split.json)
- [support data split](experiments/coverage_router/gf_fulltrain_support_k025_seed3106/data_split.json)
- [K50 data split](experiments/coverage_router/gf_fulltrain_sparse_fixedrouter_decoder_k050_seed3106/data_split.json)
- [matched data split](experiments/coverage_router/gf_fulltrain_dense_decoder_continue_seed3106/data_split.json)

`pipeline_complete.json` 记录的 final checkpoint SHA256 前缀分别为：dense
`6338a7f4`、support `3668596e`、K50 `e2f6af0a`、matched dense `d4dd8c22`；
两份 official result 的 SHA256 前缀为 learned K50 `4f7bfbf5`、matched dense
`0774891c`。完整值以链接文件为准。

训练过程审计：

| 阶段 | epoch | 关键趋势 | 证据 |
|---|---:|---|---|
| dense | 15 | loss `0.053527 -> 0.040304`，逐轮下降 | [history](experiments/coverage_router/gf_fulltrain_dense_seed3106/history.jsonl) |
| support K25 | 3 | total `2.06870 -> 1.77286`；hard `58.99% -> 64.12%` | [history](experiments/coverage_router/gf_fulltrain_support_k025_seed3106/history.jsonl) |
| K50 decoder | 5 | keep `1 -> .75 -> .5 -> .5 -> .5`；到 K50 后 loss 下降 | [history](experiments/coverage_router/gf_fulltrain_sparse_fixedrouter_decoder_k050_seed3106/history.jsonl) |
| matched dense | 5 | loss `0.040556 -> 0.040034` | [history](experiments/coverage_router/gf_fulltrain_dense_decoder_continue_seed3106/history.jsonl) |

K50 与 matched dense 的 epoch-0 loss 只差 `2.54e-10`，说明两条 decoder 分支
起点和 K100 计算对齐。没有 NaN/Inf、keep 漂移或 checkpoint 链错误。formal
refit 有意没有 validation，因此仅凭 train history不能单独排除过拟合；固定轮数
来自此前 split-clean 三 seed 结果。

最终 official test 的 4,782 张图全部 single-head。主结果见 1.2 节，完整原始
表见
[comparison.md](experiments/coverage_router/gf_fulltrain_official_test_seed3106/comparison.md)。

从现在起不再：

- 根据 official test 修改 K、route block、epoch 或选择另一个 checkpoint；
- 在 official test 上补 random/untrained control；
- 把 `pipeline complete` 误写成 accuracy gate pass。

---

## 5. 尚未完成的实验阶段

### 5.1 A1：Router 输入消融——当前第一优先级

**状态：⬜ 代码尚不支持；投稿前必须完成。**

固定相同 split、support 训练预算、K50 decoder curriculum 和 matched dense，比较：

| variant | 输入 | 回答的问题 |
|---|---|---|
| `geometry_only` | bbox-relative geometry | 简单几何 FOV 是否已经足够？ |
| `head_geometry` | head appearance + geometry | 头部朝向信息能否替代候选位置的 scene semantic？ |
| `full` | scene + head appearance + geometry | 当前完整 Router 是否确有必要？ |

完成标准：

- CLI、checkpoint `model_config`、manifest 和 evaluator 都记录 `router_input_mode`；
- 单元测试证明关闭分支后没有偷偷读取对应输入；
- seed 3106 的三种 variant 使用同一 split 和训练预算；
- 报告 support coverage、K50 gaze tuple、相对 full 和 matched dense 的差值；
- 只有 full 优势清晰且跨 epoch 一致时才补 3107/3108；若三者接近，应简化方法，
  不能强行宣称所有分支都有贡献。

当前下一动作：实现 `router_input_mode`、validator、双卡 runner，并把服务器
`nohup + CUDA 单卡` 命令写回实现文档；现在没有可安全运行的现成命令。

### 5.2 A2：预算 Pareto

**状态：⬜ 正式同协议表未完成；投稿前必须完成。**

在 split-clean validation 上补 K25、K75，与已有 K50、K100/matched dense 组成：

```text
keep ratio -> exact/hard coverage -> AUC/L2 -> B4/B8 latency/throughput
```

完成标准：所有 K 复用同一 learned support 起点、相同选模协议和 matched decoder
训练预算；先完成精度，再只对有意义的 Pareto 点测正式效率。该实验是消融，不得
回到 official test 重新选择主 K50。

### 5.3 A3：Early-exit fill 消融

**状态：⬜ 未开始；投稿前必须完成。**

比较：

- 当前 `early_exit_fill`：未选位置使用 route-layer 的浅层特征；
- `zero_fill`：未选位置直接置零。

该实验直接检验“保留浅层空间信息”是否是 sparse decoder 能接近 dense 的关键，
也把本 idea 与用户此前 ACMMM 中的多层空间信息洞察连接起来。

### 5.4 A4：路由位置

**状态：⬜ 未开始；强烈建议。**

比较 `route_after_block=2/5/8`，同时报告精度、Router coverage 和真实延迟。
目标不是事后寻找最好数字，而是说明：越早路由节省更多计算但语义不足，越晚路由
精度更稳但加速更小，block 5 是何种折中。

### 5.5 A5：多人 union 的正式效率

**状态：🟡 grouped 路径完成，效率对照未完成。**

固定每图 `1/2/4` 人，比较：

- per-image max-union：一张图只运行一次 prefix/suffix；
- per-person repeat：同一张图为每个人重复 backbone。

同时在 split-clean multi-head 子集报告 single/two/three-plus 分层精度。完成后
才能把“单分支共享 mask 比多分支 FOV 更简单、更省计算”写成实验证据。

### 5.6 A6：逐样本统计和定性分析

**状态：⬜ 未开始。**

需要在 validation 保存逐样本结果，而不是重新读取 official test：

- K50-vs-dense paired bootstrap/置信区间；
- head size、目标距离、边界、遮挡、人数分桶；
- support、hard mask、最终 heatmap 可视化；
- exact-point miss 但 gaze 正确、exact-point hit 但 gaze 错误的失败案例；
- Router 是否形成宽、多峰 support，而不是固定 cone。

### 5.7 A7：第二数据集

**状态：⚠️ VAT 当前失败；高风险，不阻塞 GazeFollow 主线。**

若投稿目标必须有第二数据集，优先级不是继续扫 K，而是设计固定 K50 总预算的
context-preserving mask。它必须继续保持单 mask、单 suffix，不回到多分支 FOV。
在 source-video-disjoint holdout 通过后，才讨论新的 formal full-train/official
协议。

---

## 6. 论文全链路进度

### 6.1 论文暂定定位

暂定工作标题：

> **Look Where It Matters: Coverage-Aware Spatial Routing for Efficient Gaze Target Estimation**

标题只是工作版本。最终应根据 Router 输入、回填和 VAT 结果决定突出
`coverage-aware routing`、`context preservation` 还是 `efficient gaze target`。

当前最稳妥的一句话 claim：

> We introduce a person-conditioned coverage router that prunes half of the
> patch tokens in the latter half of a DINOv3 gaze-target backbone, preserving
> most gaze accuracy while improving batched inference throughput.

这句话仍需在最终论文中明确：GazeFollow、K50、blocks 6–11、B4/B8，以及小幅
精度代价。

### 6.2 论文结构与当前状态

| 章节 | 当前已有内容 | 仍缺内容 | 状态 |
|---|---|---|---|
| Abstract | 一句话 claim 与正式主数字 | 完整背景、方法、结果、限制四句式摘要 | ⬜ |
| Introduction | 结构性计算错配、free-form support 动机 | 多篇文献证据、贡献定稿、与 FOV/object-aware 的关系 | 🟡 |
| Related Work | 零散讨论 | 系统文献表与三组 related-work prose | ⬜ |
| Method | 公式、模块、训练 recipe、代码均存在 | 论文图、伪代码、复杂度公式、消融后的最终版本 | 🟡 |
| Experimental Setup | split、指标、硬件、formal 协议已有 | 外部 baseline/SOTA 统一口径、实现细节精简 | 🟡 |
| Main Results | GazeFollow official 与正式效率已有 | 组合成主表并加入公开方法数字 | 🟡 |
| Ablations | learned-vs-untrained、matched dense、三 seed 已有 | 输入、预算、fill、route block | 🟡 |
| Multi-person | grouped 路径与分层数据已有 | union-vs-repeat 正式效率与最终 checkpoint 结果 | 🟡 |
| Cross-dataset | VAT 负结果完整 | 新方法若通过，或明确 limitation | ⚠️ |
| Qualitative | 无 | support/mask/heatmap/失败案例图 | ⬜ |
| Limitations | B1、显存、VAT、single-head official 已明确 | 组织成论文段落 | 🟡 |
| Conclusion | 无 | 与最终 claim 一致的结论 | ⬜ |
| Supplement | 详细命令、manifest、validator 已存在 | 整理额外表、更多可视化、匿名复现说明 | 🟡 |

### 6.3 计划中的图和表

| 编号 | 内容 | 数据是否就绪 | 状态 |
|---|---|---|---|
| Fig. 1 | dense gaze backbone 与 coverage-aware sparse backbone 对比 | 方法已定，未绘图 | ⬜ |
| Fig. 2 | Router：scene/head/geometry → per-person support → union Top-K | 方法已定，未绘图 | ⬜ |
| Fig. 3 | K50 accuracy–latency Pareto | 缺 K25/K75 正式点 | 🟡 |
| Fig. 4 | support/mask/heatmap 成功与失败案例 | 缺逐样本导出 | ⬜ |
| Table 1 | GazeFollow 与公开 SOTA 主表 | 本方法数字已有，外部数字未整理 | 🟡 |
| Table 2 | B1/B4/B8 latency、throughput、memory | 已就绪 | ✅ |
| Table 3 | Router 输入、fill、route block 消融 | 大部分未运行 | ⬜ |
| Table 4 | 三 seed 与 learned/untrained/matched dense | 已就绪 | ✅ |
| Table 5 | 多人分层与 union efficiency | 部分数据已有 | 🟡 |
| Appendix | VAT limitation、旧诊断、协议与额外表 | 证据已有，待整理 | 🟡 |

### 6.4 Claim—证据对应关系

| 论文 claim | 当前证据 | 充分性 |
|---|---|---|
| Router 学会了 gaze-relevant 区域 | learned vs untrained K25/K50 | ✅ 充分 |
| learned routing 对最终 gaze 有贡献 | K50 learned vs untrained | ✅ 充分 |
| K50 接近 matched dense | 三 seed holdout + formal official | ✅ 可写小幅代价，不可写无损 |
| K50 有真实计算收益 | K100 control + B4/B8 正反序 | ✅ 充分，限定批量场景 |
| scene/head/geometry 都必要 | 尚无输入消融 | ⬜ 不充分 |
| early-fill 保留浅层空间信息 | 尚无 zero-fill 对照 | ⬜ 不充分 |
| block 5 是合理折中 | 尚无 route-depth 表 | ⬜ 不充分 |
| 多人共享比多分支 FOV 更高效 | grouped loader 已通过，缺正式 latency | 🟡 不充分 |
| 跨数据集普适 | VAT accuracy gate 失败 | ⚠️ 当前不能写 |

### 6.5 投稿前写作与复现清单

- [ ] 建立独立的 gaze-target/efficient-ViT 最新文献 MD；
- [ ] 冻结标题、摘要 claim 和 contribution 数量；
- [ ] 完成 Router 输入、预算、fill 三项必要消融；
- [ ] 决定是否补 route-depth 和多人效率；
- [ ] 导出 validation 逐样本结果、bootstrap 和可视化；
- [ ] 整理公开 SOTA 的同协议结果，不混用不同输入/骨干/数据；
- [ ] 写 Introduction、Related Work、Method 初稿；
- [ ] 形成主表、效率表、消融表和方法图；
- [ ] 写 Experiments、Limitations、Conclusion；
- [ ] 整理匿名代码、依赖、数据说明、命令和 checkpoint provenance；
- [ ] 将正式小型结果复制到受版本控制的 immutable evidence bundle，并记录 SHA；
- [ ] 选择 venue，检查页数、模板、匿名政策和 checklist；
- [ ] 完成 supplement、内部审阅和最终复现实验抽查。

---

## 7. 决策与风险登记

### 7.1 已冻结的决定

1. GazeFollow 主点固定为 K50、`route_after_block=5`、single mask；
2. formal official K50/dense 结果已冻结，不再用于调参；
3. 成功训练采用 staged recipe，不再复跑失败的全 joint K25 recipe；
4. B1 只报告持平，效率 contribution 限定在 B4/B8；
5. 不声称显存收益；
6. VAT 不继续 K87.5 或 full-train official test。

### 7.2 当前主要风险

| 风险 | 影响 | 应对 |
|---|---|---|
| Router 组件消融后 full 与简化版接近 | contribution 变弱 | 简化方法并收缩 claim，不隐藏结果 |
| K50 只有 GazeFollow 成立 | 泛化性不足 | 尝试固定预算 context preservation；失败则明确 limitation |
| official test 无逐样本输出 | 无法做 formal paired CI | 在固定 validation 上完成统计，不重新读取 official test |
| official test 全 single-head | union claim 缺主测试证据 | 使用 split-clean multi-head + controlled latency |
| 成功 recipe 只训练 decoder | 非 end-to-end 质疑 | 如实描述 staged training；不要把失败 suffix run 包装为成功 |
| B1 无加速、显存略增 | 效率 claim 受限 | 定位为 batched throughput，不写实时/低内存 |
| 最新文献链尚未系统整理 | motivation 易显得只针对单篇论文 | R1 必须在 Introduction 前完成 |

### 7.3 不应进入论文主表的旧证据

- `gf_support_k025_seed3106`、`gf_sparse_fixedrouter_*` 等 protocol-cleanup 前的
  official-test-per-epoch 结果；
- 首轮 `benchmark_b1_amp*.json` 中受 autocast cache、顺序或 GPU 占用污染的数字；
- `query_unit_k100_comparison.json` 的首次 `passed=false` 结果，它已被 matched
  batch/precision 的
  [query_unit_k100_b4_amp.json](experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/query_unit_k100_b4_amp.json)
  取代；
- 8-sample sparse smoke 的 accuracy；
- VAT K50/K75 不能被选择性写成“AP 提升”，而忽略 AUC/L2 gate 失败。

---

## 8. 从现在开始的执行队列

### Now：当前唯一第一任务

实现 Router 输入消融：

```text
router_input_mode = geometry_only | head_geometry | full
```

同时完成 config/checkpoint/validator/test/runner/MD 命令闭环。代码完成前不要手工
启动新训练，因为现有 CLI 只会重复 `full`。

### Next：输入消融完成后

1. seed 3106 同协议比较三个 Router 输入；
2. 根据结果决定是否补 3107/3108；
3. 复用 frozen support 运行 K25/K75 validation Pareto；
4. 实现 early-fill vs zero-fill；
5. 再决定 route-depth 和多人效率的算力顺序。

### Later：论文闭环

1. 导出逐样本 validation 结果和可视化；
2. 决定是否进行 VAT context-preserving 新方法；
3. 完成 literature MD 和公开 SOTA 表；
4. 写论文初稿、图表、supplement 和匿名复现材料；
5. 决定 venue 并进入投稿检查。

---

## 9. 核心证据索引

### 9.1 正式主证据（A）

- [GazeFollow formal comparison](experiments/coverage_router/gf_fulltrain_official_test_seed3106/comparison.json)
- [GazeFollow learned K50 official result](experiments/coverage_router/gf_fulltrain_official_test_seed3106/learned_k50_image_b16_fp32.json)
- [GazeFollow matched dense official result](experiments/coverage_router/gf_fulltrain_official_test_seed3106/matched_dense_image_b16_fp32.json)
- [Formal pipeline completion and SHA chain](experiments/coverage_router/gf_fulltrain_official_test_seed3106/pipeline_complete.json)
- [Formal efficiency directory](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/formal_efficiency/)

### 9.2 正式 validation 证据（B）

- [Split-clean dense](experiments/coverage_router/gf_splitclean_dense_seed3106/summary.json)
- [Split-clean learned support](experiments/coverage_router/gf_splitclean_support_k025_seed3106/summary.json)
- [Split-clean learned K50](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/summary.json)
- [Split-clean untrained-router K50](experiments/coverage_router/gf_splitclean_sparse_untrainedrouter_decoder_k050_seed3106/summary.json)
- [Split-clean matched dense](experiments/coverage_router/gf_splitclean_dense_decoder_continue_seed3106/summary.json)
- [GazeFollow query-group audit](experiments/coverage_router/gazefollow_train_query_groups_seed3106.json)
- [VAT dense/support/K50/K75/matched directories](experiments/coverage_router/)

### 9.3 工程、诊断与负结果（C/D）

- [Sparse smoke](experiments/coverage_router/sparse_smoke/history.jsonl)
- [K25 joint-training failure](experiments/coverage_router/gf_sparse_b5_k025_seed3106/history.jsonl)
- [K100/K50/K25 initialization controls](experiments/coverage_router/gf_sparse_init_controls/)
- [Stage profile forward](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/stage_profile_forward_k100_k50_b1_amp.json)
- [Stage profile reverse](experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/stage_profile_reverse_k50_k100_b1_amp.json)

维护本文件时，应先更新对应原始结果，再更新阶段状态和派生结论；不要只改表格里的
数字而不增加证据链接。
