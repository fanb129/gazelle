# COTB 实验设计（预注册版）

## 1. 研究问题

本实验不再检验 hierarchy disagreement、SASA dynamicity 或 GGSF geometry。唯一问题是：

> 当场景、对象和显著性完全不变，只把被查询人物从 A 换成 B 时，gaze heatmap 是否能稳定切换到 B 自己的目标？

可证伪假设：

- **H1（failure existence）**：至少两个已训练模型在 VAT far-target pair 上有不低于 `10%` 的 swap error。
- **H2（method effect）**：COTB 相对 matched `λ=0` control，使 far-target swap error 绝对下降至少 `2` 个百分点，且按 VAT sequence 聚类的 paired 95% bootstrap CI 不跨 0。
- **H3（no metric trade-off）**：VAT L2 恶化不超过 `0.003`，in/out AP 下降不超过 `0.005`。
- **H4（gaze-specific effect）**：own-target rank、diagonal margin 同时改善，且 far/crowd 子集的变化不弱于 overall；否则可能只是普通 heatmap sharpening。

## 2. 数据协议

### 主数据：VideoAttentionTarget

- `train_preprocessed.json` 原生按 sequence→frame→heads 保存，能恢复同一帧的所有 observer queries。
- 训练/验证都来自官方 train；按完整 VAT sequence 划分 `90%/10%`，固定 `validation_seed=9102`。
- 官方 test 只在 checkpoint 和 `λ` 确定后运行；full evaluation 使用 `frame_sample_every=1`。
- 训练为了速度使用 `frame_sample_every=6`，control 与 COTB 完全相同。
- 几何增强只做“整帧一致”的水平翻转；不复用旧逐人物 random crop，因为不同 query 得到不同场景会破坏反事实中的 fixed-scene 条件。bbox jitter 仍可逐 head 施加，但 pair 有效性始终依据 jitter 前的原始框。

### 辅助数据

- GazeFollow：只训练同架构的初始化 checkpoint，并报告标准 single-query 指标；不把不完整同图标注强行用于 binding loss。
- GOO-Real：当前预处理缺少可靠同帧多人分组，只做后续 OOD localization，不作为 COTB 主证据。
- ChildPlay：若之后取得完整多人/目标标注，再作为外部 social/shared-attention 验证；不阻塞当前 pilot。

## 3. Pair 构造与防伪控制

对每帧所有 in-frame query：

1. gaze point 距离不超过 `0.06` 的目标做 single-link clustering；同 cluster 表示共同注视。
2. 只保留不同 target cluster 的 query pair。
3. cluster center 距离小于 `0.10` 的 pair 不训练，避免 heatmap 分辨率导致不可区分。
4. 两个 head bbox IoU 不低于 `0.80` 时视为疑似重复 track，不进入 pair loss。
5. out-of-frame 和缺失目标只参与 in/out loss，不进入 binding loss。

对 query `i` 和目标 cluster `c`，在预测 heatmap 上用 `σ=0.04` 的 Gaussian 局部平均得到 `S(i,c)`。pair margin 为：

```text
Δij = S(i, ci) + S(j, cj) - S(i, cj) - S(j, ci)
Lbind = max(0, margin - Δij), margin=0.20
```

每个 query 在等式两侧各出现一次，单纯把整张 heatmap 放大不能绕过该约束。

## 4. 指标与统计

### 标准指标

- VAT AUC、L2、in/out AP。

### Primary binding 指标

- `swap_error = mean(Δij <= 0)`，越低越好。
- primary subset：target-cluster separation `>=0.30` 的 far pairs。

### Secondary binding 指标

- diagonal accuracy、mean diagonal margin；
- own-target rank、ownership accuracy/margin；
- overall、crowd `>=4`、crowd `>=5`、far & crowd `>=4`。

### 统计单位

- 所有 control-vs-COTB 比较先按 `(frame path, query_i, query_j)` 严格配对。
- bootstrap 以 VAT sequence 为 cluster，而不是把连续视频帧当成独立样本。
- final confirmation 使用 seeds `3106/3407/4508`；同时保留每个 seed 的 cluster CI，并报告 seed-level mean±std。

## 5. 分阶段实验

### E0：Annotation audit（无训练）

目的：确认 far pairs `>=1000` 且有效 sequences `>=30`，并检查 shared target、too-close pair、重复 bbox 的比例。若不满足，主张缺少统计支撑，直接 No-Go。

输出：

- `train_annotation_audit.json`
- `test_annotation_audit.json`

### E1：Failure-existence diagnostic（无训练）

用统一 COTB evaluator 重新评价：

- 历史 single-layer v0 448；
- 历史 GazeSpot 512。

目的不是比较二者优劣，而是确认 swap failure 不是某个单一实现的偶发现象。必须使用 raw heatmaps。

### E2：一天 mechanism pilot

为节省时间，使用同一个历史 GazeSpot GF checkpoint 初始化两个 VAT run：

| Run | Loader | Architecture | COTB weight | Epoch |
|---|---|---|---:|---:|
| grouped control | frame-grouped | SASA+GGSF | 0.00 | 2 |
| COTB pilot | frame-grouped | SASA+GGSF | 0.10 | 2 |

二者随机种子、初始化、sequence split、增强、batch 和 optimizer 完全相同。这个阶段只决定 COTB objective 是否值得继续，不把 SASA/GGSF 写回论文贡献。

### E3：Weight screen

E2 为 GO 后，在 clean base 上固定 seed `3106`，比较：

```text
λbind ∈ {0.00, 0.05, 0.10, 0.20}
```

只用 train-validation 的 `joint = L2 + 0.1 × overall swap_error` 选一个非零 `λ`。禁止看 test 选择权重。

### E4：Clean three-seed confirmation

最终模型固定为：

```text
DINOv3 four-layer raw concat + no GGSF + no SASA + original Gazelle decoder
```

对 `λ=0` 和选定 `λ>0` 运行三种子、8 epochs；按同一个 joint rule 从 train-validation 选 `best_joint.pt`，最后仅一次在完整 VAT test 上评价。

### E5：必要消融（E4 成立后）

- shared-target aware (`radius=0.06`) vs near-naive (`radius=0`)；检验共同注视误负例是否关键。
- far-only reporting vs all eligible pairs；验证结果不是 close-target 像素噪声。
- `λ={0.05,0.10,0.20}`；验证不是单点超参巧合。
- 按 people count、target separation、head size 分层；不能只展示最有利子集。
- 至少再接一个公开强 base model；否则 novelty 容易被评价为 Gazelle-specific loss trick。

## 6. Go/No-Go

`compare.py` 的 primary far subset 同时通过以下门槛才是 GO：

1. matched pairs `>=1000` 且 sequences `>=30`；
2. control swap error `>=10%`；
3. COTB 绝对改善 `>=2` 个百分点；
4. paired cluster-bootstrap CI 上界 `<0`；
5. L2 delta `<=+0.003`；
6. in/out AP delta `>=-0.005`。

如果只降低 L2、却不改善 swap error/own-target rank，本 idea 被证伪为“普通定位增强”；如果只改善训练 base、无法迁移到第二个 base，则不能支撑方法级投稿。

## 7. 预期主表

| Method | VAT AUC ↑ | VAT L2 ↓ | AP ↑ | Far Swap Error ↓ | Far Diag Margin ↑ | Ownership Acc ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Clean grouped control |  |  |  |  |  |  |
| + COTB |  |  |  |  |  |  |
| Second public base |  |  |  |  |  |  |
| Second public base + COTB |  |  |  |  |  |  |

论文最关键的图不是模块结构图，而是同一帧中 query 从人物 A 切到 B 时，两张 raw heatmap 是否随之交换到正确目标。
