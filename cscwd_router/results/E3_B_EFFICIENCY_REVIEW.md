# E3-B：GF-B / VAT-B 正式效率审查

日期：2026-11-24。12 份原始 JSON/MD 经服务器提交 `85210b55b4832e31f7980a18e45397be29e80cfc` push/pull 回传；本次只做 CPU 工件、源码与汇总核验，没有重跑模型或计时。检查记录：[e3_b_efficiency_server_checks.json](e3_b_efficiency_server_checks.json)。

结论：K50 相对同一路由实现的 K100，在 B8 有约 1.25× 加速；相对官方 GazeLLE，B1 明显更慢，B8 的 GF-B 基本持平、VAT-B 仅小幅更快。B8 峰值 allocated memory 高约 59.5%。当前两个 B 模型不支持“比 GazeLLE 明显更快且更省显存”的行文；继续保留定位收益与批量成本接近的实测权衡，下一步先定位和减少实现开销。

## 1. 完整系统延迟

同一模型按正反两种顺序分别求比，再取几何平均。时间单位 ms，**B8 是整批 8 张图的延迟**。speedup >1 表示 K50 更快。

| 模型 | Batch | 官方 GazeLLE | K100 | K50 | K50 vs 官方 | K50 vs K100 |
|---|---:|---:|---:|---:|---:|---:|
| GF-B | 1 | 17.9309 | 23.3171 | 22.9234 | 0.7822× | 1.0172× |
| GF-B | 8 | 53.3757 | 67.3777 | 53.8184 | 0.9918× | 1.2519× |
| VAT-B | 1 | 17.9408 | 21.8600 | 21.7323 | 0.8255× | 1.0059× |
| VAT-B | 8 | 51.6012 | 63.2172 | 50.3369 | 1.0251× | 1.2559× |

GF-B B1 延迟增加 27.84%，VAT-B B1 增加 21.13%。B8 GF-B 延迟增加 0.83%，VAT-B 降低 2.45%（speedup 1.0251×）。VAT 的 2.5% 速度比收益是这次矩阵的实际小幅变化，尚不能写成显著或稳定的大幅跨模型加速。

正反序的跨模型 speedup：

| 模型 | Batch | Forward | Reverse |
|---|---:|---:|---:|
| GF-B | 1 | 0.7871× | 0.7774× |
| GF-B | 8 | 0.9563× | 1.0285× |
| VAT-B | 1 | 0.8226× | 0.8285× |
| VAT-B | 8 | 1.0238× | 1.0264× |

GF-B B8 两种顺序跨过 1，当前按“基本持平”处理。VAT-B B8 两种顺序均略高于 1，保留其小幅收益；未做跨独立运行的显著性判断。

## 2. 吞吐与峰值显存

吞吐为 images/s，本轮每图一个真实 GT 查询。显存为 PyTorch peak allocated MiB，不是 reserved memory 或整卡驱动占用。

| 模型 | Batch | 官方 FPS | K50 FPS | 官方 peak MiB | K50 peak MiB | Peak 变化 |
|---|---:|---:|---:|---:|---:|---:|
| GF-B | 1 | 55.72 | 43.59 | 376.35 | 415.00 | +10.27% |
| GF-B | 8 | 149.09 | 147.68 | 537.50 | 857.16 | +59.47% |
| VAT-B | 1 | 55.74 | 45.52 | 376.47 | 419.12 | +11.33% |
| VAT-B | 8 | 154.89 | 158.77 | 537.63 | 857.35 | +59.47% |

K50 与 K100 的峰值几乎相同；token 稀疏化没有在本次完整系统中转化为峰值显存下降。完整网格回填、多层输出和 decoder 的张量保留是需要分阶段测量的具体对象；当前报告没有峰值发生阶段的独立证据。

## 3. 精度与效率使用同一模型

- GF-B 为已审查的第 5 轮 best，FP32 AUC/Avg/Min L2 `0.956236/0.101550/0.043420`。本次 K100/K50 都加载同一个 checkpoint，官方效率加载原 `gazelle_dinov2_vitb14.pt`。与既有官方 GF-B FP32 报告相比，定位误差改善、AUC 基本持平；本次 B1 代价更高、B8 成本接近。
- VAT-B 使用完整 GF→VAT 的第 1 轮 best，最终逐帧 FP32 AUC/L2/AP `0.940033/0.100131/0.883183`，不是旧较低精度 VAT-B。官方效率加载原 `_inout.pt`；本次不是官方 VAT 精度复评，保留原精度报告与 AP 短板。
- FP16 效率与 FP32 精度分别标注；计时输入来自 train 的真实 RGB/GT 框，不读取 gaze 坐标进行选样，没有改变 checkpoint 选模。

## 4. 计算图和口径核验

两组各 B1/B8 × 正反序 4 单元，每单元独立进程；official→K100→K50 与反序。均 RTX3090、Torch2.6.0+cu124、CUDA12.4、FP16、inference mode、autocast cache 关闭、TF32 关闭。每 variant 50 warmup、200×5 synchronized latency、500×5 throughput，均保存 1,000 个 latency samples 的汇总、5 个 repeat median 与 FPS。完整 forward 包括 backbone、Router、gather/scatter、多层融合和 decoder；读取/预处理/H2D 在计时外。

- 官方：448 输入，12 个 blocks 每层 1,025 tokens，最终单层输出 `[11]`。
- Router K100：512 输入，12 层均 1,029 tokens（1,024 patches + 5 special tokens），输出 `[2,5,8,11]`。
- Router K50：前 6 层 1,029 tokens、后 6 层 517 tokens，输出仍回填完整网格。实际少算了 suffix token，没有误跑成 K100。

K100 是同一个已训练 K50 模型、同一路由实现的全 token 控制，仍包含 Router/选择/回填开销，不能称为单独训练或原生实现的 DINOv3 dense。约 1.25× 是相对这一控制的收益。

GF-B 与 VAT-B 分别在 GPU0/GPU1 上比较，各自官方与 K50 使用同一 GPU UUID、同一原始 RGB 和真实头框；不跨卡汇总绝对延迟。完整来源与 SHA 在原始 cell JSON 中。

## 5. 工件与源码核验

12 份回传报告 SHA、本机 50 个按记录 Git 版本读取的源码文件检查通过。8 个 source/weight 引用路径（其中 6 个 PT/PTH）、2 份 train annotation 和 16 个选中 RGB 路径 SHA 均核对。24 组 variant token traces、8 个单元设置/输入、checkpoint 与原精度选择一致；summary JSON 与 MD 重新计算完全相同，另按原始两序数据独立核算 speedup 一致。

GF 运行源码为 `4162da7`，VAT 修复源码为 `29c6607`；两组记录的源码差异仅为修复 VAT 数据解析的 `dinov3_efficiency.py`，模型、计时器与原生 DINO 库一致。每组内部四个单元源码完全相同。计时器在每个 repeat 检查目标 GPU 的其他 compute PID，任务正常完成，报告保存单元前后 GPU 状态。

逐次 latency 原始数组没有保存，本次不能独立重算每 variant 的 median/p95；已核对其采样数、有限值、repeat 数与 throughput median，并完整重算矩阵汇总。所有原始报告保留，没有替换接近 1 或小于 1 的 speedup。

数据提交通过已空闲的 VAT 修复 worktree 完成，再恢复其 `29c6607`；原目录 HEAD4162da7 和核心源码不变，GF-L 效率与 VAT-L 训练继续。大权重未下载或入 Git。

## 6. 推进方向

首先在空闲 GPU1 上对同一 checkpoint 做阶段 profile，并列保留完整 forward 测量：dense prefix、Router、gather/suffix/scatter、多层融合、decoder。补一个相同 task weights 的 native DINOv3 full-token 推理控制，分清 Router 自身与多层底座的成本；不新增 dense 训练。

静态代码已有具体候选：`gazelle/model.py` 的 raw-concat 路径先按人物复制四层特征，`spatial_prior=none` 仍创建全一掩码并执行四次乘法，再拼接投影；`repeat_tensors` 即使单人也执行 repeat/cat。官方单层 decoder 先投影再按人复制。它们确实存在于当前路径，但占多少毫秒和峰值显存仍需测量。

随后只在 `cscwd_router/code/` 做保持权重和语义的推理快路径，分别在 FP32、FP16 与单人/多人设置比较原版和快路径输出，再做完整精度与效率复测。当前不预写优化收益；保留这轮作为实际对照，GF-L/VAT-L 继续原任务。原 DINOv2 T4 的 gate 不套用到本轮 DINOv3 结果。

原始结果：[GF-B summary](efficiency_dinov3_seed0_20261124/gazefollow/vitb/summary.json)、[VAT-B summary](efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/summary.json)。
