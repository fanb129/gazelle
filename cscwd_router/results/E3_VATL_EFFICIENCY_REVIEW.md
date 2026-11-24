# E3-L：VAT-L 效率结果审查

2026-11-24。**任务已完成，审查通过。** 用户实际改用 GPU 0；四个单元均记录 RTX 3090 UUID `GPU-9abd86a3-e5fb-bca1-4843-b29aefbfbd46`，执行代码为 `0a37505`。原日志名仍为 `efficiency_vat_l_gpu1.log`，其文件名不代表实际卡号。日志有最终 `COMPLETE`，四个 JSON 单元和汇总 JSON/MD 齐全，服务器已无 Router 进程。

六份原始报告和 CPU 检查经提交 `afcc39e` push/pull 到本地。本轮只读核对既有结果，没有重新运行模型、训练或计时。原始汇总为 [summary.json](efficiency_dinov3_vatl_seed0_20261124/vat/vitl/summary.json)，检查记录为 [e3_vatl_efficiency_server_checks.json](e3_vatl_efficiency_server_checks.json)。

## 延迟

同一 GPU、同一真实 RGB/GT 头框，FP16 完整 forward；正反序各取 median 后用几何平均汇总。**Batch 8 的 ms 是整批 8 张图的时间。** speedup >1 表示 K50 更快。

| Batch | 官方 GazeLLE-L ms | 同权重 K100 ms | K50 ms | K50 vs 官方 | K50 vs K100 |
|---|---:|---:|---:|---:|---:|
| 1 | 33.4730 | 39.4368 | 38.7860 | 0.8630× | 1.0168× |
| 8 | 147.5857 | 181.8345 | 138.5573 | 1.0652× | 1.3123× |

- Batch 1 相对官方延迟增加 **15.87%**，相对同一路由实现 K100 的延迟降低 1.65%。
- Batch 8 相对官方延迟降低 **6.12%**（速度比 1.0652×），相对 K100 降低 **23.80%**（速度比 1.3123×）。
- 正/反序相对官方的速度比分别为 B1 `0.8602/0.8659×`、B8 `1.0703/1.0600×`。两种顺序都支持 B1 较慢、B8 小幅更快；当前是一组矩阵的实测结果，没有跨独立运行的显著性结论。

## 吞吐与显存

吞吐为 images/s，每图一个真实 GT 查询。显存为 PyTorch peak allocated MiB，包含模型和该完整前向路径的张量，不是整卡驱动占用。

| Batch | 官方 FPS | K50 FPS | 官方 peak MiB | K50 peak MiB | Peak 变化 |
|---|---:|---:|---:|---:|---:|
| 1 | 29.7193 | 25.8815 | 1215.75 | 1271.19 | +4.56% |
| 8 | 53.7989 | 57.3449 | 1423.95 | 1824.21 | +28.11% |

Batch 8 吞吐提高 6.59%。K100/K50 的峰值显存几乎相同，本次稀疏 suffix 没有降低完整系统的峰值；报告未单独定位峰值出现在哪个阶段。

## 精度、计算图与工件

使用此前验收的完整 GF-L Router→VAT-L、第 1 轮选中 checkpoint，SHA `6d12190c7cdf24fe0389eede68b27f0f05d35cb4c753072f62a3d0d1d8445bb5`。其独立 FP32 全量 test AUC/L2/AP 为 `0.948264/0.089676/0.891592`，选择与指标均与效率报告逐值一致。K100/K50 加载同一权重；K100 是带 Router 的全 token 计算控制，没有新增 dense 训练。本次未重测官方 VAT 精度，也未新增 FP16 精度结果。

实际图为：

| 变体 | 输入 | 24 个 blocks 的 tokens | 输出层索引 |
|---|---|---|---|
| Official DINOv2 ViT-L/14 | 448×448 | 全部 1025 | `[23]` |
| DINOv3 K100 | 512×512 | 全部 1029 | `[5,11,17,23]` |
| DINOv3 K50 | 512×512 | 前12层1029，后12层517 | `[5,11,17,23]` |

K50 实际保留 512 patches + 5 special tokens，输出回填完整网格后解码。计算缩减的实现证据明确；具体速度和显存以上表的完整系统测量为准。此处没有报告整网 FLOPs 实测值。

核验内容：

- 四个单元使用同一 GPU UUID、同一源码/输入/权重，seed 0、50 warmup、200×5 synchronized latency、500×5 throughput。Torch `2.6.0+cu124`、CUDA12.4、inference mode、autocast cache 和 TF32 关闭。读取、预处理和 H2D 在计时外，backbone/Router/decoder 都计时。
- 对照现行同版本代码重建四个请求完全一致；27 项源码 hash（25 个文件和两份 native DINO 源码树）、5 个来源/权重引用、VAT train annotation、8 张唯一 RGB 的实际 SHA 均核对。正反序使用相同原始 RGB 与 GT 头框。
- 12 组 block token trace、每变体 1000 个 latency sample 的报告计数、5 个 repeat median/FPS、数值有限性与吞吐 median 均核对通过。GPU 检查函数在每个 repeat 检测目标卡其他 compute PID，任务正常完成；四单元启动前快照均为 15 MiB、0% 利用率。
- summary JSON 和 MD 重新生成完全相同；独立核算速度比差值为 0。本机六份原始报告 SHA、按执行 commit 读取的25个源码文件 SHA 和延迟/吞吐几何平均全部吻合。
- 逐次 latency 数组没有导出，不能从已有报告独立重算 variant 的 median/p95；已完整核对保存的统计量结构和矩阵汇总。

## 论文口径与下一步

VAT-L 可写为“相对同权重 routed K100，batch 8 加速 1.31×；相对官方 GazeLLE-L，batch 8 有 1.065× 的小幅速度收益”。同时列出 B1 的代价和更高峰值显存。现有精度报告中的 AP 短板保留，不写成精度、延迟和内存全面占优。

下一项回到已确定的近期强方法精度—成本比较：优先官方 TextGaze B/L 任务权重的推理对照，保持 seed0、train/test 标签和输入差异说明；目前尚无与这些外部方法的同机速度比。不新增 dense 训练。本轮已完成的官方/K100/K50 矩阵保留为实际基线。
