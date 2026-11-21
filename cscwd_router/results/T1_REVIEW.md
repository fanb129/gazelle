# T1 服务器结果审查 · 2026-11-21

结论：T1 验收通过，进入 T2。

已通过 `git pull --ff-only origin codex/coverage-aware-router` 拉取结果提交 `32224be`。原始证据：[t1_dinov2_checkpoint_checks.json](t1_dinov2_checkpoint_checks.json)，SHA256 为 `2c0846dce19409473916dd3dedc2946ec1805be8fb186d14fa1767ed8cf6f105`，与 `RESULT_PROVENANCE.md` 登记一致。

实际执行环境：GPU 1、NVIDIA GeForce RTX 3090、PyTorch 2.6.0+cu124、CUDA 12.4。输入为官方 DINOv2 ViT-B/14 权重和用户指定的官方 ViT-B GazeFollow decoder。两份权重的实际路径及完整 SHA256 已登记在 provenance。

| 检查 | 实际结果 |
|---|---|
| 合成单测 | 9/9 通过，无 error/failure/skip |
| 检查矩阵 | B1/B2/B8 × FP32/FP16，共 6 组全部通过 |
| K100 backbone | 每组 `[B, 1024, 768]`；`max_abs_diff=0.0`，不一致元素 0，数值有限 |
| K100 原版 decoder | 每组逐值一致；B1/B2/B8 分别输出 1/3/12 人 heatmap |
| K50 后段长度 | 每组实际输入 513 tokens，即 512 patches + CLS |
| K50 early-exit fill | 每组未选位置逐值一致，最大差值 0.0 |

FP16 指开启 FP16 autocast：报告中的末尾 patch 特征保留 FP32，decoder heatmap 为 FP16。核对时使用实际 dtype 和逐值比较结果，没有放宽误差容限。

源码核对覆盖官方 backbone、官方 decoder、稀疏 backbone、合成测试、检查入口这五个文件，报告 SHA256 均与本机一致。报告保留了运行时 `git_dirty=true` 的事实；验收依据为上述实际源码 SHA、输入权重 SHA 和逐组数值结果。

该运行完成的是官方权重上的 T1 工程验收，没有产生 GazeFollow 精度或效率测量。下一步为 T2 的 DINOv2 block-5 support/union 接入与服务器合成可训练性检查。
