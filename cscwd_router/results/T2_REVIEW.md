# T2 服务器结果审查 · 2026-11-21

结论：T2 可训练性与结构验收通过，进入 T3。

已 pull 结果提交 `a08e944`。原始证据：[t2_dinov2_support_checks.json](t2_dinov2_support_checks.json)，SHA256：`b7f6561e4322311e1b6f8e0937528356080739754609658a8316786a051bfc66`，与 provenance 一致。八个相关源码文件与本机 SHA256 匹配，backbone 权重 SHA 与 T1 相同。

实际环境为 GPU 1、RTX 3090、PyTorch 2.6.0+cu124、FP32；两张合成 RGB、三个人，冻结的官方 block-5 特征形状 `[2, 768, 32, 32]`。80 步只更新 Router，所有 history 数值有限，步号连续且包含初始值与第 80 步。

| 指标 | step 0 | step 80 |
|---|---:|---:|
| coverage loss | 3.465735674 | 0.000011941 |
| total loss | 3.493860722 | 0.003062376 |
| soft coverage | 0.031250000 | 0.999988079 |
| hard coverage | 0.011285764 | 0.961666822 |

第 2 步的 scene/head/geometry 梯度 L1 分别为 `13.757266 / 3.529962 / 0.017381`，均非零。backbone 可训练参数与带梯度参数数均为 0。

K25/K50/K100 分别保留 256/512/1024 patches；三种预算的按图共享 suffix、逐人映射、精确 max-union、固定预算与排序全部通过。K25/K50 回填逐值一致；K100 与官方 backbone 逐值一致，最大差值 0.0、不一致元素 0。合成单测 20/20 通过。

这些结果证明 Router 在冻结官方 DINOv2 特征上的合成可训练性及共享执行路径，没有产生真实 GazeFollow AUC/L2 或效率结果。下一步为 T3 的官方 decoder 接入、固定 3+5+5 训练、标准指标和 checkpoint 闭环。
