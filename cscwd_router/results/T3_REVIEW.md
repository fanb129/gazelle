# T3 服务器训练闭环审查 · 2026-11-21

结论：T3 工程验收通过，进入 T4 真实 GazeFollow 主实验。

已 pull 结果提交 `de1488d`。原始报告：[summary.json](t3_synthetic_seed3106/summary.json)，SHA256 为 `4ffe9e3d342368f0b1b301e932fd9c6ef19a4b00ca9089e9a3f90c36cb91d872`；独立评测 [standalone_sparse_eval.json](t3_synthetic_seed3106/standalone_sparse_eval.json) 的 SHA256 为 `ebcf8fc0d0c0d361225b4ccc0c2d3482de9a210209d6d0746b61a7a93bcb3d30`。二者均与 provenance 一致，18 个相关源码文件的 SHA 与本机一致。

实际环境：GPU 1、RTX 3090、PyTorch 2.6.0+cu124、FP32，官方 DINOv2 ViT-B 与官方 GazeFollow decoder，合成两图三人。

| 阶段 | epoch / 实际更新 | 累计逐人样本 | 实际更新范围 |
|---|---:|---:|---|
| support | 3 / 3 | 9 | 仅 Router |
| sparse decoder | 5 / 5 | 15 | 仅 decoder |
| matched dense | 5 / 5 | 15 | 仅 decoder |

- 初始 dense/K100 与原版 GazeLLE 输出逐值一致，最大差值 0.0、不一致元素 0。
- 稀疏课程实际为 `1.0 / 0.75 / 0.5 / 0.5 / 0.5`。两条 decoder 分支的起点权重 SHA、更新次数、人数预算与逐 epoch seed 相同。
- backbone 在所有阶段的可训练参数和带梯度参数数均为 0；support 没有改变 decoder，两个 decoder 阶段没有改变 Router。
- sparse/matched checkpoint 重新加载后输出均逐值一致，最大差值 0.0。
- 独立 sparse 评测的整个 metrics 字典与训练报告中的 sparse validation 完全相同，checkpoint SHA 和训练 signature 相同。
- 34/34 单测通过；逐阶段 history 和单独 summary 与总报告一致，所有 loss 有限。七个 PT 的路径及 SHA 已在服务器登记，权重留在服务器。

总报告的十项 `engineering_checks` 全部为 true。该运行验证了合成训练/评测/保存读取链路，没有产生真实 GazeFollow 精度或效率结论。下一步执行 T4 固定 full-train/no-eval recipe、一次 official test 和正反序效率矩阵。
