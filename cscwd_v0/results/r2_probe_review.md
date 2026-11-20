# R2 feature probe 本地验收

2026-11-17，结果提交 `139a2c7`。**probe 验收通过，接受现有契约，可以执行实现计划 §9.3；无需改源码或重跑 probe。**

来源：[r2_feature_probe.json](r2_feature_probe.json)；结构化核对：[r2_probe_review.json](r2_probe_review.json)。原机器报告保持不变。

| 项目 | 真实结果 |
|---|---|
| 抽样 | 32 张唯一开发 train 图像 / 69 人物帧 |
| 特征 | layer index 11，768×32×32，原生网格，FP16 存储；不缓存可学习投影 |
| FP32 原预测对齐 | heatmap max abs 9.83476639e-07；In/Out 4.17232513e-07；阈值 1e-5，argmax 变化 0 |
| FP16 量化 | max abs 0.001953125；RMSE 8.29578761e-05；relative L2 0.000204183689（0.02042%） |
| 抽样头框覆盖 | 最小框面积折合 2.504444 个 patch 面积；69 框中无 patch 中心命中的框为 0 |
| 全量缓存估算 | 92,023,554,048 bytes，约 85.7036 GiB，覆盖 58,507 张唯一图像 |
| 探测时可用空间 | 2,731,964,395,520 bytes，约 2.4847 TiB |

代码 SHA256、R1 机器/人工报告 SHA256、split/index 身份和 checkpoint/backbone 身份与本地既有记录一致；抽样均来自开发 train，含全部 R1 案例锚点。原报告的 8 项机器检查通过，另核对了抽样路径/计数、头框面积/中心覆盖和存储算式。
FP32 原预测对齐与 FP16 特征量化是两项独立检查。缓存原始 heatmap/In/Out 保持 float32；不将 FP16 特征称为无损，也不据量化误差宣称定位/AP 无损。头框覆盖结论只针对这批样本，模型仍保留分数面积池化。
本地没有真实 RGB/权重/特征，因此没有重新前向计算数值；核对的是回传来源链和独立一致性关系。该 probe 不代表全量缓存完成或新模块已训练。

下一步：先按 §9.3 写完整场景特征；看到 `R2 feature cache completed` 且报告 passed 后，按 §9.4 运行固定一个 epoch 的普通联合头 smoke。缓存入口会再次检查剩余磁盘、probe/source/code/transform/runtime 一致性。cache 与 smoke 小报告回传后，再验收 R2；当前不进入 R3。
