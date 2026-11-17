# R3 本地实现与检查

2026-11-17，基于 `58be651`。**R3 本地验收完成：24 项新增检查和 18 项 R2 回归通过。** 结构见 [r3_mechanism_design.md](r3_mechanism_design.md)，源文件 SHA256 与实际检查结果见 [r3_local_checks.json](r3_local_checks.json)。

| 检查 | 实际结果 |
|---|---|
| r3_model | 16 tests，0.721 秒，OK |
| r3_integration | 8 tests，2.826 秒，OK |
| r2_model_regression | 10 tests，0.701 秒，OK |
| r2_data_regression | 8 tests，0.077 秒，OK |
| Python 3.10 语法 | 四个新增 Python 文件全部通过 |
| 同 seed 共享模块初始化 | 普通 GRU 与 R3 的共享参数逐项相同 |
| 原生输入前向 | batch 2，768×32×32 FP16 特征，FP32 模块，两步有限输出/状态通过 |
| 参数量 | 普通 GRU 347,522；R3 372,677，增加 25,155 |
| 既有来源 | R2 报告引用的模型、数据、训练、指标源码 SHA256 全部保持一致 |

环境为本机 `/private/tmp/gazelle-r2-venv`，Python 3.13、Torch 2.6.0、CPU。命令中的线程数仅限制本地检查的 CPU 并发，不写入模型或服务器运行配置。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gazelle-r2-venv/bin/python -m unittest discover -s cscwd/code/stag -p test_history_model.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gazelle-r2-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_history_integration.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gazelle-r2-venv/bin/python -m unittest discover -s cscwd/code/stag -p test_temporal_model.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gazelle-r2-venv/bin/python -m unittest discover -s cscwd/code/stag -p test_temporal_data.py -v
```

集成实际调用既有 `training_inputs / train_epoch / evaluate_model`，将 R3 模型传入并完成临时合成数据的梯度更新、完整轨迹验证、预测缓存、checkpoint/预测状态重载和诊断 NPY/index 写入读回。改变 GT 标签和点不会改变前向输入、重置或预测历史。诊断按原全局索引与紧凑验证帧逐一对齐，padding 不写入；漏写、重复或分区外索引不发布成功结果。

移动 patch 和受控 gate 检查证明位置更新、保留、刷新、释放在实现中可表达；两次优化后双头、投影、GRU、gate 和对应增益具有有限非零梯度。它们不证明真实注视切换已被识别。本次没有新增 VAT 性能数字或服务器作业，临时合成 NPY/checkpoint 未入 git。

下一步实现 `r4_compact_v2` 正式训练入口，按相同预算运行普通 GRU 与新机制；五个 epoch 是后续计划，不是已跑结果。
