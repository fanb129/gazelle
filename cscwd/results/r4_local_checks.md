# R4 第一轮本地实现与检查

2026-11-17，基于 `292c567`，计划 `r4_compact_v2`。**本地入口完成，31 项新增检查通过，等待用户手动运行服务器第一轮。**

新增固定预算/选择协议、同次前向完整验证、单模型训练、比较报告与顺序运行 shell 入口。原 R1/R2/R3 来源和模型源码保持不变，R4 计划 SHA256 与 R3 记录一致。

| 检查 | 实际结果 |
|---|---|
| r4_protocol | 11 tests，0.002 秒，OK |
| r4_validation | 5 tests，2.559 秒，OK |
| r4_training | 6 tests，8.455 秒，OK |
| r4_comparison | 9 tests，6.762 秒，OK |
| Python 3.10 语法 | 8 个新增 Python 文件通过 |
| Shell 语法 | 顺序脚本及 §10 的 5 个命令块通过 |

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gazelle-r2-venv/bin/python -m unittest discover -s cscwd/code/stag -p test_compact_protocol.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gazelle-r2-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_compact_validation.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gazelle-r2-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_train_compact_joint.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gazelle-r2-venv/bin/python -m unittest discover -s cscwd/code/scripts -p test_compare_compact_joint.py -v
bash -n cscwd/code/scripts/run_compact_round.sh
```

环境：本机 CPU、Python 3.13、Torch 2.6.0，解释器 `/private/tmp/gazelle-r2-venv/bin/python`。Torch CLI 子进程检查使用沙箱外本地合成执行，解决本机 OpenMP 共享内存限制；生产代码未加入环境绕过。

实际跑通两种模型各五个 epoch 的临时合成训练与完整验证，核对共享初始化、双输出更新、checkpoint 与预测重载、全部 epoch 和选中/最后 checkpoint、诊断写入、完整 CLI JSON/MD、开发选择快照及 provenance。受控验证场景覆盖最佳 epoch 不等于最后 epoch、无合格模型、原始帧事件配对、来源或 checkpoint 不一致。结果没有将“作业成功”或 Pareto 成员视为方法有效。

源文件 SHA256 见 [r4_local_checks.json](r4_local_checks.json)。这些是本地检查，不是 VAT 实验；临时预测、checkpoint 和合成报告未入 git。当前没有 R4 真实性能数字。下一步由用户按 [实现计划 §10](../implementation_plan.md#10-精简-r4-第一轮服务器运行当前入口) 手动运行并回传。
