# ConvLSTM基线本机检查

2026-11-20，分支 `codex/coverage-aware-router`。本记录只验证实现，不代表VAT性能。

本机Python：`/private/tmp/gazelle-cscwd-check/bin/python`，临时venv复用已有 `/opt/anaconda3/envs/micro_expression`；Python 3.10、torch 2.12.0、torchvision 0.27.0、timm 1.0.31、NumPy 2.0.2、scikit-learn 1.7.2、pytest 9.2.0。

实际执行：

```bash
/private/tmp/gazelle-cscwd-check/bin/python -m pytest -q \
  cscwd/code/data_prep/test_task_cache.py \
  cscwd/code/stag/test_model.py \
  cscwd/code/scripts/test_baseline.py
```

结果：**9 passed**。原 `gazelle/utils.py:133` 在NumPy 2下产生5条 `DeprecationWarning`，不影响通过；原包保持未修改。

| 检查 | 已验证行为 |
|---|---|
| 冻结特征wrapper | 原heads输入钩子重放与原输出精确一致；全模型冻结/eval；forward仅RGB和bbox；FP32缓存与SHA读回 |
| 真实GazeSpot代码路径 | 使用合成backbone运行原GGSF、SASA、3层人物Transformer及原heads；真实任务特征尺寸下，零残差时热图和InOut均与底座逐元素一致 |
| 梯度 | 冻结heads始终eval但允许input梯度；首步双输出投影有梯度，更新后所有adapter层梯度非零且有限，原heads权重不变 |
| 因果与状态 | 未来输入扰动不改变前缀；跨chunk detach不改变数值；单slot reset等于新状态，其他slot继续历史 |
| 分chunk与覆盖 | 同人顺序不乱，跨chunk保留轨迹；跨边界帧号缺口、人物切换、padding重置；每个数组row只访问一次 |
| loss与计分 | MSE仅GT画内，BCE/AP全有效人物帧；全画外chunk损失有限；极低预测InOut不删除GT画内定位样本 |
| 两轮集成训练 | 合成图片/多人轨迹→缓存→训练→逐轮test→指标/history/选模；跨chunk的H/C确实detach后传入下一chunk |
| 文件读回 | 每轮、selected、last和history轮次一致，model state dict可加载；每个checkpoint SHA与provenance一致 |
| 选模 | AUC降序、L2升序、AP降序、较早epoch；完整一轮，不拼列 |

两个CLI的 `--help` 已实际运行，`git diff --check` 通过。新结果子目录JSON可被Git加入，PT仍忽略。`gazelle/`无改动，新代码无 `cscwd_v0/` import。

未连接服务器或运行真实VAT缓存、训练、评测。真实增益需要用户手动完成服务器作业后才能确认。
