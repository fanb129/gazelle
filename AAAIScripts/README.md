# AAAI 2027 Experimental Scripts

完整执行顺序、3090 服务器命令、输出说明、Go/No-Go 标准与结果占位表见：

- [`AAAI2027_DIAGNOSTIC_AND_PILOT_PLAN.md`](./AAAI2027_DIAGNOSTIC_AND_PILOT_PLAN.md)
- [`AAAI2027_P3_EFFECT_FIRST_ALCHEMY_PLAN.md`](./AAAI2027_P3_EFFECT_FIRST_ALCHEMY_PLAN.md)：停止 P21 后的效果优先融合/损失搜索计划

新候选模型位于仓库根目录的 `AAAIModules/`；历史 `gazelle/` 只作为依赖导入，没有在本轮修改。

建议先运行 P0，不要直接启动 P1 长训练：

1. `failure_taxonomy.py`
2. `compare_failure_reports.py`
3. `hierarchical_feature_probe.py`
4. `audit_sasa_ggsf.py`
5. 根据 P0 结果决定是否运行 `train_gazelle_control.py` 与 `train_person_router.py`

所有真实服务器命令统一使用：

```text
/home/fb/anaconda3/envs/py310/bin/python
```

文档中的数据、checkpoint、输出和日志均使用服务器绝对路径；不要求预先设置环境变量。完整推理与训练使用 `nohup` 后台运行，smoke test 和汇总命令保持前台。
