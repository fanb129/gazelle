# CSCWD：GazeSpot + ConvLSTM

2026-11-20，从新目录开始。目标是在GazeSpot上加入因果时序模块，提高VAT的AUC、L2、AP，保持官方train/test。

ConvLSTM先作为时序基线。基线有效后，再围绕具体不足设计一个新模块，以新增收益支撑创新点。

```text
cscwd/
├── implementation_plan.md  # 当前方案、执行约定与进展
├── code/
│   ├── data_prep/
│   ├── stag/
│   └── scripts/
├── results/                # 新实验的小JSON/MD及provenance
└── latex/                  # 新论文材料
```

当前只完成目录重建和方案初始化。下一任务按 [implementation_plan.md](implementation_plan.md) 实现任务特征缓存、ConvLSTM与完整训练入口。

旧目录全部归档到 `../cscwd_v0/`，其代码、文献、论文材料和结果均保留；旧根进度见 `../cscwd_v0/PROGRESS_SNAPSHOT.md`。历史脚本对应归档前提交 `a77c8be`，内部原路径/SHA保持原样；新代码独立建立，不从归档代码导入。

本机只实现与检查，服务器由用户手动运行；代码与小结果经Git传递。大型特征/checkpoint留服务器，路径和SHA记入 `results/RESULT_PROVENANCE.md`。
