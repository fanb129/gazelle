# 真实可视化图片入口

2026-11-25。GPU1已完成首批和三个GT均在画内的补充组，两组各导出10案例、6张PNG/6张SVG；原图保留。模型、GT、support、实际Top-K及预测均来自真实前向。GPU0的scene-only→scene-head队列继续训练。

## 论文优先查看

| 图 | 入口 | 展示内容 |
|---|---|---|
| 人物条件与多人共享 | [多人机制图](../latex/figures/final_inframe3_seed0/vat_multi_00_mechanism.png) · [SVG](../latex/figures/final_inframe3_seed0/vat_multi_00_mechanism.svg) | 同一RGB切换GT查询人物，support、独立选区、max-union共享选区与最终预测；三人GT均在画内 |
| GF预测对比 | [PNG](../latex/figures/final_inframe3_seed0/gazefollow_comparison.png) · [SVG](../latex/figures/final_inframe3_seed0/gazefollow_comparison.svg) | 六行：RGB、GT、support、实际保留patch、官方热图、Router热图 |
| VAT预测对比 | [PNG](../latex/figures/final_inframe3_seed0/vat_comparison.png) · [SVG](../latex/figures/final_inframe3_seed0/vat_comparison.svg) | 同样六行，保留较高误差样例及真实p(in) |
| 独立与共享预测 | [PNG](../latex/figures/final_inframe3_seed0/vat_multi_00.png) · [另一案例](../latex/figures/final_inframe3_seed0/vat_multi_01.png) | 同权重逐人K50与共同预算K50的真实预测和误差 |

GT为彩色星号，预测为白色叉号；连线只连接真实GT或模型预测。变暗patch表示该位置跳过suffix，dense prefix仍处理整幅RGB，之后回填完整解码网格。三人独立为1536个suffix patch槽位，共享为512；特殊token另外保留。完整模型的时间数据由多人效率实验另行测量。

GF/VAT单人各从seed0固定32候选中选两例较低误差与两例较高误差。补充多人先要求固定三个GT查询都在画内，再从seed0固定16候选按support JS差异选两例；两例都来自同一采访序列的不同帧。该筛选只用于机制展示，量化多人评测保留原5,182帧和画外查询。

## 数据与重画

- [补充组manifest](final_visualizations_inframe3_seed0/manifest.json)、[服务器核对](w7c_inframe_visualizations_server_checks.json)。
- [首批manifest](final_visualizations_seed0/manifest.json)、[服务器核对](w7c_visualizations_server_checks.json)，包含画外案例。
- 每个case保留小RGB相对路径、checkpoint/源码SHA、GT、原始预测点/误差及小尺寸数组；PNG/SVG已回本地，可用CPU渲染入口继续排版。

GPU0训练日志：`/home/fb/src/paper/gazelleV1/logs/cscwd_router/final_ablation_queue_resume_gpu0.log`。队列PID640838、训练子进程643229（本次启动时）；当前恢复到scene-only decoder第2轮，之后自动接VAT8及scene-head完整分支。
