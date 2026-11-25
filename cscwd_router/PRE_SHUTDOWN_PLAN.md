# 维护前实验收口

2026-11-25：用户同意缩减剩余训练。14:31停止scene-head及队列；停止时scene-only GF/VAT已经完整完成，因此两组结果保留。不再等待训练或自动启动其他分支。主模型仍为GF/VAT×ViT-B/L，K50、seed0、冻结backbone、仅train/test。

| 工作 | 已完成内容 | 后续 |
|---|---|---|
| 表4条件化打分消融 | scene-only、scene＋geometry各自GF support3＋decoder5→VAT8，full复用现有同预算结果 | 使用三个完成条件；scene-head不再训练、不列入正式表 |
| 表5真实多人 | VAT共同5,182帧，N=1/2/3，三个策略共9个FP32质量单元 | 计时尚未执行，须独占GPU；不据patch槽位推算实际加速 |
| 同权重K100 | 四个原K50 checkpoint各一次完整test | 已回传，不扫描其他预算 |
| 真实可视化 | 两组各10案例、6张PNG/6张SVG；补充组两个多人案例均3个GT画内 | 本机CPU继续排版，入口 `results/QUALITATIVE_FIGURES.md` |
| 结果归档 | 小报告/图片经Git回本地，大PT/gzip留服务器 | 路径/SHA见 `results/RESULT_PROVENANCE.md` |

表4的消融对象为learned scoring输入。scene-only仍保留与full相同的真实头框及head/escape优先规则；geometry关闭时同时移除4维绝对坐标和6通道相对几何，appearance关闭时移除头部池化外观。各VAT条件继承各自的GF，不能用full Router代替初始化。

完整模型相对scene-only的GT点覆盖在GF/VAT分别增加4.30/1.38个百分点；最终定位指标差异很小，AUC/AP并非全面提高，按真实数值陈述。GF/VAT×B/L原有效率四组保留；多人耗时仍为空，不为赶维护窗口在共享GPU上计时。

本轮停止记录与结果核对见 [W7-D审查](results/W7D_ABLATION_REVIEW.md)。历史训练命令保留在 [训练入口](RUN_FINAL_ABLATIONS.md)，不再按其resume命令恢复队列。已完成的推理与图入口见 [评测入口](RUN_FINAL_EVALUATION.md)。没有新增dense训练、预算扫描、时序模块或新数据来源。
