# R4-D4 当前帧无历史对照：本地检查

2026-11-19。版本 `r4_current_frame_v1`，沿用 `r4_official_train_test_v1`。

23项CPU合成检查通过：模型7项（0.145秒），脚本16项（14.531秒；训练/实际比较CLI6、小报告比较7、wrapper3）。完整命令、运行环境与源码SHA256见同名JSON。

- 每帧无条件清空hidden/过去heatmap/过去InOut，忽略传入state/reset并返回None；帧顺序、之前输入与chunk边界不影响当前预测。
- 与原GRU空状态单步和共享初始化一致；当前投影、双输出头与bias有梯度。零历史权重任务梯度为零，但AdamW仍可令其变化。
- 实际合成五轮训练/完整test/比较CLI走通，全部epoch/checkpoint/选轮清单保留；改变test标签不改变五轮训练权重，重载匹配预测。
- 比较核对原首轮小报告来源、初始化、预算、单帧基线、选中轮，保留全部同epoch差值；旧轮大数组/权重不要求重读。
- wrapper验证GPU 1继承、含空格路径、单模型训练→比较与任一步失败停止。
- 七个新增Python文件通过3.10 AST语法检查；wrapper和§15两个服务器命令块通过bash语法检查。原已跟踪代码保持，结果JSON可进git。

768通道/hidden128下名义参数347,522，其中74,112个权重元素对应结构性零历史输入。该计数不表示有效自由度或FLOPs相同；区别记录于 `parameter_usage`。

未运行真实VAT D4训练，没有新的性能数字。下一步由用户按实现计划§15.1在物理GPU 1手动运行五轮，每轮完整test，按AUC→L2→AP→较早epoch选整轮；test参与选轮。新实验目录为 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/`。回传 `r4_current_frame.json/.md`、comparison JSON/MD、selection JSON及provenance；NPY/PT/完整明细留服务器。
