# R4-D5 训练集候选效用：本地实现检查

2026-11-19，版本 `r4_history_utility_v1`。28项CPU合成检查通过：候选生成12项（0.084秒）、计分9项（0.693秒）、最终实际CLI集成7项（1.866秒）。精确命令、环境和六个源码SHA见同名JSON。

- 五个候选固定为current/current_grid/previous_raw/EMA0.75/transport_previous；source→target cosine-softmax传输temperature0.1，过去raw空间质量仅一跳；原InOut保持。
- 原生768×32×32 FP16特征、FP32计算、手算传输方向/质量、EMA与原实现逐值一致、人物/缺口重置、未来隔离及64网格argmax通过。
- 四象限、互斥GT分组和分母、固定状态阻断、oracle并列优先当前/排除分辨率控制、原时序/状态函数与不修改输入通过。
- 真实合成CLI在删除全部test文件后完成；全量候选落盘/只读重开后才计分。改变GT不改候选，改变未来不改此前输出；来源篡改、重复输出、版本错误会停止。
- 工件SHA、原始全局行映射/坐标读回、state与leave独立计分核对通过。六个Python文件通过3.10 AST检查，§16两个bash命令块语法通过；原代码保持。

没有执行真实VAT D5分析或服务器任务，没有新的性能数字。下一步用户按实现计划§16.1在GPU1手动运行一次train分析，回传 `r4_history_utility.json/.md` 与provenance。大文件留 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/`，没有新checkpoint。GT最佳选择仅为固定集合的离线参考，不反馈记忆、不作为模型成绩。
