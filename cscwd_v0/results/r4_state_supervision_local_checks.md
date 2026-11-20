# R4-D3状态时序监督：本地实现检查

2026-11-18，版本 `r4_state_supervision_v1`。**29项CPU合成检查通过**；没有运行真实VAT D3训练或服务器任务。

| 部分 | 检查数 | 主要内容 |
|---|---:|---|
| loss与epoch | 12 | 手算L1/BCE、独立分母/空mask、已发生翻转窗口、reset/padding、跨chunk计数/梯度detach、两模型实际更新、GT不进入前向 |
| 训练/完整比较CLI | 6 | 两模型从头五轮、监督覆盖、逐轮完整test/保存/选轮、checkpoint重载、test标签不改变权重、实际比较CLI与SHA |
| 比较小报告 | 7 | 固定新监督设置、前轮同源/预算/初始化/结构、整轮身份、差值、旧大文件不重读、summary/manifest |
| 后台入口 | 4 | GRU→新机制→比较顺序、物理GPU 1继承、空格路径、任一步失败停止 |

实际整体执行为12项（0.649秒）与17项（18.646秒）；完整命令和八个新增源码的SHA见同名JSON。环境为Python 3.13.0、Torch 2.6.0 CPU、NumPy 2.2.6、scikit-learn 1.9.1；七个新Python文件通过3.10语法解析，wrapper及§14两段命令通过 `bash -n`。

loss固定为 `L_heatmap + 0.1 * (L_BCE + 0.1 * (L_stable + L_transition))`。同GT状态相邻分数采用L1；真实状态翻转当前及随后两帧增加当前状态BCE，分别归一化。监督carry独立于模型预测历史，跨chunk边界detach，跨人物/缺口重置；GT只计算loss。

两种D3模型各五轮临时合成训练及比较CLI实际完成，保存全部epoch/selected/last与同次诊断；监督计数与独立metadata计数一致。改变全部test状态标签/注视点后，五轮训练权重逐bit相同。原架构、评测/选轮函数、缓存和既有源码均未修改；新旧选中输出按同一来源/预算对比，本轮不使用D2校准。

检查仅验证实现主路径与数据/梯度边界，合成指标不作为真实VAT性能。下一步由用户按 [实现计划§14.1](../implementation_plan.md#141-唯一后台启动命令物理-gpu-1) 在GPU 1手动运行；回传 `r4_state_supervised_*` 小报告、selection与provenance后验收实际效果。
