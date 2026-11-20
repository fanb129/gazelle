# 官方 train/test 新入口本地检查

日期：2026-11-18。协议：`r4_official_train_test_v1`。按用户指定完整 train 训练、每 epoch 完整 test，按 AUC 更高、L2 更低、AP 更高、较早 epoch 选择同一轮整套结果。

新增官方数据加载、test 评测包装器、AUC 选模协议、test 特征缓存、训练/比较入口及顺序后台脚本；全部新代码位于 `cscwd/code/`。旧 `gazelle`、R2/R3/原R4源码与既有结果保持。计划见 `cscwd/r4_official_train_test_plan.md`，实际源码/计划 SHA256 见同名 JSON。

36 项本地检查通过：

| 组 | 项数 | 覆盖 |
|---|---:|---|
| 官方数据 | 5 | 全部 train/test、无开发manifest依赖、只读缓存、来源/覆盖 |
| test 特征 | 6 | 首批FP32预测对齐、原生FP16全量流写/读回、旧预测不变、CLI参数 |
| 选模协议 | 5 | AUC优先、L2/AP并列处理、整轮结果、无旧precision门槛 |
| 训练 | 6 | 两模型五轮、逐轮独立test、共享初始化、test不更新参数/梯度、checkpoint重载 |
| 比较 | 8 | 十个候选/选择重算、SHA来源、完整比较CLI、test-selected manifest |
| 后台入口 | 6 | 缓存或复用→两模型→比较，GPU1继承，含空格路径，各阶段失败停止 |

整合后的脚本组31项实际运行22.977秒，选模协议5项通过；12个新Python文件通过Python3.10语法解析，shell入口与§11两段命令语法通过。Torch子进程的本机OpenMP共享内存限制通过沙箱外CPU合成检查处理，生产代码无环境绕过。

另用既有真实小报告核对8项兼容关系通过：train/test模型、checkpoint/backbone、预测协议一致，R2全train特征对应既有train预测index，完整train/test覆盖与原生768×32×32 FP16契约一致。这些核对没有读取服务器大文件或产生新预测。

训练将复用全部train特征与train/test原预测，新建test特征与新实验目录；每轮保存指标、checkpoint、预测、历史控制诊断和history，最终保留 `selected.pt`/`last.pt`。结果和论文设置明确test参与epoch选择。

以上均为本机CPU合成/来源检查，没有新增真实VAT训练、CUDA结果或论文性能数字。下一步由用户服务器pull后，按实现计划§11.1在物理GPU1执行唯一后台命令，再回传 `r4_official_*` 小报告及provenance。
