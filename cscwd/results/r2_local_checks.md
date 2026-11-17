# R2 本地实现与合成检查

日期：2026-11-17。R1 验收提交 `f3e38d2` 已推送。本文只记录本地合成检查，未运行真实 VAT 特征 probe/cache 或新模块训练，不给出新方法性能数字。

| 检查组 | 实际通过数 | 覆盖 |
|---|---:|---|
| 特征探测/缓存 | 7 | 唯一图像、原生 shape、FP16 量化值读回、冻结前向对齐、来源契约、离线读取不导入 Torch |
| 普通 GRU 模型与损失 | 10 | 因果 prefix、连续/分 chunk 等价、人物 reset、极小框面积池化、自身预测历史、GT/padding 掩码及双分支梯度 |
| 连续轨迹加载 | 8 | 每人物帧恰一次、固定 slot 续接、gap/跨人重置、共享图像、标签不影响观察输入、旧高斯目标对齐 |
| 来源与训练 CLI 集成 | 6 | R1 来源链、来源不符拒绝、临时合成训练、紧凑 validation 映射、检查点重载、输入缓存只读 |
| 合计 | 31 | 不含重复执行的测试 |

本地环境：`/private/tmp/gazelle-r2-venv`，Torch 2.6.0、torchvision 0.21.0、NumPy 2.2.6。测试以 CPU 临时合成图像、特征和预测运行，不加载真实 DINOv3 权重。本机 OpenMP 子进程共享内存限制通过沙箱外运行合成 CLI 检查处理，生产脚本没有加入绕过环境变量。

原包 `gazelle`、R1 原始 float32 预测和已验收报告保持不变。Python 3.10 语法及脚本 help 已检查；服务器命令见 [implementation_plan.md](../implementation_plan.md) §9。

实现文件：

- `code/stag/feature_cache.py`、`code/scripts/cache_frozen_features.py`：先 probe 当前帧原生末层特征，与 R1 原始 float32 预测对齐；再按通过的契约保存每唯一图像的一份 FP16 特征。FP16 是有损特征存储，原 heatmap/In/Out 数组不改写。
- `code/stag/temporal_model.py`：普通因果 GRU，当前人物/场景/预测区域特征和自身过去输出进入状态；热图/状态两个残差头零初始化。未实现 R3 的显式对应或历史释放。
- `code/stag/temporal_data.py`：固定 batch slots 的连续轨迹片段；chunk 间只截断梯度，同一人物连续状态保留。
- `code/scripts/train_temporal_joint.py`：固定 smoke 配置的新增模块训练、完整轨迹验证和 last checkpoint；不选择最优配置、不使用 test。

下一步先运行 §9.2 的 probe，回传真实 shape、量化、头框覆盖、原始前向对齐和存储报告；核对后再运行 cache 和一个 epoch 的 smoke train。R2 尚未整体验收，R3 未开始。
