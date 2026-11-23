# CSCWD Router · T1 本机合成检查

日期：2026-11-21。结果：[t1_local_checks.json](t1_local_checks.json)。

本轮使用官方 DINOv2 源码构建随机小模型：24 维、12 blocks、patch 14、无 block chunk；CPU attention 使用官方 `Attention`。源码 checkout 位于 `/private/tmp/gazelle-router-t1-dinov2`，提交 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。未加载官方预训练 checkpoint，未读取真实数据。

本机环境：`/private/tmp/gazelle-cscwd-check/bin/python`，Python 3.10.20、PyTorch 2.12.0、torchvision 0.27.0、timm 1.0.31。

执行命令：

```bash
cd /Users/fanb/src/gazelle
PYTHONPATH=/private/tmp/gazelle-router-t1-dinov2:/Users/fanb/src/gazelle \
  /private/tmp/gazelle-cscwd-check/bin/python -m unittest -v \
  cscwd_router.code.stag.test_backbone_sparse_dinov2
```

9 项检查全部通过：

- K100 对官方 `forward_features` 的逐值一致性：batch 1/2、位置插值、矩形网格、route block 0/5/10。
- 448×448 输入的完整 32×32 网格：`[1, 1024, 24]` patch 输出，实测 `max_abs_diff=0.0`、`mismatched_values=0`，数值有限。
- prefix 的原始 token 与官方 blocks 0–5 输出一致，供 Router 使用的特征按官方 norm 处理。
- 每图不同 patch 索引保留原位置信息，CLS 保留，未选位置与 block-5 early-exit 特征逐值一致，prefix state 未被修改。
- hook 确认 K50 只让 `K+1` 个 token 进入后六层；K100 也经过相同后六层和 scatter。
- gather/scatter 的逐图索引及梯度正确。
- 冻结权重时，梯度可经拆分前向到达输入。
- `model.*` state-dict 键与官方 backbone 对齐，checkpoint 序列化与加载成功。
- 服务器检查函数已用原版 `DinoV2Backbone.forward`、原版 `GazeLLE.forward` 和随机小模型执行两图多人查询，K100 heatmap 逐值一致；只替换测试中的权重加载 I/O。

服务器真实 ViT-B/官方 decoder 检查已由用户手动运行并验收通过，结果见 [T1_REVIEW.md](T1_REVIEW.md)。复现命令见 [implementation_plan.md 第 7 节](../implementation_plan.md#7-t1-服务器运行与结果回传)；本节记录本机合成工程结果。

## T2 本机结构与梯度检查（2026-11-21）

结果：[t2_local_checks.json](t2_local_checks.json)。T1 回归 9 项 + T2 新增 11 项，共 **20/20 通过**。环境和官方 DINOv2 临时源码 checkout 与上述 T1 本机检查一致；T2 使用 56×56 随机 RGB 输入、24 维/12 blocks 小模型、16 维 Router，得到 4×4 的 block-5 网格。

```bash
cd /Users/fanb/src/gazelle
PYTHONPATH=/private/tmp/gazelle-router-t1-dinov2:/Users/fanb/src/gazelle \
  /private/tmp/gazelle-cscwd-check/bin/python -m unittest -v \
  cscwd_router.code.stag.test_backbone_sparse_dinov2 \
  cscwd_router.code.stag.test_router_dinov2
```

T2 检查覆盖：

- 直接使用现有 `CoverageAwareSpatialRouter`，实际输入等于规范化后的 DINOv2 block-5 特征。
- 两图、三人对齐为 `[0, 0, 1]`；按人预测 support，按图精确 max-union，Top-K 排序且预算固定。
- hook 确认 backbone 按图执行：K50 时前六层收到 `[2, 17]`，后六层收到 `[2, 9]`，没有按三个人重复 suffix。
- support-only 不调用后六层；`train()` 后 backbone 继续冻结并保持 eval。
- zero-init support 的 coverage loss 有有限、非零的 score-head 梯度；在测试中将末层设为非零后，scene/head/geometry 三条分支均能收到梯度。检查仅做 backward，未执行优化器更新。
- 人物顺序交换后 support 对应交换，union 与选中 patch 不变；缺失头框不会被替换为占位框。
- K100 继续逐值等于官方 `forward_features`，未选位置与 block-5 特征逐值一致。
- 服务器路由检查函数已在本机小模型上运行 K25/K50/K100 三种预算并全部通过，实际每图保留 4/8/16 patches。
- 合成数据生成器的 RGB、头框与逐人目标对齐且可复现。

服务器入口 `code/scripts/check_router_dinov2.py --help` 和四个新增 Python 文件的编译检查通过。本机没有执行 support 拟合；coverage loss 的下降数值由服务器脚本实际运行后写入 `t2_dinov2_support_checks.json`。

## T3 本机结构、协议与 checkpoint 检查（2026-11-21）

T2 的服务器拟合已通过，见 [T2_REVIEW.md](T2_REVIEW.md)。本轮结果：[t3_local_checks.json](t3_local_checks.json)。

T1 9 项 + T2 11 项 + T3 新增 14 项，**34/34 全部通过**。T3 随机小模型为 DINOv2 24 维/12 blocks/patch 14、56 输入，decoder 为 32 维/1 层，Router 为 16 维；本机环境仍为 Python 3.10.20、PyTorch 2.12.0、CPU。

```bash
cd /Users/fanb/src/gazelle
PYTHONPATH=/private/tmp/gazelle-router-t1-dinov2:/Users/fanb/src/gazelle \
  /private/tmp/gazelle-cscwd-check/bin/python -m unittest -v \
  cscwd_router.code.stag.test_backbone_sparse_dinov2 \
  cscwd_router.code.stag.test_router_dinov2 \
  cscwd_router.code.stag.test_training_dinov2
```

T3 检查内容：

- 新模型的原始 dense 和零初始化 K100 输出均与原版 `GazeLLE.forward` 逐值一致，实测最大差值 0.0、不一致元素 0。
- support 的 backward 只到 Router，且不调用 decoder；sparse/matched 的 backward 只到 decoder。backbone 全程冻结，训练模式切换正确。
- matched dense 不读取 Router 参数；大幅改变 Router 权重后，dense 输出仍逐值一致。
- compact checkpoint 只存 decoder/Router，保留外部 DINOv2 来源；稀疏预测读写逐值一致，优化器与 scheduler 初始状态可读回。
- 训练调度与恢复单测在临时目录内注入参数状态扰动，验证中断恢复不重复已完成 epoch、分支从同一官方 decoder 起点开始、两条 decoder 分支预算一致。该测试没有执行优化器更新，临时模拟 history 没有保存到结果目录。
- 多标注 AUC/Avg L2/Min L2 与现有 `gazefollow_auc`、`gazefollow_l2` 一致，按人汇总，保留同图多人查询。
- 人工生成的 RGB/annotation 文件验证了图像分组 holdout、逐人训练 collate 和全头框 validation；full-train 不构造 validation/test，synthetic 不要求真实数据路径。
- official-test 请求拒绝 holdout、support 和非 fixed-final checkpoint。

两个入口的 `--help` 和八个新增 Python 文件的编译检查通过。本机未加载真实 checkpoint，也未执行真实优化；T3 的实际 3+5+5 更新、标准指标计算和 GPU checkpoint 往返由服务器 synthetic smoke 运行后记录。

## T4 本机执行图、汇总与命令检查（2026-11-21）

T3 服务器验收已通过，见 [T3_REVIEW.md](T3_REVIEW.md)。本轮本机结果：[t4_local_checks.json](t4_local_checks.json)。

此前 34 项 + T4 新增 10 项，共 **44/44 通过**。未执行本机训练、真实 GazeFollow 推理或 GPU timing。

```bash
cd /Users/fanb/src/gazelle
PYTHONPATH=/private/tmp/gazelle-router-t1-dinov2:/Users/fanb/src/gazelle \
  /private/tmp/gazelle-cscwd-check/bin/python -m unittest -v \
  cscwd_router.code.stag.test_backbone_sparse_dinov2 \
  cscwd_router.code.stag.test_router_dinov2 \
  cscwd_router.code.stag.test_training_dinov2 \
  cscwd_router.code.stag.test_main_experiment
```

新增检查：

- 原版 dense、routed K100、routed K50 的实际执行图：随机 24 维/12 blocks 小模型、448 输入、32 维/1 层 decoder；实际保留 1024/1024/512 patches，K50 后六层实际为 513 tokens，dense/K100 输出逐值一致。
- benchmark 输入从生成的 RGB/头框 annotation 文件选择，转为 RGB，仅用真实头框字段；fixture 不含 gaze 字段，选择可重复，B1 为 B8 的同一前缀样本。
- 主命令固定 full-train、validation fraction 0；独立 test 指向 final checkpoint；六个 benchmark 子命令固定 FP16 和预声明正反序。
- 缓存 official test 必须匹配 checkpoint SHA、训练 signature、annotation SHA 和 batch；训练 summary 尚未登记工件哈希时会继续完成登记再进入 test。
- 正反序比例的几何平均、AUC/latency 两个独立门槛、含等号边界、缺失或混用证据的拒绝路径、percentile 计算通过单测。此处使用的 accuracy/timing 数字只是单测 fixture，没有导出为实验结果。

两个新增脚本的 `--help` 和四个新增 Python 文件的编译检查通过。当前没有真实 T4 AUC、latency 或 speedup 数字，实际结果由服务器主实验生成。

## E0 DINOv3 双尺寸 train/test 入口（2026-11-22）

本轮新增检查与现有 Router 回归检查共 **32/32 通过**，Python 3.10.20、PyTorch 2.12.0、CPU。报告见 [e0_dinov3_train_test_checks.json](e0_dinov3_train_test_checks.json)。

```bash
cd /Users/fanb/src/gazelle
/private/tmp/gazelle-cscwd-check/bin/python -m pytest -q \
  cscwd_router/code/stag/test_dinov3_train_test.py \
  tests/test_coverage_router.py tests/test_routed_backbone_utils.py
```

- 验证原有 B/L factory 的四层 projection、路由点及四份初始化注册；四种合成任务完整闭环，base seed 0。
- 验证无 validation loader、support 仅执行 prefix、backbone 冻结、分阶段梯度、全画外 batch 与末尾累积更新。
- 新精度计算与原有 GF/VAT evaluator 一致，单一 test checkpoint 选模、逐查询导出与原子 epoch 恢复通过。改变 seed 的恢复请求会拒绝；中断恢复与连续合成运行结果一致。
- 两条完整 nohup 命令通过 `bash -n`，两个队列与四个子任务的参数解析均通过，seed 均为 0。

服务器另只读核对四个 decoder、两份 backbone 与四份数据标注；GF-B/GF-L/VAT-B/VAT-L 的任务 tensor 匹配为 40/40/45/45，缺失、多余、shape mismatch 均为 0。权重路径/SHA 与标注统计见工程报告，未加载到本机。合成测试的数值不作为论文实验结果；本轮没有新真实数据训练或 GPU timing。

## E2-T 完整 GF Router→VAT（2026-11-23）

新增 12 项与此前 39 项，共 **51/51 通过**，Python 3.10.20、PyTorch 2.12.0、CPU。报告为 [e2t_gf_to_vat_local_checks.json](e2t_gf_to_vat_local_checks.json)。

```bash
cd /Users/fanb/src/gazelle
/private/tmp/gazelle-cscwd-check/bin/python -m pytest -q \
  cscwd_router/code/stag/test_dinov3_transfer.py \
  cscwd_router/code/stag/test_dinov3_continue.py \
  cscwd_router/code/stag/test_dinov3_train_test.py \
  tests/test_coverage_router.py tests/test_routed_backbone_utils.py
```

验证真实 B/L factory 的全部共享 tensors 逐值继承、仅新增 5 个 in/out tensors；Router 在首轮冻结、联合阶段经 coverage loss 更新，decoder/inout 更新、backbone 不变；关闭 coverage 项不会把 hard Top-K 当成 heatmap 的可微选择器。全画外 batch、B/L 合成闭环、原 GF 工件不变、逐轮 checkpoint、严格来源 SHA 与中断恢复均通过。完整 nohup 命令通过 shell 与 CLI 解析；旧 E0/续训源码 SHA 不变。未在本机加载真实权重或数据，没有新的真实 VAT 训练或 GPU timing。
