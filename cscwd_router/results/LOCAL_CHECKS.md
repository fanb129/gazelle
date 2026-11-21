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

服务器真实 ViT-B/官方 decoder 检查由用户按 [implementation_plan.md 第 7 节](../implementation_plan.md#7-t1-服务器运行与结果回传) 手动运行。该检查将记录服务器实际使用的 DINOv2 源码和 checkpoint SHA256；本文件记录的是本机合成工程结果。
