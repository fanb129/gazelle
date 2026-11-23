# L 初始化语义与当前实验取舍

日期：2026-11-23。用户指出 L 的历史实验可能是复现 GazeLLE 的单层版本。本轮只读核对 archived checkpoint、W&B 日志/配置和记录的 Git 源码，没有停止训练或运行新模型评测。

## 1. 权重文件提供的事实

证据：[l_initializer_semantics_20261123.json](l_initializer_semantics_20261123.json)。

| 项目 | GF-L epoch 14 | VAT-L epoch 7 |
|---|---|---|
| checkpoint 目录 | `train_gazefollow_vitl/2026-01-30_13-14-15` | `train_vat_vitl/2026-02-02_09-44-48` |
| W&B model 名称 | `gazelle_dinov3_vitl16` | `gazelle_dinov3_vitl16_inout` |
| `linear.weight` | `[256,4096,1,1]` | `[256,4096,1,1]` |
| 同目录 epoch 0 投影 | 同为 4096 输入通道 | 同为 4096 输入通道 |
| `pos_embed` | `[256,32,32]` | `[256,32,32]` |
| 非 backbone task tensors | 40 | 45 |
| 日志可训练参数数 | 3,678,720 | 3,712,001 |
| SASA / GGSF tensors | 无 | 无 |

普通 ViT-L 的单层特征为 1024 维；现存投影接受 4096 维，与四路 1024 维拼接兼容。四个投影分块的权重也不相同，不是简单重复同一组权重。它们不能按通常的 1024 输入通道单层 decoder 归类。运行配置和 factory 记录指向 DINOv3-L，没有发现指向 DINOv2 的运行配置。

这些是权重结构与日志事实，不能单独恢复实际选了哪几层。

## 2. 历史配置确实没有闭合

GF-L W&B 记录提交 `c516de35e4fdb212642052a7ab2a020ee7f5c487`。该提交的 backbone 用最后一层，factory 输入为 448；这与实际 4096 通道投影、32×32 网格及日志 3,678,720 个可训练参数不一致。运行目录没有保存 `.py` 或 patch 快照，因此无法仅凭该提交还原真实运行代码。运行时存在未提交改动或后续工件生命周期变化都能解释这种矛盾，当前没有证据选择其中一种。

VAT-L W&B 记录提交 `3ff2907922ee73d338d1a31b0707d3f9f7ab0fa9`，其中明确拼接 DINOv3 多层、输入 512，L 选层为 `[4,11,17,23]`；当前 Router 的 L 选层为 `[5,11,17,23]`。这说明记录的源码与当前特征配置有差异，也不等于封存了当时未提交工作区的真实层号。

因此应修正此前“L 与当前四层配置完全一致”的表述：**keys/shape 兼容已经核验，历史精确特征语义尚未确认**。不把兼容性检查当作历史配置认证，也不把这两个文件直接判成已证实的单层 checkpoint。

原 E0 工程报告及其 SHA 保留；其中 tensor keys/shapes 匹配和当前 B/L factory 的检查仍成立。本文件补充其无法覆盖的历史语义。没有改旧初始化注册或 trainer 源码，避免改变正在运行任务的恢复指纹。

## 3. 当前取舍

1. **GF-B Router→VAT-B 继续作为首要实验。** B 来源是用户确认的正确 raw-concat 模型，已验收 GF-B 后完整迁移；L 的来源疑点不影响这条链。
2. **暂不因“单层初始化”假设直接停止 GF-L。** 当前运行代码本身是 DINOv3-L、四层 `[5,11,17,23]`、block-11 Router。它使用结构兼容的 decoder warm start，不能因此自动视为无效训练。先保留当前任务及 checkpoint，等 K75 轮的 test 看适配结果；L 暂作为独立 warm-start 实验，不作为“已确认同配置 dense 初始化”的证据。
3. **现在不新增两套 L dense 训练。** 即使后续确需补清楚的 L 起点，训练链也应优先 GF-L 正确表示 → GF-L Router → 完整迁移 VAT-L，不必先单独训练 VAT-L dense 再学习 Router。时间先用于 B 的两个数据集和必要机制/效率证据，L 是补充线，不挤掉 VAT-B。

这次没有执行 stop/kill/pause；如果后续因预算停止 L，应保留原子 last checkpoint 和原 recipe 以便恢复，不重写现有 run。

## 4. 是否转用 GazeSpot

技术上可以做一个明确复用 GazeSpot 的 sparse Router 分支，但当前不建议切换主线：

- 当前初始化器要求 raw-concat、无 SASA/GGSF 的准确 key 集合，不能仅替换成 GazeSpot PT 路径；也不能静默丢弃 SASA/GGSF 后仍称作完整 GazeSpot 迁移。
- 真正复用时要保留其融合/几何模块，保证 Router 决策仍只用路由点前信息，适配 sparse feature fill，并与同一 GazeSpot dense checkpoint 比较。B 用 raw-concat、L 用完整 GazeSpot时，不能把两者写成仅 backbone 规模不同的同一配置。
- 在审 ICASSP 已覆盖的多层表示、SASA/HCSG 等不再次作为新贡献；新增贡献要由实际 sparse execution、人物条件化分配和相应实验承担。IEEE 通用政策要求披露相关已投稿/已发表工作并清楚说明差异，引用本身不代替实质区别。[IEEE Conference Submission Policies](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/)

当前继续 raw-concat + Router 的 B 主线，完整 GF→VAT 迁移结果出来后，再决定是否值得投入 GazeSpot 分支。
