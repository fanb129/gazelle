# E2-TL：完整 GF-L Router→VAT-L 结果审查

2026-11-24。结果提交 `2eb4275`、服务器 CPU 检查提交 `33e459a` 均已 push/pull 到本地。**审查通过，接续 GPU 1 效率测试。** 本轮没有新训练、模型 forward 或 GPU 测速。

| VAT / K50 | AUC ↑ | L2 ↓ | AP ↑ |
|---|---:|---:|---:|
| 旧 VAT-L Router 初始化实验 | 0.923034 | 0.097970 | 0.916599 |
| 完整 GF-B Router→VAT-B | 0.940033 | 0.100131 | 0.883183 |
| **完整 GF-L Router→VAT-L，本次** | **0.948264** | **0.089676** | **0.891592** |

同为逐帧 test、31,978 个查询（20,655 个画内）、FP32、person query。相对旧 L，AUC 提高 0.025231，L2 降低 8.47%，AP 下降 0.025007。相对当前完整 GF→VAT-B，三项均改善。当前定位精度可用，AP 仍是较弱的一项。

## 来源、训练和选模

- 来源为已由用户收口的 GF-L K50 `sparse_decoder/best_test.pt`，SHA `0e3c848ce62731c53762696488e4911e86272dd4824a2db6b69a2f0e9ee5bebf`。保留 GF 实际 support 3 + decoder 4、`completed=false, closed_by_user=true`，没有改写为跑满 5 轮。
- 初始化完整迁移 GF 的 52 个 Router/heatmap decoder 张量，逐值一致；只新增 5 个 in/out 张量。DINOv3 ViT-L/16、多层 `raw_concat`、block 11 后 K50，空间先验 `none`，backbone 始终冻结。纯 RGB、GT 头框、单帧，没有引入时序模块。
- seed 0，8 epochs，train/test 采样率 6；22,641 个 train 查询、5,461 个选模 test 查询，不另切 val。首轮固定 Router，后 7 轮联合适配。Router/decoder/inout LR 为 `1e-4/1e-5/1e-3`，batch 4、累积 8、cosine。实际 decoder/inout 各 5,664 次 optimizer 更新，Router 4,956 次。
- 按预先固定的 test L2、AUC 平局规则，选中 **第 1 轮**（零基 `epoch=0`）。rate-6 选模指标为 `0.948027 / 0.091864 / 0.892195`。选中模型的 Router 与 GF-L 逐值一致，decoder/inout 已更新；后 7 轮联合适配均未改善 L2。这是 GF 训练好的 Router 在 VAT 上的迁移，不代表 Router 没有训练。
- 第 5 轮的 rate-6 AP 为 `0.904636`，但其 L2 为 `0.093317`，不替换选中模型，也不把该 AP 拼入最终行。随机 in/out 的 `initial_test` 按原方案不参与 checkpoint 选择。
- 最终选中 checkpoint 在 rate-1 test 上报告上表完整行，保持 `test_selected` 标签。该全量 test 与 rate-6 选模评估的采样率不同，数值分列。

## 工件核对

CPU 证据为 [e2tl_gfl_to_vatl_server_checks.json](e2tl_gfl_to_vatl_server_checks.json)，可复现入口为 `code/scripts/review_vat_transfer.py`。

- 20 个来源/输出工件的实际 SHA 与登记一致，其中 14 个大工件留服务器。核对 manifest fingerprint、22 项源码 hash（21 个文件和 native DINOv3 源码树）、两份标注 hash；本机 21 个源码文件和 5 份原始报告的 SHA 同样吻合。
- 初始 GF 共享参数、初始化 digest、8 个 epoch 权重与对应测试记录、last/history、best 与 last 内嵌 best 全部一致。首轮 Router 无变化，后 7 轮 12 个 Router 张量实际更新；各轮 backbone 可训练参数为 0。
- 逐条核对 31,978 个导出查询的顺序、图像、head index、gaze 坐标和 in/out 标注。导出 AUC/L2 的均值、in/out AP 和人数重聚合，与最终 summary 的差值全部为 **0**。这是既有预测的 CPU 核对，没有重新运行推理。
- 当前 VAT-L `best_test.pt`：`/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/best_test.pt`；SHA `6d12190c7cdf24fe0389eede68b27f0f05d35cb4c753072f62a3d0d1d8445bb5`。逐查询导出和其余 PT 路径/SHA 见 [RESULT_PROVENANCE.md](RESULT_PROVENANCE.md)。

## 下一步：GPU 1 效率

现有 runner 已支持 L，无需改动模型或计时实现。B1/B8 × forward/reverse 四个真实 VAT 输入请求均通过 CPU 预检；采用包含 VAT 顶层序列展开修复的独立 worktree，完整命令见 [RUN_VATL_EFFICIENCY.md](../RUN_VATL_EFFICIENCY.md)。测试 official/K100/K50 的 FP16 完整 forward，参数与 B 正式结果相同，输出写 `efficiency_dinov3_vatl_seed0_20261124/vat/vitl/`。

最后一次 GPU 快照中，GPU 1（UUID `GPU-2a918114-c4d4-6fc2-23fb-e065099e45cb`）有其他作业占用 23,821 MiB、利用率 89%。本轮未启动 GPU 测速；待卡空闲后由用户手动运行，完成后回传实测结果。

文献表已补本次 VAT-L：相对 Mi ECCV2026 公开值 `.948/.095/.923`，AUC 同量级、L2 低 5.60%，AP 低 0.031408；这些是不同论文协议的报告值比较，尚无同机效率结论，不能写成全面胜出。近期方法对照见 [SOTA_LANDSCAPE_20261124.md](SOTA_LANDSCAPE_20261124.md)。
