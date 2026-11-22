# CSCWD Router · 结果来源

## 输入 checkpoint（2026-11-21 用户确认）

- 服务器仓库：`/home/fb/src/paper/gazelleV1`。
- DINOv2 ViT-B/14：`/home/fb/src/paper/gazelleV1/checkpoints/dinov2_vitb14_pretrain.pth`。
- 本轮 GazeLLE GazeFollow decoder：`/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`。
- 两份 checkpoint 的 SHA256 由服务器 T1 检查脚本读取实际文件后计算并追加到本文件。权重留在服务器，不进 Git。

## T1 本机合成结果（2026-11-21）

- 小结果：`cscwd_router/results/t1_local_checks.json`。
- JSON SHA256：`ba545dfc7688e564f5086b114575b5784e61450dc1fb6aa9aedb27d97c508e11`。
- DINOv2 源码：`https://github.com/facebookresearch/dinov2`，提交 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`，本机临时路径 `/private/tmp/gazelle-router-t1-dinov2`。
- 执行入口：`cscwd_router.code.stag.test_backbone_sparse_dinov2`。9/9 通过；448 输入随机小模型 K100 实测最大差值 0.0、不一致元素 0。具体环境和实现源码 SHA256 位于 JSON；命令见 `LOCAL_CHECKS.md`。
- 本机仅生成随机输入、随机小模型和内存中的 checkpoint 序列化数据，未加载真实 checkpoint，未生成需保存的 PT/NPY。

## 服务器结果登记方式

`code/scripts/check_sparse_dinov2.py` 实际运行后写 `results/t1_dinov2_checkpoint_checks.json`，并向本文件追加时间、代码提交、输入 checkpoint 路径及 SHA256、结果 JSON 路径及 SHA256。该 JSON 与本文件由用户经 Git 回传。

后续任务产生的大型 PT/NPY 仅登记服务器完整路径和 SHA256，小型 JSON/MD 直接保存于本目录并纳入 Git。

## T1 · 2026-11-21T06:17:26.326417+00:00

- code commit: `be2b4fcbdd3652cb1b48eb7f7523c027d8b7b075`; dirty: `True`
- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/t1_dinov2_checkpoint_checks.json`; SHA256: `2c0846dce19409473916dd3dedc2946ec1805be8fb186d14fa1767ed8cf6f105`; passed: `True`
- input: `/home/fb/src/paper/gazelleV1/checkpoints/dinov2_vitb14_pretrain.pth`; SHA256: `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`
- input: `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`; SHA256: `525a69495c0db475fa8522b164aba3cfabeaa0dd60adb7be5e13dd73e56d70e1`

## T1 结果审查与 T2 本机检查（2026-11-21）

- 拉取结果提交：`32224becb1a65a92742e5743c6e3202dcaf03387`。T1 报告 SHA256、五个相关源码文件 SHA256、六组数值结果核对通过；结论见 `results/T1_REVIEW.md`。
- T2 小结果：`cscwd_router/results/t2_local_checks.json`；SHA256：`6798a76b795acda5c6f85b75b410c70d21b064b65a89deaaa48639b2799198c1`。
- 本机使用官方 DINOv2 源码提交 `7764ea0f912e53c92e82eb78a2a1631e92725fc8` 的随机小模型，T1/T2 共 20/20 结构与 backward 检查通过；未加载真实权重，未执行 support 拟合。完整源码 SHA256 和环境在 JSON。
- T2 服务器入口 `code/scripts/check_router_dinov2.py` 将读取同一官方 backbone 权重，进行合成 support 拟合与 K25/K50/K100 路由检查。实际结果写 `results/t2_dinov2_support_checks.json`，并自动向本文件追加权重路径、权重 SHA、结果 SHA 和实际 coverage loss。此 smoke 不保存 PT/NPY。

## T2 · 2026-11-21T06:39:24.292574+00:00

- code commit: `3babdbdcc680f54d8d167e4d76e8b582255d06fe`
- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/t2_dinov2_support_checks.json`; SHA256: `b7f6561e4322311e1b6f8e0937528356080739754609658a8316786a051bfc66`; passed: `True`
- input: `/home/fb/src/paper/gazelleV1/checkpoints/dinov2_vitb14_pretrain.pth`; SHA256: `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`
- synthetic coverage loss: `3.465735673904419` → `1.1940904187213164e-05`; steps: `80`

## T2 结果审查与 T3 本机检查（2026-11-21）

- 拉取结果提交：`a08e944`。T2 报告 SHA256：`b7f6561e4322311e1b6f8e0937528356080739754609658a8316786a051bfc66`；与本文件登记一致，八个相关源码 SHA 匹配，backbone SHA 与 T1 相同。审查见 `results/T2_REVIEW.md`。
- T3 本机小结果：`cscwd_router/results/t3_local_checks.json`；SHA256：`15c34b9dcedeef43162e0d4fa501a90af29dcb64fd63a73698c4a6395885ec37`。
- 本机 T1/T2/T3 共 34/34 结构、backward、协议与序列化检查通过。随机小模型的 dense/K100 与原版 GazeLLE 逐值一致，最大差值 0.0、不一致元素 0；源码 SHA 与环境在 JSON。本机优化器更新数为 0；恢复调度单测使用临时状态扰动，临时文件已删除。
- T3 服务器入口为 `code/scripts/train_router_dinov2.py`，实际大文件写到 `experiments/cscwd_router/t3_synthetic_seed3106/`，关键小报告写到 `results/t3_synthetic_seed3106/`。运行后自动登记两份输入权重、四个 final checkpoint、三个 last checkpoint 的完整路径与 SHA256，权重不进 Git。
- 独立评测入口 `code/scripts/eval_router_dinov2.py` 重新加载 checkpoint，沿用现有 GazeFollow 指标；输出小 JSON 的路径与 SHA256 也由实际运行追加到本文件。

## T3 · 2026-11-21T07:16:14.173822+00:00 · synthetic

- code commit: `658f1742c30576b5fb59402582c9a2e99a3e9387`; precision: `fp32`
- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/t3_synthetic_seed3106/summary.json`; SHA256: `4ffe9e3d342368f0b1b301e932fd9c6ef19a4b00ca9089e9a3f90c36cb91d872`; engineering passed: `True`
- input backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov2_vitb14_pretrain.pth`; SHA256: `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`
- input official_decoder: `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`; SHA256: `525a69495c0db475fa8522b164aba3cfabeaa0dd60adb7be5e13dd73e56d70e1`
- checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/dense_official/final.pt`; SHA256: `48ea8e3727385e8109e9389de31252322bd340a3f123b5b15b9797a10ec5e2f4`
- checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/matched_dense/final.pt`; SHA256: `76724916a42f6b52ee37e8693f689e74ad270422f142963b59a78b4758487767`
- checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/matched_dense/last.pt`; SHA256: `277a6f7df872fe960ba9a0768110e277e21851247953d8ea9ea6617b5d3707eb`
- checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/sparse_decoder/final.pt`; SHA256: `c63c7c2185d824dca37b7fe60480453661c28afe0ccaf41b5e96b859d9fe4cb7`
- checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/sparse_decoder/last.pt`; SHA256: `fca0277c5d6b117c772a4629bd05ee135260a7a8b14164a2ca02befa1dbde64c`
- checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/support/final.pt`; SHA256: `f7047379ac0d1087d1417133c238637d02f7676733a3e91c5c79da8b34f8d88e`
- checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/support/last.pt`; SHA256: `3b2bca7ef1d965779ce69a6366a052964fa8934a55c2bfd0c44546bf632411d6`

- evaluation (synthetic, sparse_decoder): `/home/fb/src/paper/gazelleV1/cscwd_router/results/t3_synthetic_seed3106/standalone_sparse_eval.json`; SHA256: `ebcf8fc0d0c0d361225b4ccc0c2d3482de9a210209d6d0746b61a7a93bcb3d30`
- input checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/sparse_decoder/final.pt`; SHA256: `c63c7c2185d824dca37b7fe60480453661c28afe0ccaf41b5e96b859d9fe4cb7`

## T3 结果审查与 T4 本机检查（2026-11-21）

- 拉取结果提交 `de1488d`。T3 summary 与独立评测 SHA 分别为 `4ffe9e3d342368f0b1b301e932fd9c6ef19a4b00ca9089e9a3f90c36cb91d872`、`ebcf8fc0d0c0d361225b4ccc0c2d3482de9a210209d6d0746b61a7a93bcb3d30`；二者登记、18 个源码 SHA、3/5/5 更新和 checkpoint 往返均核对通过，见 `results/T3_REVIEW.md`。
- T4 本机小结果 `cscwd_router/results/t4_local_checks.json`，SHA256：`bfa0662c497237f76da482381bf4b029ab10de5e827334dad0bd6469a8e46a2e`。44/44 CPU 结构与规则检查通过，本机优化器更新数 0，未测真实 accuracy 或 GPU timing。
- T4 当前入口 `code/scripts/run_main_experiment.py` 将把大文件保存到服务器 `experiments/cscwd_router/t4_gf_fulltrain_seed3106/`，训练器实际写出后自动追加七个 PT 的路径与 SHA。小报告集中在 `results/t4_gf_fulltrain_seed3106/`。
- 主入口记录源码/输入 SHA、预声明 gate、完整子命令与证据链；每个 benchmark JSON 记录实际 checkpoint、真实 train RGB/头框样本 SHA、GPU 状态、token 执行图和计时设置。完成后自动登记 comparison 和所有小证据的路径及 SHA。恢复复用已核对的 official test 结果。

## W1 · 三稿复核与写作证据摘录（2026-11-22）

- 写作文件：`cscwd_router/latex/PAPER_STORY.md`、`cscwd_router/latex/PAPER_REVIEW.md`。完整读取 MM 9 页、rebuttal 1 页、ICASSP 5 页；渲染核对 MM 图 2、rebuttal 和 ICASSP 方法图，未修改这些 PDF。
- 证据摘录：`cscwd_router/results/WRITING_EVIDENCE_20261122.json`；SHA256：`3bb33508eb926d8301f7d465129f34ee6f3d15d97d9d341cd6e326db5dd7edad`。包含 19 份现有实验 JSON、6 个实现源码、3 份 PDF 与图素材的路径/SHA，摘录真实 GF/VAT 指标并按正反序比值几何均值重算旧效率汇总。
- PDF：`rebuttal/2026ACMMM-fanb-final-3.pdf`；SHA256：`48fb0bd2bdacd69437442a3765e5a6bb36602137a2fec12164c4a4329d80fdb9`。
- PDF：`rebuttal/rebuttal.pdf`；SHA256：`983d1d9e36f8f5cd1c88d4f70c345ed5d91d0dc15e563b9d179d0effd831313c`。
- PDF：`icassp2027/Fan.pdf`；SHA256：`4c5b37bc7409d6cb080e38aa062dc687c41570e86f520e95cb8907c1c94ad26e`。
- 旧 GF accuracy checkpoint 引用（从既有评测 JSON 读取，本机未加载 PT）：`/home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_fulltrain_sparse_fixedrouter_decoder_k050_seed3106/final.pt`；SHA256：`e2f6af0a2e1668196d28d2f11c4fdb16779e39fa07103b682efd2c2ee9d331ee`。
- 旧 GF matched dense checkpoint 引用：`/home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_fulltrain_dense_decoder_continue_seed3106/final.pt`；SHA256：`d4dd8c22cfa3824628d67441c5efb702f353a3422ec54be113cecbec577f7612`。
- 旧效率使用 split-clean `best_val_selection.pt`，并非上述两个 full-train final checkpoint；仅报告相对其自身 DINOv3 dense 的速度，未登记为相对官方 GazeLLE 的加速。
- 本轮没有训练、模型推理、GPU 测速或新的 PT/NPY。新 `test-selected` 协议仅完成文档决定，现有训练代码未更改。服务器 T4 原始小报告仍待用户经 Git 回传；会话中的只读结果不冒充已归档源报告。

## W2 · 限时实验计划的初始化引用（2026-11-22）

- W1 写作与证据文件已提交到本地 Git：`2a3bcaf`。W2 仅新增 `cscwd_router/EXPERIMENT_PLAN.md` 并同步当前文档，没有训练/推理/测速产物。
- 计划复用 GF support：`/home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_fulltrain_support_k025_seed3106/final.pt`；既有报告登记 SHA256：`3668596eb5e927c6a9e7144f3be23cfecd6231a591ce5678a29769d1595133d6`。
- 上条来源：本地既有 `experiments/coverage_router/gf_fulltrain_official_test_seed3106/pipeline_complete.json`；该 JSON SHA256：`76543cefd6dbeee9723eab3902ba9c2c22133462b5be0c32bafa63dede3bf6aa`。
- 计划复用 VAT 初始化：`/home/fb/src/paper/gazelleV1/experiments/coverage_router/vat_splitclean_dense_seed3106/best_val_selection.pt`；既有报告登记 SHA256：`bc946c6a07b230f7e677c15d52d2b984ccb65060f456fbe22ea997f7d234f050`。
- 上条来源：本地既有 `experiments/coverage_router/vat_splitclean_support_k025_seed3106/run_manifest.json`；该 JSON SHA256：`75987a88c08e5336e9fdea6c6b2257c2bbeb4abd1fcaaceda88fea23307b5617`。
- 两份 PT 仅作为初始化引用，本轮未在本机加载，运行前由服务器入口核对文件与 SHA。计划 GF 5、VAT 3+5 epochs 是待执行预算，不是新实验成绩。新 run 不划 val，按完整 train/test 及 `test-selected` 协议运行；旧初始化来源与旧 holdout 结果保留原标签。

## E0 · DINOv3 双尺寸 seed0 入口检查（2026-11-22）

- 新入口覆盖 GF/VAT × ViT-B/ViT-L；base seed 0，四组均从旧 raw-concat decoder 重新学习 support 3 + sparse decoder/inout 5 epochs，不新增 dense 训练。W2 的 seed3106 support 复用计划已被本轮要求替代；历史 decoder seed 不改写。
- 工程报告：`cscwd_router/results/e0_dinov3_train_test_checks.json`；SHA256：`c6a142c5398b50d4dfe78e07ead6b755722d6362d5fb7dad94c32d33ca1aa747`。7 个新源码 SHA、32/32 CPU 检查、两条运行命令的 shell/参数检查、四份数据标注的服务器只读检查均在 JSON 中。
- 本机只运行合成 CPU 工程闭环，服务器只读检查权重/标注与首张图片存在性；没有新真实数据训练、GPU 推理或测速。decoder 与真实 Router factory 的任务 tensor 匹配 GF-B/GF-L/VAT-B/VAT-L 为 40/40/45/45，缺失、多余、shape mismatch 均为 0。
- input gazefollow/vitb decoder: `/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt`; SHA256: `26010ecc293cd51d2ce69f95e8111ca46632e425e1463b22fb0980aacf67fbef`。
- input gazefollow/vitl decoder: `/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitl/2026-01-30_13-14-15/epoch_14.pt`; SHA256: `7dbe0f2e07d003f3d8c479751dd08c3bbc642d8ab6558c11847a6e5d38fe58af`。
- input vat/vitb decoder: `/home/fb/src/paper/gazelleV1/experiments/train_vat_baseline/2026-03-11_19-58-20/epoch_7.pt`; SHA256: `db09c5605300cd31c5ff7030d8e2f468aafb1e215f556c1a05daf8fb034bff0d`。
- input vat/vitl decoder: `/home/fb/src/paper/gazelleV1/experiments/train_vat_vitl/2026-02-02_09-44-48/epoch_7.pt`; SHA256: `979ddf804cc6b91821e2899c5bb9f1e92dec6a1136f25b26cb3d62a41d25b673`。
- input backbone vitb: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。
- input backbone vitl: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitl16_pretrain.pth`; SHA256: `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`。
- 待用户手动运行的命令见 `cscwd_router/RUN_DINOV3_TRAIN_TEST.md`：GPU0 GF-B→L，GPU1 VAT-B→L。新 PT 与压缩逐查询明细留服务器 `experiments/cscwd_router/testselected_k050_seed0/`，实际生成后自动登记路径/SHA；小报告写 `cscwd_router/results/testselected_k050_seed0/`。
