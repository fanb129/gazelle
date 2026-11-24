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

## DINOv3 test-selected · vat / b
- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/vat/vitb/summary.json`; SHA256: `ffaf51153d32c8519ea60bdb92dca15f2ce9172841e244499bb7a2e1cdfd86f6`
- protocol: official train/test, no validation; FP32 accuracy; frozen backbone
- initialization: `/home/fb/src/paper/gazelleV1/experiments/train_vat_baseline/2026-03-11_19-58-20/epoch_7.pt`; SHA256: `db09c5605300cd31c5ff7030d8e2f468aafb1e215f556c1a05daf8fb034bff0d`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- support_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitb/support/final.pt`; SHA256: `dd3204c41e9b806e97a032132025780d8df50104205aa2280ccc91eb2e4112bc`
- support_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitb/support/last.pt`; SHA256: `52f492f3fe459e35197e0ffa9d87f98f7bb0a204975f24f4f5128f4e8e5f280f`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitb/sparse_decoder/best_test.pt`; SHA256: `e30cb35b572dc129b766cb2e6d9500345594f312664d71d748393ced8a430261`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitb/sparse_decoder/last.pt`; SHA256: `f36e4ecc95cb550b8f7f88152b9c101de84d10e2e82a3b4a208d5edf780d7d28`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitb/predictions.jsonl.gz`; SHA256: `446b51cdadb3c6319e410180d6e7e88bfb0f8db9a516636a9971e1a6a303da40`

## DINOv3 test-selected · vat / l
- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/vat/vitl/summary.json`; SHA256: `e30a2e445e2daea3b65e610780431d4c727db5a9d8c48f44e428cd6864793e9d`
- protocol: official train/test, no validation; FP32 accuracy; frozen backbone
- initialization: `/home/fb/src/paper/gazelleV1/experiments/train_vat_vitl/2026-02-02_09-44-48/epoch_7.pt`; SHA256: `979ddf804cc6b91821e2899c5bb9f1e92dec6a1136f25b26cb3d62a41d25b673`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitl16_pretrain.pth`; SHA256: `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`
- support_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitl/support/final.pt`; SHA256: `d31e8212f134945dc17e2d2fa5827f75bd41d784f5e9d14d2b848cb8b97423b8`
- support_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitl/support/last.pt`; SHA256: `76275e9981c0fc26e9c03baa4e3f8cf89838ebc53bfcbecd1ee46dd9b3600ddf`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitl/sparse_decoder/best_test.pt`; SHA256: `6e3c799f2a9bb24f5083896e788935b0623e728e69a8a6fa966fb22ccefae1c8`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitl/sparse_decoder/last.pt`; SHA256: `8dc27dd226688d3163e73199d86c2334410f1b08853a65b675decb9bb078d922`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitl/predictions.jsonl.gz`; SHA256: `f7b6d402eba51e802e152cebca9e7a31f4093c11d143828d7d7ccce5f03efa69`

## E2 · VAT seed0 B/L 回传核验（2026-11-23）

- 服务器结果提交 `0fc284430297d38244f4583c4adae17a486a0bf7` 已 push，本机已 pull；8 份原始 JSON 的 SHA 与服务器核验记录一致。
- CPU 工件检查：`cscwd_router/results/e2_vat_seed0_server_checks.json`；SHA256：`5139136ad98fca1ee43100521078cfdf55735ffc156635d33a2894611fdb5d9f`。14 个大工件实际 SHA 全部匹配，19 项源码/源码树摘要与数据标注一致；两组各 31,978 条逐查询身份/坐标核对通过，AUC/L2/AP 重聚合差值均 0。
- 审查：`cscwd_router/results/E2_VAT_SEED0_REVIEW.md`；SHA256：`f34db2e400e8b4b9135354979de6db15f0e8edbd5fd3298158eeb6b44d6b7d85`。B/L 分别选 decoder 第 4/2 轮（从 1 开始），最终 rate-1 test AUC/L2/AP 为 `0.913548/0.110971/0.901738` 与 `0.923034/0.097970/0.916599`。
- 本轮只执行只读 CPU 工件/数值检查，未新增模型 forward、训练或测速；PT 与逐查询 gzip 留服务器，路径/SHA 已由上述两组实际运行记录登记。

## E2-C · VAT 初始化追溯与续训入口（2026-11-23）

- report: `cscwd_router/results/e2_vat_initialization_trace.json`; SHA256: `3f024dd05448063bb2fa4ac27571c612292ac8555c11ef25ef132c4020c45dfc`。
- report: `cscwd_router/results/e2_vat_continue_local_checks.json`; SHA256: `02fbe7e4ce6a1ffdc7f3c204de366b28f2360a26cd99bf9e425bf1c936466502`。
- 追溯只读服务器两组精确 W&B config、summary、output，源文件路径/SHA 与白名单配置写入 trace；确认 GF dense→VAT dense→新 Router。旧末轮 rate-6 AUC/L2 为 B `0.939097/0.103012`、L `0.945401/0.093716`，不改写为当前逐帧复测。MM Table 4 的 B raw-concat `0.937/0.103` 作为原稿参照。
- 新入口本机 39/39 CPU 通过（7 项新增、32 项回归），原 E0 七个源码 SHA 未改。三个新源码 SHA 与环境在工程报告；命令见 `RUN_VAT_CONTINUE.md`。本轮未新增真实训练或 GPU 推理，没有新 PT/NPY。
- 待执行续训从原 VAT last 出发，固定 K50，各加 8 epochs，decoder LR `1e-5`、in/out LR `1e-4`，base seed 0；旧 best 保留参与选模。实际新工件写服务器 `experiments/cscwd_router/vat_continue_k050_lr1e5_seed0/`，实际生成后登记路径/SHA。

## DINOv3 test-selected · gazefollow / b
- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json`; SHA256: `9d6af43099b54e5d7ea58d6f74ad7269047e6de510ab3384487c6e4ea8a5e7d0`
- protocol: official train/test, no validation; FP32 accuracy; frozen backbone
- initialization: `/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt`; SHA256: `26010ecc293cd51d2ce69f95e8111ca46632e425e1463b22fb0980aacf67fbef`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- support_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/support/final.pt`; SHA256: `7b76f7901b2a9df359e28700b5205cb962251aa7d8c44a4f8e9663a2851fd4dd`
- support_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/support/last.pt`; SHA256: `25fb2a77961a27e15c47f9db733815f346c4f8813837b3d5f5003c33379b5e35`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/sparse_decoder/best_test.pt`; SHA256: `8297c6f621ceeeef6222e31a4e1782ee92d2fb50b697e9da31144cfe5b2a5b14`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/sparse_decoder/last.pt`; SHA256: `f1d1bdef1288299c03cb437494336bbf7fcfe8db60a47c904c857e4bd7facd5e`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/predictions.jsonl.gz`; SHA256: `4de860fbf3a1458299725764f0299783b0f2ea5d4818456a186c5144526bafe9`

## DINOv3 K50 continuation · vat / b

- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_continue_k050_lr1e5_seed0/vitb/summary.json`; SHA256: `b5ec2cb26d85e3af5304a95e9f8e6965bfe5c49ac50ef227abcfc7d12abadd6f`
- parent_summary: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/vat/vitb/summary.json`; SHA256: `ffaf51153d32c8519ea60bdb92dca15f2ce9172841e244499bb7a2e1cdfd86f6`
- parent_last: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitb/sparse_decoder/last.pt`; SHA256: `f36e4ecc95cb550b8f7f88152b9c101de84d10e2e82a3b4a208d5edf780d7d28`
- parent_best: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitb/sparse_decoder/best_test.pt`; SHA256: `e30cb35b572dc129b766cb2e6d9500345594f312664d71d748393ced8a430261`
- initialization: `/home/fb/src/paper/gazelleV1/experiments/train_vat_baseline/2026-03-11_19-58-20/epoch_7.pt`; SHA256: `db09c5605300cd31c5ff7030d8e2f468aafb1e215f556c1a05daf8fb034bff0d`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_continue_k050_lr1e5_seed0/vitb/best_test.pt`; SHA256: `fb71bae80105fba1d825df7d027cbe428eb93b13765b2c4255470102fb361136`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_continue_k050_lr1e5_seed0/vitb/last.pt`; SHA256: `8199198372bf754162191cf0bff39450481174184f2605dbc00401071d04b8fb`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_continue_k050_lr1e5_seed0/vitb/predictions.jsonl.gz`; SHA256: `37a3514a1d02b7b2322e251824eb1be0daba299bf456b38a150be78886cf91fb`
- probes: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_continue_k050_lr1e5_seed0/vitb/probes.json`; SHA256: `7f9a890cb727bdc3447adbe461d212538272c35a441dc601582e1e609f0221fe`

## DINOv3 K50 continuation · vat / l

- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_continue_k050_lr1e5_seed0/vitl/summary.json`; SHA256: `bfc810c6891875d1b4d40590c5fb4fca98f52aa719a8b8bf50080bac3d82fb23`
- parent_summary: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/vat/vitl/summary.json`; SHA256: `e30a2e445e2daea3b65e610780431d4c727db5a9d8c48f44e428cd6864793e9d`
- parent_last: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitl/sparse_decoder/last.pt`; SHA256: `8dc27dd226688d3163e73199d86c2334410f1b08853a65b675decb9bb078d922`
- parent_best: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/vat/vitl/sparse_decoder/best_test.pt`; SHA256: `6e3c799f2a9bb24f5083896e788935b0623e728e69a8a6fa966fb22ccefae1c8`
- initialization: `/home/fb/src/paper/gazelleV1/experiments/train_vat_vitl/2026-02-02_09-44-48/epoch_7.pt`; SHA256: `979ddf804cc6b91821e2899c5bb9f1e92dec6a1136f25b26cb3d62a41d25b673`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitl16_pretrain.pth`; SHA256: `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_continue_k050_lr1e5_seed0/vitl/best_test.pt`; SHA256: `917827fa57c93a3a3cc0941888feb328c54d389cfbe37d7809a6a6af131de704`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_continue_k050_lr1e5_seed0/vitl/last.pt`; SHA256: `ebc9fac2fa0864e366f789ee22aea9540512ef82ee31f0840d5aa3e6cbd88c61`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_continue_k050_lr1e5_seed0/vitl/predictions.jsonl.gz`; SHA256: `3d846164c55ff72568a77b9f6509c7fb6fae0c7e19960ccc824fdb6b04f9a152`
- probes: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_continue_k050_lr1e5_seed0/vitl/probes.json`; SHA256: `c8ff5ea247e283aec0a9e694dca951dc61922fc0748d76f93d3222938ac7cdf1`

## E1-B / E2-C · GF-B 与 VAT 续训审查（2026-11-23）

- 服务器结果提交 `8f2ae9cddd27799a3b2923754e47d21d5d07e962` 已 push，本机已 pull，13 份回传 JSON SHA 核对通过。GF-L 继续运行，不将其未完成产物纳入本次结果。
- checks: `cscwd_router/results/e1_gfb_e2c_server_checks.json`; SHA256: `ecdb2b43ca41800e3c4ef1ffdd2da4917a0ecb4c4047284865ed42b561cf13b2`。29 个工件/标注路径（21 个 PT/PTH/gzip）SHA 匹配，三组逐查询身份/标注与重聚合指标核验通过，差值均 0；VAT 各确有 5,664 次更新、44 个 decoder tensors 改变，保留旧 best 符合原选择规则。
- review: `cscwd_router/results/E1_GFB_E2C_REVIEW.md`; SHA256: `ce284ca93f06d6c92ba7a70a30e393b71d4c88583e2af272c9077204ca3285a8`。GF-B AUC/Avg/Min L2 `0.956236/0.101550/0.043420`；VAT B/L 各新增 8 轮未改善 L2，最终逐帧指标不变；原始 dense/K100 三项精度差为 0。本轮未新增模型 forward、训练或测速。
- 既有官方 GF-B 精度报告已归档：`cscwd_router/results/t4_gf_fulltrain_seed3106/accuracy/dense_official.json`; SHA256: `0a4a9b3c8ead1907aae746c57e5496d02af1d0c434c0292496e148b059d9e4d4`。只回传这一份 T4 baseline accuracy，不作为 T4 总验收。
- baseline compact checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/t4_gf_fulltrain_seed3106/dense_official/final.pt`; SHA256: `26d1c421689ad827c5bd633392cc31896a6593834d67ea122fbd134b776ec851`。decoder tensors 与官方原始权重逐值一致。
- official decoder: `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`; SHA256: `525a69495c0db475fa8522b164aba3cfabeaa0dd60adb7be5e13dd73e56d70e1`，与 T1 登记一致。大文件留服务器，未下载到本机。

## E2-T · 完整 GF Router→VAT 实现（2026-11-23）

- 本机工程报告：`cscwd_router/results/e2t_gf_to_vat_local_checks.json`; SHA256: `a19f716389105a20d2016bab7638715fb43f21f9ec5c86539f68756d8d8c98e6`。51/51 CPU 通过（12 项新增、39 项回归），B/L 共享 tensors 逐值继承、仅新增 5 个 in/out tensors、分阶段梯度与恢复通过。三个新源码 SHA、环境和完整命令验证在 JSON 中；旧 E0/续训源码未改。
- 已有 GF-B source checkpoint：`/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/sparse_decoder/best_test.pt`; SHA256: `8297c6f621ceeeef6222e31a4e1782ee92d2fb50b697e9da31144cfe5b2a5b14`。来源 summary：`cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json`; SHA256: `9d6af43099b54e5d7ea58d6f74ad7269047e6de510ab3384487c6e4ea8a5e7d0`。沿用该完整 Router/heatmap decoder，不加载旧 VAT task weights；backbone 仍使用原 SHA 固定的 DINOv3。
- 当前计划固定 K50、seed 0、8 epochs（1 轮固定 Router + 7 轮联合适配），LR `1e-4/1e-5/1e-3`。实际命令见 `RUN_GF_ROUTER_TO_VAT.md`；尚未执行真实 VAT 迁移训练，未生成新 PT/NPY 或精度/效率结果。
- 实际新大工件将留服务器 `experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/`，含 initial、每轮、last/best PT 和逐查询 gzip；生成后自动登记路径/SHA。小报告写 `cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/`，由用户经 Git 回传。

## GF Router to VAT · vitb

- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json`; SHA256: `c092d90ca2b094e575a94a70c21a61042611a2cd217332d0f10c35f277bb0f99`
- gf_summary: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json`; SHA256: `9d6af43099b54e5d7ea58d6f74ad7269047e6de510ab3384487c6e4ea8a5e7d0`
- gf_manifest: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/manifest.json`; SHA256: `edda71a597986a348c0eb95c2194086f753f83bdfbfbe0ee60a0914703c7ee66`
- gf_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/sparse_decoder/best_test.pt`; SHA256: `8297c6f621ceeeef6222e31a4e1782ee92d2fb50b697e9da31144cfe5b2a5b14`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`
- initial_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/initial.pt`; SHA256: `55f377fa92e9eb69fccb08d786aa5acb26894490a61aa21b8a0b8fe43c99ae22`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/best_test.pt`; SHA256: `c4268b27989f6a6049918e97ec73c066cba23d06e19f30379b6e7cc72e48309b`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/last.pt`; SHA256: `ddb893ed4bd634350efa907d05bc16a82d1d104f8585dde0457aa9d4314b3d8a`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/predictions.jsonl.gz`; SHA256: `69902091898a541651604366e9b9b3d3a7570d4fd91fae023d403dba785bba49`
- manifest: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/manifest.json`; SHA256: `3cd3f27e3cebffe92bce50ad2470e73c1aaf0a963d2b203a92323baa4e70c405`
- history: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/history.json`; SHA256: `81f4dfa37c1194eafccb2506196a119135181e8a78b1c3ceea18f7c61a0a9d57`
- initialization_report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/initialization.json`; SHA256: `1f9d5661b2e511d6a81bd8849eed90bc8d13fb43f49f3e20cf558dab22c1be3b`
- initial_test_report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/initial_test.json`; SHA256: `70ea02421a6800e7e043bf12522b0efd01285085675b30311b07f7fd1cf037b7`
- epoch_0: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/epoch_0.pt`; SHA256: `f7a14b65f2d194a82fb279bc4b6f6dc39a188b0c22f82ae18e25377789daad3b`
- epoch_1: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/epoch_1.pt`; SHA256: `be2bd1ef12ed9def898f810b479b72a227460620877dae2d0e748e0fa6522dc2`
- epoch_2: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/epoch_2.pt`; SHA256: `f1cd1c8e566322afff5cca7d17886851e4c66403c8494f75331e4f16275c5e9b`
- epoch_3: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/epoch_3.pt`; SHA256: `1e22cdf626afb3573baa8bb6a39896f7d2aa186a82739ef0682150de07ff700f`
- epoch_4: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/epoch_4.pt`; SHA256: `0654c3cdb09530bb4e0e3a9491c11fcc318ecb6615c16f4e1b545afeeff9a912`
- epoch_5: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/epoch_5.pt`; SHA256: `17c81564c684c181ca382a9a78649dee3423b7b57024a4715d9593af322fd3dd`
- epoch_6: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/epoch_6.pt`; SHA256: `866750b1165ff18c3dcaf179fb9c0522ae1e6e9c00ec8907fd92d193f763ca20`
- epoch_7: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/epoch_7.pt`; SHA256: `45e9bd166acbb82eb2981b02b2a0fcd6ec80829f60cc0b540861cdde996cd1c8`

## DINOv3 GF-L interrupted closure · 3 support + 4 decoder epochs

- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json`; SHA256: `fd83f2f17424edb559cacf65fead5fc21a8f89e53385249abd7ec4abdc9289f1`
- protocol: seed 0; test-selected by Min L2 among 4 committed decoder epochs; K50, FP32, 4782 queries. Epoch 5 partial state discarded after server shutdown; user declined resuming. Original 3+5 manifest and last checkpoint unchanged.
- metrics: AUC/Avg/Min L2 0.960007098565/0.094061416913/0.037721491817; selected decoder epoch 2. Values read from completed epoch test; no new forward or per-query export. Training is not marked complete.
- initialization: `/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitl/2026-01-30_13-14-15/epoch_14.pt`; SHA256: `7dbe0f2e07d003f3d8c479751dd08c3bbc642d8ab6558c11847a6e5d38fe58af`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitl16_pretrain.pth`; SHA256: `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`
- support_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitl/support/final.pt`; SHA256: `62d979a6d68753835ea14c704c95044094fb1947b47d340dfacd3fdb2b9f9832`
- support_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitl/support/last.pt`; SHA256: `cd147b2a8feeb88400c0f4547c40955e6c4b6feeedd35adb27c5d996505b2933`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitl/sparse_decoder/best_test.pt`; SHA256: `0e3c848ce62731c53762696488e4911e86272dd4824a2db6b69a2f0e9ee5bebf`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitl/sparse_decoder/last.pt`; SHA256: `589417842fffe946a17eb99dc7ad8b8b5da8141a471b2c6496fda99548c916ca`

## L 初始化语义补充（2026-11-23）

- checks: `cscwd_router/results/l_initializer_semantics_20261123.json`; SHA256: `d244cf0615c2dda69d50e97f5934a19aa0358bc358a5006c1617da4d0159a7ee`。只读服务器 archived PT、W&B config/output 与记录 Git 源码；GF-L/VAT-L 首末轮投影均 `[256,4096,1,1]`，无 SASA/GGSF，参数数与日志相符。运行记录指向 DINOv3-L，普通 1024 通道单层 decoder 不匹配现存权重结构。
- decision: `cscwd_router/results/L_INITIALIZER_DECISION.md`; SHA256: `ec704574d5bfdfb130dcaf1bdcde320dae0a7f45d23b35bf9f697e51da785bda`。GF 记录提交的 last-layer/448 代码与 checkpoint 不一致，VAT 记录层号 `[4,11,17,23]` 与当前 `[5,11,17,23]` 不同；无 runtime 源码快照，精确历史层号未闭合。原 E0 keys/shape 检查不改写，但不再当作历史同配置认证。
- PT 路径/SHA 与以往登记一致：GF-L `/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitl/2026-01-30_13-14-15/epoch_14.pt` / `7dbe0f2e07d003f3d8c479751dd08c3bbc642d8ab6558c11847a6e5d38fe58af`；VAT-L `/home/fb/src/paper/gazelleV1/experiments/train_vat_vitl/2026-02-02_09-44-48/epoch_7.pt` / `979ddf804cc6b91821e2899c5bb9f1e92dec6a1136f25b26cb3d62a41d25b673`。工件留服务器，未下载权重到本机。
- 本轮未修改 trainer/初始化注册、未停止进程、未新增训练或 model forward。GF-B→VAT-B 已由用户启动并继续优先；L 暂保留 warm-start 补充线，不新开两个 L dense，不切换 GazeSpot 主线。相关投稿政策来源链接与访问日期记录在报告中。

## E2-T · GF-B→VAT-B 结果回传与审查（2026-11-23）

- 服务器原始结果提交 `170ace87b6e0f80062d85fddf80b55439085e688` 已 push/pull；仅包含 5 份当前迁移 JSON 与对应 provenance block。5 份报告 SHA、本地 20 个源码 SHA 与服务器记录一致；本地 L 文档已 rebase 保留，原有目录迁移未纳入提交。
- checks: `cscwd_router/results/e2t_gfb_to_vatb_server_checks.json`; SHA256: `98c4afe5ac4a142ab1939cde14e740cb041157c604f39801bc49dc10320352b2`。服务器只读 CPU 核对 20 个工件路径/SHA（14 个 PT/PTH/gzip）、52 个 GF 共享 tensors、8 轮真实更新和单 checkpoint 选模；31,978 条逐查询身份/标注一致，AUC/L2/AP 重聚合差值均 0。大工件路径与 SHA 见上方 `GF Router to VAT · vitb`，未下载或入 Git。
- review: `cscwd_router/results/E2T_GF_TO_VAT_REVIEW.md`; SHA256: `584cd910b09fa0836cbf9ce361e7620b53cb8ab76feb6d066ca273466392a25b`。最终逐帧 AUC/L2/AP `0.940033/0.100131/0.883183`，相对旧 VAT-B 差 `+0.026485/-0.010840/-0.018555`，L2 相对降低 9.77%。选中第 1 轮，Router 仍为 GF 权重；后 7 轮联合适配未改善 L2。
- 本轮未新增训练、模型 forward 或测速。下一任务 E2-H 优先仅适配 `inout_head`，冻结 Router、共享 decoder 与 `inout_token`，尚未实现或运行；效率等空闲卡。

## E2-W · GF-L 等待期快照与后续顺序（2026-11-23）

- snapshot: `cscwd_router/results/gfl_interim_20261123.json`; SHA256: `b2b70e8366c2281c0c78dcbfe2bbc4bed77f5f0c50c1109bf97481ba8515bf81`。观察时间 `2026-11-23T06:39:26.130300+00:00`，只读服务器三个小报告和日志；原始报告路径、SHA 与内容均在快照中，无新增训练/推理/测速。
- GF-L 已提交 2 个 decoder epochs，第 2 轮 train K75、test K50，AUC/Avg/Min L2 `0.960007/0.094061/0.037721`；第 3 轮 K50 训练中，尚无最终 summary，不当作最终成绩。
- 用户接受当前 VAT-B，将分类头适配后移。等 GF-L 完成，可用则正式效率与完整 Router→VAT-L；不理想则先训练来源明确的 DINOv3-L 多层 GF 底座。等待期预算扫描、scene-only 与真实多人 union 的顺序记录在 `EXPERIMENT_PLAN.md`，尚未启动。

## E1-L · GF-L 中断收口审查（2026-11-23）

- 服务器结果提交 `459651a8a6485a2a2ab2b5ccb418ae742817e7a8` 已 push/pull，4 份回传 JSON 与 18 个本地源码 SHA 一致；本地 3 条提交与用户目录迁移保留。只回传当前 GF-L 与其 provenance，未提交旧 T4 或其他工作。
- checks: `cscwd_router/results/e1l_interrupted_server_checks.json`; SHA256: `c870c2d4af0051fa9ecd5466cfd6c19167e019fc0e1c7056c54f5e0588ed80ad`。6 个 PT/PTH、2 份标注、18 个源码及原生树 SHA 均核对；support/decoder 已提交更新 85,095/14,184，选模与冻结状态一致。52 个导出 best tensors 与 last 内嵌 best 逐值一致，原 last SHA 未变。没有新增 GPU forward、独立 test 或逐查询重聚合。
- review: `cscwd_router/results/E1L_GF_INTERRUPTED_REVIEW.md`; SHA256: `78255235b08317bd13e57541779dda4caf66a11382b5ebe02d410314f7e178d6`。用户决定不补最后一轮，实际 support 3 + decoder 4 收口，按 Min L2 选第 2 轮 K50，AUC/Avg/Min L2 `0.960007/0.094061/0.037721`。收口报告明确 completed=false / user-closed，原 3+5 manifest 保留；第 5 轮局部状态不计入完成预算。
- best PT 路径/SHA 见上方 `DINOv3 GF-L interrupted closure`，留服务器未入 Git。当前精度可接效率与完整 GF-L→VAT-L，不先新增 GF-L dense；新入口需显式支持收口来源，再由用户手动运行。

## E3 / E2-TL · 三卡效率与 GF-L→VAT-L 入口（2026-11-24）

- local checks: `cscwd_router/results/e3_efficiency_vatl_local_checks.json`; SHA256: `d4fa9d5511782f3ae4aa95cae06ef51d4bd5736a9b8d002966b57c98feb86802`。68/68 CPU 工程检查通过（16 项效率、1 项收口迁移新增，51 项旧回归）；8 个相关源码 SHA、三条完整 CLI 解析与 bash 语法在报告内。数值为工程合成检查，不是模型效率/迁移成绩。
- runbook: `cscwd_router/RUN_EFFICIENCY_VATL.md`; SHA256: `2dd2c8c534db48ade31a3fc0977a245cef7c77331d986e45f78b2995db67cd1f`。GPU0 顺序 GF-B/GF-L，GPU1 当前 GF→VAT-B，GPU2 完整 GF-L→VAT-L；B1/B8 正反序，official/K100/K50、FP16、50/200/500/5，实际训练为新 L 目录 8 epochs K50、batch4×累积8、seed0。
- GF-L 收口原 JSON 和原权重不改，显式 --allow-closed-gf 核对完整 history/last/best 后接受；源 completed=false 与实际3+4写入新 VAT manifest/summary。官方 B/L/B-inout decoder 与两份 DINOv2 backbone 已服务器只读核对；旧 gazelle/ 未改。
- 本轮没有启动真实 GPU forward/训练/测速，无新速度或 VAT-L 精度数字。效率小报告将写 results/efficiency_dinov3_seed0_20261124/；新大型 VAT-L PT/gzip 将留服务器 experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/，运行时自动登记路径/SHA。

## E3-F1 · VAT frames 采样修复（2026-11-24）

- code commit: `29c660731e4deb8ad5746c7504b48a065e479691`，已 push；服务器独立 worktree `/home/fb/src/paper/gazelleV1_efficiency_vat_fix` 固定该提交，读取原权重/数据。原目录 HEAD `4162da7` 与三个核心源码 SHA 保持不变，GPU0/2 继续原任务。
- checks: `cscwd_router/results/e3_vat_schema_fix_checks.json`; SHA256: `044447d94efe7002433b76a5da833a5bf2f8e731feb1c2b3866f4c25b680fc6d`。70/70 本机 CPU 通过；真实 VAT 475 序列、58,507 帧，8 个 RGB/GT 框匹配官方 loader，B1 为 B8 前缀，GF B8 metadata 与旧版一致，B1/B8 正反序四个请求 --check-only 通过。没有 GPU forward/计时或新的模型指标。
- runbook: `cscwd_router/RUN_VAT_EFFICIENCY_FIX.md`; SHA256: `ddd3ad5343fca648505b39a7d7601740cb611dec21002ed0ab6313aaea6e0e83`。完整 GPU1 命令已过 CLI/ shell 语法检查；用户手动启动，结果写回原仓库 results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/，不混入旧 GF 版本单元。
- 原错误发生在计时之前；旧失败日志和已有 GF 单元保留。修复仅按 dataset 复用官方 loader 展开 VAT frames，不修改 gazelle/、迁移训练或现有权重，不复制大型工件。

## DINOv3 efficiency · gazefollow / vitb

- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124/gazefollow/vitb/summary.json`; SHA256: `e1291a857968c3e4336feecbad1c0ba41f428b23d1b1d83f1b81844687b7479e`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124/gazefollow/vitb/b1_forward.json`; SHA256: `770e8eb06292601c266d9a6cebf823df7c6bd9eaef4cca13cc045b58822ab9bb`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124/gazefollow/vitb/b1_reverse.json`; SHA256: `58fa07e0565702c025d812859f0e27d55fcee6ac9665456f7f7672714bc4863c`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124/gazefollow/vitb/b8_forward.json`; SHA256: `e14a41a4d540596aef9f48d9214abc26a8eae9a89acf08cbff54763dfc1a537d`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124/gazefollow/vitb/b8_reverse.json`; SHA256: `1413bf7ec854eb8498d6a36fa52d89299c26e920c783d3a5099caf717d43e450`

## DINOv3 efficiency · vat / vitb

- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/summary.json`; SHA256: `b16a417d4c5a8a5882161e8d1f7a3efc819bebc85f54714e4629fbd1f60916cb`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/b1_forward.json`; SHA256: `0f0d4c23252c7f781b5eefca6cb2392de0e552770ee3ecf648e61895ada3e654`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/b1_reverse.json`; SHA256: `f8569cf04be5bc9928719507a1c0ec41926f88813605b68768abbd1bc367800b`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/b8_forward.json`; SHA256: `04a0c03b68ee7e1a104bcd13de6669075d3d09ce1daa3d4df95a3f0468e5b3e3`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/b8_reverse.json`; SHA256: `05221fb0aafc131d47c5436e3656be7b0e7e5d2fc762e0345a087a5f8ddea958`

## E3-B · GF-B / VAT-B 正式效率审查（2026-11-24）

- 原始结果提交 `85210b55b4832e31f7980a18e45397be29e80cfc` 已 push/pull，含两组各4个cell + summary JSON/MD，共12份小报告。通过空闲VAT修复worktree选择性提交后恢复29c6607，原目录HEAD4162da7及核心源码未改；GF-L/VAT-L按原任务接续。
- checks: `cscwd_router/results/e3_b_efficiency_server_checks.json`; SHA256: `891eb2102797b6bc1272959cf686465d975d3fe03b5969477eed56e78ad70099`。8单元、24组token trace，8个source/weight路径（6个PT/PTH）、2份标注、16个RGB路径SHA与50个按记录Git版本检查的源码文件均一致；summary JSON/MD重算完全相同。逐次latency数组未保存，不声称独立重算单variant median/p95；已核对采样数/repeats/throughput median及矩阵公式。
- review: `cscwd_router/results/E3_B_EFFICIENCY_REVIEW.md`; SHA256: `d0c3fedf2437139d275facf0c9bbb10d26ef4fd550865695557e7d82e6a15fb3`。相对官方latency speedup：GF B1/B8 `0.7822/0.9918×`，VAT `0.8255/1.0251×`；相对同路径K100的B8为 `1.2519/1.2559×`。B8峰值allocated memory约857 vs538MiB，高59.5%，不写成明显跨模型加速或省显存。
- 本轮只读CPU核对既有数据，没有新GPU forward/训练/计时。下一项优先阶段profile与同权重native full-token推理控制，再验证none全一乘法、四层repeat/cat等候选的等价快路径；不预写收益、不新增dense训练。
- benchmark weight: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/sparse_decoder/best_test.pt`; SHA256: `8297c6f621ceeeef6222e31a4e1782ee92d2fb50b697e9da31144cfe5b2a5b14`。留服务器，未入Git。
- benchmark weight: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`; SHA256: `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。留服务器，未入Git。
- benchmark weight: `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`; SHA256: `525a69495c0db475fa8522b164aba3cfabeaa0dd60adb7be5e13dd73e56d70e1`。留服务器，未入Git。
- benchmark weight: `/home/fb/src/paper/gazelleV1/checkpoints/dinov2_vitb14_pretrain.pth`; SHA256: `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`。留服务器，未入Git。
- benchmark weight: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/best_test.pt`; SHA256: `c4268b27989f6a6049918e97ec73c066cba23d06e19f30379b6e7cc72e48309b`。留服务器，未入Git。
- benchmark weight: `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14_inout.pt`; SHA256: `451f76e5a1fa14429283d8612785d9aa94362f4754cb8a27d650482e3a2ac00e`。留服务器，未入Git。

## E3-P0 · 推理流程与效率解释（2026-11-24）

- explanation: `cscwd_router/results/E3_INFERENCE_AND_OPTIMIZATION.md`; SHA256: `6d661935acec7312a127acbf64c1edc4f00cbd7ac5c609b42855de0fd720668b`。
- evidence: `cscwd_router/results/e3_efficiency_explanation_evidence.json`; SHA256: `994d19df83f06f1f4e81fe6577439b3aca49d45e6099bab4b047a2d68e3475d4`。包含既有B矩阵的延迟差额重算、两份历史B1诊断的元数据/median/源路径/SHA，以及明确标成非实测的blocks主要MACs公式推导。
- historical source: `/home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/stage_profile_forward_k100_k50_b1_amp.json`; SHA256: `37bc4231fb47fab40c5af85c80cfe7aaea30295cf6676a012a972f08ba037118`。
- historical source: `/home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106/stage_profile_reverse_k50_k100_b1_amp.json`; SHA256: `c883f2445bbd211d20ff6fecf9769d01968f1fb9af3140993d6bd062d7148cfa`。
- 本轮仅CPU源码/既有结果分析，无新GPU forward、训练或计时。历史profile是旧seed3106、dummy输入、B1、AMP cache开启；当前正式矩阵是seed0真实输入、cache关闭。历史阶段不替代当前checkpoint分段结果。优化尚未实施，不填写收益。

## SOTA-1 · 近期方法与比较目标（2026-11-24）

- review: `cscwd_router/results/SOTA_LANDSCAPE_20261124.md`; SHA256: `d65cf2f124a70eb07e290c66bd11d70e3e29c76c39e82cd214034ff52c32f812`。
- evidence: `cscwd_router/results/sota_landscape_20261124.json`; SHA256: `cb9fe1a385b68b0283ba9cf3ef76502472aa942f2024fecc0f89ce8d2f2e4729`。8行原文/官方公开指标、4项单独标注线索、3份本地选中报告SHA，含论文版本/PDF SHA、官方checkpoint发布元数据、基于公开舍入值的差额；不作统一复测或显著性判断。
- TextGaze官方 `checkpoints-v1.0` release于2026-11-11列出GF/VAT×B/L四个PT；API的文件大小与发布方digest已登记，未下载大权重、未独立校验二进制。OmniGF官方HF模型库revision `28d4c78343028573149cc04a1d3b0ac1423732ae`列出GF/VAT safetensors；仍需单独Qwen3-VL底座，未下载/执行。
- Mi ECCV2026论文与共同作者主页未找到官方实现/权重；GitHub标题检索0项，不作为“绝无开源”的证明。FGI原表受订阅限制、GazeLoom GF距离标签有歧义、GeoDistill未核完整原表，均留在单独线索区；GazeAnywhere任务/测试子集不同，不混入标准GF/VAT。
- 本轮只读网页/发布API/原文和本地结果，服务器仅检查VAT-L最终summary是否存在，结果未见；没有启动或停止任务、没有新训练/GPU forward/计时。目标转为近期强方法精度—成本，保留GazeLLE既有证据，下一项优先TextGaze官方权重推理对照。

## SOTA-2 · 架构与GazeLLE关系（2026-11-24）

- updated review: `cscwd_router/results/SOTA_LANDSCAPE_20261124.md`; SHA256: `59c42c6fa088808a8a037e5d2b43bae3a2279656f7888225eec73b3fcc9bb34c`。新增第7–9节：框架关系表、16项模型/参照的结构、backbone多深度融合判据及PaGE/GazeLLE-v3补查。上一条SOTA-1的review SHA为提交9186de4时的历史版本。
- evidence: `cscwd_router/results/architecture_lineage_20261124.json`; SHA256: `ad45841f7e8829691631abb51eda561d1f6af30006632200fae230660cad3d87`。含9个官方/作者仓库的29份固定commit源码路径/URL/SHA、6份本地只读源码SHA、架构分类及证据范围；只存元数据和归纳，不复制外部实现。
- GazeCoT作者PDF: `https://pi.cs.tsinghua.edu.cn/wp-content/uploads/2026/04/3772318.3790922.pdf`; SHA256: `8120e419859cd2340aa663d9c9af26a65bf19223b4336ec7c0ccf48e97b5e393`。§3.2/Table1核对GazeLLE-v3替换DINOv3及GF公开结果；完整agent与独立gaze estimator分开，不认定已核实对应checkpoint。
- PaGE官方HF：`Octopus1/page-vitb` revision `b991d8bcb2890a97db611a2080ca250ecd943952`、`Octopus1/page-vithplus` revision `4433e4dbc7ff4267c0f20ad8b9ab0a69048c3992`，API确认task safetensors及config文件存在；未下载大权重/执行模型。论文从GazeLLE出发，场景/头部DINOv3各取最终map；SFT与蒸馏的联合数据范围已标注。
- 本轮仅本机文献/源码核对与文档更新，无服务器操作、模型forward、训练或测速。未修改旧gazelle代码、训练配方或选中checkpoint；PaGE公开强结果补入，不提前声称当前模型全面SOTA。


## GF Router to VAT · vitl

- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/summary.json`; SHA256: `78b45b94d7fe65cdff17f3bd86f2afd99f7ec6b2add5c3cb83c30d19550f4c9f`
- gf_summary: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json`; SHA256: `fd83f2f17424edb559cacf65fead5fc21a8f89e53385249abd7ec4abdc9289f1`
- gf_manifest: `/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitl/manifest.json`; SHA256: `572744a7c125f36b04ce3671dafdd95f943f52f64a092a4cddf721c6ceab6041`
- gf_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitl/sparse_decoder/best_test.pt`; SHA256: `0e3c848ce62731c53762696488e4911e86272dd4824a2db6b69a2f0e9ee5bebf`
- backbone: `/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitl16_pretrain.pth`; SHA256: `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`
- initial_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/initial.pt`; SHA256: `df6358c12a05316929ba9fd97c7a6b89681d330476247078a8d6367dd616be09`
- best_test_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/best_test.pt`; SHA256: `6d12190c7cdf24fe0389eede68b27f0f05d35cb4c753072f62a3d0d1d8445bb5`
- decoder_resume_checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/last.pt`; SHA256: `fc4df18e262b42a21d90671bbd9a8a9f89627daa08c4a8c9536cb4691e983641`
- per_query_predictions: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/predictions.jsonl.gz`; SHA256: `3b4c5b12d313dfc076b74b84bb8e88dc38a0c80930a48dd1d8b2876dcb2ea56a`
- manifest: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/manifest.json`; SHA256: `a35a0ea4cbc75223867b3d9b231494344f9e790c7f8fad37ec164f5c6d4f9acd`
- history: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/history.json`; SHA256: `1ca76434348fbddeee21b58d739982c81328bc0feaa1734e99ef83b5dfcc0d00`
- initialization_report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/initialization.json`; SHA256: `ad6db25445e301f0d6e1f14ca67f7f571e70ff0b1448db953c39b79429052dd7`
- initial_test_report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/initial_test.json`; SHA256: `4b3843192dd1a740f6e47d87403f3eca03bf294aefe4865d64e35f5aa2ed2306`
- epoch_0: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/epoch_0.pt`; SHA256: `a60ae17136665979340a96de1f5789db62b53e3e493abccb698e55f4bff1aba1`
- epoch_1: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/epoch_1.pt`; SHA256: `cbf726b112e2e98d5812ab81260a4523c43e8057e2a98ca8550718cc9dcbcaa2`
- epoch_2: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/epoch_2.pt`; SHA256: `42f04039c775d98bf27d6cd653abd579f87765f46ef9617c659a9a8e49f428bc`
- epoch_3: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/epoch_3.pt`; SHA256: `f2903b35d9d2968ec32a5ce2bac336653d45359acd116448b48e725c31116d0d`
- epoch_4: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/epoch_4.pt`; SHA256: `878df462af6935aae0e9dde2ad957e1bd7799c0242242b63290fd7389ab4bc3a`
- epoch_5: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/epoch_5.pt`; SHA256: `9cee78c55a319cc0a4507ee1d4ce0552d553bff71bf3254a7011816c1fb2c21c`
- epoch_6: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/epoch_6.pt`; SHA256: `a14b45a4a4240dd4b4ef44e6420a8746a27f15f5fe3ab13863976bbf56d9776f`
- epoch_7: `/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/epoch_7.pt`; SHA256: `1a9526bbc846c1a06d50d97b2fde02ce737c765fa230d49aa982071a1b26cd12`

## E2-TL CPU review and E3-L efficiency preflight · 2026-11-24

- report: `/home/fb/src/paper/gazelleV1_results_vatl_20261124/cscwd_router/results/e2tl_gfl_to_vatl_server_checks.json`; SHA256: `ab43dd992b58f45acd29783945dd52f5af4353e2b766fc79d686e1e30f4cc9d8`
- review script: `cscwd_router/code/scripts/review_vat_transfer.py`; SHA256: `84ed4e0196f0df11695abb5f9b96ee5855c82d600dbbf7b81d8eebe0152ee96a`
- Scope: CPU artifact/tensor/selection/export checks and four benchmark input preflights; no GPU forward or timing. Raw VAT-L reports committed in `2eb4275`, CPU proof in `33e459a`.
- Prepared GPU1 efficiency output: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_vatl_seed0_20261124/vat/vitl/`; manual launch after GPU1 is idle. No timing report or efficiency value exists yet for this new matrix.


## DINOv3 efficiency · vat / vitl

- report: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_vatl_seed0_20261124/vat/vitl/summary.json`; SHA256: `8f50d149cf686df53909da41988bec86f88e026add25b6862c84f1b6195a641f`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_vatl_seed0_20261124/vat/vitl/b1_forward.json`; SHA256: `6b62e143c09c02128f5c29ffcd6cf4d1945bb996974270d9e24c121dc0038db0`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_vatl_seed0_20261124/vat/vitl/b1_reverse.json`; SHA256: `2d26777dba5567a02bb63a053ef6ac2fa558705bb79a367d562077a3aef2a378`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_vatl_seed0_20261124/vat/vitl/b8_forward.json`; SHA256: `2a248f72083991780f3873d30ca5f31e4afd408c71883daf619d0475399a89f6`
- cell: `/home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_vatl_seed0_20261124/vat/vitl/b8_reverse.json`; SHA256: `eb93512200cfbd25b90ac9e3d3e5ba7ba2e0dffd374a435b5f1adda9d4260f4b`

## E3-L VAT-L completed GPU0 efficiency review · 2026-11-24

- CPU check: `/home/fb/src/paper/gazelleV1_results_vatl_20261124/cscwd_router/results/e3_vatl_efficiency_server_checks.json`; SHA256: `553f79aec85853d2c0751317bf76433b1b07dcd1835e67d0f39d66624b758461`
- Result transfer commit: `afcc39eacf34f450beb8eded4e5ea8e4c01e0bf0`; execution code: `0a375054feca2920133e72e26fcc8c1f7afd73f9`; GPU0 UUID: `GPU-9abd86a3-e5fb-bca1-4843-b29aefbfbd46`. Original log retains the name `efficiency_vat_l_gpu1.log`.
- Six original reports and 25 archived source file hashes verified locally; 27 source entries, five source/weight references, annotation and eight RGB hashes checked on server. Summary JSON/MD and independent speedup recomputation match. This review launched no GPU forward, timing or training.
