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
