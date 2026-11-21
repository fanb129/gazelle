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
