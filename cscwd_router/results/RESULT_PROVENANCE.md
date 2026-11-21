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
