# CSCWD 时序方向 · 结果来源（provenance）

> 每个写进论文的数字，在这里记来源。内部写作辅助，不进论文。
> 格式：数字 → 来源脚本/运行 → checkpoint → 结果文件路径 → SHA（如适用）。
> 只记真实跑过的；没跑的不写。

## T1 track 恢复运行约定

- 脚本：`cscwd/code/data_prep/preprocess_vat_tracks.py`；服务器运行命令见 `cscwd/implementation_plan.md` 的 T1 记录。
- 计数核对通过后，脚本在本文件末尾追加真实轨迹/图像帧/人物帧数、命令、代码版本及完整 track JSON 路径/SHA256。
- 小报告：`cscwd/results/t1_track_recovery.json`，随本文件经 git 回传；完整 JSON 留服务器数据集下 `cscwd_tracks/`。

## 单帧 baseline vs STAG 主表（VAT test，真实框，纯 RGB，track 级）

| 配置 | AUC↑ | L2↓ | In/Out AP↑ | 来源 |
|---|---:|---:|---:|---|
| 单帧 GazeSpot | ? | ? | ? | 待填 |
| +STAG 时序门控 | ? | ? | ? | 待填 |

- 单帧主缓存 checkpoint（用户于 2026-11-12 确认，指定速度优先选 ViT-B）：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；模型 `gazelle_dinov3_vitb16_inout`，GGSF + SASA。后续单帧/STAG 对照统一此底座。
- ViT-L checkpoint（用户同期确认的备用配置）：`/home/fb/src/paper/gazelleV1/experiments/train_vat_vitl_sasa_ggsf/2026-03-25_14-35-38/epoch_7.pt`；模型 `gazelle_dinov3_vitl16_inout`。
- T2 运行脚本：`cscwd/code/scripts/cache_single_frame.py`；服务器核对通过后自动追加实际覆盖、checkpoint/骨干/NPY 路径与 SHA256，回传小报告 `cscwd/results/t2_single_frame_cache.json`。
- T3 运行脚本：`cscwd/code/scripts/run_stag.py`；固定初始权重 `[-2,20,2,10]`，同一 T2 缓存上的单帧/STAG 共用标准指标函数。运行成功后追加真实对比数字及平滑/诊断 NPY 路径与 SHA256，回传 `cscwd/results/t3_stag_comparison.json`。
- 评测脚本：`cscwd/code/scripts/eval_temporal.py`，在 track 级 JSON 上跑。
- 结果文件：`cscwd/results/*.json`。

## 时序指标（贡献）

- 帧间抖动 Jitter↓ / 切换响应延迟↓ / 稳定占比↑：脚本 `cscwd/code/stag/metrics.py`。
- 诚实校验：抖动同报真实注视点位移作参照。

## 消融（单帧 / 纯EMA / 规则门控 / 可学习门控）

（待填）

## 待填

## T1 track 恢复（服务器运行）

- UTC：`2026-11-12T11:18:45.752989+00:00`；git commit：`c36cead8a80786fb2a7e9706964afbf818f32685`。
- 脚本 SHA256：`9755f2544cb473134d4561239a2933786b3f667a428a91eb58a3470356449a2a`。
- 命令：`/home/fb/anaconda3/envs/py310/bin/python cscwd/code/data_prep/preprocess_vat_tracks.py --data-path /newhome/fb/dataset/videoattentiontarget --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_tracks --legacy-dir /newhome/fb/dataset/videoattentiontarget --splits train test --report /home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`。
- 小结果：`/home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.json`；框输入：当前帧原始真实框，未作时序平滑。
- train：1033 tracks / 58507 图像帧 / 132563 人物帧；与原 CSV 和拍平 JSON 核对通过。
  - track JSON（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_tracks/train_preprocessed_tracks.json`；SHA256：`96e2392c6b4470c6101525921b8e2ab5b03f326a26bb0fccf7b2650099a78fff`。
  - 原始标注 manifest SHA256：`d58446902da36061928d57a079cf78820a5f8132b3b9a4c14b822f1651f26946`；原拍平 JSON SHA256：`9831fe491277083dfc82cb843729a37e622c88c510e9c2c9e345158c03389e43`。
- test：298 tracks / 13127 图像帧 / 31978 人物帧；与原 CSV 和拍平 JSON 核对通过。
  - track JSON（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_tracks/test_preprocessed_tracks.json`；SHA256：`ee3a7514aa72507849c04c299f0394ee4b31721c3fbb1d05147708196cc4de82`。
  - 原始标注 manifest SHA256：`0625854fe565cd8484072205c8bd9b46998011e25fd8bdbb1d44bbce0ef2d065`；原拍平 JSON SHA256：`75c1097ac5ba417326b158f5f177720ae1c834b00d82f2588b3c28abc3b2c4aa`。

## T2 单帧预测缓存（服务器运行）

- UTC：`2026-11-12T11:47:03.066166+00:00`；git commit：`8e239ba2142adfc2c1770e01f605f94c16720ebb`。
- 模型：`gazelle_dinov3_vitb16_inout`，GGSF + SASA，冻结推理；checkpoint：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；SHA256：`058793f58a1ad3f6e08a721d776a2a0412a2f0ebc5917b11dd0f78b29672f9ea`。
- 骨干预训练：`/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`；SHA256：`73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。
- 输入 track JSON：`/newhome/fb/dataset/videoattentiontarget/cscwd_tracks/test_preprocessed_tracks.json`；SHA256：`ee3a7514aa72507849c04c299f0394ee4b31721c3fbb1d05147708196cc4de82`。
- 命令：`/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/cache_single_frame.py --data-path /newhome/fb/dataset/videoattentiontarget --track-json /newhome/fb/dataset/videoattentiontarget/cscwd_tracks/test_preprocessed_tracks.json --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt --model gazelle_dinov3_vitb16_inout --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test --batch-size 8 --num-workers 4 --device cuda:0 --report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`。
- 覆盖：298 tracks / 13127 图像帧 / 31978 人物帧；float32 热图、画内概率和 argmax 坐标原样读回通过。
- 小报告：`/home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json`；缓存目录（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test`。
- 缓存 index.json SHA256：`57b161c434a0760322c1707a2e57269b0977f3bd58b3f7814cfd7618c2f175f6`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test/heatmap.npy`；SHA256：`092fef89534b93892fa6d5caa3cfb22dc136d91a93869f4a3f99287a8a0ce01c`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test/inout.npy`；SHA256：`234dbab7f8310e9f330b3d1ea21a2066de576d1277f69a34e5df2b6a9335fcbd`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test/argmax_xy.npy`；SHA256：`1bb588501a83258e6854f98836d2662fd386caa01571b034281ea4bd7729c9cc`。

## T3 STAG 初始规则门控对比（服务器运行）

- UTC：`2026-11-12T12:10:33.387170+00:00`；git commit：`4c24e3b9e676a2783250e3e2149d218d9a289139`。
- 来源模型：`gazelle_dinov3_vitb16_inout`；checkpoint：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；SHA256：`058793f58a1ad3f6e08a721d776a2a0412a2f0ebc5917b11dd0f78b29672f9ea`。
- 输入缓存：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test`；index SHA256：`57b161c434a0760322c1707a2e57269b0977f3bd58b3f7814cfd7618c2f175f6`。
- 门控权重 [bias, d_head, trans, d_heat]：`[-2.0, 20.0, 2.0, 10.0]`，测试前固定；仅平滑热图，画内概率保持单帧输出。
- 命令：`/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/run_stag.py --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test --cache-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test --weights -2 20 2 10 --report /home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`。
- 小报告：`/home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json`；输出缓存（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test`；index SHA256：`7d6f8f0e6ae17001dfd64631615fcf1cf555b7f55f6c6d90455b69de5b48fe9c`。
- single_frame：AUC `0.9401816577151993` / L2 `0.09899483221303321` / In-Out AP `0.8868010398787617`。
- stag：AUC `0.9368598071467544` / L2 `0.09919693983541644` / In-Out AP `0.8868010398787617`。
- delta_stag_minus_single_frame：AUC `-0.0033218505684449173` / L2 `0.0002021076223832352` / In-Out AP `0.0`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test/heatmap.npy`；SHA256：`d51dbf364edf30a1389e5d735a8eeb3b7514e30bd496acd40b80b65efffc2eb2`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test/inout.npy`；SHA256：`234dbab7f8310e9f330b3d1ea21a2066de576d1277f69a34e5df2b6a9335fcbd`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test/argmax_xy.npy`；SHA256：`9a60f5d8c461e2ae6c2d01627488e7b96ce9baf23fc34ba67988b156eadfd6ea`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test/alpha.npy`；SHA256：`fe820064a7b37ba8c72ef24a78c60c367a5669aca1d2be2268df22973468c509`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test/signals.npy`；SHA256：`5c2f2fdd5c3873b64455d1cd7194c5927fd7e6a8a1ed84ff06b2cb8dcae00ce7`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test/reset.npy`；SHA256：`3e96b3883517d0dff7b24ae40e90163a405c3c8847f19aafdfe73b034123be0d`。
