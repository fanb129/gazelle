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
| 单帧 GazeSpot ViT-B | 0.940182 | 0.098995 | 0.886801 | `t3_stag_comparison.json`，T3 同口径重算 |
| +STAG 初始规则门控 | 0.936860 | 0.099197 | 0.886801 | `t3_stag_comparison.json`，固定 `[-2,20,2,10]` |

- 单帧主缓存 checkpoint（用户于 2026-11-12 确认，指定速度优先选 ViT-B）：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；模型 `gazelle_dinov3_vitb16_inout`，GGSF + SASA。后续单帧/STAG 对照统一此底座。
- ViT-L checkpoint（用户同期确认的备用配置）：`/home/fb/src/paper/gazelleV1/experiments/train_vat_vitl_sasa_ggsf/2026-03-25_14-35-38/epoch_7.pt`；模型 `gazelle_dinov3_vitl16_inout`。
- T2 运行脚本：`cscwd/code/scripts/cache_single_frame.py`；服务器核对通过后自动追加实际覆盖、checkpoint/骨干/NPY 路径与 SHA256，回传小报告 `cscwd/results/t2_single_frame_cache.json`。
- T3 运行脚本：`cscwd/code/scripts/run_stag.py`；固定初始权重 `[-2,20,2,10]`，同一 T2 缓存上的单帧/STAG 共用标准指标函数。运行成功后追加真实对比数字及平滑/诊断 NPY 路径与 SHA256，回传 `cscwd/results/t3_stag_comparison.json`。
- T4 运行脚本：`cscwd/code/scripts/eval_temporal.py`；协议与固定阈值见实现计划。两侧标准指标重现 T3 后写出时序汇总；回传 `cscwd/results/t4_temporal_comparison.json`。逐轨迹/事件明细留服务器，路径/SHA256 自动追加。
- 评测脚本：`cscwd/code/scripts/eval_temporal.py`，在 track 级 JSON 上跑。
- 结果文件：`cscwd/results/*.json`。

## 时序指标（贡献）

- 帧间抖动 Jitter↓ / 切换响应延迟↓ / 稳定占比↑：脚本 `cscwd/code/stag/metrics.py`。
- 诚实校验：抖动同报真实注视点位移作参照。

T4 真实结果摘要（源文件 `t4_temporal_comparison.json`，结果提交 `7f15cd7`）：

| 指标 | 单帧 | 初始 STAG |
|---|---:|---:|
| Jitter mean↓ | 0.016647 | 0.012135 |
| 相同帧对 GT 位移 mean | 0.005922 | 0.005922 |
| 稳定占比↑ | 77.34% | 86.67% |
| 准确且稳定占比↑ | 52.75% | 58.15% |
| 全部可评估事件响应率↑ | 59.21% | 58.52% |
| 可评估 gaze_shift 响应率↑ | 58.23% | 55.06% |

- Jitter 降低 27.10%，准确且稳定占比提高 5.40 个百分点，均从上述源文件真实值计算。
- 共同响应的 337 个事件：STAG−单帧延迟均值为 +0.13353115727002968 帧；其中共同响应的 87 个 gaze_shift 为 +0.27586206896551724 帧。gaze_shift 单帧独有响应 5 个、STAG 独有响应 0 个。
- 当前结果支持稳定性改善，切换响应未改善；门控相对普通 EMA 的额外价值由后续 T5 比较。

## 消融（单帧 / 纯EMA / 规则门控 / 可学习门控）

- T5 首轮入口：`cscwd/code/scripts/run_ema_ablation.py`；固定 EMA 为 `sigmoid(-2)`、`0.25/0.5/0.75`，与单帧及当前 STAG 共六种方法，全部报告。
- 小结果：`cscwd/results/t5_ema_ablation.json`、`cscwd/results/t5_ema_ablation.md`；沿用 T4 协议，分别保存 EMA−单帧、EMA−STAG 的同事件配对延迟。
- 完整 EMA 缓存和逐轨迹/事件明细留服务器 `cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/`；实际运行后由脚本追加真实数字、路径与 SHA256。

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

## T4 标准与时序指标（服务器运行）

- UTC：`2026-11-12T14:39:32.455983+00:00`；git commit：`833270a4bda8434da0cad09dc0b9bc8f9b3a729c`。
- checkpoint：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；SHA256：`058793f58a1ad3f6e08a721d776a2a0412a2f0ebc5917b11dd0f78b29672f9ea`。
- STAG 权重：`[-2.0, 20.0, 2.0, 10.0]`；评测协议和阈值完整保存在小报告。
- 命令：`/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/eval_temporal.py --single-frame-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test --single-frame-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --stag-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test --stag-report /home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json --details-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/temporal_rule_v1_test --shift-threshold 0.15 --response-radius 0.10 --stable-threshold 0.02 --hold-frames 3 --max-response-frames 30 --report /home/fb/src/paper/gazelleV1/cscwd/results/t4_temporal_comparison.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`。
- 小报告：`/home/fb/src/paper/gazelleV1/cscwd/results/t4_temporal_comparison.json`。
- 逐轨迹/逐事件明细（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/temporal_rule_v1_test/temporal_details.json`；SHA256：`49e3b7e24b3e51fb4ef141ea6968f64b16f968cc2f4df5dac147faa9bafde95f`。
- single_frame 来源报告：`/home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json`；SHA256：`5f2da6594925aec1443fdf38b1ac3881e11af65b5114b19c3c7f968e6fd15022`；缓存 index SHA256：`57b161c434a0760322c1707a2e57269b0977f3bd58b3f7814cfd7618c2f175f6`。
- stag 来源报告：`/home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json`；SHA256：`84ab4a3f6cf56589222e167f2d7125eeea734dbc5aa26ce875d93b1daa1cc10c`；缓存 index SHA256：`7d6f8f0e6ae17001dfd64631615fcf1cf555b7f55f6c6d90455b69de5b48fe9c`。
- single_frame：AUC `0.9401816577151993` / L2 `0.09899483221303321` / AP `0.8868010398787617`。
- single_frame：Jitter mean `0.016646765511375713`，同帧对 GT 位移 mean `0.0059218882811410915`；稳定占比 `0.7734390786017378`，准确且稳定占比 `0.5275308143059204`。
- single_frame：切换代理事件 `688`，响应率 `0.5920826161790017`，已响应起始延迟 mean `2.991279069767442` 帧，未响应 `237`，短窗口 `107`。
- stag：AUC `0.9368598071467544` / L2 `0.09919693983541644` / AP `0.8868010398787617`。
- stag：Jitter mean `0.012135128302046163`，同帧对 GT 位移 mean `0.0059218882811410915`；稳定占比 `0.8667407557082238`，准确且稳定占比 `0.5815316225500101`。
- stag：切换代理事件 `688`，响应率 `0.5851979345955249`，已响应起始延迟 mean `2.9911764705882353` 帧，未响应 `241`，短窗口 `107`。

## T5 固定 EMA 对照（服务器运行）

- UTC：`2026-11-12T15:10:50.325679+00:00`；git commit：`76b7c0b7d4c2ec3794dff186fcbd0041ce7349df`。
- checkpoint：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；SHA256：`058793f58a1ad3f6e08a721d776a2a0412a2f0ebc5917b11dd0f78b29672f9ea`。
- 命令：`/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/run_ema_ablation.py --single-frame-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test --single-frame-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --stag-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test --stag-report /home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json --temporal-report /home/fb/src/paper/gazelleV1/cscwd/results/t4_temporal_comparison.json --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test --ema-alphas 0.25 0.5 0.75 --report /home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`。
- 小结果：`/home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.json`；Markdown：`/home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.md`。
- 逐轨迹/事件明细：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ablation_details.json`；SHA256：`9a64353d1b784febd6c06511e69b76e1aae25cb372089bae6713bcc0720c2ad2`。
- 来源 single_frame：`/home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json`；SHA256：`5f2da6594925aec1443fdf38b1ac3881e11af65b5114b19c3c7f968e6fd15022`。
- 来源 stag：`/home/fb/src/paper/gazelleV1/cscwd/results/t3_stag_comparison.json`；SHA256：`84ab4a3f6cf56589222e167f2d7125eeea734dbc5aa26ce875d93b1daa1cc10c`。
- 来源 temporal：`/home/fb/src/paper/gazelleV1/cscwd/results/t4_temporal_comparison.json`；SHA256：`3431b84beef1e0040519cafae0cf03c9d2008f863f7ffba455b22793d4e53315`。
- single_frame：AUC `0.9401816577151993` / L2 `0.09899483221303321` / AP `0.8868010398787617` / Jitter `0.016646765511375713` / 准确稳定占比 `0.5275308143059204` / gaze_shift 响应率 `0.5822784810126582`。
- stag：AUC `0.9368598071467544` / L2 `0.09919693983541644` / AP `0.8868010398787617` / Jitter `0.012135128302046163` / 准确稳定占比 `0.5815316225500101` / gaze_shift 响应率 `0.5506329113924051`。
- ema_bias_matched：AUC `0.9327503218436328` / L2 `0.11058682202016828` / AP `0.8868010398787617` / Jitter `0.007274233812445743` / 准确稳定占比 `0.5808244089715094` / gaze_shift 响应率 `0.4620253164556962`。
- ema_bias_matched 缓存：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_bias_matched`；index SHA256：`1b94af0c942b8ca405af6c3da820b51cd4df87750cc8139bc33d431e23c10422`；α=`0.11920292202211753`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_bias_matched/heatmap.npy`；SHA256：`f0780b5e5d608e74d2708b57f6b5e45cc5edea63d3e3e6ef3c8b4611700a8d4c`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_bias_matched/inout.npy`；SHA256：`234dbab7f8310e9f330b3d1ea21a2066de576d1277f69a34e5df2b6a9335fcbd`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_bias_matched/argmax_xy.npy`；SHA256：`a80017a039c141b8c6ec19e9d07010aa250ba1b5a6210a3d12d825ee8abc1c24`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_bias_matched/alpha.npy`；SHA256：`6a8284783983c17e86439b9e824de8cb5de4170246349b9491442f4354aa2c1e`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_bias_matched/reset.npy`；SHA256：`3e96b3883517d0dff7b24ae40e90163a405c3c8847f19aafdfe73b034123be0d`。
- ema_alpha_0p25：AUC `0.9376132785477839` / L2 `0.10259948881789266` / AP `0.8868010398787617` / Jitter `0.009248240715306048` / 准确稳定占比 `0.5730955748636088` / gaze_shift 响应率 `0.5063291139240507`。
- ema_alpha_0p25 缓存：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p25`；index SHA256：`93a9c82f5dd9ff17fa347614a21f7c54964cfb2e5f1164622bd75b439b3f52fd`；α=`0.25`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p25/heatmap.npy`；SHA256：`1f2d1f5fa3b5b8d7bc8b99b2a67679151c32493f310888c0b76fea21c796f32e`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p25/inout.npy`；SHA256：`234dbab7f8310e9f330b3d1ea21a2066de576d1277f69a34e5df2b6a9335fcbd`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p25/argmax_xy.npy`；SHA256：`11e64716b625813f8b23fb25fa564903aa5dedac2d88973fbf3d71d1ca1250b7`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p25/alpha.npy`；SHA256：`2c8fb7040739d9dbc5e45eafd77dfed1f8d6fc0223b69af47d5ba84b7c69d4e6`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p25/reset.npy`；SHA256：`3e96b3883517d0dff7b24ae40e90163a405c3c8847f19aafdfe73b034123be0d`。
- ema_alpha_0p5：AUC `0.9396895318318765` / L2 `0.09963527150651678` / AP `0.8868010398787617` / Jitter `0.011698977406574896` / 准确稳定占比 `0.5571832693473429` / gaze_shift 响应率 `0.5506329113924051`。
- ema_alpha_0p5 缓存：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p5`；index SHA256：`8b3d28198843ecece72700cb03c41dc1ecb731a85eaa1b9c2de1e84103a6560f`；α=`0.5`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p5/heatmap.npy`；SHA256：`9bc7b6e143e489b34b5f3b01f4a8ec291fafe10a8040df5b9cc7b05bb8fdb593`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p5/inout.npy`；SHA256：`234dbab7f8310e9f330b3d1ea21a2066de576d1277f69a34e5df2b6a9335fcbd`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p5/argmax_xy.npy`；SHA256：`961582403e06b065c3460cb82b6b2529d5c3c60fb0b36839bf149ebaf1220050`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p5/alpha.npy`；SHA256：`1c2ae3d90c5bfabc2573f51fc2a5520c8388f800a43f471376150b8181808c98`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p5/reset.npy`；SHA256：`3e96b3883517d0dff7b24ae40e90163a405c3c8847f19aafdfe73b034123be0d`。
- ema_alpha_0p75：AUC `0.9401198060239975` / L2 `0.09894134421564804` / AP `0.8868010398787617` / Jitter `0.01422873674966188` / 准确稳定占比 `0.5425338452212568` / gaze_shift 响应率 `0.569620253164557`。
- ema_alpha_0p75 缓存：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p75`；index SHA256：`c9abfbc07e171a8ec964bd67dab9f681a256ae7ab84080797eececf6cc041a91`；α=`0.75`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p75/heatmap.npy`；SHA256：`911d77da6915653d0c49d7bd147976975e9e07ed73aabbff75dc6008b0ddbb81`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p75/inout.npy`；SHA256：`234dbab7f8310e9f330b3d1ea21a2066de576d1277f69a34e5df2b6a9335fcbd`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p75/argmax_xy.npy`；SHA256：`98c140d2089f71f95b9c380ddc4c7fa50d2e9a9ead0953fe56fbcea4b9f5683c`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p75/alpha.npy`；SHA256：`ac7dc624af5913ee9cdd2b9f8056af92cbd760ad4a11f8a1a4e9434faa5539d5`。
  - 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/ema_alpha_0p75/reset.npy`；SHA256：`3e96b3883517d0dff7b24ae40e90163a405c3c8847f19aafdfe73b034123be0d`。
