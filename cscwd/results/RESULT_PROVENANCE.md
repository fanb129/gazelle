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

T5 本地验收与解读（2026-11-13，结果提交 `3de06cd`，来源 `t5_ema_ablation.json/.md`）：

- 六种配置、完整来源/实现 SHA256、协议/参考结果重现、覆盖与计数、Markdown/JSON 一致性检查通过。
- STAG vs EMA `0.5`：准确稳定占比 **58.1532% vs 55.7183%**，L2 **0.099197 vs 0.099635**，AUC **0.936860 vs 0.939690**，Jitter **0.012135 vs 0.011699**；gaze_shift 响应率均 **55.0633%**。EMA−STAG 的直接配对延迟均值 **+0.15476190476190477 帧**（84 个共同事件），双方各有 3 个独有响应。
- STAG vs EMA `0.25`：准确稳定占比高 **0.8436 个百分点**、gaze_shift 响应率高 **4.4304 个百分点**；EMA−STAG 的直接配对延迟均值 **+0.5625 帧**（80 个共同事件），STAG 独有响应 7 个，EMA 独有响应 0 个。EMA 的 Jitter/AUC 更好。
- 偏置匹配 EMA−STAG 的直接配对延迟均值 **+1.4931506849315068 帧**（73 个共同事件），STAG 独有响应 14 个；两侧平滑强度不同。
- EMA `0.75` 实测 AUC **0.9401198060239975**、L2 **0.09894134421564804**、Jitter **0.01422873674966188**、gaze_shift 响应率 **0.569620253164557**；保留为已跑配置，不选作 test 最优模型。
- 结论：STAG 有部分门控折中收益，尚未全面优于普通 EMA，也未优于单帧切换响应。下一步 T6 定位成功与失败机制，再决定模型改动。

## T6 案例诊断运行约定

- 入口：`cscwd/code/scripts/visualize_temporal.py`，只读 T5 六种方法缓存及已存事件明细，确定性选择五类案例，全部单帧独有切换失败均保留。
- 小结果：`cscwd/results/t6_visualization.json/.md` 与 `cscwd/results/t6_previews/`；完整候选清单、逐帧 JSON 和全分辨率图留服务器 `cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/`，实际路径/SHA256 由脚本成功运行后追加。
- 原始热图固定 `[0,1]` 色标，GT 仅作标记/选样依据；所有预测仍为原缓存。真实图像和案例结果由服务器运行产出。

T6 本地验收与观察（2026-11-13，结果提交 `9e0fdd0`）：

- 已核验 `t6_visualization.json` 来源/实现 SHA256、候选与选择计数，以及全部 10 张本地预览 PNG 的 SHA256/大小。完整输出 13 例，五类候选数 7/5/11/726/16；已实际查看五类各一例的帧图/时间线，其余完整图与逐帧 JSON 仍在服务器。
- `stag_over_ema_96f0fc50a3877070`：CBS `2517`，单帧/STAG 延迟 0，EMA `0.25/0.5` 在 3 帧窗口内 censored。
- `stag_lag_bbc34ab341e623ee`：CBS `12144`，单帧 +8 帧、STAG +12 帧、EMA `0.25` +13 帧、EMA `0.5` +12 帧。
- `single_frame_only_a2f1575f30b3f20b`：Downton `13190`，单帧 +6 帧，STAG 未形成连续 3 帧成功而 censored；预览中仍有短暂低误差段。
- 后续先缓存 train 的冻结预测并按 video 组划分 train/validation，校准已有门控参数；不把上述测试案例直接用来选权重。新缓存小报告为 `t5_calibration_train_cache.json`，实际数字由服务器运行后追加。

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

## T6 时序案例诊断（服务器运行）

- UTC：`2026-11-12T17:47:23.334761+00:00`；git commit：`0fafc8fd59263150b1046837991daa43563bf55f`。
- T5 报告：`/home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.json`；SHA256：`552f97dbd99c6078a514d745e70ea152b2ca705f28a0bb089e6bdae8d378edac`。
- checkpoint：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；SHA256：`058793f58a1ad3f6e08a721d776a2a0412a2f0ebc5917b11dd0f78b29672f9ea`。
- 命令：`/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/visualize_temporal.py --data-path /newhome/fb/dataset/videoattentiontarget --ablation-report /home/fb/src/paper/gazelleV1/cscwd/results/t5_ema_ablation.json --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics --preview-dir /home/fb/src/paper/gazelleV1/cscwd/results/t6_previews --cases-per-group 2 --preview-cases 5 --context-before 10 --context-after 20 --min-heads 4 --snapshots 5 --dpi 140 --report /home/fb/src/paper/gazelleV1/cscwd/results/t6_visualization.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/t6_visualization.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`。
- 小结果：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_visualization.json`；摘要：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_visualization.md`；预览：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews`。
- 完整候选清单：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/selection_manifest.json`；SHA256：`87ac4516f3d03c26af4eaeab90d75760dee568a174ed6875850acd395ae1bfde`。
- stag_over_ema：候选 7，完整输出 2。
- single_frame_only：候选 5，完整输出 5。
- stag_lag：候选 11，完整输出 2。
- stable_gain：候选 726，完整输出 2。
- crowded：候选 16，完整输出 2。
- 案例 `stag_over_ema_96f0fc50a3877070`：test/CBS This Morning/2487_2818/s01 / 原帧号 2517。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_over_ema_96f0fc50a3877070/case_data.json`；SHA256：`db2628701265dbcd96ec24746635d43dd5db07a59e9cfc9a98f104a685974665`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_over_ema_96f0fc50a3877070/frames.png`；SHA256：`80c116ab319c09fc8802058e133e2f0b3fca614d0e7193a62cdaff5c63ee0f9b`。
  - preview_frames：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/stag_over_ema_96f0fc50a3877070_frames.png`；SHA256：`6ab1a5b7c958093f719cfeba92d3bd606d09d15b97176d4e014fbd297e86abaf`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_over_ema_96f0fc50a3877070/timeline.png`；SHA256：`b8056c698fc84ded410987a308c62e1d1bdec512a10e2b7a16d39e2ffe6ead41`。
  - preview_timeline：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/stag_over_ema_96f0fc50a3877070_timeline.png`；SHA256：`4c956027632b43a32a034433b40c8895ea678311899e366244be2d57d858dfa8`。
- 案例 `stag_over_ema_1c421f5b19eb9de7`：test/CBS This Morning/3446_3567/s00 / 原帧号 3551。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_over_ema_1c421f5b19eb9de7/case_data.json`；SHA256：`1543654e48e2320dcd38a3099834aadf55bf068de4306de9ac377176b295987f`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_over_ema_1c421f5b19eb9de7/frames.png`；SHA256：`3b44b5f297515dadd7436b90f5e62624c957df88777f18f1e66b50200700dffb`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_over_ema_1c421f5b19eb9de7/timeline.png`；SHA256：`211e02b0204f868b7c5d0d80097f315f5f057f7ec276a4380c249e3d51c038f6`。
- 案例 `single_frame_only_a2f1575f30b3f20b`：test/Downton Abby/13066_13217/s02 / 原帧号 13190。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_a2f1575f30b3f20b/case_data.json`；SHA256：`ad90f2e51eb4ae914d1d8087c73db7ef957680db3c981e8ab15fb4bbfa12a6e5`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_a2f1575f30b3f20b/frames.png`；SHA256：`8dc1f3181cb7e4681e2395d6334a5334166e0a02ab8cd34852f7e6f98c8839ab`。
  - preview_frames：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/single_frame_only_a2f1575f30b3f20b_frames.png`；SHA256：`95b5a94d3215ad4e31944b568210328f1a8064914d7de10d72407f6c5d10969d`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_a2f1575f30b3f20b/timeline.png`；SHA256：`b949db63917540123fccb9b4561112e599e8cd7aded2d119fe71c76dbeb30a8e`。
  - preview_timeline：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/single_frame_only_a2f1575f30b3f20b_timeline.png`；SHA256：`080bf824d96709ad9bb7a13092d4fada5cb44bae42d94b85d241a3800c691ac1`。
- 案例 `single_frame_only_9bd1cc7f8ed30e35`：test/Hell's Kitchen/104235_104266/s01 / 原帧号 104254。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_9bd1cc7f8ed30e35/case_data.json`；SHA256：`6c54d0a6dff4f4a50a361d269e9ea6815f79022a6abf102aa645bfb44f13a891`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_9bd1cc7f8ed30e35/frames.png`；SHA256：`0cfc7204d7569fac9ea528526ed2c0fa56bcb95830ab8120ef328963c16150a0`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_9bd1cc7f8ed30e35/timeline.png`；SHA256：`c68891f122801531279ca9cbe2c9c23862ff723178a6d8cc9127b537d74af926`。
- 案例 `single_frame_only_679c8fbd8c09f1d7`：test/It's Always Sunny in Philadelphia/46470_46710/s01 / 原帧号 46655。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_679c8fbd8c09f1d7/case_data.json`；SHA256：`da2a1ce637c0be9c1dc685841c190bfe16f36e7622bacac18d273d986981abdc`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_679c8fbd8c09f1d7/frames.png`；SHA256：`d1195a9451ff0b0e41c3eae8ff6ee5d4e6fa33e943eaeaa05f3a447f70228093`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_679c8fbd8c09f1d7/timeline.png`；SHA256：`3345487288d8cc0ce0989305a5453ec5102ef4b7591893f1e374041cf29b6845`。
- 案例 `single_frame_only_4b76dfb4660a5d91`：test/It's Always Sunny in Philadelphia/46470_46710/s01 / 原帧号 46696。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_4b76dfb4660a5d91/case_data.json`；SHA256：`c8ee7c9685e4ed2d4bf925a55e2203de67914e8204a1c35621586a0a3d63fc55`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_4b76dfb4660a5d91/frames.png`；SHA256：`83c690ba387985e77986bd0e12d0424d640fdc0b6e8f4f30829582bb059dabfa`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_4b76dfb4660a5d91/timeline.png`；SHA256：`6a228b3c1e4f76588bfe3860c7d0fbe4aed630a0dc07bd85d42e5b45e89234d0`。
- 案例 `single_frame_only_5377cd78e01d64ff`：test/It's Always Sunny in Philadelphia/46470_46710/s02 / 原帧号 46704。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_5377cd78e01d64ff/case_data.json`；SHA256：`18fdb4c1137c944d3f1e3d57b09af7df0b65ffe92d58d05db810ebcd9ffe2ba6`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_5377cd78e01d64ff/frames.png`；SHA256：`1d39824fb95081a7e565e947b62573cabbb3ed6398ee3977c04d1f3e6c116dce`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/single_frame_only_5377cd78e01d64ff/timeline.png`；SHA256：`4cef427b5ac803e8c8ade5f0c7c33a7492733d00a2f568d296bc23a28dee0763`。
- 案例 `stag_lag_bbc34ab341e623ee`：test/CBS This Morning/12137_12258/s03 / 原帧号 12144。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_lag_bbc34ab341e623ee/case_data.json`；SHA256：`76093001eb07384fc8d5ece0e438ddf4d135223199baa1e417403ee78bbf8efb`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_lag_bbc34ab341e623ee/frames.png`；SHA256：`d6b0d8fdc575ecd3f1d0b2d1de899674d745ed2ee066513348758611af3024bc`。
  - preview_frames：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/stag_lag_bbc34ab341e623ee_frames.png`；SHA256：`86b4b0a3f900200a573bdd0898cf4ec8d1a254ef4ef7b4a5f823acbfb8f3adfd`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_lag_bbc34ab341e623ee/timeline.png`；SHA256：`66e5c3aefe21714e183b7028cf02b904dcb6adf91692c3820359fc31aac10783`。
  - preview_timeline：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/stag_lag_bbc34ab341e623ee_timeline.png`；SHA256：`be873df10afcc2f413eae7d5dcdef95695db7b29ed70b4bf515f25f438915e6b`。
- 案例 `stag_lag_1346752560acc288`：test/CBS This Morning/12407_12648/s03 / 原帧号 12474。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_lag_1346752560acc288/case_data.json`；SHA256：`305c2816a24e43c142e26ca41c32e249577c72548dab23c9ddb0214b583c42ca`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_lag_1346752560acc288/frames.png`；SHA256：`ac7c933ad7b5c00b6837ec2152ee74309bbaca5c33bb173e319794a3fdf62b02`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stag_lag_1346752560acc288/timeline.png`；SHA256：`4c6a6d071767d303944dd66709dd58f0a6eb968ff849cd58c0b3efcb9ca1484d`。
- 案例 `stable_gain_84f76562c98320b6`：test/CBS This Morning/11118_11359/s00 / 原帧号 11128。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stable_gain_84f76562c98320b6/case_data.json`；SHA256：`b9f688f41f925010492f68d24641d3f8255ce60b0dc063207f3e9d98191d7c4d`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stable_gain_84f76562c98320b6/frames.png`；SHA256：`d2aead2f5ed5db44b777e2b44b92582bf08b7b1887964e2eef6e094dbbdc6cfa`。
  - preview_frames：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/stable_gain_84f76562c98320b6_frames.png`；SHA256：`d108bac5c9af076ea7996b39af785e34bf3df4654ad8c4dec1a4479cb0ecbf04`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stable_gain_84f76562c98320b6/timeline.png`；SHA256：`ddaacb9a7338ffd430423a84c6966bfe5c2681eeb0ada95a94dc575042d40038`。
  - preview_timeline：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/stable_gain_84f76562c98320b6_timeline.png`；SHA256：`a2890de742fee3206c74dcfdb801737826217f54a1fd381704a23a54ba20e44b`。
- 案例 `stable_gain_3204f959c5946bb9`：test/CBS This Morning/11118_11359/s00 / 原帧号 11184。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stable_gain_3204f959c5946bb9/case_data.json`；SHA256：`7f1eefd46b450d3ee88104cf1309036093455777f61d45457c476dbfd2a76fcc`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stable_gain_3204f959c5946bb9/frames.png`；SHA256：`171bc0140774a1896f7a1d79bf9dc20452d0e04a18016d06f5d00d8e2278087d`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/stable_gain_3204f959c5946bb9/timeline.png`；SHA256：`9d2fa07421aa97fd4d271acd13d8b5f1181a6b6c71593b1c277dbc5ee74766c8`。
- 案例 `crowded_7e45e09419c673db`：test/CBS This Morning/11118_11359/s00 / 原帧号 11121。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/crowded_7e45e09419c673db/case_data.json`；SHA256：`8cf98dbd4a00ecda71f4b62f0aebcdbf4444ade58334e1ffe62b39eb3a971b4e`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/crowded_7e45e09419c673db/frames.png`；SHA256：`fcbcdec38aa0c107d2632b1d05c22cb77ed04213b28879fadb86d7f4807024e8`。
  - preview_frames：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/crowded_7e45e09419c673db_frames.png`；SHA256：`bd95edecc37cc0332cd054254d69a3d3e28a49c97ccea4fb90191ea6516a9f76`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/crowded_7e45e09419c673db/timeline.png`；SHA256：`e7e1a1eb60e8101037a269ca05e2f9dcabc60abfd164a05bc101468e3e6da2d0`。
  - preview_timeline：`/home/fb/src/paper/gazelleV1/cscwd/results/t6_previews/crowded_7e45e09419c673db_timeline.png`；SHA256：`bf3fe19bd3e274e02e01b449f5a09f80999a376e604d7726c7a5bf299b9ed4fe`。
- 案例 `crowded_ba4dcbbdf191947c`：test/CBS This Morning/12137_12258/s00 / 原帧号 12137。
  - data：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/crowded_ba4dcbbdf191947c/case_data.json`；SHA256：`7961de3a9fc15d7c2008229317c84603defe4dbb2f567acac2363f7d893f6f47`。
  - full_frames：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/crowded_ba4dcbbdf191947c/frames.png`；SHA256：`09025117839c129a8f6fdb541d4694794c5215d2b3225461cffff9d4a5ef2e50`。
  - full_timeline：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/cases/crowded_ba4dcbbdf191947c/timeline.png`；SHA256：`1d54bb0c0e5cce5f2e9ef48f62c246acc6d15b7571dba36294415127b87ca46f`。

## T2 单帧预测缓存（服务器运行）

- UTC：`2026-11-17T05:37:50.081389+00:00`；git commit：`909a81e0afe55e6ed535ef74fc6ac10caa601b82`。
- 模型：`gazelle_dinov3_vitb16_inout`，GGSF + SASA，冻结推理；checkpoint：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`；SHA256：`058793f58a1ad3f6e08a721d776a2a0412a2f0ebc5917b11dd0f78b29672f9ea`。
- 骨干预训练：`/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`；SHA256：`73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。
- 输入 track JSON：`/newhome/fb/dataset/videoattentiontarget/cscwd_tracks/train_preprocessed_tracks.json`；SHA256：`96e2392c6b4470c6101525921b8e2ab5b03f326a26bb0fccf7b2650099a78fff`。
- 命令：`/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/cache_single_frame.py --data-path /newhome/fb/dataset/videoattentiontarget --track-json /newhome/fb/dataset/videoattentiontarget/cscwd_tracks/train_preprocessed_tracks.json --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt --model gazelle_dinov3_vitb16_inout --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions --batch-size 8 --num-workers 4 --device cuda:0 --report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`。
- 覆盖：1033 tracks / 58507 图像帧 / 132563 人物帧；float32 热图、画内概率和 argmax 坐标原样读回通过。
- 小报告：`/home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json`；缓存目录（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions`。
- 缓存 index.json SHA256：`aadb2646d5b99ac57191025f88155e767fc203999b71ee05ea60edce72da7770`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions/heatmap.npy`；SHA256：`a26d901918e5969cea32d3e1d22f6fcc84ebef47d44baff742a7672ef6d9b0a6`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions/inout.npy`；SHA256：`9910c4811b1ac24d4a7f444ded3e77c301f54817657e648d06fe12b2381ce533`。
- 大文件：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions/argmax_xy.npy`；SHA256：`56f7743df02018f7570911e2d3b59fef4fe749104911854bb88753956e33ce6c`。

## R1 开发集分组与单帧诊断（服务器机器准备）

- UTC：`2026-11-17T06:50:17.188790+00:00`；commit：`909a81e0afe55e6ed535ef74fc6ac10caa601b82`。
- 命令：`/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/prepare_development.py --data-path /newhome/fb/dataset/videoattentiontarget --cache-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions --cache-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --recovery-report /home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.json --reference-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_development_v1 --preview-dir /home/fb/src/paper/gazelleV1/cscwd/results/r1_previews --validation-fraction 0.2 --seed 20261117 --cases-per-category 3 --preview-cases 12 --context-before 10 --context-after 20 --snapshots 5 --report /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`。
- 小报告：`/home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json`。
- 机器检查：`True`；阶段状态：`awaiting_local_review`；人工诊断未完成。
- split_manifest（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_development_v1/split_manifest.json`；SHA256：`5bfe98c29f739c44b0fa55ec26bf2ea4a64979379fbd02d19abffb2c925bf28d`。
- selection_config（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_development_v1/selection_config.json`；SHA256：`1508c885c4a0bec61a51653f656ebb88d817ba256c18da181efbc9b478ca9534`。
- diagnosis_manifest（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_development_v1/diagnosis_manifest.json`；SHA256：`3cc82fd84ca5bb837889586db5644a8d79a3b4114d7c1c1ab5b03307a63cc074`。
- details（服务器保留）：`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_development_v1/baseline_details.json`；SHA256：`95d477eb2b9bf9e446fce23bb904f26a8f53e53bfcc52b8ea1175330d281e27c`。
- cache_report：`/home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json`；SHA256：`dbbb121a36b883b46d242b44e86047ed59096275e173128bb6f5c02406a72c6a`。
- recovery_report：`/home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.json`；SHA256：`2ae463a901f9c0da5337f976bf86af86676479a0fb7faf6942bf21444355aac0`。
- checkpoint_reference_report：`/home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json`；SHA256：`5f2da6594925aec1443fdf38b1ac3881e11af65b5114b19c3c7f968e6fd15022`。

## R1 本地验收与人工诊断（2026-11-17）

- 结果提交：`d6adba41b3f9e89dc138400e3f15fb576bbf3289`；服务器实现：`909a81e0afe55e6ed535ef74fc6ac10caa601b82`。
- 来源小报告、9 个实现文件、12 张本地 PNG 的 SHA256/大小、标准/joint/state 协议、分组种子与计数关系的本地核对通过。服务器 NPY/index/完整 manifest/逐事件数据留原路径，本地未重新计算全量预测指标。
- 覆盖：1,033 tracks / 132,563 人物帧 / 58,507 图像帧。按 annotation-video/program 保守分组：train 32 组、827 tracks、98,362 人物帧；validation 8 组、206 tracks、34,201 人物帧。该留出仅约束新增模块，冻结 VAT 底座此前已见过官方 train。
- validation 单帧：AUC `0.9542055630729601` / L2 `0.07960456904362263` / AP `0.9609226821493625`，不是新方法增益。
- validation 画外/画内错误率：`0.25733344048862816` / `0.027713944296350768`；state-only enter `179/189`、leave `93/169`；joint gaze_shift `135/221`。state-only 与 joint enter 的可评估分母不同。
- 已查看全部 12 张回传快照图，人工标签可重叠：prediction_error 7、attention_target_change 2、unresolvable 8；属于诊断观察，不作为全量事件标签或模型历史输入。原机器报告和空人工模板保持原样，后续验收由下列记录给出。
- 本地验收产物：`cscwd/results/r1_local_verification.json`；SHA256：`3d5ec952f1feebd4ddfef87bc62c10c0ed33a598e7d8838118e6c6a70a0544d1`。
- 本地验收产物：`cscwd/results/r1_manual_review.json`；SHA256：`d831059ac1233051d44b9f69f286a01692d316bd50586ebdcdcfc701020d1285`。
- 本地验收产物：`cscwd/results/r1_manual_review.md`；SHA256：`83727ef5c883a4ccc16dab7eada4d8762ac4869b78db84cb631b9ef63a1ec8e3`。
- R1 验收通过，下一步 R2 冻结视觉特征契约与普通因果联合头；本次未启动 R2。R4 补记 train 拟合的全局单调状态校准对照，尚未运行，不回写原 R1 selection manifest。

## R2 本地实现与合成检查（2026-11-17）

- R1 验收提交 `f3e38d2` 已推送；R2 本地新增唯一图像原生末层特征探测/FP16缓存、普通因果GRU联合头、固定slot连续轨迹加载及smoke训练/完整验证入口。旧 gazelle 包和已有R1预测/报告未修改。
- 新增合成检查31项通过：特征7、模型10、数据8、来源/训练CLI6；另6项R1入口回归通过。合成指标不作为VAT性能。Python3.10语法、CLI help与服务器命令语法检查通过。
- 本地检查记录：`cscwd/results/r2_local_checks.md`；SHA256：`6375e64ee0e35a876d85496dd5596a7b8e0a3d0992b54fa91db85be0f25d1072`。
- 真实R2 probe/cache/VAT新模块训练尚未运行。先回传probe确认shape、量化、前向与存储契约，再缓存和固定一个epoch smoke；各真实数字与大文件路径/SHA256由服务器成功运行后追加。

## R2 frozen feature probe

- UTC: `2026-11-17T09:21:54.065569+00:00`; commit: `01c18f5ecd586016729bc7ac45d653f7103c4554`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/cache_frozen_features.py --mode probe --data-path /newhome/fb/dataset/videoattentiontarget --r1-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json --r1-review /home/fb/src/paper/gazelleV1/cscwd/results/r1_manual_review.json --cache-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --recovery-report /home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.json --reference-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --probe-images 32 --seed 20261117 --batch-size 8 --num-workers 4 --device cuda:0 --report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_probe.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_probe.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_probe.json`; SHA256: `fc451694487f2038ad659d04a8a77361f6ea4ab1cfa37ca4e3c0560768098c53`.
- Contract: `{"layer_index": 11, "channels": 768, "grid": [32, 32], "storage_dtype": "float16", "source_dtype": "torch.float32", "layout": "NCHW", "spatial_pooling": "none", "image_unit": "unique_current_RGB", "learned_projection_cached": false}`; original FP32 predictions retained.

## R2 特征 probe 本地验收（2026-11-17）

- 结果提交 `139a2c7` 已拉取；原 probe 报告 SHA256 `fc451694487f2038ad659d04a8a77361f6ea4ab1cfa37ca4e3c0560768098c53`，原报告与源码保持不变。
- 验收记录：`cscwd/results/r2_probe_review.json/.md`；结构化记录 SHA256 `e4790cd93c0b9d444a0a7a44e82c59eace0f715e47085a0a14587bf481b7c779`。来源/代码身份、开发 train 抽样、头框覆盖与存储算式共 37 项本地一致性核对通过，接受 layer 11、768×32×32、原生网格 FP16 契约。
- 真实 probe：32 图像 / 69 人物帧；FP32 原预测 heatmap/In/Out max abs 为 9.83476639e-7 / 4.17232513e-7，argmax 变化 0；FP16 特征 relative L2 误差 0.02042%。这些是探测结果，不是新模块性能。
- 58,507 张唯一图像的特征估算 92,023,554,048 bytes（85.7036 GiB），探测时磁盘可用 2,731,964,395,520 bytes（2.4847 TiB）。全量 NPY 目标路径为 `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16/scene_features.npy`；文件尚未生成，无文件 SHA256。
- 下一步用户手动执行实现计划 §9.3 全量缓存，成功后 §9.4 固定一个 epoch 联合头 smoke；不重跑 probe，cache/smoke 回传后整体验收 R2。

## R2 frozen feature cache

- UTC: `2026-11-17T10:30:43.658009+00:00`; commit: `ef646e1bd588a96571d17bd8eb4c8611c2292a10`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/cache_frozen_features.py --mode cache --data-path /newhome/fb/dataset/videoattentiontarget --r1-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json --r1-review /home/fb/src/paper/gazelleV1/cscwd/results/r1_manual_review.json --cache-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --recovery-report /home/fb/src/paper/gazelleV1/cscwd/results/t1_track_recovery.json --reference-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt --probe-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_probe.json --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --batch-size 8 --num-workers 4 --device cuda:0 --report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json`; SHA256: `cfbd01019bc0ff976cdcea4598211656e76101863495d1b55b8cf810e7205437`.
- Contract: `{"layer_index": 11, "channels": 768, "grid": [32, 32], "storage_dtype": "float16", "source_dtype": "torch.float32", "layout": "NCHW", "spatial_pooling": "none", "image_unit": "unique_current_RGB", "learned_projection_cached": false}`; original FP32 predictions retained.
- Server artifact: `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16/scene_features.npy`; SHA256: `82145bc436614802c7e4500dbdf39fdddb02ae442c091c5f63ada587df288c9c`.
- Cache index: `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16/index.json`; SHA256: `f817826cc271b7b257600d2f6db2fac3659f3b74111a7cabb0d84fab359feeb1`.

## R2 causal joint-head smoke training

- UTC: `2026-11-17T12:06:52.273231+00:00`; commit: `ef646e1bd588a96571d17bd8eb4c8611c2292a10`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/train_temporal_joint.py --feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --r1-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json --r1-review /home/fb/src/paper/gazelleV1/cscwd/results/r1_manual_review.json --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r2_gru_smoke_v1 --epochs 1 --learning-rate 0.0001 --lambda-state 0.1 --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 --device cuda:0 --report /home/fb/src/paper/gazelleV1/cscwd/results/r2_joint_smoke.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r2_joint_smoke.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r2_joint_smoke.json`; SHA256: `6ddc825b88fe73c1d42db261177937232c55487ac5551d6ac2dab28fdc15a611`.
- Server checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r2_gru_smoke_v1/last.pt`; SHA256: `aae319c7ef652af053305df6b96cdcc365cc29d4caae0ef74e89f505535c63c6`.
- Validation cache: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r2_gru_smoke_v1/validation_epoch_001`; index SHA256: `460eadfbfbc753044e0ef65a8da8ee72e6fefc9fb4c1008dd0181c3a0734ecac`.

## R2 全量缓存与 smoke 本地验收（2026-11-17）

- 结果提交 `85d9050` 已拉取，R2 最小冻结缓存/训练/验证主路径验收通过；本地记录 `cscwd/results/r2_local_review.json/.md`，JSON SHA256 `e172a980bf319abcc506bbfb9b6a224c71013bf47598fb284738aa2dc738344b`。原服务器报告保留。
- 回传来源和源码 SHA256、probe 契约、R1 baseline、覆盖与 loss/指标算式等 151 项一致性核对通过。完整 NPY/checkpoint/逐事件文件留服务器，路径与 SHA256 见上述两条运行记录及原报告；本地没有重算全量预测或配对延迟。
- 缓存 58,507 张唯一图像，`[58507,768,32,32]` FP16，文件 92,023,554,176 bytes。固定一个 epoch 训练 98,362 帧（画内 67,161）、821 chunks，347,522 个新增参数，验证 34,201 帧，两个头均更新。平均 MSE/BCE/total 为 0.003275220 / 0.301476298 / 0.033422850。
- validation 单帧→smoke：AUC 0.954205563→0.954325432，L2 0.079604569→0.079601696，AP 0.960922682→0.960799192；准确稳定 60.1874%→59.4586%，状态错误 11.1254%→10.5289%，state enter 179/189→175/189、leave 93/169→97/169，joint gaze_shift 135/221→135/221。
- R2 工程验收完成；当前配置未满足 R1/R4 的 state enter 响应下限，不选为最终模型。下一步 R3 本地实现历史目标视觉对应与有效性控制；R4 按共同预算比较普通 GRU、无历史修正头和单调状态校准，本次未启动新训练。

## 精简 R4 计划与 R3 本地实现（2026-11-17）

- 用户授权精简 R4，计划 `cscwd/r4_compact_plan.md`，版本 `r4_compact_v2`，SHA256 `13e8dccba14c6ade7825b55f505c1fee8c295239f50e3a785479108cf4a65bb4`。首轮普通 GRU/新机制各 seed 0、单配置、五个 epoch；选择规则与指标取舍在真实新训练前固定，原 R1/R2 结果和验收保留。
- R3 新增 `cscwd/code/stag/history_model.py`、`history_diagnostics.py` 及两套测试；来源 SHA256 见 `cscwd/results/r3_local_checks.json`，该 JSON SHA256 `2abe5e8ab5029807cdbbf64d84bb24d96c15b0ca3e924de05491657301d26b8d`。机制/状态/消融说明为 `cscwd/results/r3_mechanism_design.md`。
- 本机 CPU、Torch 2.6.0 实际构造 R3 372,677 参数，普通 GRU 347,522，共享 seed 初始化一致；native 768×32×32 FP16 输入的两步 FP32 模块前向通过。24 项新增合成检查（16 模型、8 集成）及18项R2回归通过，Python3.10语法通过。
- 合成临时数据实际跑通梯度更新、完整验证、checkpoint/预测记忆重载及诊断 NPY/index 写入读回；临时大文件未入 git。旧 gazelle、R2模型/加载器/训练入口与已有预测/结果未改。
- R3 本地验收完成，本次未运行真实 VAT 训练/评测或创建服务器大文件。下一任务实现精简 R4 训练、逐 epoch 验证/诊断、选择与报告入口，提供用户手动执行的完整 nohup 命令。

## R4 第一轮本地入口与检查（2026-11-17）

- 基于 `292c567` 实现精简 R4 首轮训练、同次前向验证/诊断、checkpoint 选择、比较与顺序 shell 入口，计划仍为 `r4_compact_v2`，计划 SHA256 `13e8dccba14c6ade7825b55f505c1fee8c295239f50e3a785479108cf4a65bb4`。原 R2/R3 源码与历史报告保持原样。
- 本地记录 `cscwd/results/r4_local_checks.json/.md`，JSON SHA256 `34a71be716c2bcbbc492c2e611254378570cbb9370f32e94d4d081870ceeea97`；新增源码 SHA256、命令及各组结果见该 JSON。31 项新增检查通过（协议11、验证5、训练6、比较9），8个Python文件的3.10语法及顺序脚本/命令块shell语法通过。
- CPU临时合成数据实际跑通两种模型各五个epoch、完整validation、双头/历史门更新、checkpoint重载、选中/无合格分支、诊断及完整训练/比较CLI。Torch子进程使用沙箱外本地检查解决OpenMP共享内存限制，生产代码不加环境绕过。
- 这些是本地功能检查，没有新增真实VAT指标或服务器大文件。§10给出用户手动运行的完整GPU/Python/参数/日志/输出路径命令；等待回传后审核第一轮效果，开发选择快照此时不冻结test配置。

## R4 中断后完整性复核（2026-11-17）

- 复核实现提交 `bda1501`，原R4九个源码文件SHA256与本地检查记录一致，无需修改源码；完整记录 `cscwd/results/r4_completeness_review.json/.md`，JSON SHA256 `580af8d2c085a7c42dd44b9afee88a975e8654947f7efc553a0733f7b97dcc5e`。
- 31项CPU合成检查重新通过（协议11、验证5、训练6、比较9）；另用临时替身程序验证三种后台入口场景、失败停止和带空格路径传参。替身不运行真实训练。
- 训练/验证/选模/比较/日志/回传清单与实现计划§10一致，两份独立只读源码复核无实际问题；没有新增真实VAT/CUDA结果。下一步仍按§10.1用户手动运行并回传。

## R4 compact training: ordinary_gru

- UTC: `2026-11-17T15:12:39.902326+00:00`; commit: `7459abc2182489247c71797aa8ec68720015720d`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/train_compact_joint.py --model-id ordinary_gru --feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --r1-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json --r1-review /home/fb/src/paper/gazelleV1/cscwd/results/r1_manual_review.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_compact_plan.md --epochs 5 --learning-rate 0.0001 --lambda-state 0.1 --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1 --device cuda:0 --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_ordinary_gru.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_ordinary_gru.md`.
- Plan: `/home/fb/src/paper/gazelleV1/cscwd/r4_compact_plan.md`; version: `r4_compact_v2`; SHA256: `13e8dccba14c6ade7825b55f505c1fee8c295239f50e3a785479108cf4a65bb4`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_ordinary_gru.json`; SHA256: `ffcb7168ba1f42c1de28b16d7a91dcb571289a6fd26f3628513f3a96e3eaf8d6`.
- Last checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/last.pt`; SHA256: `d8008bfa957002706e2264f460d451e4ae5e07f4dc6356db754ee8e0ee0058cf`.
- Selected epoch: `1`; checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/selected.pt`; SHA256: `a76fcbb48c9880bb887c84f9fcd61da8f71388d8ca0ce9b8e02f86bbeeb0c717`.
- Epoch 1: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/epoch_001.pt` SHA256 `a76fcbb48c9880bb887c84f9fcd61da8f71388d8ca0ce9b8e02f86bbeeb0c717`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/validation_epoch_001` index SHA256 `18a90a66a3ef893f55a25c0a796362f892f9e197bfd778ac30f3d60fd25dc29d`.
- Epoch 2: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/epoch_002.pt` SHA256 `da8d5c63c6f9ea50be216083e4a2c68221ac4dc2140696165d91e2732510871a`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/validation_epoch_002` index SHA256 `4375c9aeeb235e1b25c90b211866a3035d9ea58c4ac388df9349d21e51ddda32`.
- Epoch 3: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/epoch_003.pt` SHA256 `0de507029712111913d6014be3b5be65c630c2692313247ae882de9e16c2aa67`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/validation_epoch_003` index SHA256 `87264f73f62a4e80b07c20a067fb586ecce98ebd492c7104437c62e1f4f9d8a2`.
- Epoch 4: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/epoch_004.pt` SHA256 `e6e8e493ae40ee6696a194ab0c9e1942df89791f22049ef6a69e4ab1845a8188`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/validation_epoch_004` index SHA256 `3d8d985dd27afad7504106b99241b06354d48e4f0074b7a5b72d30716b9bfda2`.
- Epoch 5: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/epoch_005.pt` SHA256 `d8008bfa957002706e2264f460d451e4ae5e07f4dc6356db754ee8e0ee0058cf`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/ordinary_gru/validation_epoch_005` index SHA256 `033d8a16ab33d3271b5acbe81a4c81d6ff12722c938e48bb7d4d1883a00a9aff`.

## R4 compact training: history_validity

- UTC: `2026-11-17T15:53:10.562078+00:00`; commit: `7459abc2182489247c71797aa8ec68720015720d`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/train_compact_joint.py --model-id history_validity --feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --r1-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json --r1-review /home/fb/src/paper/gazelleV1/cscwd/results/r1_manual_review.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_compact_plan.md --epochs 5 --learning-rate 0.0001 --lambda-state 0.1 --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1 --device cuda:0 --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_history_validity.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_history_validity.md`.
- Plan: `/home/fb/src/paper/gazelleV1/cscwd/r4_compact_plan.md`; version: `r4_compact_v2`; SHA256: `13e8dccba14c6ade7825b55f505c1fee8c295239f50e3a785479108cf4a65bb4`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_history_validity.json`; SHA256: `7487a87de2985fdb3038c718f2981f67800cb96b71acbf8a00d171c5d3a96a81`.
- Last checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/last.pt`; SHA256: `330fb32711b094c54a99deab445616cae201a12cf41de8a5d4ed710879c858eb`.
- Selected epoch: `1`; checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/selected.pt`; SHA256: `6b478e92607efc0215956ec49464cde3b771c7200a4346f48b203da2d1c58ea0`.
- Epoch 1: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/epoch_001.pt` SHA256 `6b478e92607efc0215956ec49464cde3b771c7200a4346f48b203da2d1c58ea0`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_001` index SHA256 `d890940290819625ade8d67ce8240de0aed33ea5da2e6772621e15489c42f91a`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_001/history_diagnostics`; index SHA256: `a9cada93ecc560e26b7efe72b29f0c44513b4770af5f1f99810a3db3a6ba9a2c`.
- Epoch 2: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/epoch_002.pt` SHA256 `e83d914bc842e6a24f964cf66dcdf3092da5726d11186a48f7932899f10119bb`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_002` index SHA256 `d0a53c9b149c8d13189bf7148288414797e7cd96e024e7be6f457684324d7083`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_002/history_diagnostics`; index SHA256: `3eeb9daa26b51b0efee00d31f78b8ef582edd3e6d931da2197f0135d795b689e`.
- Epoch 3: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/epoch_003.pt` SHA256 `76452d9f5445fac0ebcee76981a41bd52800526f88f828e50f7693a8b7fdad2e`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_003` index SHA256 `4aaf7900d47009230a6a5fad3aaa9697107ac3de581412edce4b73544a9f4b8d`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_003/history_diagnostics`; index SHA256: `ba4f095a720f3770344afba0f6bc1de5cedeec30c5492b43c9fc24e678ed8f76`.
- Epoch 4: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/epoch_004.pt` SHA256 `e28ba8b75c52c962bc022736fd08c446f8a1937c9165b60b5fad3d8f53ddadb9`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_004` index SHA256 `664f595d6c9356f0e22853405444f4ae38a1525c0cd882c296d7bf9293483ccb`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_004/history_diagnostics`; index SHA256: `7b753eb97d44d535373ad63dca8f38bd492fa26dd1a480fabdf566c6680c3532`.
- Epoch 5: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/epoch_005.pt` SHA256 `330fb32711b094c54a99deab445616cae201a12cf41de8a5d4ed710879c858eb`; validation `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_005` index SHA256 `e2dfad46341ffcc67b1be43b68e76d0f789b51cd5ed4f851fa0f923086fbd6dd`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_compact_v2/history_validity/validation_epoch_005/history_diagnostics`; index SHA256: `18c26d99f8eb15184d9c1584734cbdfe4fefcfddaf73dde496ed58f87afab93a`.

## R4 compact validation comparison

- UTC: `2026-11-17T15:53:17.467411+00:00`; commit: `7459abc2182489247c71797aa8ec68720015720d`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/compare_compact_joint.py --run-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_ordinary_gru.json /home/fb/src/paper/gazelleV1/cscwd/results/r4_history_validity.json --r1-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_split_and_diagnosis.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_compact_plan.md --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_validation_comparison.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_validation_comparison.md --selection-manifest /home/fb/src/paper/gazelleV1/cscwd/results/r4_selection_manifest.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_validation_comparison.json`; SHA256: `695d3ece133231181112c54e88d03893171257b951f83537cbc857777f252508`.
- Summary: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_validation_comparison.md`; SHA256: `731adc47cde8054199de710ae3781a36f812a50d7e52b1272b27020db88ccb31`.
- Development selection snapshot: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_selection_manifest.json`; SHA256: `b6dfb1f820461c91c235e07dbf07a4459b1d472f171204f4d7fd9fc5ed45b7b3`.
- Test configuration frozen: `false`; awaiting development review.

## R4 第一轮真实结果本地验收（2026-11-18）

- 已拉取结果提交 `f1d41e4`，对应运行源码 `7459abc`、预定计划 `r4_compact_v2`。产出 `cscwd/results/r4_local_review.json/.md`，JSON SHA256 `52a5e26ee820d6cc720db8c0cea6d11f728cc8d64ec6c240f89689821ab00a05`，Markdown SHA256 `aa581f743fa3ff145b0f62021606f69150384a77b64ea9e8b3c8c65391d4adfa`。
- 本地235项来源/源码/预算/覆盖/损失与比例算式/选模/Pareto/manifest一致性核对通过。原服务器报告保留；完整NPY/PT/index/逐事件数据留服务器，本地未重算其大文件SHA或全量逐帧指标。
- 普通GRU与新机制均完成五轮，均按训练前固定规则选择epoch 1，selected与epoch_001的checkpoint SHA相同；总体validation首选仍为单帧。实际checkpoint、每轮预测和诊断路径/SHA见上述服务器运行条目与原报告。
- 同一validation单帧→新机制：AUC 0.954205563→0.954289944、L2 0.079604569→0.079562421、AP 0.960922682→0.960808102；准确稳定60.1874%→59.4964%，状态错误11.1254%→10.5114%。新机制仅比普通GRU多8个准确稳定窗口、少6帧状态错误，当前保留状态纠偏折中结论。
- R4分区来自official train的开发validation，原GazeSpot底座曾见过official train；不能将本轮数字与ViT-L旧VAT test 0.944/0.092/0.912直接比较。当前各模型开发表统一使用epoch 1整套指标，正式test主表留待冻结配置后的R5实测。
- 下一步复用epoch-1缓存分解定位/状态贡献并汇总门控诊断，按既定条件补train拟合的严格单调状态校准，再决定训练调整。首轮不需重跑，test配置继续未冻结；本次未启动服务器作业或新增模型评测。

## 官方 train/test 新协议与本地入口（2026-11-18）

- 用户明确指定取消后续validation，只用官方train/test，每epoch完整test并选最佳一轮；用户确认AUC优先，并列L2更低、AP更高，再取较早epoch。新计划 `cscwd/r4_official_train_test_plan.md`，协议 `r4_official_train_test_v1`，SHA256 `b33216abf59f95d3500ec7efc5b8cd99eb1b2581377cd0b19489ca7c68785a99`。旧 `r4_compact_v2` 与validation机器报告保持原样。
- 新增官方数据/评测/选模模块、test特征缓存、训练/比较/顺序入口及测试，13个新代码文件全部位于 `cscwd/code/`。原 `gazelle` 及已验收模型/训练/指标源码未修改；新运行复用完整train特征与train/test原预测，新增参数从头初始化。
- 本地记录 `cscwd/results/r4_official_local_checks.json/.md`，JSON SHA256 `7d898ed7ed905578964b9a482fd3043f65dfecdefd4631c090ba7c140e295972`，Markdown SHA256 `10939cd9af1bc3bc861333753fd1e837d33c7946e4d23247c13dda0f3c4dd525`。36项CPU合成检查通过（数据5、特征6、协议5、训练6、比较8、后台6）；12文件Python3.10语法、shell/命令块、8项既有小报告来源兼容核对通过。
- 合成数据实际走通两模型五轮、每轮完整test、test无反传、AUC整轮选择、checkpoint重载、全部候选/来源/诊断核对与比较CLI；后台临时替身验证GPU1继承、缓存复用与四阶段失败停止。不保存合成指标为VAT性能结果。
- 当前没有新增真实VAT/CUDA运行或服务器大文件；完整服务器作业由用户按实现计划§11.1手动启动，物理GPU1。后续真实大文件路径及SHA由脚本记录，小报告使用独立 `r4_official_*` 名称。论文与报告明确test参与epoch选择，旧validation结果不更名为test。

## 官方 train/test 服务器只读检查（2026-11-18）

- 实际SSH连接 `fb@3090.lab`，服务器仓库 `/home/fb/src/paper/gazelleV1`，检查时提交 `f1d41e4`。仅查询文件、JSON/index SHA、NPY header/大小与全部RGB路径，没有运行模型或缓存作业。记录 `cscwd/results/r4_official_server_preflight.json/.md`，JSON SHA256 `10fd4e044d59c2e7b3eacda39dc533f747072e28ceeca15fb222917d93267444`，Markdown SHA256 `537a252fe8e71137e1d230147cb5bab904277603db9ca3d99446b3253452b3dd`。
- 完整官方train为1,033tracks/132,563人物帧/58,507图像，test为298/31,978/13,127，全部71,634图像路径存在且分区不交叉。R1清单仅逻辑拆分train，完整train特征与train/test原预测均存在，索引/文件头等10项检查通过；没有重算大型NPY的完整SHA。
- 新test视觉特征 `r4_official_test_native_last_fp16/` 的目录/index/NPY及 `r4_official_test_features.json` 均不存在，需由新wrapper首先缓存；test原预测不能替代它。服务器尚未pull `run_official_round.sh`，下一步由用户先pull再按§11.1手动运行GPU1唯一入口。

## Official VAT test frozen features

- UTC: `2026-11-17T17:19:08.320446+00:00`; commit: `0de03307ea01bb53b6e51b808d037de52f067021`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/cache_official_test_features.py --data-path /newhome/fb/dataset/videoattentiontarget --prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --train-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt --output-dir /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16 --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_test_features.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_test_features.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md --batch-size 8 --num-workers 4 --device cuda:0`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_test_features.json`; SHA256: `b606681223edcc6d6735d4a1b815f0d23863f9485266b6e04a63349be7ed3524`.
- Cache index: `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16/index.json`; SHA256: `99eaa78fae7611fe62e61a68ae08d1f1dd97f0c87d337300c5f6aa7bed038cd3`.
- Server artifact: `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16/scene_features.npy`; SHA256: `692fe34fb05d556066b718ee01399e3ba37734bff5ddc1d6d98b1f6e3d29e780`.

## R4 official train/test: ordinary_gru

- UTC: `2026-11-17T19:42:34.640686+00:00`; commit: `0de03307ea01bb53b6e51b808d037de52f067021`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/train_official_joint.py --model-id ordinary_gru --train-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --train-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --test-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16 --test-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_test_features.json --train-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --test-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_official_train_test_plan.md --epochs 5 --learning-rate 0.0001 --lambda-state 0.1 --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1 --device cuda:0 --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_ordinary_gru.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_ordinary_gru.md`.
- Protocol: `r4_official_train_test_v1`; plan SHA256: `b33216abf59f95d3500ec7efc5b8cd99eb1b2581377cd0b19489ca7c68785a99`.
- Test used for epoch selection: `true`; independent held-out estimate: `false`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_ordinary_gru.json`; SHA256: `d075f5309dc6aba39e8c11b44b480c983c3cf3f461c7e1a4f407aa8ba81c3494`.
- Last checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/last.pt`; SHA256: `b1511f1c7bdc9d136956de4f4a42648b7a0ea4ed714b6a18e53685aacb58c96d`.
- Selected epoch: `5`; checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/selected.pt`; SHA256: `b1511f1c7bdc9d136956de4f4a42648b7a0ea4ed714b6a18e53685aacb58c96d`.
- Single-frame test details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/baseline_details.json`; SHA256: `2e1f25a338bcf8a530a0dd169f071563747c425b3ef86c690cfd6e14e4d35798`.
- Epoch 1: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/epoch_001.pt` SHA256 `03f479034be3752390fb65d9bc1c30e36b6cbc81411ca8ce48431b88ed1d9065`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/test_epoch_001` index SHA256 `a23d7b5080d4699b5e2e4058049cb1121f9e3bfafda2e053a05cc1e758667c1f`.
- Epoch 2: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/epoch_002.pt` SHA256 `9d85fa3ca1a4583e732da3f25de7bba73e2f1709cf2a341ca4cd2b16df537810`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/test_epoch_002` index SHA256 `7967d5ea787b021f9836917728a42af303a4770d8a2520f20b8e33fab926d898`.
- Epoch 3: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/epoch_003.pt` SHA256 `b1186bcb78755c9a8b77ba0320f76739e326862e9fe6a7b47a15b0eff75b9c8c`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/test_epoch_003` index SHA256 `d2aa797ab4d3907e2340d3ceb5a5d6d43874ff281d24033776e4506bf5963794`.
- Epoch 4: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/epoch_004.pt` SHA256 `42351479521b76267692c46cc5347944300c670efec435e7ffd5d510a2376c81`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/test_epoch_004` index SHA256 `a661e7a7200f0d762f289fbd5cb8d6077f293eaf5b8c55ee5677f8ac08abab58`.
- Epoch 5: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/epoch_005.pt` SHA256 `b1511f1c7bdc9d136956de4f4a42648b7a0ea4ed714b6a18e53685aacb58c96d`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/test_epoch_005` index SHA256 `919f900b107c4a8970233523322f9f70f3e35a30408991b3a9cef311da1cfde2`.

## R4 official train/test: history_validity

- UTC: `2026-11-17T22:56:54.714404+00:00`; commit: `0de03307ea01bb53b6e51b808d037de52f067021`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/train_official_joint.py --model-id history_validity --train-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --train-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --test-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16 --test-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_test_features.json --train-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --test-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_official_train_test_plan.md --epochs 5 --learning-rate 0.0001 --lambda-state 0.1 --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1 --device cuda:0 --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_history_validity.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_history_validity.md`.
- Protocol: `r4_official_train_test_v1`; plan SHA256: `b33216abf59f95d3500ec7efc5b8cd99eb1b2581377cd0b19489ca7c68785a99`.
- Test used for epoch selection: `true`; independent held-out estimate: `false`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_history_validity.json`; SHA256: `ff5c5b74615d03977c44412e07d81f1b8ae3318c1d274c2f9175c5c6e3399635`.
- Last checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/last.pt`; SHA256: `506795906103f048de8fe33d8865aad60ed01a5acc267490c88f5ef51946dbfc`.
- Selected epoch: `5`; checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/selected.pt`; SHA256: `506795906103f048de8fe33d8865aad60ed01a5acc267490c88f5ef51946dbfc`.
- Single-frame test details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/baseline_details.json`; SHA256: `2e1f25a338bcf8a530a0dd169f071563747c425b3ef86c690cfd6e14e4d35798`.
- Epoch 1: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/epoch_001.pt` SHA256 `ee6bea9673ae40e320064b2a88e726dbcdcfbc26e2c3fc87bd7d3794b1c037d0`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_001` index SHA256 `a38a99723de61381c9b2ee2dde1186c35e9ca7557034b21ac20e91a8b72a9bfd`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_001/history_diagnostics`; index SHA256: `8d42b6fffffecb0bf1ddd67b0797fdc6c2853e3f35f4bf8793d0c75cfff61173`.
- Epoch 2: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/epoch_002.pt` SHA256 `3c99d22c204a3ff7607f742ec0ef91ef866b0c0b567c2086aeb5f879901242a2`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_002` index SHA256 `5f0640bdc5572aa2a870834fb08decd08d1d7c8b506d1fbd7dba835f1ac5669f`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_002/history_diagnostics`; index SHA256: `198d0d5ac3355f1c6784005d64e82b81678e7f129ec534e2d330004a3c30157e`.
- Epoch 3: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/epoch_003.pt` SHA256 `4819c6febb31fa38da156cc8b9a697fb23d569b0c13873bac7941e99870a9ca2`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_003` index SHA256 `f95c8be43c1385db190a94e1f4d28202314e200e43c0366a449b1529ebb80c6d`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_003/history_diagnostics`; index SHA256: `a77f0f9e923d576c686089874c3e3fb33427f2897736dd7ba9b4747952ad2398`.
- Epoch 4: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/epoch_004.pt` SHA256 `90905bc80248c46f765d4cfe0ce29981d4fc66803bbb8847e286562bfaf272b6`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_004` index SHA256 `c5694c88738b0be0d40bc2f251a673ebe10a5871097929c3f042a35bd8d93ce5`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_004/history_diagnostics`; index SHA256: `044c47a0335df2e70c21f7d9b1f3d7a8bd595d56f61555f4d182ae793cc53f9f`.
- Epoch 5: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/epoch_005.pt` SHA256 `506795906103f048de8fe33d8865aad60ed01a5acc267490c88f5ef51946dbfc`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_005` index SHA256 `8baa8ef1790bd3be332ff1fd5ca8ec070d919cf0862bb7b1c2e3e1f3bbaf0a17`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/test_epoch_005/history_diagnostics`; index SHA256: `34345c24fb96864aa35190c3b42ebca5f0854df824b02ba7cef28a3fae2b1a22`.

## R4 official test-selected comparison

- UTC: `2026-11-17T22:57:34.110733+00:00`; commit: `0de03307ea01bb53b6e51b808d037de52f067021`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/compare_official_joint.py --run-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_ordinary_gru.json /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_history_validity.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_official_train_test_plan.md --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.md --selection-manifest /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_selection.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.json`; SHA256: `47c18572f60feec382d0e7b0623d6e5af160603c42eef0be44161801a4601317`.
- Summary: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.md`; SHA256: `9d16ae70558c659c0d2236139fdc8ad05636b447704e0e5c98cb6e018dee6cfd`.
- Test-selected checkpoint manifest: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_selection.json`; SHA256: `6684ce0d8bd8bc0082f5343e681c68febf187f938b4bd2ab2f0bd7059541ae29`.
- Full official test evaluated after every epoch and used for checkpoint selection; no independent held-out claim.

## 官方train/test首轮本地验收（2026-11-18）

- 结果提交 `aa0ab55` 在服务器已提交、GitHub当时尚无此提交；本地通过只读Git/SSH fetch取回，与文档提交 `c966239` 合并，保留双方历史。本次没有修改服务器文件或启动作业。作业源码为 `0de0330`，协议计划SHA `b33216abf59f95d3500ec7efc5b8cd99eb1b2581377cd0b19489ca7c68785a99`。
- 验收记录 `cscwd/results/r4_official_local_review.json/.md`，JSON SHA256 `669b03b8f1d5c4f6850217623f8c55a2972fdd8911fbe185a9eece625ce4cba4`，Markdown SHA256 `1ce8c0b8371cf7038ce28155f3d62e7f4bcb91b8a6b7f96b163557f693e0737d`。283项本地来源/源码/计划/覆盖/损失/比例/候选/差值/选模与manifest核对通过；完整NPY/PT/index和逐事件数据留服务器，未在本地重算。
- test特征缓存完成，13,127唯一图像，`[13127,768,32,32]` FP16，20,646,985,856 bytes；首批8图/32人物帧FP32 heatmap/InOut最大误差1.400709152e-6/1.788139344e-7、argmax变化0。大文件路径/SHA见原服务器条目。
- 两模型均完整train五轮，每轮训练132,563帧（画内88,919）、test31,978帧（定位20,655）；均按AUC规则选epoch 5，selected/last/epoch_005的SHA相同。普通GRU AUC/L2/AP **0.940540935 / 0.099481836 / 0.892941195**，新机制 **0.940466415 / 0.099243910 / 0.892777853**；同一test单帧 **0.940181658 / 0.098994832 / 0.886801040**。
- 单帧→新机制准确稳定52.7531%→50.3334%，gaze_shift92/158→88/158，状态错误6,241→6,181，状态翻转592→798；leave91/192→107/192。新机制比GRU的L2更低、leave多1个，但AUC/AP、准确稳定与状态错误不占优，当前核心机制额外优势尚未建立。
- 下一步用现有缓存交叉分解坐标/状态贡献，并汇总validity/refresh与历史对应诊断，依据结果决定一次针对性调整；本次没有新增训练、评测或修改选轮规则。论文当前版本整行使用各模型epoch 5，保留test参与epoch选择的协议说明。

## 缓存贡献与门控诊断本地入口（2026-11-18）

- 新计划 `cscwd/r4_cache_diagnosis_plan.md`，版本 `r4_cache_diagnosis_v1`，SHA256 `4ba64cb72e45f309eaaaa4d2344a7614f27e99758f3226844eb6a5d464893b34`。只诊断官方首轮选中的既有输出，官方训练协议、epoch 5与旧指标实现不改。
- 新增交叉输出核心、历史控制汇总、CPU入口及三套测试，六个源码文件SHA见 `cscwd/results/r4_cache_diagnosis_local_checks.json`；JSON SHA256 `a69de9bff562f0fd3a7c85705f08be10ee02abf8eaaa26795add7112440a1d0e`，Markdown SHA256 `b37291c1583256118b8f77f2cb98595cfc765f21641e418d4e5fa580ca46a324`。
- 23项CPU合成检查通过（交叉11、门控8、入口4），覆盖手算分解/失败组合、源行/人物/原帧对齐、缺口、门值/对应分组、浮点常量关联null、两端重现、原缓存不变以及禁止Torch导入的完整CLI；Python3.10语法和§12命令语法通过。
- 真实服务器诊断尚未运行，没有新增VAT指标或模型权重。下一步由用户按§12.1手动运行CPU脚本，产出独立 `r4_official_cache_diagnosis.json/.md`；逐轨迹/事件大明细留服务器新目录，实际路径/SHA由入口追加记录。

## Official test cache diagnosis

- UTC: `2026-11-18T05:13:06.448222+00:00`; commit: `ba746f8fc0f53b487a58519e0901afa379fd6eb8`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/diagnose_official_cache.py --comparison-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.json --single-frame-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_cache_diagnosis_plan.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_diagnosis_v1 --event-radius 3 --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_cache_diagnosis.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_cache_diagnosis.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_cache_diagnosis.json`; SHA256: `3b7c522bcb35b2750e470e2366e1672178b624b22203abe91e8c5d00953802ff`.
- Summary: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_cache_diagnosis.md`; SHA256: `e576e047659072e809c521842ef5e2c6789c131f654d5890b28a395426140ed2`.
- ordinary_gru cross details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_diagnosis_v1/ordinary_gru_cross_details.json`; SHA256: `cd1773005b1e2b5e114333b65d130f986285d609d2bd3ab4118bc664e98c00c9`.
- history_validity cross details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_diagnosis_v1/history_validity_cross_details.json`; SHA256: `216713cafe28207d00459f0e627931500fdd0371b58bfa1be52336c103ba1f73`.

## 缓存贡献与门控诊断本地验收（2026-11-18）

- 已拉取 `287d063`，诊断作业源码 `ba746f8`；169项本地来源/源码/计划/端点/窗口分解/状态错帧/门控分母核对通过。记录 `cscwd/results/r4_cache_diagnosis_local_review.json/.md`，JSON SHA256 `c0fe55f1461744b06a612e4cd0373eb38d9a046b0cb5e6290489105e1216c99b`，Markdown SHA256 `24f4680ea79fe6a4cd63e09113513d16ed47bef38c141f266079ce43e1e47c39`。原JSON不改，服务器NPY与逐事件明细未在本地重算。
- 新机制准确稳定窗口M00/M10/M01/M11=10,443/10,399/9,992/9,964，总−479；坐标−44、状态分数−451、交互+16；对称坐标−36、分数−443，状态输出项占净下降约92.48%。普通GRU为10,443/10,442/10,081/10,078，状态项也主导下降。此为输出替换数值分解，不当作重训结构因果贡献。
- 新机制GT画内引入1,155帧画外误判、纠正40帧，净增1,115错帧；画外净少1,175错帧，全体只净少60错帧。其失去631/获得152个准确稳定窗口，失去窗口中431个仅状态条件失败。
- 有历史31,680帧、无历史298帧；有历史validity均值0.671961、范围0.496976–0.896991，refresh均值0.610451。门控聚合未显示清晰的事件低门释放形态，相关与组均值不直接作因果证据。
- 下一步R4-D2为train拟合的一维严格单调输出状态校准，对单帧/GRU/新机制保持现有epoch 5/坐标/原递归历史，最终分数单独校准；核对AP排序不变并评价状态/时序取舍。D2细节在实现与运行前固定，本次未拟合参数、重训或启动服务器作业。

## R4-D2单调状态校准本地入口（2026-11-18）

- 计划 `cscwd/r4_state_calibration_plan.md`，版本 `r4_state_calibration_v1`，SHA256 `7dadbc8290e882cdc68a9c7fdca3cd749615ebc6f7b0d6b5c806f6b029fa89fe`。固定原epoch 5/坐标/递归历史，三个全局偏置只用完整train拟合；原训练协议和真实报告不改。
- 18项CPU合成检查通过（数学10、冻结前向4、入口4）；记录 `cscwd/results/r4_state_calibration_local_checks.json` / `cscwd/results/r4_state_calibration_local_checks.md`，SHA256分别为 `76e141db36f1b66811c888eb5a8ec9bd17a482c92f6d6c0c599ba1c929edb432` / `2604e2338886fe8a21b3faf6b03e62a658320ff0939801330af8f8b5061d617f`；六个新源码的SHA见JSON。
- 覆盖float32极值保序/ties/AP、有限目标、GT不进入预测、人物/缺口与chunk状态、原行序、来源/权重不变、端点重现、失败检查和完整CLI/provenance；Python3.10语法、服务器命令bash语法通过。
- 未运行真实VAT或服务器作业，没有新的真实校准指标/权重。下一步用户按实现计划§13.1用GPU 1手动运行，输出目录 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/`；实际NPY/参数/明细路径与SHA由入口追加。

## R4-D2 train-only monotone state calibration

- UTC: `2026-11-18T06:48:05.608063+00:00`; commit: `639e8b40a5bd479e65334b753fefba4d9a3a6416`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/calibrate_official_state.py --comparison-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.json --train-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --test-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --train-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --train-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_state_calibration_plan.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1 --device cuda:0 --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_calibration.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_calibration.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_state_calibration.json`; SHA256: `a94f5130d4baf7f1e0f499abd68d68c420f1073c495dcc7f67fc6b2b9f24eeb6`.
- Summary: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_state_calibration.md`; SHA256: `cb8051b18a7c6ad9e7eaa1470e5bfbef8522bab092e2b56f9f5858d38edae2cf`.
- single_frame calibration: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/single_frame/calibration.json`; SHA256: `1662dba8a00b6c657dd650b7252d6f51f799304a33be00098429c0b7756bde09`.
- single_frame calibrated_test_scores: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/single_frame/test_inout_calibrated.npy`; SHA256: `bb936fb33f0b59f7bfc8eb6224df218a47fecb2ca3f801824a6d84166da7dfc2`.
- single_frame details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/single_frame/test_details.json`; SHA256: `92248bcda069e4037b799dba175dce654697d6a585bed187bd97220f17432811`.
- ordinary_gru calibration: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/ordinary_gru/calibration.json`; SHA256: `4d4bccc2d955bd30e89216933bde25f893793f5350a8dd503ac6e2ccc95172e6`.
- ordinary_gru calibrated_test_scores: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/ordinary_gru/test_inout_calibrated.npy`; SHA256: `92a557b09a1655aacccec3a9ace85095124f18a71b66f7b9dd1ae36a6a744a2d`.
- ordinary_gru details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/ordinary_gru/test_details.json`; SHA256: `ab7b54635edab4b078d585b0d5d339486683c17afaa1427fc5e558023c2a78aa`.
- ordinary_gru frozen train scores: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/ordinary_gru/train_scores`; index SHA256: `0b9adf03ee1d716eee37e3d33c5d93e44127f405648b4697368baa44c8912883`; inout SHA256: `680a4484a8e995ece743e972bfa87d70daf47c2744651480314ca2539ddf1fcc`.
- history_validity calibration: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/history_validity/calibration.json`; SHA256: `7b40d3e5309210d2a483db29f638e1035d0dd19e9c0a604223d84b8bd1d33465`.
- history_validity calibrated_test_scores: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/history_validity/test_inout_calibrated.npy`; SHA256: `ed6ff15f98ba2ef85bf0e61c18e1c8a1509cd549121f58d9a598381194492bce`.
- history_validity details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/history_validity/test_details.json`; SHA256: `859d5b266d8f65fb5dbbebaeb6cc16656bb7b79c755e1ccabc6d172edd8e5522`.
- history_validity frozen train scores: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/history_validity/train_scores`; index SHA256: `0684db5a63077d5751e88e65c5375fd88d8cd512d3227cc2550816d0ee03e19b`; inout SHA256: `9da40c02d2c5699f7524ccf642cfe549641ffc73b48812a778d2eb10013dc3da`.

## R4-D2真实结果本地验收（2026-11-18）

- 已pull结果提交 `75f305c`，作业源码 `639e8b4`。393项小报告来源/源码/覆盖/端点/比例/配对算式核对通过；未访问服务器或下载/重算NPY、PT、index与完整逐事件数据。
- 验收记录 `cscwd/results/r4_state_calibration_local_review.json`；SHA256 `f861d111eaba366b5cf2737ea201749cb33550f43b2d08f48efd73cc6dc8273e`。
- 验收记录 `cscwd/results/r4_state_calibration_local_review.md`；SHA256 `ceb1afd435d4d3dc504728fdc64932abed213e5c64fc4a538d50174b3056e2e2`。
- train偏置单帧−0.423750、GRU+0.041511、新机制+0.090667；AUC/L2/AP保持。新机制准确稳定9,964→10,064、错帧6,181→6,055、错误翻转798→758；只补回相对原始单帧479窗口损失的20.88%，gaze_shift仍88/158。校准后比校准单帧/GRU少96/41个稳定窗口，尚无机制额外优势。
- 下一步D3为状态时序训练监督：稳定段一致性与状态变化后的响应监督，保留逐帧MSE/BCE、冻结GazeSpot、GRU/新机制同预算；先固定细节再实现。不继续test阈值扫描。本次只验收及更新计划，没有新增模型训练、权重或服务器任务。

## R4-D3状态时序监督本地实现（2026-11-18）

- 规格 `cscwd/r4_state_supervision_plan.md`，版本 `r4_state_supervision_v1`，SHA256 `14567f4bf4831fd069dced259745981e33d95a57bc95566b516b9bfc80418f6a`。固定逐帧损失+稳定同状态L1+真实翻转当前及随后两帧BCE，lambda_state/lambda_time均0.1，窗口3帧；GT仅loss，监督carry与预测历史分离。
- 29项CPU合成检查通过（loss/epoch12、训练/完整比较CLI6、比较7、wrapper4）；记录 `cscwd/results/r4_state_supervision_local_checks.json` / `cscwd/results/r4_state_supervision_local_checks.md`，SHA256分别 `89ba874e19325bfe20832ca679feb674a727c8b54573decff297911170847518` / `e36bcb2764348febd6cb76251151c3570c5d96aefb58750cd76ddf6a074eda14`。八个新源码SHA见JSON，七个新Python的3.10语法及wrapper/服务器命令bash语法通过。
- 实际跑通临时合成双模型各五轮、完整test/比较、重载、监督计数和test标签不影响五轮权重；没有真实VAT训练指标、没有服务器任务。旧源码/原训练计划/结果保持。
- 下一步用户手动运行实现计划§14.1，物理GPU 1，复用已有缓存。新实验根目录 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/`；实际checkpoint/预测/诊断路径与SHA由任务回写，NPY/PT不进Git。

## R4-D3 state supervised train/test: ordinary_gru

- UTC: `2026-11-18T13:40:41.216436+00:00`; commit: `524999f48bcf438904c92ee1988c27b77b0bf3b4`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/train_state_supervised.py --model-id ordinary_gru --train-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --train-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --test-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16 --test-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_test_features.json --train-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --test-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_state_supervision_plan.md --epochs 5 --learning-rate 0.0001 --lambda-state 0.1 --lambda-time 0.1 --response-frames 3 --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1 --device cuda:0 --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_ordinary_gru.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_ordinary_gru.md`.
- Protocol: `r4_official_train_test_v1`; supervision: `r4_state_supervision_v1`; plan SHA256: `14567f4bf4831fd069dced259745981e33d95a57bc95566b516b9bfc80418f6a`.
- Test used for epoch selection: `true`; independent held-out estimate: `false`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_ordinary_gru.json`; SHA256: `a074d32e2829516ee3b1088bd6e91d05bdd1849b8ee86fe36a00519fd979f5b1`.
- Last checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/last.pt`; SHA256: `7981a4044f922e1b406adfcf2c1200522982db88d48d9e468cf47d9b249096ab`.
- Selected epoch: `2`; checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/selected.pt`; SHA256: `aa07f010532bb34aa2aa298185b37a00aa96ad7221ffdae462d5906c50eb2af0`.
- Single-frame test details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/baseline_details.json`; SHA256: `2e1f25a338bcf8a530a0dd169f071563747c425b3ef86c690cfd6e14e4d35798`.
- Epoch 1: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/epoch_001.pt` SHA256 `72ff30b0e05db19d68ec97fcb3e655f375c5dc33a393e793e6d41eb4ec153c35`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/test_epoch_001` index SHA256 `e3c167097c31110b4ea783303b4dcc27b4fbc357b29bf74a2548f868e3a02d88`.
- Epoch 2: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/epoch_002.pt` SHA256 `aa07f010532bb34aa2aa298185b37a00aa96ad7221ffdae462d5906c50eb2af0`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/test_epoch_002` index SHA256 `be3ef9dc70e49286d5ad4833913245474552f3f919c9573338b84fa0fcc37f73`.
- Epoch 3: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/epoch_003.pt` SHA256 `736785c2691ce3846ef7574b27e2970fa913cb8f20c7dad4e1e0f2b199e016c0`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/test_epoch_003` index SHA256 `3c8c22cfa45738b193e86d8f8a3e59989d0df736c841b033d7a2703c8baac92b`.
- Epoch 4: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/epoch_004.pt` SHA256 `9587b0fc3cc00437c5343acb69b1b00f521ecc87bb170039978010fc625dab00`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/test_epoch_004` index SHA256 `b8faf78e244f55a9a69dafacb280ec3d3ab87c570edacedaed158a05ff62a8d4`.
- Epoch 5: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/epoch_005.pt` SHA256 `7981a4044f922e1b406adfcf2c1200522982db88d48d9e468cf47d9b249096ab`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/ordinary_gru/test_epoch_005` index SHA256 `e627ba09a5dab1546be84c5b7de1dbc15f24d88a7dbb98225c3d6e0e29ea095a`.

## R4-D3 state supervised train/test: history_validity

- UTC: `2026-11-18T17:08:06.175875+00:00`; commit: `524999f48bcf438904c92ee1988c27b77b0bf3b4`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/train_state_supervised.py --model-id history_validity --train-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --train-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --test-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16 --test-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_test_features.json --train-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --test-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_state_supervision_plan.md --epochs 5 --learning-rate 0.0001 --lambda-state 0.1 --lambda-time 0.1 --response-frames 3 --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1 --device cuda:0 --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_history_validity.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_history_validity.md`.
- Protocol: `r4_official_train_test_v1`; supervision: `r4_state_supervision_v1`; plan SHA256: `14567f4bf4831fd069dced259745981e33d95a57bc95566b516b9bfc80418f6a`.
- Test used for epoch selection: `true`; independent held-out estimate: `false`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_history_validity.json`; SHA256: `de639aefb8ab56f12103ef69c70b21cfd4ca5b8f254415bc782a8a68d95e25d2`.
- Last checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/last.pt`; SHA256: `7c7054a3d26a03a7b52c72e42c5b51189f98ac7898eb6f0e092671e176eb8a8d`.
- Selected epoch: `3`; checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/selected.pt`; SHA256: `51aa2767ff1e0a16df63996786d616f5f56fd289e418d8430b2fc52fa7a7eaf0`.
- Single-frame test details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/baseline_details.json`; SHA256: `2e1f25a338bcf8a530a0dd169f071563747c425b3ef86c690cfd6e14e4d35798`.
- Epoch 1: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/epoch_001.pt` SHA256 `bc352c2e137770491903459a1e3064387d4e07080fcea99dfcf0dfa4597163be`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_001` index SHA256 `ce879975bf6743d19b0c2045d9d3510f71b127327d4e9933403997e401478d52`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_001/history_diagnostics`; index SHA256: `c48daf3bdbd4cced90ba65ad880b78bbb4c2dd67012d2df6a51e05b65788d2b8`.
- Epoch 2: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/epoch_002.pt` SHA256 `0da867b345c49978e92cbcdd362cacff0193fbbe323aee5c98ada2253a9128ca`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_002` index SHA256 `bfef9683a9ad6ccf96cf4933093e25520e05755a8cac7146d75cf97013410061`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_002/history_diagnostics`; index SHA256: `8636dd75a3d2d14d61bc8316c2d812779cc189deadb154bdf7ca5c06983dbec2`.
- Epoch 3: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/epoch_003.pt` SHA256 `51aa2767ff1e0a16df63996786d616f5f56fd289e418d8430b2fc52fa7a7eaf0`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_003` index SHA256 `05c8d86a4ad83128d00597b41b7c298843eed51a4962196343bf1fc4facb15e3`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_003/history_diagnostics`; index SHA256: `4ed413f7a467c7a63eddcd2cd19177d920a19d3b5d97eba89eae78a766a18adf`.
- Epoch 4: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/epoch_004.pt` SHA256 `6dd9d6cbc6c7dce360c4f676242c8f4bf3dec7e299df4360263d81a5f3b008e9`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_004` index SHA256 `f0f3f7770baa7727671fef3ec8e5a20de2d89c6d1a8a9a5ee004ab6f935aa218`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_004/history_diagnostics`; index SHA256: `f897c9615332b4512791ab5774b1044c1fa9fa599efb9768c9c9744061113d2b`.
- Epoch 5: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/epoch_005.pt` SHA256 `7c7054a3d26a03a7b52c72e42c5b51189f98ac7898eb6f0e092671e176eb8a8d`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_005` index SHA256 `910715e6ccaf178c81c35489ffceea706438e3c6eb5fa4dbab7985c8531fef10`.
  - History diagnostics: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_supervision_v1/history_validity/test_epoch_005/history_diagnostics`; index SHA256: `7773482d4b3881b1737ef4ab1f8397ebcb94fc6807248fcb8bfde4e1bd22a168`.

## R4-D3 state-supervision comparison

- UTC: `2026-11-18T17:09:09.501164+00:00`; commit: `524999f48bcf438904c92ee1988c27b77b0bf3b4`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/compare_state_supervised.py --run-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_ordinary_gru.json /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_history_validity.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_state_supervision_plan.md --previous-comparison /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.json --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_comparison.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_comparison.md --selection-manifest /home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_selection.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Previous comparison: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.json`; SHA256: `47c18572f60feec382d0e7b0623d6e5af160603c42eef0be44161801a4601317`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_comparison.json`; SHA256: `7f6234b729fb2c53824a6023dc272fdad92473da343790c27a522c46816284e9`.
- Summary: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_comparison.md`; SHA256: `48666e66f120d717f700dedbe9f925b0d61d16bcb6381343f7e11c89e27ad7a0`.
- Selection manifest: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_state_supervised_selection.json`; SHA256: `04590b0d3c6b675b31c72da3f42be7c6edf65e7139cf213f9b174a0c685b49cf`.
- Full official test evaluated each epoch and used for selection; original outputs, no D2 calibration.

## R4-D3真实结果本地验收（2026-11-19）

- 已pull结果提交 `db239cf`，作业源码 `524999f`。240项小报告来源/源码/覆盖/损失算式/选轮/模板/provenance核对通过；服务器NPY/PT/index/完整事件未下载或本地重算。
- 验收记录 `cscwd/results/r4_state_supervision_local_review.json`；SHA256 `8083396698b2f8c4b7bc999302de49dc8c0e081b9e5ea40d6ee463a438355597`。
- 验收记录 `cscwd/results/r4_state_supervision_local_review.md`；SHA256 `e036faa8df6a829659212834ca2aaebfd5018742a66d479b932f4b7fb2ddcbf9`。
- 两模型完整五轮，train每轮132,563帧、stable pairs129,666、transition frames5,358。按原规则选GRU epoch 2/HV epoch 3；HV AUC/L2/AP为0.940481/0.099299/0.888879，准确稳定51.2225%、错帧6,168、翻转717、gaze89/158，核心机制优势尚未建立。
- 同epoch分析：HV3稳定10,232→10,140、翻转660→717；GRU2稳定10,256→10,203。两模型十个同轮AP均下降/L2均增加，稳定只有epoch4增加；不把选轮提前后的整行改善直接归功新loss，也不临时换选epoch4。
- 下一步D4只做一个当前帧无历史联合修正头，原逐帧目标/五轮预算，判定原AP收益是否依赖历史；先固定约定再实现。不继续D3权重扫描/校准。本次仅分析和计划，没有新的模型训练或服务器任务。

## 时序方向与底座选择复盘（2026-11-19）

- 分析文件 `cscwd/idea_feasibility_review.md`；SHA256 `4231c9e08d72ddb50d3e5c171251b816637ff21048a25bf09bbc03e180d5e7c9`。引用已有T5/R4/D1–D3真实报告、官方Gaze-LLE与本地GazeSpot源稿；没有新实验数字，理想结果只写比较关系。
- 官方Gazelle/DINOv2论文数值与本地GazeSpot/DINOv3结果分开；GazeSpot-L源稿记录不是CSCWD同协议的新复现。底座规模与模块增量分别评价。
- 本次仅讨论/文档更新，D4尚未实现/运行，没有切换底座、训练或评测。当前候选选择/效用监督为后续假设，原代码/结果/预声明训练计划保持。

## R4-D4当前帧无历史对照本地实现（2026-11-19）

- Control: `r4_current_frame_v1`; data/selection: `r4_official_train_test_v1`. Plan: `cscwd/r4_current_frame_plan.md`; SHA256: `09215386d2257bb6b75c9650eb4e62a2f648f36cb294f3a85b86fb4a5b45fb20`.
- Local checks: `cscwd/results/r4_current_frame_local_checks.json`; SHA256: `0e6205067d9805534c99f815832bd71fdd240c6c83402e6aef7d8a76104d4f9f`.
- Local summary: `cscwd/results/r4_current_frame_local_checks.md`; SHA256: `0eebaa14e21d336cab852ba4a8dfdc7b5601c0ecf114c6712a47153b8412b25d`.
- 23 CPU synthetic tests passed (model7, training/actual comparison CLI6, comparison7, wrapper3); final root rerun: 7 in0.145s +16 in14.531s. Seven Python3.10 AST checks, wrapper/two server command blocks bash syntax passed.
- Fresh state every frame; original per-frame MSE/BCE, five epochs/seed0. Nominal347,522 parameters;74,112 weight elements multiply structurally zero history. Effective use is not claimed identical.
- Server D4 job not started; no new VAT performance numbers. Manual GPU1 command: implementation_plan.md §15.1. New large artifacts will stay at `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/`; actual artifact hashes will be recorded by server scripts.

## R4-D4 current-frame train/test: current_frame

- UTC: `2026-11-19T10:23:43.112284+00:00`; commit: `4058054fc76d43bda7d25877138cbd52a4a84b33`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/train_current_frame.py --train-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --train-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --test-feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16 --test-feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_test_features.json --train-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --test-prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/t2_single_frame_cache.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_current_frame_plan.md --epochs 5 --learning-rate 0.0001 --lambda-state 0.1 --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1 --device cuda:0 --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Protocol: `r4_official_train_test_v1`; control: `r4_current_frame_v1`; plan SHA256: `09215386d2257bb6b75c9650eb4e62a2f648f36cb294f3a85b86fb4a5b45fb20`.
- Test used for epoch selection: `true`; independent held-out estimate: `false`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame.json`; SHA256: `6da801593e4c64f6e9f3399074cbe45ec0d54432ed8d7f601614dc072fa8df88`.
- Last checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/last.pt`; SHA256: `93a849123ec80a983143b5cb5c25f400dd00d477a74366cf174feb54c7d9599e`.
- Selected epoch: `5`; checkpoint: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/selected.pt`; SHA256: `93a849123ec80a983143b5cb5c25f400dd00d477a74366cf174feb54c7d9599e`.
- Single-frame test details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/baseline_details.json`; SHA256: `2e1f25a338bcf8a530a0dd169f071563747c425b3ef86c690cfd6e14e4d35798`.
- Epoch 1: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/epoch_001.pt` SHA256 `8313bdc3f9e3c77cea51a77827ba88900b58e3762f7683865fa37a6b50967f59`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/test_epoch_001` index SHA256 `e162ab89135330871eff727b2af69cd612bbca7f256a145e15838ba1d1d63e78`.
- Epoch 2: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/epoch_002.pt` SHA256 `d137800117464e8767d7ceb2a7c03cd8aa783961fae027448be75cc900a8e76e`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/test_epoch_002` index SHA256 `4cb91e9a87aed6e668bc0df34ef4a59a37918f1a66b8b1bc464119b79b4570b7`.
- Epoch 3: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/epoch_003.pt` SHA256 `a7767d0f87ee68a901547bbdb55185d09f44d0f2b799b37d31f18a2f397fa171`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/test_epoch_003` index SHA256 `cb9f9c463b99e59c4b45d41a0de285641ada25e18e6e1e04bf272a4ef229cdfc`.
- Epoch 4: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/epoch_004.pt` SHA256 `01c8d881ce13abfb6389fd6e55b26e7d40c73aef1e1d31dadc6a88fe9f85395c`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/test_epoch_004` index SHA256 `9512530446cd19874636824365633b1f27b0802f44ac826b192d950fc80a6109`.
- Epoch 5: checkpoint `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/epoch_005.pt` SHA256 `93a849123ec80a983143b5cb5c25f400dd00d477a74366cf174feb54c7d9599e`; test `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_current_frame_v1/current_frame/test_epoch_005` index SHA256 `0daabe6f6dae27e043fd35edebbb322f21e7fc17f1275321f8f5975fd58351d1`.

## R4-D4 current-frame comparison

- UTC: `2026-11-19T10:24:25.481313+00:00`; commit: `4058054fc76d43bda7d25877138cbd52a4a84b33`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/compare_current_frame.py --run-report /home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_current_frame_plan.md --reference-comparison /home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.json --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame_comparison.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame_comparison.md --selection-manifest /home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame_selection.json --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Reference comparison: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.json`; SHA256: `47c18572f60feec382d0e7b0623d6e5af160603c42eef0be44161801a4601317`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame_comparison.json`; SHA256: `2a75e0a6211a53ecaf1579c2d3e53947871679f28c67c75e787d9f7acb7fce49`.
- Summary: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame_comparison.md`; SHA256: `94ae3ce143ec70b16dd0074c8309e5e617a2649259ba48167a16062efae3609b`.
- Selection manifest: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_current_frame_selection.json`; SHA256: `2a55f5a6bca3774ac453bc9f560a85addd6221db44433cfb6ac9dc87fab21f19`.

## R4-D4真实结果本地验收（2026-11-19）

- Result commit: `6eeb594`; job code: `4058054fc76d43bda7d25877138cbd52a4a84b33`. 235 actual named small-report checks passed; server NPY/PT/index/full events were not downloaded or recomputed locally.
- Review: `cscwd/results/r4_current_frame_local_review.json`; SHA256: `89b611902348e3b3fa4748274281d00f0451fc427e7dcfd2cf59cf25c0c2cf52`.
- Review summary: `cscwd/results/r4_current_frame_local_review.md`; SHA256: `dd26114914caf2909c83cad7dcc6b715ede519470fe739ebb88aedd1f3e1e112`.
- Selected epoch5: AUC0.9404074807484596/L2 0.09936236130604252/AP0.8930672866412298; accurate stable10050/19796, gaze87/158, leave107/192.
- AP/leave benefits are reproduced without history; ordinary GRU retains higher AUC at all five matched epochs. Next: train-only candidate utility analysis, not further expansion of the original GRU/HV.

## R4-D5训练集候选效用本地实现（2026-11-19）

- Utility version: `r4_history_utility_v1`; plan: `cscwd/r4_history_utility_plan.md`; SHA256: `3bad42a8efbe7925198c1e9f40cd893e25053fc1a4830e33debef81a78cc8006`.
- Local checks: `cscwd/results/r4_history_utility_local_checks.json`; SHA256: `a560eee0aa3c735ad2d6baa8cbba1cea4fec55687759af7b708c8642ccdbe884`.
- Local summary: `cscwd/results/r4_history_utility_local_checks.md`; SHA256: `6b2734e83b47fdfea12321eda0626860d7ee4e1940eebd6d1288641ea4521289`.
- 28 synthetic CPU tests passed: candidate12/0.084s, scoring9/0.693s, final CLI7/1.866s; six Python3.10 AST checks and two server command blocks bash syntax passed.
- Fixed five candidates, original InOut, one-hop raw visual transport and independent EMA; complete candidate generation precedes GT scoring. Oracle never feeds memory.
- Real VAT D5 analysis/server job not run. Manual GPU1 command in implementation_plan.md §16.1; future artifacts at `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/`. No new checkpoint; actual large-artifact paths/SHA will be recorded by server script.

## R4-D5 train-only history utility

- UTC: `2026-11-19T12:50:12.595776+00:00`; commit: `8f1bf4f0d3b59a5f6f80765e93898a717da7703e`.
- Command: `/home/fb/anaconda3/envs/py310/bin/python cscwd/code/scripts/analyze_history_utility.py --feature-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16 --feature-report /home/fb/src/paper/gazelleV1/cscwd/results/r2_feature_cache.json --prediction-report /home/fb/src/paper/gazelleV1/cscwd/results/r1_train_single_frame_cache.json --plan /home/fb/src/paper/gazelleV1/cscwd/r4_history_utility_plan.md --output-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1 --device cuda:0 --report /home/fb/src/paper/gazelleV1/cscwd/results/r4_history_utility.json --summary /home/fb/src/paper/gazelleV1/cscwd/results/r4_history_utility.md --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md`.
- Plan: `/home/fb/src/paper/gazelleV1/cscwd/r4_history_utility_plan.md`; SHA256: `3bad42a8efbe7925198c1e9f40cd893e25053fc1a4830e33debef81a78cc8006`.
- Report: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_history_utility.json`; SHA256: `dc0ec7bb6b976a3ee22b9f4755abc03d267008c713b31c8a9489f95eb73cff4f`.
- Summary: `/home/fb/src/paper/gazelleV1/cscwd/results/r4_history_utility.md`; SHA256: `6ddee813877ab5069072e1b8b52df5616f099598108fc3e87aa7a37873b106fb`.
- Complete official train only; no test access, training or checkpoint; original InOut fixed.
- GT/oracle scoring follows candidate generation and never feeds memory.
- candidate_xy: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/candidate_xy.npy`; SHA256: `e5488770f50753968544ab7fa81e5ff1043b70229479bf172026df5468ea1e3e`.
- history_available: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/history_available.npy`; SHA256: `eeff757cad5e8dabb6905c39d1918477329ee7e83d4e8371aec393cec605a168`.
- errors: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/errors.npy`; SHA256: `1127c1115d5e3002113426f88f8d2850e188a248bea7630ef9ca7fdb69ebd871`.
- group_codes: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/group_codes.npy`; SHA256: `46e3c2f8569d482857169b9cedd9513091567b44bcd9b21911e51b80d9d00764`.
- oracle_choice: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/oracle_choice.npy`; SHA256: `6a22df395dcc4adf9657a23499db3420fd1550f56456a6ec54044d6791e3a8ec`.
- details: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/utility_details.json`; SHA256: `682f089dcf989403bb906a34ef7d0a05c30e16bdef9036d40c716245cb619426`.
- index: `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/index.json`; SHA256: `9e44e79f4da29493609b95348b7cf472948988eeb05b5670b5053efd341192c8`.

## R4-D5真实结果本地验收（2026-11-20）

- Result commit: `a6928e6`; job code: `8f1bf4f0d3b59a5f6f80765e93898a717da7703e`. 879 actual small-report/source/formula checks passed; server NPY/index/full events not downloaded or recomputed locally.
- Review: `cscwd/results/r4_history_utility_local_review.json`; SHA256: `53d6b6a1a422a616e99d6f954bfc292d84cd9b28107a9e6739c6e69db8bceea0`.
- Review summary: `cscwd/results/r4_history_utility_local_review.md`; SHA256: `78b9a088bd4b0081f628ea8b5b06915aaa0444c3bc7b0d551bfdffb0c9df12bb`.
- Train utility denominator88,270/current errors22,831; fixed-set GT reference rescues3,126 (2,972 with fixed raw InOut,154 blocked). Oracle L2 .08165971810370352 is not deployment performance.
- Visual transport rescue2,624/harm3,145; no-history current_grid accurate stable47,676→51,364 and gaze499→504/816. Next train selector anchored on grid with grid_EMA control.

## R4-D6成本监督选择门本地实现（2026-11-20）

- Version: `r4_utility_selector_v1`; plan: `cscwd/r4_utility_selector_plan.md`; SHA256: `59a2a083f8260533048680e1f3c7b8b70e93fa5fc9b2caa5f14e1d84aee88162`.
- Local checks: `cscwd/results/r4_utility_selector_local_checks.json`; SHA256: `5ad457c395b51e62ce1e09bf116ec1dc964c189562b13d37460f858400be168e`.
- Local summary: `cscwd/results/r4_utility_selector_local_checks.md`; SHA256: `d337dbdd427daf4b3270ad1e890e88bc3b5ab93ab54b876f428f6cbe12fbf519`.
- 38 synthetic CPU checks passed: core15/0.734s, bank7/0.724s, evaluation8/0.517s, actual full training CLI8/3.982s. Eight Python3.10 AST checks/two server command blocks passed.
- MLP296,129 parameters, six frozen tokens+17 scalars; train-only cost-weighted BCE. Hard grid/visual-history choice, raw InOut unchanged; no GT/oracle/gate-output memory feedback.
- Real VAT D6 training/evaluation/server job not run. GPU1 manual command: implementation_plan.md §17.1. New large banks/checkpoints/full outputs at `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_utility_selector_v1/`; server script will record actual paths/SHA.
