# 官方 train/test 服务器只读检查

日期：2026-11-18。实际通过 SSH 连接 `fb@3090.lab` 检查，服务器仓库为 `/home/fb/src/paper/gazelleV1`，检查时提交为 `f1d41e4`。仅查询文件/索引/NPY header及RGB路径，没有运行缓存、训练或评测作业。

官方 train/test 均已存在，无需重新划分：

| 官方分区 | tracks | 人物帧 | 图像 | 缺失RGB |
|---|---:|---:|---:|---:|
| train | 1,033 | 132,563 | 58,507 | 0 |
| test | 298 | 31,978 | 13,127 | 0 |

原始分区由 `/newhome/fb/dataset/videoattentiontarget/annotations/train/`、`annotations/test/` 指定；轨迹JSON位于 `cscwd_tracks/train_preprocessed_tracks.json` 和 `test_preprocessed_tracks.json`。原图共用 `images/<video>/<clip>/`。已逐个核对全部71,634个train/test图像路径存在，两个分区路径集合不交叉。

R1的train/validation只是完整官方train上的逻辑清单：98,362+34,201=132,563人物帧。原test一直独立存在；完整train预测和特征也一直保留，新入口读取完整index，不读取R1清单作为划分。

| 缓存 | 服务器实际状态 |
|---|---|
| 完整train预测 `r1_train_predictions/` | 存在，132,563帧，FP32 heatmap/inout/argmax |
| 完整train特征 `r2_native_last_fp16/` | 存在，shape `[58507,768,32,32]`，FP16，92,023,554,176 bytes |
| test原预测 `test/` | 存在，31,978帧，FP32 heatmap/inout/argmax |
| 新test视觉特征 `r4_official_test_native_last_fp16/` | **目录、index和scene_features.npy均不存在** |
| `cscwd/results/r4_official_test_features.json` | **不存在，需要先生成test视觉特征** |

以上缓存目录均位于 `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/`。既有预测/特征索引SHA与小报告一致，NPY header/文件大小吻合；本次没有重新读取或计算大型数组的完整SHA。ViT-B VAT checkpoint及冻结骨干权重文件均存在。10项既有数据与来源检查通过，明细见同名JSON。

test的原预测不能替代空间视觉特征。新特征预计为 `[13127,768,32,32]` FP16，脚本先缓存它再训练。检查时服务器尚未pull新实现，`run_official_round.sh`仍不存在。

下一步：服务器先 `git pull --ff-only`，再由用户手动执行实现计划§11.1的单条GPU1后台命令。入口会自动补test特征，再训练普通GRU/新机制各五轮并比较；无需另做划分或手动拆开缓存命令。
