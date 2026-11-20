# GazeSpot + ConvLSTM：结果来源

2026-11-20，基线代码已实现并完成本机合成检查，尚无本轮真实VAT缓存、训练或评测结果。

- 原始checkpoint：`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`。
- 服务器仓库：`/home/fb/src/paper/gazelleV1`；数据根：`/newhome/fb/dataset/videoattentiontarget`。
- 旧实验来源记录完整保存在 `cscwd_v0/results/RESULT_PROVENANCE.md`。旧源码/报告内的原路径和SHA不作批量改写，历史复现对应归档前提交 `a77c8be`。
- 新作业完成后在此记录实际命令、代码/数据/checkpoint SHA、结果报告和服务器大型工件路径。
- `code/scripts/train_convlstm.py` 在实际运行时自动追加输入checkpoint/轨迹、两split缓存及每轮checkpoint/selected/last的服务器路径和SHA；小报告位于 `results/<run_id>/`，大型缓存/checkpoint留服务器。首次训练开始时登记缓存，之后每轮追加checkpoint。
- 本机检查记录见 [LOCAL_CHECKS.md](LOCAL_CHECKS.md)。合成工件位于临时目录，未作为VAT结果登记。
