# W7-D：条件化消融收口

2026-11-25 14:31（Asia/Shanghai），用户同意缩减训练后停止scene-head及外层队列。执行前核对到scene-only GF已于11:23、VAT已于13:03完整完成，因此两组结果均保留。进程组640838的11个相关进程已退出，无新GPU任务、无删除工件；部分scene-head checkpoint不作正式结果。

表4采用三个已完成条件：

| Learned scoring inputs | GF AUC ↑ | GF Avg L2 ↓ | GF Min L2 ↓ | GF GT coverage ↑ | VAT AUC ↑ | VAT L2 ↓ | VAT AP ↑ | VAT GT coverage ↑ |
|---|---|---|---|---|---|---|---|---|
| scene_only | 0.956205 | 0.101614 | 0.043444 | 0.902740 | 0.940565 | 0.100693 | 0.886334 | 0.949455 |
| scene_geometry | 0.956018 | 0.101514 | 0.043611 | 0.913874 | 0.940438 | 0.100879 | 0.886571 | 0.952748 |
| full | 0.956236 | 0.101550 | 0.043420 | 0.945781 | 0.940033 | 0.100131 | 0.883183 | 0.963253 |

同一DINOv3-B多层基础、原GF decoder初始化、K50、seed0、GF support3＋decoder5、各自GF→VAT8（首轮固定Router），仅train/test。GF按Min L2选第5轮；VAT按rate-6 test L2选第1轮，最终rate-1逐帧test。GF为4,782查询，VAT为31,978查询（20,655画内）。scene-only只关闭learned scoring中的外观/几何，真实头框及head/escape优先规则与其他条件保持一致。

服务器CPU汇总已核对同配方、初始decoder、数据、权重和各自GF→VAT继承。scene-only另核对25个工件SHA、24份源码和原生DINOv3源码树、连续3/5/8轮history、test选模和backbone可训练参数为0。逐查询AUC/L2/AP重聚合差值均小于1e-12；本机9份报告及24份源码SHA一致。没有重新训练或执行模型forward。

full相对scene-only的GT点覆盖提高GF 4.30、VAT 1.38个百分点；最终定位指标非常接近，AUC/AP并非一致提高。可说明人物条件更集中地保留GT位置，同时保留真实的最终指标差异。scene-head取消后，不作两类条件完全独立的因果结论。

原工具仍输出四条件注册表，`summary.json`的`all_branches_complete=false`明确表示scene-head未完成；正文只列三个已完成条件。多人9个质量单元、四组同权重K100、两组真实图均已完成；多人计时仍为空，后续须独占目标GPU。当前不再等待训练。

结果经e68eb93回传：[汇总](final_ablation_seed0/summary.md)、[scene-only CPU核对](w7d_scene_only_server_checks.json)、[停止记录](w7d_ablation_scope_stop.json)。所有PT/gzip仅留服务器，路径/SHA见 [provenance](RESULT_PROVENANCE.md)。
