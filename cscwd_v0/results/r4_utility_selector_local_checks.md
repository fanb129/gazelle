# R4-D6 成本监督选择门：本地检查

2026-11-20，`r4_utility_selector_v1`。38项CPU合成检查通过：核心15项/0.734秒、bank7项/0.724秒、评测8项/0.517秒、实际完整CLI8项/3.982秒。精确命令、环境与八个新源码SHA见同名JSON。

- 六个冻结token+17标量顺序、原生输入4625维、296,129参数、zero init与无历史强制C通过。C=current_grid/H=一跳视觉传输；原InOut保持，整图硬选择。
- 成本加权BCE手算/梯度、float64 GT微小优势、零成本可微、mask/不重平衡通过。GT只用于train当前帧loss，不进入观察或记忆。
- raw/grid双EMA、人物/缺口/未来隔离、GT变化不改bank、原图/分数/全局坐标与分块无损读回、文件/header/index核对通过。
- 实际合成CLI完成双bank→五个固定对照→五轮训练/完整test/整轮选模；独立重算每轮成本分子/权重/监督数，全部预测/选择/同次logits和权重保留。
- 重载后bank与在线逐帧观察/选择一致；test GT/标签变化不改变五轮训练权重。原AP/state/leave保持，source文件不变，预算/plan/重复输出检查通过。
- 八个新Python文件通过3.10 AST检查，§17两个bash块语法通过；旧gazelle和已跟踪cscwd代码保持。

没有真实VAT D6性能数字，没有启动服务器。下一步用户按实现计划§17.1在物理GPU1手动运行一次，回传 `r4_utility_selector.json/.md`、selection JSON和provenance。Bank/PT/完整预测与事件留 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_utility_selector_v1/`；回传后评价相对grid/grid-EMA的定位—稳定—响应折中。
