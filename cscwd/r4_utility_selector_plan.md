# R4-D6：以强当前候选为锚的历史效用选择

版本 `r4_utility_selector_v1`，2026-11-20，在真实运行前固定。D5结果 `a6928e6` 已通过小报告验收；当前任务实现并提供一次完整服务器训练入口，不同时展开更多门控版本。

## 1. 从D5得到的决定

D5只在train计分。固定候选GT选择可救3,126/22,831个当前错误，但不是部署成绩。视觉传输单独使用救2,624、伤3,145，需要检验可学习选择。无历史 `current_grid` 的稳定55.6066%→59.9081%、L2 0.094313→0.094019，不能把重采样收益当成历史作用。

本轮只训练一个二选一MLP：锚点C为 `current_grid`，历史H为D5 `transport_previous`。候选均固定，整个GazeSpot-B和原decoder保持冻结。原InOut原样输出；本轮目标收缩到定位质量、准确稳定和切换响应，AP/状态错误/错误翻转/leave响应应保持原值。

## 2. 候选、记忆与输出

直接import D5 `step_candidates`，不修改旧代码。C是当前raw heatmap avgpool至native feature网格再bilinear回64；H是上一连续帧raw空间质量，按source→target cosine-softmax、temperature0.1传输。历史来自过去raw观察，不来自门的选择结果。

无历史时强制C；否则logit>0选完整H图，logit≤0选完整C图。两图不做数值混合，避免将raw响应幅值与归一化质量当同一尺度。采用原64网格argmax；InOut直接复用当前raw分数。

每人物起点/原帧号缺口重置，绝不跨人物或缺口，绝不看未来。状态仅为D5 raw历史、独立grid-EMA、过去人物/场景/raw预测区域描述符、raw分数和真实头框。GT/oracle/门输出不写回这些记忆。

## 3. 观察和模型

输入六个L2-normalized冻结特征向量（每个C维）：当前scene mean、当前真实head框pool、过去head pool、当前C区域pool、当前H区域pool、过去raw预测区域pool。区域pool采用非负heatmap avgpool至feature网格并归一化加权；真实head pool复用原fractional bbox pooling。无历史时过去向量为零。

再输入以下17个标量，顺序作为cache/model契约固定：

1. 当前raw InOut；2. 过去raw InOut；3. 两者绝对差；4. 真实框中心位移；5. 前后人物描述符cosine；6. C/H当前区域描述符cosine；7. H当前区域与过去raw区域cosine；8–9. C/H归一化图entropy（除log4096）；10–11. C/H图peak fraction；12. C/H argmax位移；13–14. C/H与过去raw argmax位移；15. raw/grid argmax位移；16. 前后scene cosine；17. history_available。

过去相关标量在无历史时为0。模型为 `(6*C+17) → 64 → 1`，GELU，无GRU/learned recurrent state。最后线性层权重/bias置0，初始hard输出全C。只训练这个MLP；bank观察为FP32，底座特征仍来自既有原生FP16缓存。

## 4. 训练监督和固定预算

仅train当前GT画内且有历史帧参与效用监督：`d=L2(C,GT)-L2(H,GT)`，`y=1[d>0]`（tie为C），`w=abs(d)`。损失为 `sum(w * BCEWithLogits(logit,y)) / sum(w)`；零权重batch为可微零loss。不做正负类重加权，不扫描margin、不加额外时序loss。GT只构造当前帧loss，不进入观察/候选/重置/记忆。

seed0从头，五个epoch，AdamW lr0.0001/weight_decay0.01、batch256、hidden64、gradient_clip_norm1.0，FP32、TF32关闭。每epoch遍历完整train，loss只含上述mask；记录人物帧、监督帧、权重总和与weighted BCE分子。冻结候选无需TBPTT，可按seed+epoch打乱帧批次；不把本预算称为与旧GRU有效容量相同。

官方train/test、完整test每epoch、AUC更高→L2更低→AP更高→较早epoch选择整轮的用户约定保持。test用于选轮，明确不是独立留出估计。全部五轮指标/权重/输出保留，保存selected/last；不能临时改成稳定最优轮或拼列。

## 5. Bank与固定对照

先一次性从既有train/test预测与特征生成FP32 bank，五轮不再重复读取90GB空间特征或计算传输。候选顺序固定为 `current / current_grid / ema_075 / grid_ema_075 / transport_previous`。grid-EMA对每帧C做当前权重0.75的因果EMA，人物/缺口重置。

bank保存五候选heatmap、argmax、原InOut、观察、history mask、原轨迹/来源/文件SHA；不把GT utility target混入观察。全量来源核对复用 `official_training_inputs`；来源、模型、特征契约与分区不交叉检查保持。大小按实际shape/dtype在报告记录，不写未测的耗时估计。

五个固定候选均完整test一次，使用原标准/时序/状态指标，保留逐事件明细。每epoch选择门完整test，保存每帧logit、选择、最终FP32 heatmap/score/argmax、明细和checkpoint。报告与五个对照的整行差值及配对响应，不只比较原raw。部署选择只发生在C/H之间；其余三者为对照。

## 6. 路径与执行

新根目录 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_utility_selector_v1/`，下设 `banks/train`、`banks/test`、`baselines/<candidate>`、`selector/`。全部大NPY/PT/逐事件明细留服务器；原实验不覆盖。

唯一服务器入口 `code/scripts/train_utility_selector.py` 完成双bank→五个test对照→五轮训练/逐轮test→整轮选择/比较。完整GPU1手动nohup命令见实现计划§17.1。

小报告为 `results/r4_utility_selector.json/.md`、`r4_utility_selector_selection.json`，自动追加provenance，日志 `r4_utility_selector.log`。真实数字只在用户运行回传后写。

回传后先核对来源/覆盖/损失/选择/AP和状态保持，再判断相对current_grid及grid-EMA的定位—稳定—响应折中。只有超过这些控制才支持选择门的额外作用；训练结果将回答效用是否可由观察学习，不将D5 oracle直接当作可达到的目标成绩。
