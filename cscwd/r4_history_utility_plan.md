# R4-D5：训练集因果历史候选效用分析

版本 `r4_history_utility_v1`，2026-11-19，在真实运行前固定。D4已通过小报告验收：无历史AP/出画收益可重现原收益，旧历史机制尚无核心优势。下一步先判断历史候选能否补救当前错误，再决定新选择门；本任务不训练模型。

## 输入与范围

只读取完整官方train（1,033 tracks / 132,563人物帧 / 58,507唯一图像）的原始GazeSpot-B预测和原生FP16视觉特征。读取 `r1_train_single_frame_cache.json`、`r2_feature_cache.json` 与对应cache；不读取test缓存、D4权重或R1 validation manifest。复用原 `official_data._load_split('train', ...)` 的来源/大文件SHA/覆盖核对，GT仅在生成全部候选坐标之后用于离线计分。

整个GazeSpot保持冻结，本任务直接读取缓存，不执行底座前向。每个人物轨迹独立处理，只有原帧号恰差1时历史可用；起点和缺口清空。reset只依赖metadata，不依赖GT或预测状态。历史可用性不要求过去GT画内，enter帧也纳入分析。

## 固定候选

候选生成函数只接收当前heatmap、当前冻结feature、原帧号和自身过去状态，无GT参数。计算为FP32、关闭TF32，无训练/参数扫描。

| ID | 定义 |
|---|---|
| `current` | 原始当前64×64 heatmap |
| `current_grid` | 当前heatmap avgpool至feature网格，再bilinear回64×64；仅作分辨率变化参考 |
| `previous_raw` | 本人上一个连续帧的原始heatmap，未做视觉对齐 |
| `ema_075` | 原始预测的因果EMA，当前权重0.75、过去EMA权重0.25；起点/缺口直接当前 |
| `transport_previous` | 上一连续帧原始heatmap的空间质量，按上一帧→当前帧特征对应传输 |

transport具体定义：将过去heatmap非负avgpool至过去feature网格并归一化为总质量1；过去/当前每个patch的feature做通道L2归一化。相似度 `S[p,q]=cos(F_previous[p], F_current[q])`，固定temperature=0.1，`A[p,:]=softmax(S[p,:]/0.1)`，每个过去patch将质量分配到当前patch，`m_current[q]=sum_p m_previous[p]*A[p,q]`。再bilinear上采样至64×64并归一化总质量。该图用于定位候选，不称校准后的独立gaze概率。

历史不可用时三个历史候选均为原始当前heatmap。`current_grid`保留其固定分辨率操作。所有坐标按原口径 `(column/64,row/64)`、flatten首个argmax；不使用patch中心坐标代替。

记忆只保存上一帧raw heatmap/feature和自身raw-EMA；不保存GT、oracle选择或分析结果。transport是一跳raw历史，不递归累积transport图。固定原InOut分数，空间候选无法纠正原模型画外误判。

## 计分

全部候选坐标生成并落盘后，独立计分模块才读取标签。主效用分母为**当前GT画内且有连续历史**的人物帧；正确定位为L2≤原response_radius=0.1。

每个非current候选均报告 `both_correct / current_only_correct / history_only_correct / both_wrong`，四项和必须等于分母。`history_only_correct`为补救机会，`current_only_correct`为盲目采用历史的伤害；另报L2差 `e_current-e_candidate`分布/更好、持平、更差计数，以及固定原InOut>0.5后的四象限/可补救数，单列被原画外分数阻断的补救机会。

按当前/过去GT事后形成互斥分组，阈值复用 `TemporalConfig`：

- `start_or_gap`：无连续历史，不进入主效用分母。
- `gt_out`：当前GT画外，不计定位效用。
- `enter_frame`：连续历史、过去GT画外、当前GT画内。
- `stable_pair`：两帧GT画内，GT位移≤0.02。
- `intermediate_motion`：两帧GT画内，位移(0.02,0.15]。
- `gaze_shift`：两帧GT画内，位移>0.15；这是VAT坐标代理事件，不称已核实目标切换。

另对各固定候选完整train坐标调用原 `evaluate_temporal`/`evaluate_state_temporal`，报告全量定位L2与准确稳定/响应；原InOut保持且状态摘要必须相同。仅保存坐标，不保存全量候选heatmap，因此不报告候选AUC。全train AP只由相同原InOut计算一次，不将其列为新模型提升。

## GT选择参考

固定候选集合为 `current / previous_raw / ema_075 / transport_previous`，不含分辨率控制 `current_grid`。只在离线当前GT画内帧按最小L2选候选，并列按上述顺序优先current。报告该集合的L2、当前错误中可救回数量、各候选被选次数；不计算或宣称可部署的oracle时间指标。

oracle不参与任何候选生成/记忆写入，不训练门、不在test选参数。它只是该固定集合的GT知情参考，不是全部视频信息的理论上限。若当前门控可观察的证据不足，即便oracle存在空间，也不保证可学习选择能达到它。

## 产物与后续

新服务器目录 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_history_utility_v1/` 保存 `candidate_xy.npy`（N×5×2）、`history_available.npy`、`index.json`（轨迹/候选/映射与SHA）、完整计分明细和oracle选择记录。NPY/完整明细留服务器，不保存新checkpoint或全量候选热图。

小报告为 `cscwd/results/r4_history_utility.json/.md`，自动追加provenance；日志 `r4_history_utility.log`。完整GPU 1手动nohup命令见实现计划§16.1。

回传后检查稳定/变化组的补救、伤害与固定状态阻断，并比较raw历史/EMA/视觉传输及分辨率参考。只有存在可利用空间时才固定并实现一版效用监督选择门；当前任务不同时实现后续模型或切换底座。
