# R4-D2：train拟合的单调状态校准对照

版本：`r4_state_calibration_v1`。2026-11-18，在真实运行前固定本文件。

## 目的与固定内容

D1显示，替换状态分数占新机制准确稳定净下降的主要部分。本轮检验：这个代价有多少来自状态输出的全局运行点，而非坐标变化。单帧、普通GRU、新机制各拟合一个独立的全局偏置，三者使用相同目标和规则。

普通GRU和新机制固定官方首轮已经选中的 **epoch 5**；单帧继续使用原GazeSpot ViT-B VAT epoch 7。保持模型参数、热图、坐标、原递归历史和所有时序指标定义。本轮没有新的epoch搜索或时序模块训练。原test参与过checkpoint选择的事实继续记录；偏置拟合只使用官方train。

## 数据与因果前向

- 完整官方train：1,033 tracks / 132,563人物帧 / 58,507图像。单帧分数直接复用 `r1_train_predictions/inout.npy`，不使用R1开发分组。
- 普通GRU与新机制加载各自 `experiments/cscwd_r4_official_v1/<model>/selected.pt`，核对等于epoch 5，执行 `eval/no_grad` 的完整train因果前向。
- 使用已有 `r2_native_last_fp16/` 特征和原始train预测，沿用各模型原配置 `clip_frames=16, batch_size=8`，FP32前向、TF32关闭。同人物跨chunk保留状态，跨人物/原帧缺口重置。只将当前特征、原预测、当前真实头框与自身过去状态传给 `model.step`；GT不传入模型或历史。
- 前向内部仍产生定位结果供原历史使用，但只保存train状态分数；不保存新的train热图。每帧按原prediction index的global row写回一次。
- 完整官方test：298 tracks / 31,978人物帧 / 13,127图像。直接复用既有单帧及epoch-5 test缓存，无需test视觉特征、重新前向或原始RGB。
- 启动先核对输入报告、checkpoint、旧前向源码、预测和train特征SHA256；train大特征文件的校验在前向之前完成。

## 一维参数与拟合规则

对每种方法使用同一个严格单调映射：

\[
q_b(p)=\frac{p\exp(b)}{1-p+p\exp(b)},\qquad b\in[-16,16].
\]

仅用train标签拟合 **一个** 标量 `b`，最小化逐人物帧、未加权的二元交叉熵，不另调阈值。概率在开区间时，目标等价于 `BCEWithLogits(logit(p)+b, y)`；导数为 `mean(q_b(p)-y)`。在固定区间上按导数做80轮二分；若最优在边界则直接取边界并记录，若全部为不可调整的端点则取0。没有温度、逐人物/逐视频参数或test选择分支。

数值规则在运行前固定：

- 输入float32概率提升到float64，按数值稳定的odds公式输出float64；不clip原始分数。原始 `p=0/1` 保持不变。
- 正确端点的BCE为0；错误端点贡献与偏置无关的无穷大。本轮拟合剩余有限、随偏置变化的BCE之和，除以全部train人物帧数；记录端点数量和错误端点数量。存在错误端点时，完整NLL写 `null`、`full_nll_finite=false`，不伪造有限值。
- `train_before/train_after` 记录有限目标与完整NLL；核对有限目标不增。每种方法的train/test都逐项核对不同原分数严格顺序保持、原ties保持、AP完全相等；失败则不发布成功报告。
- **先完成原模型前向，再应用校准**。校准分数只作为最终输出，不回填GRU状态、模板、validity或下一帧输入。部署同样可在每帧原模型输出后应用偏置。

## 评测与判读

状态判定仍为 `score > 0.5`。按原协议同时报告校准前/后：

- 标准AUC/L2/AP：AUC沿用相同热图的原结果并注明未重算；L2/AP重算，核对不变。单调状态映射没有新的AP排序收益。
- 准确稳定、Jitter、joint gaze_shift/enter/leave、状态错帧与画内/画外分组、稳定GT上的错误翻转、state-only enter/leave。
- 同一方法校准前后的事件配对延迟与两侧独有响应；三种方法校准后的共同事件配对。事件按轨迹和原帧号对齐。
- 分数变化与阈值跨越造成的状态错误纠正/引入；坐标变化计数应为0。

先重现每种方法原始L2/AP和完整temporal/state端点，再发布结果。全报三种方法，不按test结果重新调偏置或挑选校准强度。校准若恢复一部分稳定性，说明全局运行点可以解释相应部分；若没有恢复，则据此决定一次状态输出或训练目标调整。它本身不证明历史有效性机制；将原AP改善归因于历史仍需要无历史头对照。

## 产物与执行

唯一入口为 `code/scripts/calibrate_official_state.py`；完整服务器手动命令见 `implementation_plan.md` §13.1，物理GPU 1。

- 小报告：`results/r4_state_calibration.json/.md`，包括三个偏置、train目标值、前后完整摘要、来源与排序检查。`RESULT_PROVENANCE.md` 自动追加实际路径和SHA。
- 服务器目录：`/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/`。
- 每种方法保存 `calibration.json`、原test行序的float64 `test_inout_calibrated.npy`、校准前后逐轨迹/事件 `test_details.json`。
- GRU/新机制另存 `train_scores/inout.npy` 与 `index.json`；索引记录源预测index的路径/SHA、原行序、checkpoint、特征报告与覆盖检查。单帧train缓存直接引用。
- 不产生新的 `.pt`；不覆盖原预测、checkpoint或原结果。新目录要求不存在，避免混用不同作业。大文件与完整明细只留服务器，经Git回传小报告与provenance后本地验收。
