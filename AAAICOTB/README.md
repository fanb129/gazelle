# AAAICOTB：Counterfactual Observer--Target Binding

本目录是新 idea 的独立实验包。它不修改历史 `gazelle/` 模型代码，只把 VAT 从“逐人物采样”改为“同帧所有人物一起采样”，并在原有 heatmap/in-out loss 之外加入 COTB 配对损失。

## 一句话方法

同一张图中，若人物 A 看目标 A、人物 B 看目标 B，那么模型对正确配对 `(A→A, B→B)` 的总分必须高于交换配对 `(A→B, B→A)`。共同注视同一目标的人先聚成 shared-target cluster，不互相作为负样本。

## 输入输出

- 输入：一张 VAT 场景图 + 该帧全部人物的 normalized head bboxes。
- 模型输出：每个人一张 `64×64` gaze heatmap，以及一个 in/out probability。
- COTB 额外输出：没有新增部署输出；binding score/loss 只用于训练和评价。

## 文件

- `binding.py`：shared-target clustering、目标局部打分、diagonal-vs-swapped loss。
- `annotations.py` / `geometry.py`：无 PyTorch 依赖的标注切分、目标聚类和 bbox 审计。
- `data.py`：按帧读取 VAT，并对所有人物执行一致的水平翻转。
- `audit_dataset.py`：只审计标注是否有足够的有效反事实 pair。
- `train.py`：VAT matched control / COTB 训练，使用 train 内 sequence-level validation。
- `evaluate.py`：标准 VAT 指标和 binding 指标，保存 raw records。
- `compare.py`：相同 pair 的 paired sequence-cluster bootstrap 与 Go/No-Go。
- `summarize_seeds.py`：三随机种子的描述性汇总。
- `run_pilot.sh`：一天 pilot；历史 GazeSpot 架构上比较 `λ=0` 与 `λ=0.1`。
- `run_full.sh`：clean `raw_concat + no spatial prior` 配置的三随机种子确认。
- `EXPERIMENT_DESIGN.md`：完整假设、控制变量、指标和判据。
- `SERVER_COMMANDS.md`：服务器端可直接复制的命令。

## 最快运行

在服务器仓库根目录执行：

```bash
mkdir -p AAAIResults/COTB
setsid nohup env GPU=0 bash AAAICOTB/run_pilot.sh > AAAIResults/COTB/pilot.log 2>&1 &
```

默认 checkpoint 路径来自本仓库历史 `help.md`。若服务器位置不同，通过 `GF_INIT_CKPT`、`V0_VAT_CKPT`、`SPOT_VAT_CKPT`、`DATA_PATH` 覆盖；详见 `SERVER_COMMANDS.md`。

## 科学边界

- 主 binding 数据集是 VAT，因为其预处理标注保留同帧多人；GazeFollow 只用于初始化与标准指标。
- GOO-Real 当前预处理是逐 head/target 样本，不能作为主要 binding 证据。
- pilot 使用历史 GazeSpot 结构只是快速检查 loss 是否有信号；最终方法必须在 `raw_concat + spatial_prior=none` 的 clean control 上确认。
- 所有图表必须来自 `evaluate.py` 保存的 raw predictions，不使用历史评测脚本中的任何 GT 后处理分支。
