# E3-L：GF-L 效率补回与审查

2026-11-24。整理论文表格时发现原GF-L矩阵已完整结束，本次将六份报告与CPU检查通过服务器提交 `5e64c45` push/pull，并合并到本地草稿分支。**审查通过。** 没有重新训练、模型forward或GPU计时。

| Batch | 官方GazeLLE-L ms | routed K100 ms | K50 ms | K50 vs official | K50 vs K100 |
|---|---:|---:|---:|---:|---:|
| 1 | 33.0995 | 36.5304 | 36.0665 | 0.9177× | 1.0129× |
| 8 | 147.7829 | 179.8038 | 137.8264 | 1.0722× | 1.3046× |

B8为整批8张图。相对官方，B8延迟降低6.74%，B1延迟增加8.96%。两序B8速度比为1.0655/1.0791×，两序B1为0.9283/0.9073×。B8吞吐为57.8502 images/s，官方53.7160；峰值allocated为1824.02MiB，官方1423.83（高28.11%）。

运行GPU0，UUID `GPU-9abd86a3-e5fb-bca1-4843-b29aefbfbd46`；执行代码 `4162da7`。FP16、真实GF train RGB/GT头框、每图一查询；官方448输入、Router512输入。50 warmup、200×5 latency、500×5 throughput，正反序汇总；预处理/H2D在计时外。原始汇总见 [summary.json](efficiency_dinov3_seed0_20261124/gazefollow/vitl/summary.json)。

使用GF-L已导出第2轮best，SHA `0e3c848ce62731c53762696488e4911e86272dd4824a2db6b69a2f0e9ee5bebf`，FP32 AUC/Avg/Min L2仍为 `0.960007/0.094061/0.037721`。保留实际support3＋decoder4、`completed=false, closed_by_user=true`；CPU source reader核对原last/history和导出best通过，没有把中断来源改为跑满。K100/K50使用同一权重。

CPU检查 [e3_gfl_efficiency_server_checks.json](e3_gfl_efficiency_server_checks.json) 记录：

- 四个单元、12组token trace均一致。官方24层均1025 tokens；K100均1029；K50前12层1029、后12层517，输出层为 `[5,11,17,23]`，回填后解码。
- 核对执行版本25份源码文件、两份native源码树、五个来源/权重引用、GF annotation和八张唯一RGB的实际SHA。与当前源码的记录差异仅在VAT采样修复所在的 `dinov3_efficiency.py`；GF路径重建的输入/参数请求与原记录一致。
- 单元GPU UUID、模型选择、原精度、计时范围和参数一致；summary JSON/MD重算相同。已核对1000 latency sample的报告计数、五个repeat、吞吐median、数值有限性和显存关系。
- 本地六份原始报告SHA、按执行commit读取的25份源码SHA与独立延迟几何平均核对通过。

每次latency的原始数组没有保存，本次不能独立重算variant median/p95；只对现有统计量结构和矩阵汇总做核验。这是一组正反序矩阵，没有独立重复运行的显著性结论。

论文可报告GF-L在B8相对routed K100加速1.30×、相对官方GazeLLE-L加速1.072×；同时保留B1更慢、显存更高的结果。四组GF/VAT×B/L的完整表已填入 [CORE_DRAFT_ZH.md](../latex/CORE_DRAFT_ZH.md)。
