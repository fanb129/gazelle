# 人工案例包本机检查

2026-11-21，分支 `codex/coverage-aware-router`。只验证工具实现，不代表模型性能或已经导出真实案例。

实际执行：

```bash
/private/tmp/gazelle-cscwd-check/bin/python -m pytest -q \
  cscwd/code/stag/test_failure_cases.py \
  cscwd/code/scripts/test_failure_review.py \
  cscwd/code/stag/test_review_viewer.py
```

结果：**13 passed**。环境复用本机临时Python、NumPy、Pillow、pytest；viewer交互逻辑另以已有Node执行。CLI的 `--help` 及完整合成导出命令已实际运行。

| 检查 | 结果 |
|---|---|
| 原始缓存读取 | 按旧T2索引的start/local位置映射，未按track_id重排；FP32、原始热图argmax与分数合同验证 |
| 抽样 | 固定seed可复现，五类各有明确候选条件；人物/视频上限全类别共享，不足明确记录，不自动放宽 |
| 历史 | 同人物原帧号连续，无未来；缺口截断；最大历史窗口不重叠 |
| 导出读写 | 合成train/test缓存→选择→图片/原热图预览/头部局部→manifest/HTML/空标注→报告/ZIP完整通过 |
| 文件一致性 | 源NPY、manifest、HTML、ZIP的SHA重算与记录匹配；错误SHA拒绝；热图像素等于原概率乘255后取整 |
| 默认观察 | 中性case编号，GT/类别/L2默认隐藏，只看当前帧；类别选定后打乱展示顺序 |
| 标注 | JSON/CSV覆盖全部case；CSV逗号/引号/中文读回；同包JSON与空模板可导入，其他包/未知或重复case拒绝且不修改原标注 |
| 真实浏览 | 用本机已有Playwright/Chromium通过file://打开合成包，无服务、无外部请求、无页面错误 |
| 浏览交互 | localStorage刷新保持，train/test/类别筛选，空筛选清空旧场景并可恢复；历史窗口/播放到anchor停止，SVG叠加与原图比例一致，头部局部随帧切换 |
| GT揭示 | 首次揭示时间保留，隐藏或导入空时间均不清除；当前/历史记录及揭示后机制备注均可编辑 |

实际浏览使用18组两split合成案例，所有合成图片、缓存、导出包与截图只在 `/private/tmp/`。默认与历史/GT两种界面截图已目视检查，布局无明显溢出或遮挡。

服务器只做了已授权的只读结果核实；本轮未启动真实案例导出、模型推理或训练。真实导出命令见 [FAILURE_REVIEW.md](../FAILURE_REVIEW.md)。停止实验的实际六轮history及checkpoint来源已单独记录在 [stopped summary](convlstm_v1_stopped_summary.json)。
