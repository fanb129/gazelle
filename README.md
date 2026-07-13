# Gazelle AAAI2027 工作区

本仓库当前已经切换为 AAAI 2027 投稿周期的工作区。后续所有研究设计、代码修改、实验整理、论文写作与审稿前检查，默认都服务于 AAAI 2027。

## 当前方向

- 当前工作分支：`codex/aaai2027`
- 原 V1 / ACM MM 2026 rebuttal 工作已经不再是当前主线。
- ACM MM 2026 相关材料仅作为历史背景、问题复盘和 AAAI 2027 改稿参考。
- 从 `v1` 切换到本分支前，旧工作区的未提交内容已经保存在 Git stash 中：
  - `stash@{0}: On v1: pre-aaai2027-untracked-rebuttal-files`
  - `stash@{1}: On v1: pre-aaai2027-cleanup-from-v1`

## ACM MM 2026 历史材料

ACM MM 2026 投稿已经被拒，以下材料保留在仓库中，用于分析评审意见、总结失败原因，并指导 AAAI 2027 版本的重构与补强。

- ACM MM 2026 论文正文：`ACMMM2026rebuttal/2026ACMMM-fanb-final-3.pdf`
- ACM MM 2026 附件：`ACMMM2026rebuttal/Appendix of GazeSpot.pdf`
- ACM MM 2026 rebuttal 目录：`ACMMM2026rebuttal/2026ACMMM_rebuttal/`
- ACM MM 2026 rebuttal 策略记录：`ACMMM2026rebuttal/rebuttal_strategy.md`
- ACM MM 2026 评审意见 / OpenReview 导出：`ACMMM2026rebuttal/Gaze in the Crowd_ Frustum-Aware Feature Aggregation for Robust Gaze Target Estimation _ OpenReview.pdf`
- rebuttal 阶段实验命令：`ACMMM2026rebuttal/p0_experiment_commands.md`
- rebuttal 阶段结果：`ACMMM2026rebuttal/results/`

## 项目本地 Skills 配置

Supervisor-Skills 已经作为项目本地 Skills 安装到本仓库中。

- Skills 根目录：`.agents/skills/`
- 来源仓库：`https://github.com/HKUSTDial/Supervisor-Skills`
- 本项目只使用 `.agents/skills/` 下的项目本地 Skills。
- 不要把本项目依赖的 Skills 安装到 `$HOME/.agents/skills`、`~/.codex/skills` 或任何全局、用户级、管理员级、系统级目录。
- 每个 Supervisor-Skills skill 都保留为独立目录，并包含原始 `SKILL.md` 与相关资源。

已安装的 skill 目录：

- `.agents/skills/benchmark-paper-template/`
- `.agents/skills/deep-research/`
- `.agents/skills/drawio-reconstruction/`
- `.agents/skills/figure-designer/`
- `.agents/skills/idea-evaluator/`
- `.agents/skills/intro-drafter/`
- `.agents/skills/paper-polish/`
- `.agents/skills/paper-writer/`
- `.agents/skills/pre-submission-reviewer/`
- `.agents/skills/tech-paper-template/`
- `.agents/skills/vibe-research-workflow/`

## 开发与运行配置

- 本地环境主要用于修改代码、整理仓库、更新文档和补充测试。
- 真实训练与评测由用户在服务器上 `pull` 最新代码后手动运行，并把运行结果反馈回来。
- 服务器登录：`fb@3090.lab`
- 本地已经配置好 SSH 密钥登录。
- Python 包名：`gazelle`
- 包配置文件：`setup.py`
- 现有实验命令和历史运行片段：`help.md`

当前脚本中的默认路径主要面向服务器环境：

- GazeFollow：`/newhome/fb/dataset/gazefollow_extended`
- VideoAttentionTarget：`/newhome/fb/dataset/videoattentiontarget`
- GooReal：`/newhome/fb/dataset/gooreal_data`
- 默认 checkpoint / 实验输出目录：`./experiments`
- 默认可视化图片输出目录：`./experiments_imgs`
- 部分脚本中的默认 Weights & Biases project 仍是 `gazelleV1`，启动 AAAI 2027 新实验前需要检查是否改名。

## 数据集备注

GooReal 已经在服务器上下载到：

```text
fb@nise-server:/newhome/fb/dataset/gooreal_data
├── gooreal.zip
├── oneshotrealhumansNew.pickle
├── testrealhumansNew.pickle
├── testrealhumansSparsedNew.pickle
├── valrealhumansNew.pickle
├── wget-log
├── wget-log.1
├── wget-log.2
├── wget-log.3
└── wget-log.4
```

ChildPlay-gaze 因为无法访问 YouTube，暂时还没有下载。

## 工作规则

除非明确说明，否则之后对本仓库的讨论、修改、实验设计和写作，都默认面向 AAAI 2027。
