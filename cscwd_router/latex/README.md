# CSCWD 2027 LaTeX

2026-11-25已核对[会议投稿页](https://cscwd2027.dailyeliteevents.com.au/submission.html)：英文，IEEE proceedings格式，最多6页，包含图表与参考文献；官网延期投稿日为2026-11-30，未给出截止时刻/时区。本次只准备模板，不提交论文。

模板位于 [ieee_template/IEEEtran/bare_conf.tex](ieee_template/IEEEtran/bare_conf.tex)，类文件为 [IEEEtran.cls](ieee_template/IEEEtran/IEEEtran.cls)，BibTeX样式为 [IEEEtran.bst](ieee_template/IEEEtran/bibtex/IEEEtran.bst)。上游文件原样保留，版权和LPPL说明未改。

下载采用IEEEtran维护者通过[CTAN发布的1.8b包](https://ctan.org/pkg/ieeetran)。IEEE官网ZIP直链返回HTTP 202空响应，IEEE SPMB镜像返回HTTP 429，故改用CTAN分发的维护者版本。完整下载地址、原包SHA与36份文件SHA见 [download_provenance.json](ieee_template/download_provenance.json)。它是IEEE conference通用模板；CSCWD 2027官网没有另发专用LaTeX包。

使用 `\documentclass[conference]{IEEEtran}`，10pt、双栏，默认US Letter。2027官网未单独说明A4/Letter，本文采用IEEEtran默认Letter，不把2026年的额外格式条款当作2027要求。保持模板的字号、边距、行距和栏距。

本机验证命令：

```bash
cd /Users/fanb/src/gazelle/cscwd_router/latex/ieee_template/IEEEtran
mkdir -p /private/tmp/gazelle-template-check
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=/private/tmp/gazelle-template-check bare_conf.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=/private/tmp/gazelle-template-check bare_conf.tex
```

已实际编译两次，示例为1页、612×792 pt（Letter），字体为Type 1，页面渲染目视通过。示例作者、摘要和正文均为上游占位内容；写正式稿时另建 `main.tex`，不将示例内容作为论文。当前中文核心稿是 [CORE_DRAFT_ZH.md](CORE_DRAFT_ZH.md)。

效率主对照采用 **Full-token control (ours, K100)** 与 **Routed (ours, K50)**：同一选中权重、相同DINOv3、多层解码器和Router，区别为后段保留token的预算。K100不是官方GazeLLE，也不是单独训练的dense模型。GazeLLE作为框架来源和文献精度基线引用，官方效率原始报告保留。相关方法按输入、训练设置和任务可比性选择，不按是否低于本文决定是否列出；现有证据采用“competitive accuracy with reduced inference cost”，不声称全局SOTA。
