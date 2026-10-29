# 论文版本

**当前最新版是 [v9](v9/)：[正文 PDF](v9/manuscript.pdf) · [SM PDF](v9/supplement.pdf) · [单栏审阅 PDF](v9/review.pdf)。**

| 版本 | 说明 | 正文 | SM | 单栏审阅 |
| --- | --- | --- | --- | --- |
| [v9](v9/) | 2026-10-28，加入作者、邮箱、单位与仓库信息 | [PDF](v9/manuscript.pdf) | [PDF](v9/supplement.pdf) | [PDF](v9/review.pdf) |
| [v8](v8/) | 读者审阅意见的有范围修订 | [PDF](v8/manuscript.pdf) | [PDF](v8/supplement.pdf) | [PDF](v8/review.pdf) |
| [v7](v7/) | 正文、审阅版、SM 同步构建 | [PDF](v7/manuscript.pdf) | [PDF](v7/supplement.pdf) | [PDF](v7/review.pdf) |
| [v6](v6/) | 正文 Fig.2–4 单栏排版版本 | [PDF](v6/manuscript.pdf) | 沿用 [v5 SM](v5/supplement.pdf)，无独立新版 | — |
| [v5](v5/) | 扩充物理内容与方法内核，采用已选 A 稿 | [PDF](v5/manuscript.pdf) | [PDF](v5/supplement.pdf) | — |
| [v4](v4/) | 从内部审计表达改为公开可读正文 | [PDF](v4/manuscript.pdf) | [PDF](v4/supplement.pdf) | — |
| [v3](v3/) | Stage IV 表述及 Fig.4(d) 调整 | [PDF](v3/manuscript.pdf) | [PDF](v3/supplement.pdf) | — |
| [v2](v2/) | 扩充参考文献 | [PDF](v2/manuscript.pdf) | 无独立新版 | — |
| [v1](v1/) | 首个加入文献的整合稿 | [PDF](v1/manuscript.pdf) | 无独立新版 | — |

这里的 v1 明确指 `manuscript_prl_integrated_cited_v1`。更早的无引用稿、
单栏试验、fallback 和合订审阅包不再混入主版本列表；它们仍保留在原始
[工作流输出区](../../output/pdf/)中。不会重编号或覆盖历史审计身份。

## 每版只保留四类文件

```text
v9/
  manuscript.tex       manuscript.pdf
  supplement.tex       supplement.pdf
  review.tex           review.pdf
  references.bib
  tex/                 必需的 TeX 子文件
  figures/             实际引用的图片（保留原始矢量 PDF）
```

`tex/` 是文章源码的一部分，不是审计材料。保留原来的输入文件边界，
避免强制合并为单文件时改变 TeX 的空白、段落及换行。
所有图片和子文件引用均为相对路径，复制一个完整版本目录即可编译。
版本目录不存放 `.aux`、`.log`、`.bbl`、预览 PNG、审计 JSON、CSV 或 ZIP。
已保存的 PDF 是原版本的逐字节副本；整理不改变作者历史、排版或科学结论。

## 编译

需要 pdfLaTeX、BibTeX、REVTeX 4.2。为避免编译缓存重新混入版本目录，
先把整版复制到临时目录。在该副本中运行：

```sh
pdflatex -shell-restricted -halt-on-error supplement.tex
pdflatex -shell-restricted -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -shell-restricted -halt-on-error review.tex
bibtex review
for pass in 1 2 3; do
  pdflatex -shell-restricted -halt-on-error supplement.tex
  pdflatex -shell-restricted -halt-on-error manuscript.tex
  pdflatex -shell-restricted -halt-on-error review.tex
done
```

上例适用于 v7–v9，它们的正文和 SM 有交叉引用。较早版本跳过不存在的
`review.tex` 或 `supplement.tex`；正文仍需 BibTeX 和重复 LaTeX 编译。
使用与原工作流一致的受限模式，不启用无限制的 `-shell-escape`。

维护者可从项目根目录运行下列命令，重新检查已导出的版本；它在临时目录
编译并比较文字和页面像素，不覆盖版本内 PDF，也不运行物理计算：

```sh
python3 manuscript_langgraph/scripts/export_clean_manuscripts.py --verify
```

此检查额外需要 Python 的 `pypdf`、`Pillow` 及 `pdftoppm`。
输入哈希和检查结果集中存放在
`../../docs/report_qa/20261029_manuscript_exports/`，不混入各版稿件。
原始写作包及其 Markdown、表格、Gate 状态保持不变。
这些仍然是审阅稿；目录整理不表示科学或图件 Gate 获得批准。
