# 格式依据（访问日期：2026-09-11）

- 明确规则：PRL Letter的主体长度限制为3,750等效词；公式、图和表按专门方法折算，标题、作者、摘要、参考文献及致谢不计入这部分。本文保持可读尺寸，不按PDF页数反推合规。[APS长度指南](https://journals.aps.org/authors/length-guide)
- 补充材料以正文中的单个参考文献条目指向；SM独用文献也应进入主文参考文献表。本轮保留原50篇文献并加入第51项SM说明，未开展新科学检索；SM没有独有的新科学引用。SM标题与主文一致，署名仍待作者提供。[APS Supplemental Material Instructions](https://journals.aps.org/authors/supplemental-material-instructions)
- PRL要求正文能够独立阐明研究，End Matter只是可选安排；本轮未引入第三套正文结构或将核心问题移走。[PRL作者指南](https://journals.aps.org/prl/authors)
- REVTeX的reprint样式不等同最终出版排版，BibTeX与多次LaTeX编译用于稳定引用。正常浮动与floatfix可用；本轮撤掉逐段FloatBarrier，没有负间距补丁。[REVTeX FAQ](https://journals.aps.org/revtex/revtex-faq)
- 项目偏好而非APS硬性规定：图内普通标注优先7–9 pt、SM单栏、主文双栏并另供单栏审阅、公式/图/表/算法使用S前缀。批准资产部分细字达不到该优先目标，已如实列为待审，不声称期刊要求全部满足。

等效长度明细由本轮源文件的TeXcount和实际嵌图宽高生成，见输出目录的 `tables/length_estimate.csv`。单栏中的陈列公式按16词/行；跨栏图按600/(宽高比)+40折算。估计不是APS编辑系统的最终判定。
