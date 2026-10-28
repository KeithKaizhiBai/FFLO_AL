# 同步审阅稿（2026-09-11）

本轮交付的是可供作者审阅的配对版本，不是投稿就绪稿。旧稿、物理数据、求解代码、正文图资产均未修改。

## 文件与阅读顺序

- `manuscript_double_column.pdf`：6页，正文主输出；优先看第2–5页的图文组合。
- `manuscript_single_column_review.pdf`：8页，与双栏稿共享正文、图注和参考文献，只改变排版。末页仅有参考文献尾项，是正常文末分页，不是浮动屏障留下的图页。
- `supplement.pdf`：12页，重新从源文件编译。第1–2页模型与算法，第4–8页网络/选点细节，第9页计数，第10–11页拓扑与3D诊断，第12页复现边界。
- `latex_source_package.zip` / `source/`：便携LaTeX、BibTeX、图文件和成套构建脚本。另附完整文字/LaTeX数学表达的Markdown伴随文件、`tables/`和`figures/`。
- `AUTHOR_REVIEW.md`：科学措辞、署名、数据公开及AI披露待确认项。

## 改了什么

1. 根据2026-08-28人工选图批准记录，把Fig.1恢复为紧凑landscape的engine-v4；不是重新绘图。Fig.2/3保留冻结图，Fig.4沿用已删除(d)的scope-revision版本。
2. 撤掉逐段FloatBarrier。Fig.1–4实际置于正文第2、3、4、5页，均约179.25 mm宽，图下接正文；不再有旧版第2页大片空白或Fig.1孤立整页。
3. 清理内部术语与重复计数；将“固定mu切片”更正为有明确半宽的窄窗口。Stage-I 6,880与Stage-II 7,434的区别只在必要的正文说明处展开，详细请求/留存口径在SM。
4. SM改为正常字号单栏，算法改为输入、输出和分步循环。核对并补齐不同阶段的初始化、网络、训练与acquisition。Stage III不是额外谱网络，Stage IV不是2D选点公式原封不动加一个输入。
5. SM S1–S4用Figure Engine生成原生164-mm可读方法图，普通文字9–11 pt，登记实际文字bbox并通过lint；旧图保留。恢复原有Stage-IV六面板原始诊断为SM Fig.S5，不新增阈值线。
6. 统一主文/SM标题和自动编号；移除虚构署名占位句。保留50篇原科学文献，添加第51项SM条目。没有检索新科学文献。

## 核验

- 复用现有四节点LangGraph，新增配对profile，不另建框架。CLI默认现在使用`synchronized_v7`；重建旧流程需显式`--profile legacy_v6`。
- 每次在全新临时目录编译三份文档，成功后才发布成套PDF；不读取旧aux/bbl。
- 56项稿件测试、28项Figure Engine测试通过。新增负例覆盖旧Fig.1、旧SM、伪单栏、缺失引用/重复编号和过度缩图。
- 检查实际PDF嵌图变换矩阵、recorder依赖、统一源指纹；无undefined引用、重复编号或overfull box。
- 正文6页、SM12页及单栏8页均已渲染检查。对照图见`previews/fig1_source_vs_embedded.png`。
- 自由能表达式及其原有说明逐字节保持一致；本轮没有做物理重算或自由能正确性判定。

## 长度与未解决项

按APS方法近似合计3,676.5等效词：正文及标题层级1,621，图注209，行内公式估计36，陈列公式32，四张跨栏图约1,778.5，表0。距离3,750较近；这不是期刊系统的最终认证。今后应优先删重复措辞，不靠缩字/缩图凑长度。方法、来源和分项见`policy_notes.md`及`tables/length_estimate.csv`。

Fig.3右侧刻度、Fig.4小图和图例的一些原始细字仍低于项目7–9 pt阅读目标；本轮保留批准科学面板，未伪称全部可读性自动通过。后续若要求全图大字，需要单独批准原生布局修订/面板拆分。连续闭隙与一级极小值切换等机制措辞仍待作者确认；署名、公开仓库、部分运行绑定信息也未完成。详见`AUTHOR_REVIEW.md`。

## 来源与构建

旧正文真正入口：`manuscript_variants/scientific_depth_single_column_v6/main_barrier.tex`，调用`sections/results_barrier.tex`；运行记录为`active_learning_prl_single_column_v6`，实际编译依赖在`tmp/pdfs/scientific_depth_single_column_v6/barrier/barrier.fls`。

旧SM真正入口：`manuscript_variants/scientific_depth_v5/supplement.tex`，调用`sections/supplement.tex`；运行记录为`active_learning_prl_scientific_depth_v5`，旧输出保留其`.fls`。不是根据PDF文件名猜测。

完整输入路径、SHA-256、图版本见`input_manifest.json`；便携源文件指纹见`source/built/input_hashes.json`，实际依赖与检查见`source/built/*.fls`及`validation.json`。

```sh
# 在项目根目录，已有新SM资产时：
world_model_descriptor_dev/.venv/bin/python manuscript_langgraph/scripts/run_single_column_placement.py --profile synchronized_v7
# 需要重建本轮SM方法图时（不运行物理）：
MPLCONFIGDIR=/tmp/fflo_mpl_cache world_model_descriptor_dev/.venv/bin/python figures/build_readable_supplement_methods.py
# 解包便携source后（Python需安装pypdf，系统需REVTeX/pdfLaTeX/BibTeX）：
python3 build.py
```

最终状态：`GATE_SYNCHRONIZED_MANUSCRIPT_AUTHOR_REVIEW = HUMAN_WAIT`。未commit、push或投稿。
