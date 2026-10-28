# R01–R14 定向修订候选 v8

执行日期：2026-09-12。基准是独立评审指定的 v7 单栏正文与 SM；原 PDF、冻结图和科学数据均保留。

这不是“所有问题解决”的最终稿。已实施不依赖新科学决定的文字/方法修正、Fig.2 局部布局、SM 定义与算法排版；其他项按原编号明确停在具体作者选择。

## 本轮修改

- Fig.2 计数移离低交换条带，移除主相图与左列相挤的重复纵轴标题，增高原生画布并扩大历史图字号/间隔；保持原坐标、相标签、样本和历史曲线。Figure Engine preflight/lint 通过；自动 lint 不代表所有文字几何均被测量，也不代替人工打印尺度审阅。
- Fig.2 部分继承刻度仍为原生 6.8 pt，在 164.5 mm 审阅版中约 6.3 pt；本轮不宣称全图达到实际插入尺寸下 7 pt。图内主相条带遮挡已修，打印小字仍留审阅意见。
- 主文补固定耦合与二维化学势，解释 Fig.2 分配图的迭代颜色和虚线含义；Discussion 删除跳向未展示 learned-acquisition/world-model 的末段，保留统计限制。
- SM 补模式分派、扩窗诊断、近简并/近零配对与返回状态；分开记录已有代码与实际历史调用绑定。未确认的零温处理、调用级细化与扩窗设置不以当前默认值替代。
- 修正 discovery 模式实际关闭 cooldown 的说明；统一 active-pool 分位数对象，补概率/标准差单位、Stage III 继承关系、Stage IV bracket/jitter/边缘提案、配额余数和逐点顺序。
- 分开二维在线停止、Stage III 离线拓扑审计和 Stage IV 预算/诊断，补已有二维邻居重构的定义。未重构任何边界。
- 算法和易断裂的完整定义采用局部约束；SM 图表改为位于其语义锚点的完整块，不强制每幅图独占一页，不缩图。
- S2/S3 方法概览在 Figure Engine 下生成 164 × 75 mm 的独立候选：原文字、9–11 pt 字号和宽度不变，只压缩行间留白；不是缩放旧 PDF。原 v7 资产保留。

## R01–R14 状态

|编号|状态|本轮结果／具体剩余依赖|
|---|---|---|
|R01|证据对照完成，待选择|四窗拓扑计数/分母已核对；保留拓扑结论并补对应空间显示，或收窄为热力学描述；未替作者选择|
|R02|待科学确认|列出机制弱化的具体候选；尚无已确认同一选态分支直接路径证据；Fig.3 科学性重排联动等待|
|R03|部分修改|固定耦合和二维化学势已入主文；Ansatz、基底/矩阵和晶格单位待作者确认|
|R04|部分修改，A/B限定|补存档实现条件分支；调用级绑定、完整零温处理和 Fig.3 offline refinement 归属仍未确认；C未复现|
|R05|部分修改，A/B限定|补 discovery cooldown 实际关闭、3D建议点/配额/先后与回填；不是端到端重放或全部历史绑定|
|R06|已修文字一致性|概述与详细式均以惩罚后的 eligible $A_{\rm pool}$ 为分位数对象|
|R07|部分修改|区分Stage I预算停、II/III在线热力学停、III离线审计；补二维重构；3D完整监测实现与实际job退出绑定待恢复确认|
|R08|部分修改|Fig.2已修；Fig.3/4与S5的新布局分别依赖R02/R01/R12选择，旧版细字/裁切等问题未宣布消失|
|R09|SM重点修订|算法与完整公式不再被浮动拆开；SM图表就地成块；主文仍为宽浮动图，跨页叙事布局仍需结合待选新图联调|
|R10|部分修改|Discussion聚焦本文，Fig.2去内部存档用语；依赖R01/R02的摘要/结果措辞保留待审|
|R11|待作者确认|红色边界态限定句已备好，Fig.1不重画|
|R12|末端身份核对完成，待选择|逐行确定额外141条uniform-SC=120/21；未将末端核对泛化到所有历史；保留历史或重汇总待选|
|R13|待作者信息|匿名/身份、资助、公开入口及新图版批准不虚构；未发布数据|
|R14|部分修改|温度导数改$\partial_\tau$，邻居数解释为$k_{\rm nn}$；Fig.4旧图词语随R01新图修订；Liao姓名缺可靠原始条目，未猜改|

## 证据与审阅文件

- `AUTHOR_REVIEW.md`：具体拟改句和布局选择。
- `source_changes.diff`：逐行前后对照。
- `tables/R01_proposed_bin_topology_counts.csv`：四窗既有标签计数，不是相体积。
- `tables/R12_endpoint_reconciliation.csv` / `R12_endpoint_uniform_sc_records.csv`：末端口径与141条记录索引；未重汇总全历史。
- `input_manifest.json`：v7保护输入和候选嵌图来源。
- `method_source_hashes.json`：本轮核对方法的存档源；文件散列不是历史执行绑定证据。
- `source/built/validation.json`：实际栏宽、嵌图尺寸、页数与共同源指纹。
- 全科学内容 Markdown、表格 CSV、图 PNG 与便携 LaTeX 包同步输出。科学文字含待确认项，不称投稿就绪。

## 构建

```sh
PYTHONPATH=. MPLCONFIGDIR=/tmp/fflo_mpl_cache world_model_descriptor_dev/.venv/bin/python figures/build_fig2_reader_revision_v8.py
PYTHONPATH=. MPLCONFIGDIR=/tmp/fflo_mpl_cache world_model_descriptor_dev/.venv/bin/python figures/build_compact_methods_v8.py
world_model_descriptor_dev/.venv/bin/python manuscript_langgraph/scripts/run_single_column_placement.py --profile reader_revision_v8
world_model_descriptor_dev/.venv/bin/python manuscript_variants/reader_revision_v8/review_evidence.py
```

解包文档 source 后运行 `python3 build.py`；需要 pypdf、REVTeX/pdfLaTeX/BibTeX。不运行物理。

继续沿用现有四节点 LangGraph，未新建框架、未更改旧默认 profile、未 commit/push。最终 Gate 保持 `HUMAN_WAIT`。
