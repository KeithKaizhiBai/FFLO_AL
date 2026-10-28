# v9 identity update — 2026-10-28

Name, email and affiliation are confirmed and inserted. This resolves only the identity portion of R13. Funding, scientific choices and figure approval are not inferred. The following is the preserved v8 review record.

# v8 科学选择与依赖项

本轮并未批准任何新科学结论。下面是具体待选项；未回复时保留 v7 对应文字，不视为问题消失。

## R01：Fig.4(c) 画了什么、文字说了什么

当前图：Normal / uniform SC / FFLO 三色热力学样本。
当前文字：比较每窗 cFFLO/tFFLO 分布，并称最后一窗无 trusted tFFLO。
现有 Table S7：热力学与数值信任计数，没有拓扑列。

已从原始末端与图用 CSV 核对坐标顺序和 phase_code，不新增标签。按原条件 `abs(mu-center) <= 0.025 + 1e-12`：

|中心|全部保留|FFLO|图用可信有隙 FFLO 分母|cFFLO|tFFLO|
|---|---:|---:|---:|---:|---:|
|-0.45|272|176|176|121|55|
|0.15|154|97|97|70|27|
|0.85|223|148|148|75|73|
|1.45|142|35|35|35|0|

方案 A：保留局部拓扑结论，Fig.4(c) 用同窗的热力学/拓扑并列展示，并给上述计数表；只读既有标签。表本身只能支持数量，不能独立验证“高低交换/温度”的空间比较，故应有对应拓扑面板。

方案 B：保留现有热力学类别，结果收窄为：

> The narrow chemical-potential bins in Fig. 4(c) display the thermodynamic labels of the sampled states. Uniform-SC support extends over a broader exchange range in the bin centered at $\mu/t=0.85$, whereas paired observations in the $\mu/t=1.45$ bin are concentrated at low exchange. These nonuniformly sampled bins do not measure phase volumes.

B 不是宣称拓扑支持被证伪。两方案均未自动执行。对应摘要和 Discussion 在选择后同步。

## R02：机制语言

公开 Fig.3 提供可靠有隙标签、存档 Pfaffian 支持受限重构、异号括区和邻居数敏感性，没有确定热力学选态分支上的直接 gap-vs-path 曲线。不同数据集的单点 gap refinement 不能冒充该证据。

拟改 Eq.(2) 后：

> This index distinguishes cFFLO from tFFLO among the reliable gapped samples. A gap-closing requirement applies to a continuous Hamiltonian branch; the present reconstruction does not establish continuity of the selected thermodynamic minima.

拟改结果：

> The reliable gapped FFLO observations separate into sectors with opposite Pfaffian indices. The support-restricted contour summarizes the region between these sampled sectors; it does not by itself establish how the gap evolves along the selected thermodynamic minima.

若批准，同步 abstract/Fig.3 legend/caption；不改不变量公式、边界坐标，也不排除分支切换或未分辨无隙区。当前尚未实施。

## R03 / R11：配对空间与边界态

固定耦合和二维化学势已补主文。以下候选需作者确认后再入稿：

> The mean-field search varies one pairing amplitude and one center-of-mass momentum, corresponding to a single-plane-wave ansatz $\Delta(x)=\Delta e^{iqx}$, rather than a general LO modulation.

晶格单位、标量 Delta 对应哪个自旋矩阵及写出的 Nambu 基底必须一起确认；本轮没有重定义它们。

Fig.1 caption 候选：

> The red boundary-state line is illustrative only; no calculated boundary-state dispersion is presented as a numerical result in this work.

请确认红线意图。没有补算边界谱或重画 Fig.1。

## R12：末端逐行身份已核对，历史未重汇总

原始末端满足 stored `topology_trusted`、`spectral_status=0`、`phase!=Normal` 的记录为 4731：3265 个 $\nu=0$、1466 个 $\nu=1$。
同一个末端按 FFLO 筛选为 4590：3145/1445。额外 uniform-SC 的 141 行为 120/21，坐标和索引另存 CSV。不是仅凭数字差相等推断身份。

建议保留 S5 原历史，caption 说明 a/c 的在线 SC 口径与正文 FFLO 子集不同；正文范围不扩大。整段逐轮记录的资格与最后一行身份是不同验收层，本轮只核对末端。若希望 FFLO-only 历史，需另审逐轮数据完整性，未默认齐全。等待选择，未改历史数字或轴名。

## R13：身份与批准

待作者确认：是否有意匿名、作者/单位、资助、目标期刊、公开存档方式、AI 使用声明。
真实可用：本地 v7/v8 LaTeX 源包、输入清单、导出的 NPZ/CSV 与存档代码；没有已确认的公共下载 URL/DOI。
历史 Fig.1–3 来源批准不等于 v8 图版批准；Fig.2 本轮为新候选；Fig.4 当前 v2 不可自动套旧 Fig.4 批准；SM 图版也未自动批准。

## 方法与书目仍待核实

- R04：逐次 oracle 命令/环境覆盖、拓扑细化与 Fig.3 endpoint 的调用绑定、零温历史处理未全部恢复。
- R07：三维监测器的完整实现/版本绑定、每次实际退出原因未全部核对。
- R14：本地条目是 `Yean-an Liao`，排版显示 `Y. an Liao`；没有可靠原始书目条目支撑修正分词，本轮保持，不猜姓名。

这些不是自动科学通过项。A=文本清晰化，B=所列实现片段核对，C=未做物理或端到端复现。
