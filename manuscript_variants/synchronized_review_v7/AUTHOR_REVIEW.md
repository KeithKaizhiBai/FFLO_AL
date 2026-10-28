# 待作者确认（2026-09-11）

本轮没有改动 Hamiltonian、自由能目标、数据标签或物理计算。下面前三项没有静默应用；其原句仍在候选稿中，不能将本轮排版通过视为科学认可。

| 原句/当前表达 | 建议改句 | 理由 | 声明范围及处理 |
|---|---|---|---|
| “a change of index requires a bulk gap closing and reopening, not a change in basin depth…” | “Along a continuous gapped BdG branch, changing the invariant requires loss of the gap; a discontinuous switch of the minimizing branch must be distinguished.” | 连续谱变形与一级极小值切换不同；当前采样未排除后一种机制。 | 涉及机制；**未应用，待确认**。正文第 2 页。 |
| “Their meeting region follows a curved locus where the bulk gap closes before reopening…” / “sampled close-and-reopen structure” | “The opposite-index supports delimit a transition region; the present sampling does not by itself establish continuous gap closing along the global minimum.” | 离散标签和重构线不能独立证明沿全局极小值连续闭隙。 | 收窄机制声明；**未应用，待确认**。正文第 3–4 页。 |
| “FFLO order”作为全部有限动量变分空间的统称 | 首次说明 “Within the single-q pairing ansatz studied here, finite-q solutions are denoted FFLO.” | 代码只搜索单一配对动量；不代表完整多-q/LO空间。 | 模型范围澄清，建议作者确认全文口径；**未新增解释**。 |
| “exact many-body-inspired numerical objectives” | “direct BdG numerical objectives” | canonical MODEL_SPEC 已明确是平均场 BdG；exact不能暗示精确多体解。 | 与已定义模型一致的术语整理；已应用。 |
| “Topology-inapplicable includes all non-FFLO…” | “The topology analysis in this study is restricted to reliable gapped FFLO states; excluded uniform-SC states are outside this analysis, not universally topologically inapplicable.” | 研究掩膜不是普适拓扑禁区。 | 适用范围解释；**待确认**，没有扩大uniform-SC结果。 |
| “four exact fixed-mu cuts” | “four narrow-mu bins of exact observations” | 已有图表窗口中心为 −0.45、0.15、0.85、1.45，半宽0.025；不是严格固定参数重算。 | 不改变数据/科学范围；已应用，并加入SM计数表。 |
| Fig.1 示意图可能读作实算曲面 | 图注明确“Schematic illustration, not computed phase data” | 资产合同的schematic属性已确认。 | 不改变科学声明；已应用。 |

## 未完成信息

- 署名、排序、单位、通讯作者：未找到已确认名单。两份文档统一不放虚构署名，不再显示工作流占位句。正式署名仍未完成。
- 自由能问题：完全暂缓；保留原数学表达和说明，不宣称正确性已验证或误差可忽略。
- 科学复现：Stage I/II缺少逐次运行的commit/可执行文件hash绑定；Stage IV仍需核对实际作业override及完整环境。配置中的值不等于已证明bitwise复现。
- 总执行次数：selected、unique selected、retained三类计数不能替代包含失败和重算的完整执行账本。
- 数据发布、永久标识符、资助和贡献说明未确认。没有添加虚构仓库/DOI或“按需提供”的未经授权承诺。
- 图中细小标注：Fig.3右侧诊断刻度、Fig.4部分图例/小轴仍偏小。原图不变；后续若要求全部普通标注达到7–9 pt，需单独批准原生版面调整或科学面板拆分，不能再整体缩到一列。

## AI 使用披露建议（待作者确认后放入论文）

建议英文底稿：

> OpenAI Codex assisted with manuscript drafting and editing, source/configuration cross-checks, figure-generation code, and document assembly. The authors directed these tasks and reviewed the scientific claims, source provenance, and resulting figures. The model/tool versions and the scope of research-code assistance will be specified from the retained usage records before submission.

最后一句是待补信息，不应原样作为最终投稿披露。需核实实际工具版本，以及作者已经完成哪些检查；本轮不能替作者宣称完成全部科学核验。涉及图生成的帮助还应在图注交代；研究计算/分析中的AI帮助应放在相应方法说明；其他实质写作帮助放致谢。这是当前[APS AI政策](https://journals.aps.org/authors/appropriate-use-ai-tools)的分项披露要求，不是只声明“语言润色”即可覆盖本项目的工作（访问2026-09-11）。
