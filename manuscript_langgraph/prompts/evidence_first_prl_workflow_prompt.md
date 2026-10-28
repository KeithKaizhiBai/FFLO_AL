你正在接续 Active Learning FFLO / topology phase-diagram 项目。

QAHE3D 已经经过审计，确认其中不存在一套覆盖 Introduction → Results →
Discussion → LaTeX integration 的完整 PRL LangGraph。

QAHE3D 中可复用的只有：

1. intro_citation_loop：
   Introduction claim map、文献检索、claim-citation 映射、
   citation placement audit、paragraph blueprint 和 Introduction draft；

2. agents governance layer：
   角色、文件所有权、work order、handoff 和 approval gates；

3. PRL prompts：
   作为叙事、风格和图件规划参考，不是 executable graph。

因此本任务不是复制 QAHE3D 整包，也不得把以上三项拼接后称为完整论文模板。

目标：

在 Active Learning 项目内，受控地建立一套 evidence-first PRL manuscript
workflow，并首先完成 manuscript planning 和一个 Results-by-figure vertical slice。

==================================================
一、严格隔离
==================================================

不得修改 QAHE3D。

在 Active Learning 项目中新建独立目录，例如：

    manuscript_langgraph/

如需复用 QAHE3D 文件：

    复制到 namespaced vendor/adapted 目录；
    保留 UPSTREAM_PROVENANCE.md；
    记录原文件路径、复制日期、commit/hash；
    不做双向同步；
    不覆盖 QAHE3D。

==================================================
二、先审计，不要立即写全文
==================================================

审查 Active Learning 项目中的：

    Stage I–VI reports
    frozen datasets
    convergence audits
    figure outputs
    tables
    run manifests
    PROJECT_SUMMARY
    HPC return packages
    existing manuscript notes

输出：

    manuscript_artifact_manifest.csv
    manuscript_provenance_audit.md
    missing_artifact_report.md

每个 artifact 记录：

    stage
    run_id
    path
    hash
    status
    scientific maturity
    usable in main text / supplement / discussion / not usable

==================================================
三、提出并冻结论文 scope
==================================================

优先考虑如下论文主线：

    Cold-start active learning reconstructs thermodynamic and topological
    FFLO phase diagrams using a numerically audited BdG oracle.

推荐 scope：

    Stage I:
        2D thermodynamic discovery

    Stage II:
        oracle reliability and numerical methodology

    Stage III:
        formally converged 2D cFFLO/tFFLO topology boundary

    Stage IV:
        optional 3D extension demonstration, with explicit caveats

    Stage V–VI:
        Discussion / Outlook only unless independently supported

生成：

    MANUSCRIPT_SCOPE.md
    alternative_scope_options.md
    main_text_inclusion_matrix.csv

不要默认所有 stages 都进入主文。

设置 human gate：

    SCOPE_APPROVAL_REQUIRED

在 scope 未批准前，不生成完整 manuscript。

==================================================
四、Publication-readiness audit
==================================================

检查：

    multi-seed benchmark
    random/Sobol matched-budget baseline
    hidden-ground-truth evaluator
    final raw-data figure reproducibility
    all main claims supported by frozen runs
    cross-run numerical consistency
    unresolved convergence caveats

输出：

    publication_readiness_report.md
    publication_gap_matrix.csv

每个 gap 标记：

    blocks_main_claim
    supplement_only
    discussion_only
    optional_improvement

不得因为缺少某项就伪造结果或静默降低标准。

==================================================
五、Claim registry
==================================================

建立：

    claim_registry.yaml

每个 claim 包含：

    claim_id
    exact statement
    scientific status:
        frozen_supported
        supported_with_caveat
        provisional
        not_established
        negative_result
    supporting evidence ids
    allowed wording
    forbidden wording
    limitations
    destination:
        main text
        supplement
        discussion
        outlook
    human approval state

任何 agent 不得自行改变 claim status。

初始至少建立：

    2D thermodynamic reconstruction claim
    exact-oracle reliability claim
    2D topology convergence claim
    3D extension claim
    learned-acquisition claim
    ground-up world-model claim

其中 Stage V/VI claim 应根据实际审计状态严格限制。

==================================================
六、Evidence registry
==================================================

建立：

    evidence_registry.yaml

每个 evidence 包含：

    evidence_id
    run_id
    dataset
    report
    table / figure source
    metric
    provenance
    scientific status
    limitations
    claims supported

==================================================
七、Figure registry
==================================================

提出四张主图方案并建立：

    figure_registry.yaml

建议：

    Figure 1:
        physics problem + exact oracle + AL loop

    Figure 2:
        2D thermodynamic discovery and convergence

    Figure 3:
        topology-aware cFFLO/tFFLO discovery and convergence

    Figure 4:
        matched-budget/multi-seed benchmark if available;
        otherwise carefully caveated 3D mu extension

每张图定义：

    scientific role
    source datasets
    panels
    claims supported
    required caveats
    caption contract
    prohibited claims
    redraw requirements

输出：

    proposed_main_figures.md
    figure_evidence_matrix.csv

==================================================
八、建立新 LangGraph 设计
==================================================

设计以下 nodes/subgraphs：

    artifact_provenance_audit
    publication_readiness_gate
    claim_freeze
    figure_evidence_contract
    results_by_figure_subgraph
    methods_model_subgraph
    discussion_scope_guard
    intro_citation_loop
    abstract_title_generator
    cross_section_consistency_audit
    citation_audit
    prl_length_claim_gate
    latex_integration
    final_human_approval

注意：

论文阅读顺序不等于生成顺序。

生成顺序必须是：

    evidence/claims
    → Results
    → Methods
    → Discussion
    → Introduction
    → Abstract/Title
    → audits
    → LaTeX

Introduction citation loop 必须在 Results claims 已冻结后运行。

输出：

    graph_design.md
    state_schema.py or state_schema.yaml
    node_contracts/
    langgraph.json draft

==================================================
九、Results-by-figure subgraph
==================================================

这是新 graph 的核心。

每张图执行：

    load figure contract
    inspect raw evidence
    extract observations
    separate observation from interpretation
    map observations to frozen claims
    draft caption
    draft Results paragraphs
    run overclaim audit
    run evidence trace audit
    output approval package

每个 Results paragraph 必须附带机器可读 metadata：

    paragraph_id
    claims_used
    evidence_used
    figure panels cited
    caveats
    unsupported sentence flags

==================================================
十、Vertical slice
==================================================

不要立即实现整个 manuscript。

首先选择 Stage III topology result 作为 vertical slice。

完成：

    frozen topology claim
    Figure contract
    evidence extraction
    caption
    Results paragraph(s)
    cross-section consistency stub
    LaTeX insertion
    audit report

输出：

    vertical_slice_stageIII/
        claim_contract.yaml
        figure_contract.yaml
        evidence_extract.md
        caption_candidate.md
        results_candidate.md
        audit_report.md
        integrated_fragment.tex

设置 human gate：

    VERTICAL_SLICE_APPROVAL_REQUIRED

只有该 vertical slice 通过后，才扩展其余 Figures 和 sections。

==================================================
十一、QAHE3D assets 的适配
==================================================

适配并 namespaced 复用：

    intro_citation_loop

但不得直接运行到完整论文。

输入必须来自：

    approved manuscript scope
    frozen contribution claims
    Results-derived necessary background claims

多代理 governance 层只用于：

    role assignment
    file ownership
    work order
    handoff
    approval gates

PRL prompts 只提取为：

    journal_style_contract.yaml
    narrative_rules.yaml
    figure_rules.yaml

不得作为事实来源。

==================================================
十二、最终输出
==================================================

本轮只需交付：

    manuscript artifact/provenance audit
    scope proposal
    publication-readiness audit
    claim registry
    evidence registry
    figure registry
    graph design
    state schema
    node contracts
    Stage III vertical slice
    implementation plan for full graph

不得在本轮生成一份貌似完整但未通过 evidence gates 的 PRL 稿件。

==================================================
十三、最终报告必须回答
==================================================

1. 当前最适合的一篇论文主线是什么？
2. 哪些 stages 可以进入主文？
3. 哪些只能进入 Supplement / Discussion / Outlook？
4. 还缺哪些 publication-critical evidence？
5. 四张主图应如何组织？
6. QAHE3D 哪些代码可以真正复用？
7. 哪些必须新建？
8. Stage III vertical slice 是否成功端到端运行？
9. 是否可以进入完整 LangGraph 实现？
10. 是否存在需要用户先做科学决策的问题？

开始时先输出：

    repository audit
    proposed manuscript scope
    proposed figure plan
    publication gap summary
    implementation plan

只有遇到会改变论文中央 claim、需要新增科学计算、
或可能覆盖 frozen artifact 的关键问题时才暂停。
