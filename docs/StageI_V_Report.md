你正在接续 BdG / FFLO / topology-aware active-learning 项目。现在的任务不是继续跑 exact calculation，也不是修改 active-learning 代码，而是整理 Stage I–Stage V 的完整项目综合报告，并重绘一套清晰、易读、可用于组会/论文补充材料/项目归档的 PDF 图。

请先审查当前 repository、所有 stage 的报告、datasets、figures、logs、README、PROJECT_SUMMARY、run manifests、HPC package outputs 和已有 PDF，然后自行制定具体实施计划，完成数据收集、图形重绘、LaTeX/Markdown 报告生成、PDF 编译和最终交付。

本 prompt 规定报告目标、章节结构、图形原则和交付要求。具体文件路径、读取方式、绘图代码、LaTeX 模板和图形排版由你根据当前代码库决定。优先复用现有数据和报告，不要伪造不存在的结果。

==================================================
一、任务目标
==================================================

生成一份综合 PDF 报告：

    Stage I–V Active-Learning FFLO / Topology Project Report

核心目标：

1. 梳理 Stage I 到 Stage V 的科学目标、技术路线和结论；
2. 汇总各阶段得到的相图结果；
3. 重画各阶段的 phase diagram / topology diagram；
4. 重画机器学习流程图、exact oracle 流程图、acquisition function 设计图；
5. 用统一视觉风格表达项目从 2D thermodynamic phase learning
   到 topology-aware 2D，再到 3D high-dimensional AL，
   再到 Stage V acquisition-learning 的演进；
6. 清楚标注每一阶段哪些结论已正式通过，哪些只是 pilot/prototype；
7. 生成一份完整 PDF 和所有独立 figure PDF/PNG/SVG 文件。

本任务禁止：

    新增 exact calculation
    继续 active-learning loop
    修改 frozen datasets
    修改 phase/topology labels
    合并不同 provenance 的数据
    将 report-only diagnostic surfaces 写成 publication-grade final boundaries
    夸大 Stage IV / Stage V 的 convergence status

==================================================
二、必须首先做 repository / artifact audit
==================================================

开始时先输出一个简短 audit，列出：

    Stage I artifacts found
    Stage II artifacts found
    Stage III artifacts found
    Stage IV artifacts found
    Stage V artifacts found

每个 stage 至少记录：

    run_id
    final dataset path
    report path
    key tables path
    key figures path
    convergence / decision status
    missing artifacts
    whether results are final / pilot / provisional

如果某个 stage 的 artifact 不完整，不要猜测。
在报告中写：

    artifact missing / not found
    conclusion limited to available files

尤其注意：

    Stage V 当前本地报告可能只到 dataset_iter093，
    若 dataset_iter100 或 final merged dataset 不存在，
    Stage V 只能写作 local available return / prototype analysis，
    不得写成 final convergence result。

==================================================
三、建议 Stage 划分
==================================================

如果 repository 中已有正式 stage 命名，以 repository 为准。
若没有，则按以下工作定义组织：

--------------------------------------------------
Stage I:
    2D thermodynamic active learning baseline
--------------------------------------------------

主题：

    在 (kBT/t, J_A/t) 平面上学习：
        normal
        uniform-SC
        FFLO

重点表达：

    从随机 / Sobol 初始点开始；
    用 exact oracle 得到 thermodynamic labels；
    训练 surrogate；
    acquisition 从粗采样逐步转向 phase boundaries。

需要图：

    2D thermodynamic phase map
    normal/SC boundary
    uniform/FFLO boundary
    selected points by iteration
    convergence metrics if available

--------------------------------------------------
Stage II:
    exact-oracle numerical reliability and optimization
--------------------------------------------------

主题：

    解决 q-window、Delta refinement、local refinement 成本和 label closure 问题。

重点表达：

    robust incremental q-window expansion
    near-zero Delta refinement
    basin-level local refinement
    rank-and-cap K3
    distinction:
        q_not_applicable
        q_unresolved
        delta_unresolved
        trusted / hard-risk

需要图：

    exact oracle flowchart
    q-window expansion schematic
    local-basin refinement schematic
    K3 rank-and-cap schematic
    before/after runtime / local-box reduction if data exists
    hard-risk frontier explanation

--------------------------------------------------
Stage III:
    2D topology-aware cold-start full loop
--------------------------------------------------

主题：

    从头随机撒点，在 2D 中同时学习：
        normal / SC
        uniform / FFLO
        cFFLO / tFFLO

重点表达：

    Pfaffian Z2:
        P0
        Ppi
        P0*Ppi sign
    bulk-gap check
    no finite-area nodal-SC in current parameter set
    cFFLO/tFFLO topology boundary formal convergence

需要图：

    final 2D topology-aware phase map
    cFFLO/tFFLO contour
    topology convergence audit summary
    final-five-contours overlay
    bracket support map
    Stage III decision panel:
        Decision A
        topology_main_converged = True
        need_new_exact_calculation = False

--------------------------------------------------
Stage IV:
    3D high-dimensional topology-aware active learning
--------------------------------------------------

主题：

    扩展到 3D:
        (kBT/t, J_A/t, mu/t)
        fixed U, fixed t=1

重点表达：

    3D cold-start AL 已发现 tFFLO / cFFLO 结构；
    tFFLO 与 normal-state single-band corridor 高度相关；
    lower-mu edge range-limited；
    formal 3D convergence 尚未通过；
    high-mu slice anomaly 属于 broad-bin projection / curve-extraction artifact。

需要图：

    3D thermodynamic point cloud
    3D topology point cloud
    fixed-mu slice atlas
    single-band corridor overlap plot
    mu-edge contact plot
    convergence-failure summary
    curve-extraction fix comparison:
        old smooth curve
        support-restricted curve
        removed unsupported segments

--------------------------------------------------
Stage V:
    acquisition-function improvement and learned residual prototype
--------------------------------------------------

主题：

    改进 acquisition function：
        boundary-support base acquisition
        online learned residual
        micro-batch selection

当前结果：

    learned residual machinery runs;
    lambda_t grows;
    learned reward rank correlation exceeds A0;
    phase-map proxy stabilizes;
    but scalar reward appears dominated by normal/SC and global Sobol proposals;
    topology/P0/Ppi channels remain weak;
    tFFLO support is under-sampled;
    Stage V-v1/v2 is a method prototype, not final convergence.

需要图：

    Stage V architecture diagram
    three learning modules:
        phase / label surrogate
        physical-field surrogate
        acquisition-value learner
    A0 + learned residual formula flow
    micro-batch loop
    lambda_t and rank-correlation curves
    selected channel focus
    candidate-source mixture
    topology count growth
    phase-map proxy
    failure-mode diagnosis:
        scalar reward dominated by normal/SC
        proposal-source bias
        P0/Ppi weak

==================================================
四、报告主线
==================================================

报告不要写成简单日志堆叠。
要形成一条清晰主线：

    1. We start from a 2D thermodynamic phase diagram.
    2. We stabilize the exact oracle and convergence semantics.
    3. We add topology and demonstrate cold-start discovery of cFFLO/tFFLO.
    4. We lift the method to 3D parameter space with experimentally relevant mu.
    5. We identify limitations of high-dimensional AL:
           sparse boundary support,
           projection artifacts,
           topology surprise,
           acquisition imbalance.
    6. We develop Stage V acquisition-learning:
           boundary-support acquisition + learned residual + micro-batches.
    7. Stage V proves the machinery can learn,
       but also reveals that scalar reward must be replaced by per-boundary multi-head learning.

最终报告要表达：

    project achieved:
        2D thermodynamic convergence
        exact-oracle reliability
        2D topology boundary convergence
        3D tFFLO discovery
        acquisition-learning prototype

    project not yet fully achieved:
        formal 3D convergence
        Stage V learned acquisition outperforming all baselines
        open-set new-phase discovery

==================================================
五、必须重绘的核心图
==================================================

生成一套统一风格的 figures。
每张图输出：

    PDF
    PNG
    SVG if possible

统一要求：

    consistent fonts
    readable labels
    large enough markers
    no overcrowded legends
    no tiny captions
    colorblind-friendly palette
    avoid raw rainbow unless physically necessary
    all axes labelled with dimensionless units:
        kBT/t
        J_A/t
        mu/t
    report-only diagnostics clearly marked

--------------------------------------------------
Figure 1: Project roadmap
--------------------------------------------------

内容：

    Stage I -> Stage II -> Stage III -> Stage IV -> Stage V

每个 stage 用一个 box：

    goal
    method
    output
    status

状态颜色：

    completed / passed
    pilot / not converged
    prototype / needs v2

--------------------------------------------------
Figure 2: Exact oracle pipeline
--------------------------------------------------

流程：

    candidate point
        -> Delta-q free-energy search
        -> robust q-window expansion
        -> near-zero Delta refinement
        -> basin clustering
        -> rank-and-cap K3
        -> thermodynamic label
        -> topology oracle for SC
        -> reliability flags
        -> training append

必须显示：

    topology does not invalidate thermodynamic label
    q_not_applicable != q_unresolved
    nodal != unresolved

--------------------------------------------------
Figure 3: Hierarchical label system
--------------------------------------------------

画成树：

    exact point
        -> thermodynamic phase:
             normal / uniform-SC / FFLO
        -> if SC:
             spectral status:
                 gapped / nodal / unresolved
        -> if trusted gapped SC:
             topology:
                 trivial / topological

标注：

    normal:
        topology not_applicable

    nodal:
        Z2 not_defined

--------------------------------------------------
Figure 4: Active-learning loop
--------------------------------------------------

流程：

    initial Sobol design
        -> exact oracle
        -> dataset
        -> surrogate training
        -> acquisition scoring
        -> selected points
        -> HPC exact evaluation
        -> append / checkpoint
        -> StopController

分出：

    Stage I–III original loop
    Stage V micro-batch loop

--------------------------------------------------
Figure 5: Stage III 2D topology result
--------------------------------------------------

重画最终 2D map：

    normal
    uniform-SC
    cFFLO
    tFFLO
    cFFLO/tFFLO contour
    normal/SC contour
    uniform/FFLO contour
    exact points
    bracket support if useful

不要画 eta 作为主背景。

--------------------------------------------------
Figure 6: Stage III convergence audit
--------------------------------------------------

可组合成 2x2 panel：

    topology map change
    boundary shift p95
    boundary coverage p95
    trusted topology surprise
    final decision panel

--------------------------------------------------
Figure 7: Stage IV 3D thermodynamic and topology maps
--------------------------------------------------

显示：

    3D thermodynamic point cloud
    3D topology-aware SC point cloud
    local crossing markers
    transparent diagnostic surfaces only if readable

图注必须写：

    diagnostic surfaces are not final publication boundaries.

--------------------------------------------------
Figure 8: Stage IV fixed-mu slice atlas
--------------------------------------------------

使用 curve-extraction fix 后的窄 fixed-mu atlas：

    exact labels
    local brackets
    no unsupported smooth curve

不要使用旧 wide-bin smooth curve 作为主图。

--------------------------------------------------
Figure 9: Stage IV single-band corridor relation
--------------------------------------------------

显示：

    normal-state single-pair corridor in (mu/t, J_A/t)
    cFFLO exact points
    tFFLO exact points
    boundary proxy
    lower-mu edge band

标注：

    single-band corridor is diagnostic, not a topology label.

--------------------------------------------------
Figure 10: Stage IV convergence limitation summary
--------------------------------------------------

显示：

    topology volume-map change
    surface shift
    coverage
    trusted topology surprise
    component count
    mu-edge limitation

结论：

    tFFLO discovered
    formal 3D convergence not passed
    lower-mu range-limited

--------------------------------------------------
Figure 11: Stage V architecture
--------------------------------------------------

画三套学习模块：

    Module 1:
        phase / label surrogate

    Module 2:
        physical-field surrogate

    Module 3:
        acquisition-value learner

并显示：

    exact outputs train Module 1 and 2
    AL rewards train Module 3
    Module 3 modifies acquisition score

公式：

    A(x) = A0(x) exp(lambda_t g_theta(phi(x)))

或 Stage V-v2 proposed formula：

    A_s(x) = A0_s(x) exp(lambda_s g_s(phi_s(x)))

--------------------------------------------------
Figure 12: Boundary-support acquisition principle
--------------------------------------------------

画一个概念图：

    predicted boundary surface
    existing exact brackets
    dense boundary patch
    sparse boundary patch
    candidate selected near sparse boundary patch

标注：

    boundary likelihood
    uncertainty
    support fill-distance

公式：

    A0_s = B_s * U_s * H_s

--------------------------------------------------
Figure 13: Stage V v1 learning diagnostics
--------------------------------------------------

重画：

    lambda_t
    learned rank correlation vs A0
    selected channel focus
    selected candidate-source mixture

并加文字 annotation：

    learned residual trains successfully
    but scalar reward over-focuses normal/SC
    P0/Ppi topology channels remain weak
    global Sobol dominates

--------------------------------------------------
Figure 14: Stage V-v2 proposed improvement
--------------------------------------------------

设计图：

    scalar value learner
        -> multi-head per-boundary learner

显示：

    g_NS
    g_UF
    g_P0
    g_Ppi
    g_gap

每个 head 有：

    reward_s
    lambda_s
    alpha_s
    support metrics_s

这张图是未来工作 / next implementation design。

--------------------------------------------------
Figure 15: Summary table
--------------------------------------------------

一个横向表格：

    Stage
    Dimension
    Parameters
    Main target
    Exact samples
    Main output
    Status
    Next issue

==================================================
六、图形风格细则
==================================================

1. 相图配色建议：

    normal:
        light gray

    uniform-SC:
        blue

    FFLO / cFFLO:
        orange or teal

    tFFLO:
        magenta / red-purple

    boundary markers:
        black

    hard-risk / unresolved:
        dark gray or black cross

2. 不要使用过度透明导致看不清的 3D scatter。
   如果 3D scatter 太拥挤，提供两个 view：

        oblique 3D view
        primary projected view

3. 对 3D 结果，必须提供 fixed-mu slice atlas。
   3D 图只作为直观展示，slice atlas 才是主要科学图。

4. 不要让 figure caption 只显示文件名。
   每个 caption 必须说明科学含义。

5. 所有 report-only fitted curves 都要明确标注：

        diagnostic only

6. 对 missing artifact，图中不要空白，
   可以用 placeholder panel：

        artifact missing / not available

==================================================
七、报告章节结构
==================================================

建议 PDF 报告结构：

    Title page

    Abstract / Executive Summary

    1. Project motivation
        FFLO, topology, active learning, high-dimensional phase diagrams

    2. Physical problem and labels
        free-energy landscape
        thermodynamic phases
        Pfaffian Z2
        bulk-gap status
        reliability flags

    3. Active-learning framework
        exact oracle
        surrogate models
        acquisition
        StopController
        HPC workflow

    4. Stage I: 2D thermodynamic AL
        goal
        method
        result
        limitation

    5. Stage II: numerical reliability and oracle optimization
        q-window
        Delta refinement
        K3
        hard-risk frontier

    6. Stage III: 2D topology-aware AL
        cold-start
        cFFLO/tFFLO result
        convergence Decision A

    7. Stage IV: 3D extension in (T, J_A, mu)
        3D phase/topology result
        single-band corridor
        lower-mu limitation
        curve-extraction fix
        not formally converged

    8. Stage V: acquisition-function learning
        boundary-support A0
        learned residual
        micro-batch selection
        v1 result
        failure diagnosis
        v2 design

    9. Lessons learned
        what worked
        what failed
        what must be improved

    10. Next steps
        Stage V-v2 multi-head learning
        possible mu-window expansion
        future open-set novelty discovery
        possible 4D extension

    Appendix A: artifact table
    Appendix B: metric definitions
    Appendix C: exact-oracle parameters
    Appendix D: figure provenance

==================================================
八、必须清楚区分 status
==================================================

报告中必须使用明确 status language：

    completed / formally converged:
        Stage III 2D topology boundary

    completed as pilot / not formally converged:
        Stage IV-A 3D

    prototype / method test:
        Stage V-v1

    proposed:
        Stage V-v2 multi-head acquisition
        open-set novelty discovery
        4D extension

不得写：

    Stage IV-A already converged
    Stage V learned acquisition already outperforms baseline
    Stage V final topology map is publication-grade

除非 repository 中存在新的 formal convergence audit 支持这些结论。

==================================================
九、数据与 provenance 要求
==================================================

每张 figure 必须能追溯到：

    input dataset
    run_id
    script
    output filename
    generation timestamp

生成：

    figure_manifest.csv

字段：

    figure_id
    figure_title
    source_stage
    source_dataset
    source_report
    script_path
    output_pdf
    output_png
    caveat

生成：

    stage_artifact_table.csv

字段：

    stage
    run_id
    final_dataset
    report
    exact_samples
    status
    missing_items
    notes

==================================================
十、实现方式建议
==================================================

推荐使用：

    Python matplotlib for plots
    networkx or graphviz for flow diagrams
    LaTeX / Markdown -> PDF for final report

如果使用 LaTeX：

    use stable fonts available in environment
    avoid custom font dependency
    compile with latexmk if available
    otherwise use xelatex/pdflatex consistently

If Graphviz unavailable:

    draw flowcharts using matplotlib patches
    or generate Mermaid markdown plus rendered fallback if available

All generated PDF figures should be vector where possible.

==================================================
十一、质量检查
==================================================

Before final delivery, perform:

    Python script syntax check
    figure files exist check
    PDF compilation check
    missing references check
    broken image link check
    table path check
    run_id/provenance consistency check
    spelling consistency:
        cFFLO / tFFLO
        uniform-SC
        normal/SC
        Pfaffian
        topology-aware
    no file-name-only captions
    no unsupported convergence claims

==================================================
十二、输出目录
==================================================

Create a clean output directory, for example:

    reports/stageI_to_stageV_comprehensive_report/

Inside:

    main_report.pdf
    main_report.tex or main_report.md
    figures/
        fig01_project_roadmap.pdf/png/svg
        ...
    tables/
        stage_artifact_table.csv
        figure_manifest.csv
    scripts/
        generate_figures.py
        generate_report.py
    README.md
    build_log.txt

Do not overwrite old reports.

==================================================
十三、最终交付
==================================================

Final Codex response must include:

1. where the report was written;
2. path to main_report.pdf;
3. list of generated figures;
4. changed-files manifest;
5. build log summary;
6. missing artifact list;
7. caveats;
8. suggested next edit pass.

==================================================
十四、报告语气
==================================================

报告应当清晰、克制、科学。

重点不是把所有结果说成成功，而是准确表达：

    Stage I–III:
        method and physics were progressively stabilized.

    Stage IV:
        high-dimensional extension discovered meaningful 3D topology structure,
        but formal convergence and domain closure remain open.

    Stage V:
        learned acquisition framework begins to work,
        but v1 learns an imbalanced scalar reward;
        v2 should use per-boundary multi-head acquisition learning.

==================================================
十五、开始执行
==================================================

Start by printing:

    repository/artifact audit
    proposed report outline
    proposed figure list
    missing artifacts

Then implement figure regeneration and report writing.

Do not wait for step-by-step approval unless:

    a required source dataset is missing,
    stage identity is ambiguous,
    or a figure would require making an unsupported scientific claim.