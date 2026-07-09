Stage V:
    acquisition-function improvement and active-learning method stage

Core:
    boundary-support base acquisition
    + online learned residual
    + micro-batch selection

接续 BdG topology-aware active-learning 项目，开启 Stage V。

Stage V 的目标是开发、实现并验证下一代 acquisition function：

    boundary-support base acquisition
    + online learned residual
    + micro-batch active learning

请先审查当前 repository、Stage III/IV frozen outputs、Stage IV-A acquisition logs、
surrogate/model code、exact oracle、topology oracle、StopController、checkpoint/resume
机制和现有 HPC 打包方式，然后自行制定具体实施计划、完成代码修改、测试、
smoke run、HPC package 和最终报告。

我们期待交付可以直接上超算进行full-loop计算的超算包。

本 prompt 规定科学目标、模块边界、不可变物理语义和交付要求。
具体类、函数、模型架构、配置参数、日志格式和优化细节由你根据现有代码库决定。
优先复用现有可靠实现，不要平行重写 Hamiltonian 或 exact oracle。

==================================================
一、Stage V 定义
==================================================

Stage V 是 acquisition-function and active-learning method stage。

目标不是修补 Stage IV-A 的某个图形问题，也不是 fixed-mu 2D AL。
目标是建立一套可以从 cold start 中成长的 acquisition 机制，使 active learning
自动将 exact calculations 投向：

    sparse thermodynamic boundary support
    sparse topology boundary support
    high-surprise boundary regions
    high-shift boundary regions
    possible component-instability regions
    domain-edge / physical-corridor relevant regions

核心设计：

    1. physical-field surrogate as implicit boundary model
    2. boundary-support base acquisition A0
    3. online acquisition-value learner g_theta
    4. learned residual multiplier
    5. stochastic micro-batch selection
    6. automatic trust schedule for the learned component

建议 run_id：

    stagev_acqv2_boundary_support_learned_residual_3d_v1

Stage V first target domain:

    3D:
        x = (kBT/t, J_A/t, mu/t)

Default domain:

    use Stage IV-A domain:
        mu/t in [-0.5, 1.5]
        kBT/t and J_A/t read from Stage IV-A production config
        U fixed to Stage IV-A production U0
        t = 1

Optional config profile:

    lower-mu extension:
        mu/t in [-1.0, 1.5]

But do not silently change domain in the default run.

==================================================
二、严格冷启动与 provenance
==================================================

Stage V production run must be cold-start.

Do not use these as Stage V training initialization:

    Stage III dataset_iter018 / dataset_iter035
    Stage IV-A dataset_iter025
    Stage IV-A surrogate checkpoints
    Stage IV-A selected point coordinates
    Stage IV-A exact labels

These artifacts may be used only for:

    regression tests
    offline replay diagnostics
    report comparison
    validation of code paths
    method benchmarking after Stage V run

They must not enter Stage V initial design or model training.

Initial design:

    1024 scrambled Sobol exact points

Micro-batch size:

    default = 64
    allowed config alternatives = 128

Maximum exact budget should be configurable.

For same-budget comparison with Stage IV-A, provide a config close to:

    1024 + 96 * 64 = 7168 selected points

or equivalent 128-point micro-batch budget.

Do not hard-code these values outside config.

==================================================
三、冻结物理语义
==================================================

Do not change thermodynamic phase criterion:

    if any Delta > 0 state has free energy lower than normal:
        superconducting

    superconducting:
        q_opt distinguishes uniform-SC and FFLO

    normal:
        q_not_applicable, not q_unresolved

Continue to separate:

    thermodynamic phase reliability
    q-window / q-grid reliability
    spectral-gap reliability
    topology reliability
    response-side eta / Ic reliability

Stage V must not use eta / Ic response diagnostics in the main acquisition.

Topology:

    For trusted gapped SC:
        P0 * Ppi < 0 -> topological
        P0 * Ppi > 0 -> trivial

    For nodal SC:
        Z2 not_defined

    For normal:
        topology not_applicable

Do not assume topology exists only in FFLO after mu variation.
Spectral/topology acquisition must be allowed throughout predicted SC,
with soft physical guidance only.

==================================================
四、三类学习模块
==================================================

Implement Stage V around three learning modules.

They may share an encoder internally, but they must remain logically separated.

--------------------------------------------------
Module 1: phase / label surrogate
--------------------------------------------------

Inputs:

    normalized parameters:
        kBT/t
        J_A/t
        mu/t
        optional U/t if future config extends to 4D

Targets:

    thermodynamic phase:
        normal
        uniform_SC
        FFLO

Optional conditional heads:

    gapped / nodal
    trivial / topological

Use these heads primarily for:

    soft gates
    sanity checks
    label uncertainty
    phase-map diagnostics

Do not rely on the phase classifier alone to define physical boundary location.

--------------------------------------------------
Module 2: physical-field surrogate
--------------------------------------------------

This is the implicit boundary model.

It learns continuous fields from exact-oracle outputs.

Required or strongly preferred targets:

    Delta_opt
    q_opt
    F_SC_minus_F_normal
    P0
    Ppi
    log_bulk_gap
    Pfaffian margin
    bulk-gap margin

If current exact outputs lack F_SC_minus_F_normal,
modify new Stage V exact-output schema to save it.
Do not retroactively modify old datasets.

Define residual fields:

    m_NS:
        F_SC_min - F_normal
        with explicit sign convention in config/report

    m_UF:
        |q_opt| - q_threshold
        or project-consistent uniform/FFLO signed margin

    m_P0:
        P0

    m_Ppi:
        Ppi

    m_gap:
        log(Eg / E_gap_tol)

Boundary surfaces are zero-level sets:

    normal/SC:
        m_NS = 0

    uniform/FFLO:
        m_UF = 0

    trivial/topological:
        P0 = 0 or Ppi = 0

    gapped/nodal:
        m_gap = 0

Normal points must not be used as if q or topology were meaningful.
Use masking and training eligibility rules.

Every field prediction must provide uncertainty estimate:

    ensemble variance
    MC dropout
    bootstrap
    deep ensemble
    or another existing project-consistent method

Choose implementation after code review.

--------------------------------------------------
Module 3: acquisition-value learner
--------------------------------------------------

This is the new acquisition-function learner.

It does not directly learn phase labels.

It learns:

    how useful a candidate point is for improving boundary discovery.

Input features phi(x) should include at least:

    coordinates:
        kBT/t, J_A/t, mu/t

    phase probabilities:
        p_normal
        p_uniform_SC
        p_FFLO
        p_SC

    physical-field residual predictions:
        m_NS, m_UF, P0, Ppi, m_gap

    field uncertainties:
        sigma_NS, sigma_UF, sigma_P0, sigma_Ppi, sigma_gap

    boundary likelihoods:
        B_NS, B_UF, B_P0, B_Ppi, B_gap

    support / geometry:
        nearest exact distance
        local sample density
        boundary fill distances
        distance to local brackets
        bracket density
        local support radius
        estimated boundary curvature if available

    physics diagnostics:
        predicted Delta
        predicted q
        predicted bulk gap
        predicted Pfaffian margin
        single-band corridor indicator
        distance to single-band corridor boundary
        distance to lower/upper mu edge

    run-state features:
        iteration
        micro-batch index
        recent local surprise
        recent local boundary shift
        recent local coverage contribution
        recent component instability indicator

Output:

    predicted acquisition value:
        g_theta(phi)

Stage V final acquisition should be:

    A(x) = A0(x) * exp(lambda_t * g_theta(phi(x)))

where:

    A0:
        boundary-support base acquisition

    g_theta:
        learned residual value predictor

    lambda_t:
        online learned-residual trust coefficient

Initial lambda_t:

    0.0 before enough reward data
    then activate at 0.1
    then adapt automatically based on validation performance

Do not let g_theta fully replace A0 in early stage.

==================================================
五、Boundary-support base acquisition A0
==================================================

For each boundary type s:

    s in:
        normal_SC
        uniform_FFLO
        P0_topology
        Ppi_topology
        gap_nodal

Use residual field:

    m_s(x)

Assume field surrogate gives:

    mean mu_s(x)
    uncertainty sigma_s(x)

Define boundary likelihood:

    B_s(x) = P(|m_s(x)| < tau_s)

Define uncertainty factor:

    U_s(x) = sigma_s / (abs(mu_s) + sigma_s + epsilon)

or an equivalent calibrated form.

Define boundary support sparsity:

    H_s(x)

using distance to existing trusted local boundary-support set.

Boundary-support sets must be built from trusted exact points using local graph rules:

    normal_SC:
        local normal <-> SC brackets

    uniform_FFLO:
        local uniform_SC <-> FFLO brackets

    P0/Ppi topology:
        local opposite-Z2 trusted gapped SC brackets
        and/or local P0/Ppi sign-change brackets

    gap_nodal:
        local gapped <-> nodal brackets if nodal exists
        else small-gap support diagnostics only

Graph:

    mutual kNN or local-scale filtered neighbor graph
    normalized coordinate space
    no raw global Delaunay long edges
    no cross-empty-space long edges

H_s(x):

    high when candidate is near predicted boundary
    but far from existing exact bracket/support points

Example meaning:

    This point lies on a likely boundary patch,
    but this patch is under-supported by exact calculations.

Boundary base score:

    A_s(x) = B_s(x) * U_s(x) * H_s(x)

Combine boundary scores without manual quotas:

    A0(x) = logsumexp_s(log A_s(x) + alpha_s)

where alpha_s is an automatically updated boundary priority,
not a manually fixed quota.

alpha_s should respond to:

    recent trusted surprise for boundary s
    recent surface coverage deficit for boundary s
    recent surface shift for boundary s
    recent component instability for boundary s

but should not be hand-tuned per batch.

If a boundary is absent or not yet supported:

    do not set shift = 0
    use missing_boundary / insufficient_support status
    keep exploration through B_s uncertainty and coverage safeguards

==================================================
六、Online reward definition for acquisition-value learner
==================================================

After each micro-batch exact evaluation, assign reward to selected points.

Reward should quantify usefulness for boundary learning, not simply label class.

Define immediate reward components:

    r_bracket:
        point creates or shortens local opposite-label bracket

    r_surprise:
        previous model predicted wrong label/residual sign with confidence

    r_support:
        point reduces local boundary fill distance / support distance

    r_uncertainty_drop:
        point or micro-batch reduces local ensemble uncertainty on audit cloud

    r_component:
        point helps discover, merge, split, or stabilize a significant boundary/component

    r_margin:
        point lands in physically meaningful small residual / small margin region

Penalties:

    r_redundant:
        point is deep interior and too close to existing points

    r_untrusted:
        exact result not trusted or training-ineligible

    r_numerical_failure:
        timeout, unresolved, invalid output

Reward can be scalar:

    r = w1*r_bracket
      + w2*r_surprise
      + w3*r_support
      + w4*r_uncertainty_drop
      + w5*r_component
      + w6*r_margin
      - penalties

Also store vector rewards for future analysis.

Reward must be computed from information available after the exact point returns.
Do not use final-run future labels to train the online acquisition model.

For offline after-run diagnostics, delayed rewards may be computed separately,
but they must not be used during the live run unless available causally.

==================================================
七、Learning schedule for g_theta
==================================================

Early stage:

    Use A0 only.
    Record candidate features, scores, selected points, exact results, rewards.
    Train g_theta in shadow mode when enough reward samples exist.

Activation:

    When validation ranking performance exceeds A0 baseline,
    activate learned residual with lambda_t = 0.1.

Validation metrics:

    reward prediction rank correlation
    top-k reward enrichment
    pairwise ranking accuracy
    calibration by reward quantile
    local boundary-support improvement prediction

Trust schedule:

    If g_theta improves over A0:
        gradually increase lambda_t

    If g_theta degrades:
        reduce lambda_t

    If g_theta becomes unstable or drives bad sampling:
        lambda_t -> 0
        continue with A0

Bad sampling diagnostics:

    deep-normal selected fraction rises
    trusted exact fraction drops
    boundary coverage worsens
    topology surprise rises without boundary-support gain
    repeated duplicates or overly clustered points
    component fragmentation increases

Keep lambda_t fully logged per micro-batch.

==================================================
八、Stochastic micro-batch selection
==================================================

Replace large frozen 256-point batches with micro-batches.

Default:

    micro_batch_size = 64

Alternative:

    128

Selection method:

    candidate pool generation
    score by A(x)
    stochastic top-k selection using:
        softmax sampling
        Gumbel-top-k
        Thompson sampling
        or equivalent

Record selection probability / propensity for each selected point.

Do not deterministically always take the top raw score.
Some stochasticity is necessary for online learning and bias control.

Candidate generation may use multiple proposal families:

    global Sobol candidate cloud
    local boundary-bracket midpoint proposals
    boundary-neighbourhood jitter proposals
    sparse-surface fill-distance proposals
    single-band corridor proposals
    lower/upper mu-edge guard proposals
    SC-interior coverage proposals

Do not use fixed manual selected-point quotas.
Proposal generation weights may adapt automatically via reward/bandit statistics.

Final selection should be based on unified A(x), diversity control,
and support constraints, not hand-written per-channel quotas.

Diversity control:

    avoid duplicates
    avoid selecting many points within the same local support ball
    but do not over-dilute boundary concentration

Micro-batch loop:

    train surrogates
    generate candidates
    score candidates
    select micro-batch
    run exact oracle
    append results
    compute rewards
    update boundary supports
    update g_theta
    checkpoint
    repeat

==================================================
九、Candidate logging for learning
==================================================

Stage V must log enough data to train and audit acquisition learning.

For every selected point store:

    coordinates
    all Module 1 outputs
    all Module 2 outputs
    A0 score
    g_theta output
    lambda_t
    final A score
    candidate proposal source
    selection probability
    selected rank
    diversity penalties
    local support distances
    exact result
    reward components
    final scalar reward

For candidate pools, storing every candidate may be too expensive.
At minimum store:

    all selected candidates
    top-K unselected candidates per micro-batch
    random background control candidates
    their features and scores

This enables offline bias and acquisition-quality diagnostics.

==================================================
十、Exact oracle integration
==================================================

Exact oracle remains:

    robust incremental q-window expansion
    near-zero Delta refinement
    basin-level local refinement
    rank-and-cap K3

Each exact point must output:

    kBT/t
    J_A/t
    mu/t
    U/t
    thermo_phase
    Delta_opt
    q_opt
    free_energy_opt
    normal_free_energy
    F_SC_minus_F_normal if available
    P0
    Ppi
    pf_product
    bulk_gap
    k_at_bulk_gap
    spectral_status
    topology_label
    z2_value
    trusted_exact
    training_eligible_exact
    rerun_required
    q_unresolved
    delta_unresolved
    spectral_trusted
    topology_trusted
    runtime
    failure_reason

For retained K3 basins, save:

    basin Delta
    basin q
    basin free energy
    basin P0
    basin Ppi
    basin bulk gap
    basin topology
    delta_F_to_ground

Cache keys must include:

    kBT
    J_A
    mu
    U
    t
    all relevant fixed model parameters

==================================================
十一、StopController for Stage V
==================================================

Reuse and extend existing StopController.

Monitor thermodynamic:

    phase-volume map change
    normal/SC surface shift
    uniform/FFLO surface shift
    surface coverage
    trusted phase surprise
    component stability

Monitor topology/spectral:

    trusted Z2 volume-map change
    cFFLO/tFFLO surface shift
    topology surface coverage
    trusted topology surprise
    topology component stability
    nodal diagnostics if any
    unresolved numerical frontier

Monitor Stage V learning:

    mean reward per micro-batch
    reward enrichment over A0
    lambda_t evolution
    fraction of selected points creating brackets
    boundary support p95 improvement
    candidate diversity
    selected deep-interior redundancy rate
    learned-model validation performance

Do not declare convergence solely because a boundary is missing.

Missing boundary status:

    missing_boundary
    insufficient_support
    physically_absent
    not_yet_discovered

must be distinct.

Stage V success criteria should include method improvement:

    compared to Stage IV-A same budget,
    improve surface coverage,
    reduce trusted topology surprise,
    reduce surface shift,
    stabilize component count,
    increase bracket-support density,
    improve reward efficiency.

==================================================
十二、Stage V comparison protocol
==================================================

Stage V should produce a method comparison against Stage IV-A,
without training on Stage IV-A data.

Compare at equal or matched exact budget:

    final samples
    phase counts
    topology counts
    topology volume-map change
    surface shift
    surface coverage
    trusted topology surprise
    component stability
    boundary support density
    acquisition concentration metrics
    reward metrics

Stage IV-A artifacts may be used as frozen comparison reference only.

Also include ablations if feasible without excessive cost:

    A0 only
    A0 + learned residual
    micro-batch vs 256 batch
    with / without support sparsity H_s
    with / without physical-field residuals

If full ablations are too expensive,
implement offline replay / shadow-mode diagnostics first.

==================================================
十三、Implementation phases for Codex
==================================================

Proceed in phases.

--------------------------------------------------
Phase 0: repository and artifact audit
--------------------------------------------------

Report:

    existing surrogate architecture
    existing physical-field regressors
    exact output fields
    acquisition code path
    StopController code path
    checkpoint/resume code path
    HPC scripts
    missing fields needed for Stage V

Do not start production run.

--------------------------------------------------
Phase 1: physical-field surrogate upgrade
--------------------------------------------------

Implement or verify:

    F_SC_minus_F_normal regression target
    Delta regression
    q regression with masking
    P0 regression
    Ppi regression
    log_bulk_gap regression
    uncertainty estimates

Add tests for:

    normal q not_applicable
    SC-only topology fields
    residual sign convention
    field masking
    uncertainty outputs

--------------------------------------------------
Phase 2: boundary support extraction
--------------------------------------------------

Implement:

    local mutual-kNN boundary bracket builder
    support sets for NS, UF, topology, gap
    boundary fill-distance H_s
    support coverage metrics
    no-long-edge filtering

Test on synthetic data and Stage IV-A frozen data.

--------------------------------------------------
Phase 3: base acquisition A0
--------------------------------------------------

Implement:

    B_s boundary likelihood
    U_s field uncertainty factor
    H_s support sparsity factor
    automatic alpha_s boundary priority
    unified logsumexp boundary score
    stochastic selection-ready scores

Test:

    deep normal does not dominate
    sparse boundary patch scores high
    dense boundary patch scores lower
    unsupported far extrapolation is controlled
    topology surface can be selected inside SC interior

--------------------------------------------------
Phase 4: reward logging
--------------------------------------------------

Implement reward computation after each micro-batch:

    bracket reward
    surprise reward
    support improvement reward
    uncertainty-drop reward
    component reward
    redundancy/untrusted penalties

Implement feature logging for selected and control candidates.

--------------------------------------------------
Phase 5: acquisition-value learner
--------------------------------------------------

Implement g_theta:

    small neural network, gradient boosted trees, random forest,
    or project-consistent lightweight model

Codex may choose after repository audit.

Requirements:

    online training
    checkpointing
    shadow mode
    validation against A0
    lambda_t scheduler
    automatic fallback to A0

--------------------------------------------------
Phase 6: micro-batch loop
--------------------------------------------------

Implement:

    micro-batch exact selection
    stochastic top-k
    propensity logging
    checkpoint after every micro-batch
    resume from last valid checkpoint
    partial failure recovery
    asynchronous compatibility if existing HPC supports it

--------------------------------------------------
Phase 7: smoke tests
--------------------------------------------------

Run local/small smoke tests only:

    tiny initial design
    1 or 2 micro-batches
    no production-scale run

Validate:

    exact oracle integration
    model training
    scoring
    selection
    exact append
    reward computation
    g_theta shadow training
    checkpoint/resume

--------------------------------------------------
Phase 8: HPC package
--------------------------------------------------

Generate a Stage V HPC package.

Do not submit automatically.

==================================================
十四、Tests required
==================================================

At minimum include tests for:

1. physical-field outputs

    F_SC_minus_F_normal present or explicit fallback
    P0/Ppi match known convention
    log_bulk_gap finite for SC points
    masks correct

2. boundary support extraction

    synthetic plane/sphere boundary test
    no raw long-edge artifacts
    local bracket support correct

3. A0 acquisition

    high on sparse predicted boundary
    low in dense boundary region
    low in deep normal for spectral fields
    topology boundary inside SC interior remains selectable

4. learned residual

    can train on toy reward
    shadow mode works
    lambda_t remains zero before enough data
    lambda_t increases only when validation improves
    fallback works

5. micro-batch

    stochastic selection produces propensities
    diversity control works
    checkpoint/resume exact

6. label hierarchy

    normal is not trivial SC
    nodal has Z2 not_defined
    unresolved not nodal
    topology failure does not erase thermodynamic label

7. no data leakage

    Stage IV-A data not used for Stage V training initialization
    offline replay clearly separate from production

==================================================
十五、HPC package requirements
==================================================

Generate package:

    stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc.tar.gz

Include:

    source tree or patches
    production config
    same-window config
    optional lower-mu-extension config
    smoke-test config
    environment specification
    Slurm submit script
    resume script
    monitoring script
    checkpoint inspection script
    failed-task inspection script
    result collection script
    README
    manifest
    SHA256 checksums

Must support:

    preflight
    smoke
    production submit
    resume
    collect reports
    no automatic submission

==================================================
十六、gpuh01 exclusion
==================================================

All Slurm scripts must include:

    #SBATCH --exclude=gpuh01

including:

    smoke
    preflight
    production
    resume
    workers
    postprocessing jobs that request compute nodes

Runtime guard:

    if hostname or SLURMD_NODENAME is gpuh01:
        exit safely
        write clear error
        do not start exact calculation
        do not modify checkpoint

README examples must also include gpuh01 exclusion.

==================================================
十七、encoding and archive validation
==================================================

Before archive:

    Python compile/import check
    config parse check
    bash -n shell scripts
    UTF-8 validation
    no BOM
    LF line endings
    CRLF scan
    broken symlink scan
    tar listing
    SHA256 generation

Machine field names:

    ASCII only
    use mu, delta, pi, ja
    no Greek letters in keys
    no spaces in filenames

Shell environment:

    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    export PYTHONUTF8=1
    export PYTHONIOENCODING=utf-8

==================================================
十八、Final implementation report
==================================================

Final Codex report must include:

1. repository audit
2. implemented architecture
3. changed-files manifest
4. physical-field surrogate details
5. boundary-support extraction details
6. base acquisition formula
7. learned residual architecture
8. reward definition
9. lambda_t schedule
10. micro-batch strategy
11. tests run and results
12. smoke-test results
13. HPC package path and SHA256
14. gpuh01 exclusion verification
15. encoding validation report
16. exact commands to run smoke
17. exact commands to submit production
18. exact commands to resume
19. known risks and fallback behavior

==================================================
十九、Do-not-claim list
==================================================

Do not claim:

    Stage V production has converged before production run is completed.
    learned acquisition outperforms A0 before matched-budget comparison.
    Stage IV-A data was used for Stage V training.
    absence of nodal SC is failure.
    single-band corridor is a topology label.
    smooth surfaces are final publication boundaries.
    response eta/Ic validates thermodynamic/topology labels.

==================================================
二十、Stage V success definition
==================================================

Stage V method succeeds if, at matched or comparable budget,
it improves over Stage IV-A or A0-only baseline in:

    topology surface coverage p95
    trusted topology surprise
    topology surface shift
    component stability
    local bracket support density
    boundary support uniformity
    selected-point reward per exact calculation

Scientific convergence remains a separate question.

Start by outputting:

    repository/input audit
    implementation plan
    risk list

Then implement, test, smoke-run, package, and report.
Only pause if you encounter a key ambiguity that would change physical definitions,
mu ensemble semantics, or corrupt frozen datasets.