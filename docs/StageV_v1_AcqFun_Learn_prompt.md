接续 BdG topology-aware active-learning 项目，开启 Stage V-v2。

Stage V-v2 是 Stage V-v1 的保守修正版 acquisition-learning full loop。
目标是修复 v1 中 learned residual 学习目标偏向 normal/SC、topology channel 被淹没、
global_sobol proposal 过度支配的问题。

核心改动：

    1. multi-head per-boundary learned residual
    2. per-boundary reward normalization
    3. StopController-driven automatic alpha_s priority
    4. proposal-source density / quantile correction
    5. independent lambda_s trust gates
    6. stronger P0/Ppi topology-boundary support acquisition
    7. matched-budget cold-start full loop comparison

本 prompt 规定科学目标、必改方向、不可变物理语义和交付要求。
具体模块、函数、类结构、模型类型、日志格式和调度细节由你根据当前 repository 决定。
优先复用 Stage V-v1 的实现，不要平行重写 exact oracle、Hamiltonian 或 topology oracle。

==================================================
一、Stage V-v1 诊断基线
==================================================

Stage V-v1 return report 的关键信息如下，必须作为修正依据：

    available cumulative dataset:
        dataset_iter093.npz

    available samples:
        6892

    thermodynamic counts:
        normal = 4855
        uniform_SC = 130
        FFLO = 1907

    topology counts:
        cFFLO / trivial FFLO = 1637
        tFFLO / topological FFLO = 270
        gapless / unresolved SC = 0

    latest phase-map change proxy:
        0.00025

    supported probe fraction:
        0.997

    learned residual:
        lambda_t = 0.700
        learned model rank correlation = 0.789
        A0 rank correlation = 0.209

Observed failure mode:

    learning machinery worked,
    but the learned scalar reward was dominated by normal/SC and global exploration.

    selected-channel diagnostics show normal/SC score rising late,
    while P0 and Ppi topology channels remained weak.

    selected candidate-source mixture was dominated by global_sobol.

    tFFLO support was under-sampled compared with Stage IV-A.

Conclusion to implement:

    Do not continue Stage V-v1 unchanged.
    Implement Stage V-v2 as a new cold-start run with per-boundary learning.

==================================================
二、Stage V-v2 scientific objective
==================================================

Stage V-v2 should test whether a safer learned acquisition can improve 3D boundary learning by:

    increasing cFFLO/tFFLO support
    improving P0/Ppi topology-boundary sampling
    reducing topology surprise
    improving topology surface coverage
    reducing topology surface shift
    stabilizing topology component count
    avoiding over-selection of deep normal / generic normal/SC points
    reducing global_sobol proposal dominance

This is still a closed-set boundary-learning stage.

Do not implement open-set novelty discovery in this version.
No new phase discovery pipeline is required in v2.
That belongs to a later extension after known-boundary learning is fixed.

Suggested run_id:

    stagev_v2_multihead_boundary_learning_3d_v1

Default domain:

    same as Stage IV-A and Stage V-v1:
        x = (kBT/t, J_A/t, mu/t)
        mu/t in [-0.5, 1.5]
        kBT/t and J_A/t read from Stage IV-A / Stage V config
        U fixed to production U0
        t = 1

Do not silently expand mu range in the default run.

Optional config may exist for lower-mu extension:

    mu/t in [-1.0, 1.5]

but it must be a separate explicit run profile, not the default.

==================================================
三、strict cold-start and provenance
==================================================

Stage V-v2 production run must be cold-start.

Do not use Stage III, Stage IV-A, or Stage V-v1 data as training initialization.

Do not use:

    Stage III datasets
    Stage IV-A dataset_iter025
    Stage V-v1 dataset_iter093 or later
    Stage V-v1 surrogate checkpoints
    Stage V-v1 selected-point coordinates
    Stage IV-A selected-point coordinates

for Stage V-v2 initial training or initial design.

These artifacts may only be used for:

    code regression tests
    offline replay diagnostics
    matched-budget comparison after run
    report-only comparison
    validation of figure and metric code

Initial design:

    1024 scrambled Sobol exact points

Micro-batch size:

    default = 64

Provide config alternative:

    micro_batch_size = 128

Matched-budget target:

    comparable to Stage V-v1 / Stage IV-A,
    e.g. around 7000 exact points total.

All budget values must be configurable.

==================================================
四、frozen physical semantics
==================================================

Do not change thermodynamic phase criterion:

    if any Delta > 0 state has free energy lower than normal:
        superconducting

    SC state:
        q_opt distinguishes uniform_SC and FFLO

    normal:
        q_not_applicable, not q_unresolved

Continue to separate:

    thermodynamic phase reliability
    q-window / q-grid reliability
    spectral-gap reliability
    topology reliability
    response-side eta/Ic reliability

Do not use eta/Ic in Stage V-v2 acquisition.

Topology:

    For trusted gapped SC:
        P0 * Ppi < 0 -> topological
        P0 * Ppi > 0 -> trivial

    For nodal SC:
        Z2 not_defined

    For normal:
        topology not_applicable

Do not restrict topology search to FFLO only after mu variation.
Spectral/topology acquisition must be allowed in all predicted SC,
with soft physical gating only.

==================================================
五、learning modules
==================================================

Reuse the Stage V-v1 three-module concept:

    Module 1:
        phase / label surrogate

    Module 2:
        physical-field surrogate

    Module 3:
        acquisition-value learner

But modify Module 3 from scalar reward learning to multi-head per-boundary reward learning.

--------------------------------------------------
5.1 Module 1: phase / label surrogate
--------------------------------------------------

Input:

    normalized (kBT/t, J_A/t, mu/t)

Targets:

    normal
    uniform_SC
    FFLO

Optional conditional heads:

    gapped / nodal
    trivial / topological

Use primarily for:

    soft gates
    sanity checks
    label uncertainty
    phase-map diagnostics

Do not let phase classifier alone define final boundary surfaces.

--------------------------------------------------
5.2 Module 2: physical-field surrogate
--------------------------------------------------

This is the implicit boundary model.

Required targets:

    Delta_opt
    q_opt
    F_SC_minus_F_normal
    P0
    Ppi
    log_bulk_gap
    Pfaffian margin
    bulk-gap margin

If F_SC_minus_F_normal is missing from current output schema,
add it for Stage V-v2 exact outputs.

Define residual fields:

    m_NS:
        F_SC_min - F_normal
        sign convention must be explicit

    m_UF:
        |q_opt| - q_threshold
        or project-consistent uniform/FFLO signed margin

    m_P0:
        P0

    m_Ppi:
        Ppi

    m_gap:
        log(Eg / E_gap_tol)

Boundary surfaces:

    normal/SC:
        m_NS = 0

    uniform/FFLO:
        m_UF = 0

    trivial/topological:
        P0 = 0 or Ppi = 0

    gapped/nodal:
        m_gap = 0

Every field prediction must have an uncertainty estimate.

Normal points must be masked from q/topology training where not applicable.

--------------------------------------------------
5.3 Module 3: multi-head acquisition-value learner
--------------------------------------------------

Replace scalar acquisition-value learner:

    old:
        g_theta(phi) -> scalar reward

with per-boundary heads:

    g_NS(phi_NS)       -> normal/SC value
    g_UF(phi_UF)       -> uniform/FFLO value
    g_P0(phi_P0)       -> P0 topology value
    g_Ppi(phi_Ppi)     -> Ppi topology value
    g_gap(phi_gap)     -> gapped/nodal value if applicable

Each head predicts usefulness for that specific boundary family.

Do not train one single reward model that mixes all boundary types.

Each head has:

    own training targets
    own reward normalization
    own validation metrics
    own lambda_s trust coefficient
    own fallback to A0_s

==================================================
六、base acquisition A0_s per boundary
==================================================

For each boundary family s:

    s in:
        NS
        UF
        P0
        Ppi
        gap

Build base acquisition:

    A0_s(x) = B_s(x) * U_s(x) * H_s(x)

where:

    B_s:
        boundary likelihood
        P(|m_s(x)| < tau_s)

    U_s:
        field uncertainty factor

    H_s:
        boundary-support sparsity / fill-distance factor

Boundary support sets:

    NS:
        trusted local normal <-> SC brackets

    UF:
        trusted local uniform_SC <-> FFLO brackets

    P0/Ppi:
        trusted gapped SC local P0/Ppi sign-change brackets
        and/or trusted opposite-Z2 local brackets

    gap:
        trusted gapped <-> nodal brackets if nodal exists,
        otherwise small-gap support diagnostic only

Local graph:

    mutual kNN
    local-scale filtered edges
    normalized 3D coordinate space
    no raw global Delaunay long edges
    no cross-empty-space long edges

H_s should be high only when:

    candidate is near predicted boundary
    and that boundary patch lacks nearby trusted exact support.

==================================================
七、per-boundary reward normalization
==================================================

For each selected exact point, compute reward components separately for each boundary:

    r_NS
    r_UF
    r_P0
    r_Ppi
    r_gap

Do not collapse into one scalar before training boundary-specific heads.

Reward components may include:

    r_bracket_s:
        creates or shortens local bracket for boundary s

    r_surprise_s:
        previous model confidently predicted wrong sign / label for boundary s

    r_support_s:
        reduces local fill-distance or support-distance for boundary s

    r_uncertainty_drop_s:
        reduces local ensemble uncertainty for boundary s

    r_component_s:
        helps discover, remove, merge, split, or stabilize component for boundary s

    r_margin_s:
        lands near physically meaningful small residual for boundary s

Penalties:

    r_redundant
    r_untrusted
    r_numerical_failure
    r_deep_interior

Normalize reward per boundary:

    r_tilde_s =
        (r_s - baseline_s) / (std_s + epsilon)

Use running statistics or robust quantile scaling.

Reason:

    topology rewards are rarer than normal/SC rewards.
    Without per-boundary normalization, NS dominates learning.

Store both raw and normalized rewards.

==================================================
八、per-boundary learned residual and lambdas
==================================================

Final per-boundary score:

    A_s(x) =
        A0_s(x) * exp(lambda_s(t) * g_s(phi_s(x)))

Each lambda_s is independent:

    lambda_NS
    lambda_UF
    lambda_P0
    lambda_Ppi
    lambda_gap

Initial:

    lambda_s = 0 until enough reward data for boundary s

Activation:

    when head g_s beats A0_s in boundary-specific validation,
    set lambda_s = 0.1

Trust schedule:

    increase lambda_s only if both conditions hold:

        1. reward-prediction validation improves over A0_s
        2. boundary-specific convergence/support metrics improve

If reward improves but scientific metrics worsen:

    decrease lambda_s

Cap lambda_s by config.

Do not let good NS learning raise topology lambdas.
Do not use one global lambda_t.

Validation metrics per head:

    reward ranking correlation
    top-k reward enrichment
    pairwise ranking accuracy
    boundary-support improvement prediction
    selected bracket-yield prediction

Scientific metrics per head:

    support p95 for boundary s
    surface shift proxy for boundary s
    trusted surprise_s
    bracket density_s
    component stability_s

==================================================
九、automatic alpha_s priority from failure metrics
==================================================

Combine boundary scores without manual quotas.

Use:

    A_total(x) =
        logsumexp_s[
            alpha_s(t)
            + ranknorm(log A_s(x))
        ]

or equivalent numerically stable formulation.

Each alpha_s is automatically updated from boundary failure metrics:

    alpha_s(t+1) =
        alpha_s(t)
        + eta * (
            c1 * surprise_deficit_s
          + c2 * coverage_deficit_s
          + c3 * shift_deficit_s
          + c4 * component_instability_s
          - c5 * convergence_success_s
        )

This is not hand-written quota.
It is StopController-driven priority.

Important behavior:

    if normal/SC becomes stable:
        alpha_NS decreases

    if topology support is poor:
        alpha_P0 / alpha_Ppi increase

    if P0/Ppi channels are missing support:
        they must not stay near zero indefinitely

    if a boundary is physically absent:
        alpha should eventually decay, but only after sufficient support.

Missing boundary states must be separated:

    missing_boundary
    insufficient_support
    physically_absent
    not_yet_discovered

Do not encode missing boundary as shift = 0.

==================================================
十、per-boundary rank normalization
==================================================

Before combining scores, normalize per boundary within candidate pool.

For each micro-batch candidate pool:

    ranknorm(A_NS)
    ranknorm(A_UF)
    ranknorm(A_P0)
    ranknorm(A_Ppi)
    ranknorm(A_gap)

Then combine.

Reason:

    raw numerical scales differ.
    Stage V-v1 likely suppressed P0/Ppi channels through scale mismatch or gating.

Rank normalization must be logged.

Log:

    raw score
    rank-normalized score
    alpha_s
    lambda_s
    final contribution per boundary

==================================================
十一、proposal-source density correction
==================================================

Stage V-v1 selected candidate-source mixture was dominated by global_sobol.

Fix proposal-source bias without manual quotas.

Candidate proposal families:

    global_sobol
    sparse_fill
    boundary_support_jitter
    bracket_midpoint
    mu_edge_guard
    single_band_corridor
    SC_interior_coverage

For each proposal source q:

    track number of generated candidates
    track selection rate
    track realized reward
    track boundary-specific reward

Correct source dominance using one or more of:

    per-source score quantile normalization
    score - log(source_candidate_count)
    score - log(proposal_density)
    source-wise Thompson sampling over proposal generators
    source-aware diversity penalty

But do not set fixed manual selected-point quotas.

Goal:

    global_sobol remains available for exploration
    but cannot dominate 60-75% of selected points simply due to proposal volume.

Log selected source fractions per micro-batch.

==================================================
十二、micro-batch selection
==================================================

Keep micro-batch active learning.

Default:

    micro_batch_size = 64

Selection:

    generate candidate proposals
    compute per-boundary A_s
    apply per-boundary ranknorm
    combine with alpha_s
    apply learned residual per boundary
    apply source correction
    apply stochastic selection

Use:

    Gumbel-top-k
    softmax sampling
    Thompson sampling
    or project-consistent stochastic top-k

Always record:

    selection probability / propensity
    source
    scores
    diversity penalties
    final rank

Do not deterministically take raw top scores only.

Diversity:

    avoid duplicates and exact near-duplicates
    but do not over-dilute boundary concentration.

==================================================
十三、topology-channel safeguards
==================================================

Add explicit safeguards so topology channels cannot silently die.

Monitor every micro-batch:

    mean raw P0 score
    mean raw Ppi score
    ranknorm P0 contribution
    ranknorm Ppi contribution
    selected points with high P0/Ppi contribution
    selected points near predicted P0/Ppi zero surfaces
    new tFFLO count
    new opposite-Z2 brackets
    P0/Ppi reward heads validation
    lambda_P0 / lambda_Ppi
    alpha_P0 / alpha_Ppi

If over several micro-batches:

    alpha_P0/Ppi high
    but selected P0/Ppi contribution remains near zero

then flag:

    topology_channel_suppressed

and automatically diagnose:

    SC gate too hard?
    P0/Ppi scale too small?
    boundary support set empty?
    ranknorm bug?
    source correction suppressing topology candidates?
    model uncertainty collapsed?

Do not silently continue for dozens of micro-batches with P0/Ppi near zero.

==================================================
十四、candidate generation improvements
==================================================

Ensure candidate proposals include topology-relevant proposals:

    P0/Ppi zero-surface proposals from physical-field surrogate
    opposite-Z2 bracket midpoint proposals
    topology boundary jitter proposals
    sparse topology support fill-distance proposals
    single-band corridor proposals
    lower-mu edge topology guard proposals

These are proposal generators only.
They do not impose manual selected-point quotas.

All proposal points are scored by unified A_total.

If topology proposal generators produce no candidates,
log reason.

==================================================
十五、exact oracle integration
==================================================

Keep exact oracle:

    robust incremental q-window expansion
    near-zero Delta refinement
    basin-level local refinement
    rank-and-cap K3

Each exact point must save:

    kBT_over_t
    J_A_over_t
    mu_over_t
    U_over_t
    thermo_phase
    Delta_opt
    q_opt
    free_energy_opt
    normal_free_energy
    F_SC_minus_F_normal
    P0
    Ppi
    pf_product
    pfaffian_margin
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

Retained K3 basins should save:

    basin_Delta
    basin_q
    basin_free_energy
    basin_P0
    basin_Ppi
    basin_bulk_gap
    basin_topology
    delta_F_to_ground

Cache keys must include:

    kBT
    J_A
    mu
    U
    t
    all relevant fixed parameters

==================================================
十六、StopController and diagnostics
==================================================

Implement formal or semi-formal Stage V-v2 diagnostics.

Thermodynamic:

    phase-volume map change
    normal/SC surface shift proxy
    uniform/FFLO surface shift proxy
    surface coverage
    trusted phase surprise
    component stability

Topology:

    trusted Z2 volume-map change
    cFFLO/tFFLO surface shift proxy
    topology surface coverage
    trusted topology surprise
    topology component stability
    P0/Ppi surface support
    opposite-Z2 bracket density

Learning:

    reward per boundary
    normalized reward per boundary
    lambda_s per boundary
    alpha_s per boundary
    rank correlation per boundary
    selected contribution per boundary
    proposal source fractions
    bracket yield per boundary
    deep-normal selected fraction
    SC selected fraction
    FFLO selected fraction
    tFFLO selected fraction

Must produce convergence and learning plots every N micro-batches.

Do not declare final convergence from kNN proxy alone.
Label proxy metrics as report-only unless full StopController is implemented.

==================================================
十七、comparison protocol
==================================================

After Stage V-v2 run, compare to:

    Stage V-v1 available dataset_iter093
    Stage IV-A return run
    A0-only ablation if feasible

Comparison must be matched-budget where possible.

If Stage V-v1 final dataset_iter100 remains missing,
use dataset_iter093 as provisional comparison and state caveat.

Metrics:

    final phase counts
    final topology counts
    tFFLO count
    cFFLO/tFFLO bracket count
    topology surface coverage
    topology surprise
    phase-map proxy
    topology-map proxy
    selected normal fraction
    selected SC fraction
    selected FFLO fraction
    proposal-source fractions
    reward efficiency
    learned model validation per boundary

Primary success criteria:

    Stage V-v2 should increase topology support without destroying phase-map stability.

Expected improvement over v1:

    higher tFFLO count
    higher opposite-Z2 bracket density
    higher P0/Ppi selected contribution
    lower topology surprise
    lower topology surface coverage p95
    less global_sobol dominance
    more balanced boundary learning
    phase-map proxy still stable

==================================================
十八、tests required
==================================================

Implement tests for:

1. per-boundary reward computation

    NS reward independent of P0 reward
    P0/Ppi reward not zero when opposite-Z2 bracket appears
    reward normalization works with sparse topology positives

2. multi-head learner

    separate heads train independently
    lambda_s independent
    good NS head cannot activate P0/Ppi lambda
    failing topology head falls back to A0_P0/Ppi

3. per-boundary rank normalization

    raw-scale mismatch does not suppress P0/Ppi
    ranknorm outputs valid

4. alpha_s update

    topology coverage deficit increases alpha_P0/Ppi
    stable NS lowers alpha_NS
    missing boundary not treated as shift=0

5. proposal-source correction

    huge global_sobol pool does not dominate solely by count
    source normalization stable

6. micro-batch selection

    stochastic selection works
    propensity recorded
    diversity control works
    duplicates avoided

7. label hierarchy

    normal is not trivial SC
    nodal has undefined Z2
    unresolved not nodal
    topology failure does not erase thermodynamic label

8. no data leakage

    Stage IV-A and Stage V-v1 data not used for Stage V-v2 training initialization

==================================================
十九、HPC package
==================================================

Generate a complete HPC package.

Suggested archive:

    stagev_v2_multihead_boundary_learning_3d_v1_hpc.tar.gz

Include:

    source tree or patches
    production config
    smoke-test config
    A0-only ablation config if feasible
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

Do not submit production automatically.

==================================================
二十、gpuh01 exclusion
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
二十一、encoding and archive validation
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
二十二、implementation phases for Codex
==================================================

Proceed in phases.

Phase 0:
    repository and Stage V-v1 audit

Phase 1:
    exact-output schema check and physical-field surrogate verification

Phase 2:
    per-boundary support extraction and A0_s score

Phase 3:
    per-boundary reward and normalization

Phase 4:
    multi-head acquisition-value learner and lambda_s schedule

Phase 5:
    alpha_s StopController-driven priority

Phase 6:
    proposal-source correction

Phase 7:
    micro-batch selection integration

Phase 8:
    smoke test

Phase 9:
    HPC package

Do not start production run locally.

==================================================
二十三、final implementation report
==================================================

Final Codex report must include:

    repository audit
    Stage V-v1 failure-mode reproduction
    implemented architecture
    changed-files manifest
    exact-output schema changes
    per-boundary reward definitions
    per-boundary normalization details
    multi-head model architecture
    alpha_s update rule
    lambda_s schedule
    proposal-source correction method
    tests run and results
    smoke-test results
    HPC package path
    SHA256
    gpuh01 exclusion verification
    encoding validation report
    production submit command
    resume command
    known risks
    fallback behavior

==================================================
二十四、do-not-claim list
==================================================

Do not claim:

    Stage V-v2 has converged before production run.
    learned acquisition outperforms v1 before matched-budget comparison.
    Stage IV-A or Stage V-v1 data was used for training.
    absence of nodal SC is failure.
    single-band corridor is a topology label.
    smooth surfaces are final publication boundaries.
    response eta/Ic validates thermodynamic/topology labels.

==================================================
二十五、success definition
==================================================

Stage V-v2 method succeeds if, at comparable budget to Stage V-v1 / Stage IV-A:

    tFFLO support increases significantly relative to v1
    P0/Ppi selected-channel contribution no longer remains near zero
    topology bracket density improves
    topology surprise decreases
    topology surface coverage improves
    topology component stability improves
    global_sobol selected fraction is controlled
    normal/SC no longer dominates learned residual after phase map stabilizes
    thermodynamic phase-map stability is retained

Start by outputting:

    repository/input audit
    implementation plan
    risk list

Then implement, test, smoke-run, package, and report.
Only pause if you encounter a key ambiguity that would change physical definitions,
mu ensemble semantics, or corrupt frozen datasets.