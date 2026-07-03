接续现有 BdG topology-aware active-learning 项目，开启第四阶段：

    Stage IV-A:
    cold-start three-dimensional active learning
    in (kBT/t, J_A/t, mu/t)

请先审查 repository、Stage III frozen run、exact oracle、topology oracle、
acquisition framework、StopController、checkpoint/resume 机制和既有 HPC package，
随后自行制定具体实施计划，完成代码、测试、配置和 HPC 打包。

本 prompt 规定科学目标、参数范围、冻结语义和验收条件。
具体类、函数、模型结构、候选池大小和并行实现由你根据现有代码决定。
优先做最小可靠改动，不要平行重写已经验证的核心物理代码。

==================================================
一、第四阶段科学目标
==================================================

建立一次真正从头开始的 3D topology-aware full active-learning loop：

    3D scrambled-Sobol initial design
    -> exact thermodynamic and topology evaluation
    -> train 3D surrogates
    -> acquisition scoring
    -> iterative boundary-surface discovery
    -> joint thermodynamic/topology convergence

三维参数空间：

    x = (kBT/t, J_A/t, mu/t)

固定：

    t = 1
    U = Stage III production value U0
    all other model parameters = Stage III production values

本阶段研究以下三维边界曲面：

1. normal / superconducting surface；
2. uniform-SC / FFLO surface；
3. trivial / topological gapped-SC surface；
4. 若实际出现，则包括 gapped / nodal surface。

重点问题：

    topology-aware active learning 能否从无相图先验的随机空间填充样本出发，
    在三维参数空间中发现并收敛这些相边界曲面；

    tFFLO 是否主要出现在 normal-state single-band occupancy corridor；

    tFFLO 区域如何随 mu 和 J_A 演化、生成、终止或改变连通性。

建议新 run_id：

    active_phase_topology_3d_t_ja_mu_from_scratch_v1

==================================================
二、严格 cold-start 与 provenance
==================================================

新 3D full loop 不得使用以下数据作为训练初始化：

    dataset_iter018
    dataset_iter035
    Stage III topology-derived datasets
    Stage III surrogate checkpoints
    Stage III selected-point coordinates

Stage III 数据只允许用于：

    unit/regression tests
    numerical tolerance reference
    hidden validation at the original fixed mu slice
    final post-run comparison

不得将 Stage III exact points加入 3D training dataset。

所有新输出必须使用独立：

    run_id
    output directory
    configuration
    checkpoint namespace
    dataset provenance

==================================================
三、参数范围
==================================================

kBT/t：

    直接读取 Stage III production config 中的完整范围
    不要根据报告图片人工估计

J_A/t：

    直接读取 Stage III production config 中的完整范围
    不要根据报告图片人工估计

mu/t production range：

    [-0.5, 1.5]

U：

    固定为 Stage III production value U0

t：

    固定为 1，作为能量单位

确认 Stage III 原始固定 chemical potential：

    mu_reference

必须位于新的 production range 内。

若不在该范围内：

    停止 production package 生成
    输出明确的 domain inconsistency report

所有 surrogate 输入在内部可归一化到 [0,1]，
但所有 dataset、日志和科学图必须保留真实无量纲坐标。

==================================================
四、Stage IV-0：mu 参数化与能带预审计
==================================================

在启动昂贵 3D full loop 前，完成一次自动化 preflight。

--------------------------------------------------
4.1 mu 传播审计
--------------------------------------------------

确认 mu 不再是隐藏固定常量，并正确进入：

    normal-state Hamiltonian
    BdG Hamiltonian
    thermodynamic free energy
    normal-state reference free energy
    Delta-q exact oracle
    Pfaffian P0 and Ppi
    full-BZ bulk-gap oracle
    q-window expansion
    local refinement
    cache keys
    task hashes
    checkpoint metadata
    output datasets
    acquisition model inputs
    report generation

任何 cache key 或 exact-task identity 都必须包含：

    kBT
    J_A
    mu
    U
    relevant fixed model parameters

不得因 cache key 缺少 mu 而复用其他 chemical-potential point 的结果。

--------------------------------------------------
4.2 grand-canonical 语义
--------------------------------------------------

本阶段将 mu 作为外部实验控制参数。

采用：

    fixed input chemical potential
    grand-canonical calculation

不得在 exact oracle 内部重新调整 mu 以保持固定粒子数，
除非当前项目本来就明确采用另一种定义。

若代码中存在 fixed-density mu solver：

    本 run profile 必须禁用
    并在 metadata 中记录 ensemble = grand_canonical

--------------------------------------------------
4.3 normal-state band-count scan
--------------------------------------------------

执行廉价 normal-state preflight，扫描：

    J_A/t = full production range
    mu/t   = [-1.0, 2.0]

使用足够密的 k grid 和二维 (J_A, mu) diagnostic grid。

计算并保存：

    normal-state band energies
    number of bands crossing the Fermi level
    number of Fermi-point pairs
    minimum direct band gap
    minimum indirect band gap
    single-band occupancy mask
    multi-band occupancy mask
    no-Fermi-surface mask
    band extrema versus J_A

必须根据项目实际 Hamiltonian 定义“single band”，
不得仅凭 mu 的大小推测。

如果存在多个等价定义，例如：

    one nondegenerate band crossing
    one pair of Fermi points
    one helical channel

应在报告中分别列出并说明 production 使用哪个诊断。

--------------------------------------------------
4.4 mu-range 判断
--------------------------------------------------

production window 保持：

    [-0.5, 1.5]

除非出现以下硬失败：

    production window 完全不包含 single-band corridor
    mu_reference 不在 window 内
    Hamiltonian 在 window 内数值不稳定
    exact oracle 在 window 端点无法工作

preflight 必须报告：

    single-band corridor 是否被 production window 覆盖
    corridor 到 lower mu boundary 的最小距离
    corridor 到 upper mu boundary 的最小距离
    mu_reference 所处 band-count region
    是否存在可能遗漏的另一 disconnected single-band corridor

single-band mask 只能作为：

    physical diagnostic
    coverage stratification
    capped soft acquisition prior

不得作为：

    topology label
    exact phase label
    hard exclusion mask
    tFFLO existence criterion

最终 topology 仍由 self-consistent SC solution、
bulk gap 和 Pfaffian Z2 决定。

==================================================
五、exact oracle 与标签体系
==================================================

完整保留 Stage III 已验证的 production exact oracle：

    robust incremental q-window expansion
    near-zero Delta refinement
    basin-level local refinement
    rank-and-cap K3

thermodynamic criterion 保持不变：

    任一 Delta > 0 且自由能低于 normal state
        -> superconducting

    superconducting state:
        q_opt 区分 uniform-SC 与 FFLO

    normal state:
        q_not_applicable，不是 q_unresolved

继续严格分离：

    thermodynamic phase reliability
    q-window and q-grid reliability
    spectral-gap reliability
    topology reliability
    response eta/Ic reliability

本阶段不研究 response-side eta 或 Ic。

--------------------------------------------------
5.1 每点标签
--------------------------------------------------

thermo_phase：

    normal
    uniform_SC
    FFLO

spectral_status：

    not_applicable
    gapped
    nodal
    unresolved

topology_label：

    not_applicable
    trivial
    topological
    unresolved

normal：

    spectral_status = not_applicable
    topology_label = not_applicable

trusted gapped SC：

    根据 P0 * Ppi 定义 Z2

trusted nodal SC：

    Z2 = not_defined

numerically unresolved：

    不得当作 nodal physical phase

--------------------------------------------------
5.2 Pfaffian
--------------------------------------------------

复用 Stage III 已验证的解析和数值实现。

mu 必须使用当前 exact point 的输入值。

保持：

    P0 =
        [mu - t cos(q/2)]^2
        + Delta^2
        - alpha_y^2 sin^2(q/2)
        - [J_A cos(q/2) + alpha_z sin(q/2)]^2

    Ppi =
        [mu + t cos(q/2)]^2
        + Delta^2
        - alpha_y^2 sin^2(q/2)
        - [J_A cos(q/2) + alpha_z sin(q/2)]^2

trusted gapped SC：

    P0 * Ppi < 0:
        topological

    P0 * Ppi > 0:
        trivial

不要硬编码只有 FFLO 可以是 topological。

改变 mu 后可能出现：

    topological uniform-SC
    additional topology islands
    changed topology-boundary connectivity

这些都必须允许被发现。

--------------------------------------------------
5.3 bulk gap
--------------------------------------------------

复用 full-Brillouin-zone bulk-gap oracle：

    float64 / complex128
    coarse k scan
    small-gap trigger
    Nk doubling where required
    local k refinement

不得仅检查 k = 0 和 pi。

nodal 与 unresolved 必须分开。

==================================================
六、初始设计与运行预算
==================================================

使用：

    1024 scrambled Sobol exact points

原因：

    1024 = 2^10
    适合作为空间填充的三维 cold-start design

要求：

    configurable deterministic seed
    保存 Sobol metadata
    完整覆盖 3D box
    不注入 Stage III boundary points
    不人为向已知二维边界聚集

默认 acquisition batch size：

    256

默认 maximum acquisition batches：

    24

允许 StopController 提前停止。

所有数量放入 config，不能分散硬编码。

只运行一个 production seed：

    seed configurable
    default seed documented

多 seed benchmark 留到 Stage IV 后续，
但代码和 HPC profile 必须支持更换 seed。

==================================================
七、三维 surrogate 结构
==================================================

保持分层结构，不建立单一扁平多分类器。

至少包括：

A. thermodynamic phase model

    normal
    uniform_SC
    FFLO

B. Delta surrogate

    复用 Stage III 已验证逻辑

C. Pfaffian regressors

    P0(kBT, J_A, mu)
    Ppi(kBT, J_A, mu)

D. bulk-gap regressor

    suitable transformed Eg(kBT, J_A, mu)

第一版不要求独立 Z2 classifier。

Z2 可由 trusted gapped SC 上的 P0/Ppi signs 导出。

所有模型必须：

    接受三维输入
    使用固定、记录的输入归一化
    具有 uncertainty estimate
    具有 cold-start sample sufficiency guard
    支持 checkpoint/resume
    记录 training eligibility

模型不足以训练时：

    对应 quota 回流到 coverage/exploration
    不伪造 topology labels
    不因单类样本导致 full loop 崩溃

==================================================
八、Acquisition：保持简单的三通道结构
==================================================

只保留三个顶层 channel：

    A_phase
    A_spectral
    A_coverage

不要因升维增加大量独立 acquisition heads。

--------------------------------------------------
8.1 A_phase
--------------------------------------------------

复用 Stage III production full thermodynamic acquisition：

    normal/SC uncertainty
    uniform-SC/FFLO uncertainty
    B_delta gating
    active-pool narrowing
    sampling-power annealing
    exploration annealing
    R_obs
    R_batch
    high-confidence thermodynamic interior handling

A_phase 负责：

    normal / SC surface
    uniform-SC / FFLO surface

不得用 quasiparticle bulk gap 代替 normal/SC thermodynamic criterion。

--------------------------------------------------
8.2 A_spectral
--------------------------------------------------

A_spectral 由：

    bulk-gap boundary uncertainty
    Pfaffian-zero/sign uncertainty

组成。

Bulk-gap 部分寻找：

    gapped / nodal surface

Pfaffian 部分寻找：

    P0 = 0
    Ppi = 0
    trivial / topological surface

分别建模 P0 和 Ppi，
不要只建模乘积。

A_spectral 使用 soft SC gate：

    predicted SC probability
    predicted Delta > 0
    model-support reliability

改变 mu 后，不得把 spectral acquisition 硬限制在 FFLO。

允许对 predicted FFLO 给予温和优先级，
但必须保留 uniform-SC spectral exploration，
以发现可能的 topological uniform-SC。

Stage III 的 high-confidence thermodynamic phase-interior penalty
不得直接压制 A_spectral。

因为 topology boundary 可能位于高度可信的 SC interior。

bulk-gap 和 Pfaffian score：

    先独立归一化或 rank-normalize
    再简单组合
    任意一个高即可入选
    二者同时高可获得有限奖励

normal-state single-band score：

    最多作为 capped soft bonus
    或 coverage stratification

不得成为 hard gate。

--------------------------------------------------
8.3 A_coverage
--------------------------------------------------

三维阶段始终保留 coverage。

初始建议 quota：

    15%

语义上覆盖：

    global 3D parameter-space maximin coverage
    predicted SC-interior coverage
    single-band-corridor coverage
    mu-boundary guard coverage

使用：

    Sobol proposals
    kNN fill distance
    maximin distance
    local support distance

不要在三维中依赖原始全局 Delaunay tetrahedralization
作为主要 coverage 或边界判据。

==================================================
九、Batch 配额
==================================================

early / joint-discovery stage：

    A_phase       45%
    A_spectral    40%
    A_coverage    15%

thermodynamic surfaces 连续若干轮稳定后，
允许切换到：

    A_phase       25%
    A_spectral    60%
    A_coverage    15%

阶段切换依据：

    thermodynamic map stability
    phase-boundary surface stability
    trusted phase surprise

不得仅根据固定 iteration number 切换。

各 channel：

    生成超额候选
    去重
    执行 observation-distance control
    执行 batch-diversity control
    quota shortage 时按明确规则 backfill

保存每个 selected point 的：

    primary channel
    raw component scores
    normalized scores
    gates
    acquisition rank
    diversity penalties
    backfill reason
    single-band diagnostic status

==================================================
十、三维候选生成
==================================================

每轮候选应来自以下组合：

1. global scrambled-Sobol candidate cloud；
2. local opposite-label bracket proposals；
3. local boundary-neighbourhood jitter proposals；
4. maximin/fill-distance coverage proposals。

local opposite-label pairs 至少包括：

    normal vs SC
    uniform-SC vs FFLO
    trivial vs topological
    gapped vs nodal

使用：

    mutual kNN
    local-scale-filtered neighbour graph

不要使用跨越大空洞的长 Delaunay edges。

boundary jitter 应同时包含：

    approximate surface-normal refinement
    approximate surface-tangential exploration

避免只在一条法向上重复二分，
却没有沿三维边界曲面扩展。

==================================================
十一、mu-domain 边界防护
==================================================

正式运行期间持续监控：

    trusted tFFLO points 是否接触 mu = -0.5
    trusted tFFLO points 是否接触 mu = 1.5
    trivial/topological surface 是否接触 mu boundaries
    normal/SC surface 是否在 mu boundary 仍显著移动
    acquisition 是否持续堆积在 mu boundary
    surrogate uncertainty 是否在 mu boundary 很高

输出两个分离状态：

    within_domain_converged
    mu_domain_complete

若主边界在三维 box 内收敛，但显著 topology surface
或 tFFLO region 接触 mu boundary：

    within_domain_converged = true
    mu_domain_complete = false
    mu_range_limited = true

不得把这类情况写成完整 chemical-potential phase diagram closure。

不要在同一个 production run 中自动扩展 mu 范围，
以免破坏归一化、Sobol provenance 和 StopController 语义。

只输出建议扩展方向：

    lower
    upper
    both

==================================================
十二、三维 StopController
==================================================

不要使用密集三维 tensor grid 作为唯一审计方法。

建立固定的 3D Sobol audit cloud，
所有 iteration 使用完全相同的 audit points。

监控：

Thermodynamic：

    phase-volume map change
    normal/SC surface shift
    uniform/FFLO surface shift
    surface coverage
    trusted phase surprise
    connected-component stability

Spectral/topology：

    trusted Z2 volume-map change
    trivial/topological surface shift
    topology-surface coverage
    trusted topology surprise
    topology-region connected components
    topology-surface connected components
    nodal-region diagnostics
    unresolved numerical frontier

Surface shift 使用：

    bidirectional nearest-surface distance
    median
    p90
    p95
    strict max as diagnostic

缺失 surface：

    status = missing_surface

不得自动记为 shift = 0。

停止要求：

    thermodynamic_main_converged
    AND
    topology_main_converged
    AND
    no unresolved significant component change

同时单独报告：

    mu_domain_complete

少量 hard-risk points 不得无限阻止主相图收敛，
但必须保留 numerical-frontier audit。

具体阈值应从 Stage III 的物理分辨率推广到各轴归一化距离，
并全部配置化。

==================================================
十三、Stage III hidden validation slice
==================================================

从 Stage III frozen production config 读取：

    mu_reference

Stage III 的二维结果只作为隐藏验证，不参与 3D 训练。

在每个后期 checkpoint 和最终 checkpoint 上，
提取 3D surrogate 的：

    mu = mu_reference

二维切片。

与 Stage III frozen结果比较：

    normal/SC boundary
    uniform-SC/FFLO boundary
    cFFLO/tFFLO contour
    thermodynamic phase map
    topology map
    connected-component count

计算：

    slice map change
    bidirectional boundary p95 shift
    boundary coverage
    topology-region overlap
    missed-component count

不得为了改善 hidden-slice score
将 Stage III samples加入训练。

hidden validation 失败时：

    不得宣称 3D capability benchmark 通过

但应区分：

    3D model error
    insufficient 3D sampling
    differing numerical configuration
    provenance mismatch

==================================================
十四、输出科学图与数据
==================================================

至少输出：

1. 3D exact-sample cloud by thermodynamic phase；
2. 3D exact-sample cloud by topology label；
3. normal/SC boundary surface point cloud or mesh；
4. uniform-SC/FFLO boundary surface；
5. trivial/topological surface；
6. acquisition points colored by iteration；
7. acquisition points colored by primary channel；
8. normal-state single-band corridor in (J_A, mu)；
9. tFFLO support projected onto (J_A, mu)；
10. overlap between tFFLO and single-band corridor；
11. fixed-mu 2D slice atlas；
12. fixed-J_A 2D slice atlas；
13. phase/topology volume fraction versus mu；
14. topology-surface shift and coverage versus iteration；
15. mu-boundary contact diagnostics；
16. Stage III hidden-slice comparison；
17. hard-risk and unresolved frontier maps。

推荐固定 mu slices：

    lower boundary
    25% point
    mu_reference
    75% point
    upper boundary

另外可以自动选择：

    tFFLO volume maximum slice
    topology-component transition slices
    largest-surrogate-change slices

三维 surface 不得仅以一张不透明 mesh 表达。
同时输出可检查的二维 slice atlas 和 surface point data。

==================================================
十五、输出数据字段
==================================================

每个 exact point 至少保存：

    point_id
    acquisition_iteration
    kBT_over_t
    J_A_over_t
    mu_over_t
    U_over_t
    t

    thermo_phase
    Delta_opt
    q_opt
    free_energy_opt
    normal_free_energy

    P0
    Ppi
    pf_product
    pfaffian_margin

    bulk_gap
    k_at_bulk_gap
    spectral_status
    topology_label
    z2_value

    normal_band_crossing_count
    fermi_point_pair_count
    single_band_diagnostic

    trusted_exact
    training_eligible_exact
    rerun_required
    q_unresolved
    delta_unresolved
    spectral_trusted
    topology_trusted

    acquisition_channel
    acquisition_scores
    gate_values
    local_coverage_distance

    runtime
    backend
    failure_reason
    provenance

保存 retained K3 basins 的对应：

    Delta
    q
    free energy
    P0
    Ppi
    bulk gap
    topology
    delta_F_to_ground

==================================================
十六、测试要求
==================================================

由你制定具体测试计划，但至少包含：

1. mu propagation test

    不同 mu 必须产生不同 Hamiltonian、
    free energy、Pfaffian 和 cache key

2. Stage III regression at mu_reference

    新参数化代码在固定 mu_reference 时，
    与原 Stage III oracle 在数值 tolerance 内一致

3. domain-corner exact tests

    mu = -0.5
    mu = 1.5
    low/high kBT
    low/high J_A
    representative interior points

4. single-band diagnostic tests

    crossing count 与直接 spectrum plot 一致
    k-grid doubling 稳定

5. label hierarchy tests

    normal is not trivial SC
    nodal has undefined Z2
    unresolved is not nodal
    topology failure does not erase trusted thermodynamic label

6. 3D acquisition tests

    deep normal 不被 spectral channel 主导
    SC interior Pfaffian-zero candidate 可被选择
    topology surface candidates 可出现在 uniform-SC 和 FFLO
    coverage points 覆盖完整 3D domain
    single-band prior 不会硬排除其他区域

7. one-iteration 3D integration smoke test

    small initial design
    exact evaluation
    train
    candidate scoring
    batch selection
    append
    checkpoint
    resume

8. StopController tests

    missing surface 不能成为 zero shift
    surface edge contact 可触发 mu_range_limited
    hidden-slice validation 不泄漏训练数据

不要在本地启动正式 production loop。

==================================================
十七、HPC package
==================================================

生成可直接上传但不自动提交的 HPC package。

建议包名：

    active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz

包内包含：

    source tree or patches
    production config
    preflight config
    smoke-test config
    environment specification
    Slurm submit script
    resume script
    monitoring script
    checkpoint-inspection script
    failed-task inspection script
    result collection script
    README
    manifest
    SHA256 checksums

支持：

    preflight-only run
    full production submit
    resume
    partial batch recovery
    clean stop
    final report generation

不得自动提交 production job。

==================================================
十八、必须排除 gpuh01
==================================================

所有 Slurm scripts 必须包含：

    #SBATCH --exclude=gpuh01

包括：

    preflight
    smoke test
    production
    resume
    worker arrays
    postprocessing jobs that request compute nodes

同时加入运行时 hostname guard：

    如果 hostname 或 SLURMD_NODENAME 为 gpuh01
        立即安全退出
        写清晰错误日志
        不启动 exact calculation
        不修改有效 checkpoint

README 的所有 sbatch 示例也必须保留 gpuh01 exclusion。

==================================================
十九、编码与跨平台审计
==================================================

HPC package 必须执行编码检查：

Python/YAML/JSON/Markdown：

    UTF-8
    no BOM
    LF line endings

shell/Slurm：

    ASCII-safe
    no BOM
    LF line endings
    no smart quotes
    no Unicode dash
    no full-width punctuation

文件名、目录名和 machine keys：

    ASCII only
    no spaces
    use mu, delta, pi, ja
    do not use Greek characters as machine field names

shell 环境设置：

    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    export PYTHONUTF8=1
    export PYTHONIOENCODING=utf-8

archive 前执行：

    Python compile/import check
    bash -n
    UTF-8 validation
    BOM scan
    CRLF scan
    broken symlink scan
    config parse
    tar listing
    checksum generation

==================================================
二十、正式决策输出
==================================================

最终报告必须分别回答：

1. 3D thermodynamic surfaces 是否收敛？
2. 3D trivial/topological surface 是否收敛？
3. 是否发现 topological uniform-SC？
4. 是否发现 nodal-SC？
5. tFFLO 是否主要位于 single-band corridor？
6. tFFLO 在 mu 方向的存在区间是什么？
7. topology surface 是否在 mu 上下边界处被截断？
8. mu range [-0.5, 1.5] 是否足够？
9. Stage III mu_reference hidden slice 是否被恢复？
10. 是否需要扩大 mu range？
11. 是否需要新增 exact calculation？
12. 是否可以进入 Stage IV-B 或 multi-seed benchmark？

decision JSON 至少包含：

    thermodynamic_main_converged
    topology_main_converged
    within_domain_converged
    mu_domain_complete
    mu_range_limited
    recommended_mu_extension
    hidden_slice_passed
    significant_component_count
    need_new_exact_calculation
    recommended_next_action

==================================================
二十一、禁止事项
==================================================

禁止：

    将 Stage III 数据用于 3D cold-start 训练
    使用固定-density solver 覆盖输入 mu
    把 single-band diagnostic 当作 topology label
    把 single-band mask 作为 hard acquisition mask
    只在 FFLO 中搜索 topology
    将 nodal 与 unresolved 合并
    使用 raw high-dimensional Delaunay 长边定义边界
    在 topology surface 接触 mu edge 时宣称完整 mu closure
    因没有发现 nodal phase 而自动判定失败
    修改旧 frozen datasets
    自动提交 HPC production job
    在 gpuh01 上运行
    静默改变 mu production range

==================================================
二十二、验收条件
==================================================

任务完成必须满足：

    mu 已成为 exact oracle 和 surrogate 的真实第三输入
    U 固定且记录
    t = 1
    production mu range exactly [-0.5, 1.5]
    cheap guard scan covers [-1.0, 2.0]
    initial design exactly 1024 scrambled Sobol points
    new loop is strict cold-start
    Stage III slice is hidden validation only
    three acquisition channels remain simple and separate
    spectral acquisition searches all SC, not FFLO only
    3D coverage is preserved
    mu-edge range limitation is explicitly monitored
    checkpoint/resume works
    3D smoke test passes
    gpuh01 is excluded in every scheduler path
    encoding/archive validation passes
    HPC archive and SHA256 are produced
    no production job is automatically submitted

开始时先输出：

    repository audit
    Stage III provenance audit
    mu parameterization audit
    normal-band preflight plan
    implementation plan

随后直接完成：

    code changes
    tests
    smoke test
    HPC package
    final implementation report

除非发现会改变科学定义、mu ensemble 语义或损坏历史数据的关键歧义，
不要等待用户逐步批准内部编码决定。