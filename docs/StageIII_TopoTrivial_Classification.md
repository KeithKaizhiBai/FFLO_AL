你现在接续一个 active-learning BdG 相图项目，执行一次“现有数据离线拓扑分类与计算量审计”。

====================
一、项目冻结状态
====================

冻结输入数据集：

    dataset_iter035

已有热力学标签：

    total = 7434
    normal = 1867
    uniform-SC = 715
    FFLO = 4852

现有热力学主相图已经收敛。本任务不得修改原始：

    normal / uniform-SC / FFLO

标签，也不得启动新的 active-learning loop 或新的 Δ-q exact search。

本次 run_id：

    topology_pass_dataset_iter035_v1

任务目标是：

1. 在现有超导 exact points 上计算 Pfaffian Z2；
2. 独立计算全 Brillouin zone bulk quasiparticle gap；
3. 初步分类 trivial / topological / gapless；
4. 判断现有采样对 topo/trivial 边界的覆盖程度；
5. 生成下一轮 topology-aware acquisition 的诊断材料；
6. 不把现有稀疏点插值得到的轮廓宣称为最终拓扑边界。

====================
二、计算资源决策
====================

本任务优先在本地工作站运行。

本地资源包含：

    NVIDIA RTX 4090
    多核 CPU
    足够的系统内存

先执行一个 stratified pilot benchmark：

    pilot point count = 256

pilot 点应覆盖：

    uniform-SC
    FFLO
    低温
    高温
    小 q
    大 q
    thermodynamic trusted points
    少量 hard-risk points

分别测试：

    CPU vectorized backend
    GPU batched backend

测试精度：

    float64 / complex128

测试 Nk：

    512
    1024
    2048

记录：

    wall time
    points per second
    k-Hamiltonians per second
    peak RAM
    peak VRAM
    CPU/GPU agreement
    Nk convergence

根据 pilot 外推全部 5567 个 SC 点的运行时间。

资源决策规则：

    projected full runtime <= 6 hours:
        本地执行

    projected full runtime > 6 hours:
        先检查是否存在 Python scalar loop、重复构造 Hamiltonian、
        保存完整谱、未分块等实现问题

    优化后仍 projected > 12 hours:
        输出超算提交建议，但不要自动迁移

预计本任务应当可以在本地完成。

注意：4090 的双精度小矩阵 eigensolver 不一定比 CPU 快，
必须以 pilot benchmark 结果决定后端，而不是默认 GPU。

====================
三、输入数据审计
====================

首先定位 dataset_iter035 和对应 exact-oracle 配置。

确认每个数据点是否包含：

    point_id
    kBT
    J_A
    Delta_opt
    q_opt
    mu 或全局 mu 配置
    thermo_phase
    trusted_exact
    training_eligible_exact
    rerun_required
    q_unresolved
    delta_unresolved
    free_energy_opt
    normal_state_free_energy

确认 Hamiltonian 还需要的固定参数：

    t
    alpha_y
    alpha_z
    其他模型参数

不得重新猜测 Hamiltonian、Nambu basis、q convention 或单位。

必须直接复用当前 exact free-energy oracle 中使用的 BdG Hamiltonian，
或者调用同一个 Hamiltonian builder。

如果 topology 代码需要重写 Hamiltonian，则必须先与原 oracle
在随机点逐元素比较，要求达到 double-precision tolerance。

====================
四、Pfaffian Z2 oracle
====================

实现独立模块：

    TopologyPfaffianOracle

输入：

    Delta
    q
    mu
    J_A
    t
    alpha_y
    alpha_z
    其他必要参数

根据项目采用的 Majorana-basis Hamiltonian和论文 Eq. (5) 计算：

    P0  = Pf[H'(k=0)]
    Ppi = Pf[H'(k=pi)]

解析公式形式为：

    Pf[H'(0/pi)]
      = [mu ± t cos(q/2)]^2
        + Delta^2
        - alpha_y^2 sin^2(q/2)
        - [J_A cos(q/2) ± alpha_z sin(q/2)]^2

不得凭直觉指定哪个符号对应 k=0、哪个对应 k=pi。

必须从项目的 Majorana-basis Hamiltonian直接验证：

    sign convention
    q convention
    k=0 convention
    k=pi convention

定义：

    pf_product = P0 * Ppi

在有隙且可靠时：

    pf_product < 0:
        z2 = 1
        topology = topological

    pf_product > 0:
        z2 = 0
        topology = trivial

    P0 == 0 or Ppi == 0 within numerical tolerance:
        z2_status = boundary_or_unresolved

输出：

    P0
    Ppi
    pf_product
    z2_value
    pfaffian_margin
    z2_status

建议定义无量纲 Pfaffian margin：

    pfaffian_margin =
        min(abs(P0), abs(Ppi)) / E_scale^2

其中 E_scale 应根据模型参数自动构造，不要使用任意固定单位。

====================
五、Pfaffian 单元测试
====================

解析 Pfaffian 必须与数值 Pfaffian进行交叉验证。

从现有 SC 数据中选择至少 100 个随机点：

1. 构造 k=0 和 k=pi 的 Majorana-basis antisymmetric matrix；
2. 检查：

       ||A + A^T|| / ||A||

3. 数值计算 Pfaffian；
4. 比较解析 P0、Ppi 和数值 Pfaffian；
5. 允许整体固定符号 convention 差异，但 P0*Ppi 的符号必须一致。

还需测试：

    q = 0
    Delta 接近 0
    小 J_A
    大 J_A
    P0 接近 0
    Ppi 接近 0

验收条件：

    analytic and numerical Pfaffian product signs agree
    on all non-boundary validation points

若不一致，停止 full pass，输出诊断，不得继续分类。

====================
六、全 Brillouin zone bulk-gap oracle
====================

Pfaffian Z2 只有在 bulk spectrum 有隙时才可作为正式拓扑标签。

实现：

    BulkGapOracle

计算：

    Eg = min_{k,n} abs(E_n(k))
    k_gap = argmin_k min_n abs(E_n(k))

建议 coarse grid：

    Nk_coarse = 2048
    k domain = one full Brillouin zone

必须与项目已有 k convention 完全一致。

优先复用项目已有的 BdG eigenvalue 函数。

如果已有可靠的解析能谱，优先使用解析能谱；
否则使用 Hermitian eigvalsh。

计算必须：

    vectorized or batched
    chunked
    float64 / complex128
    not store full spectra for all points

每个点只长期保存：

    minimum gap
    k at minimum
    optional second-lowest local minimum
    convergence diagnostics

在 coarse grid 上识别若干最低局部极小区间，并在这些区间执行
局部自适应 refinement。

不能只检查 k=0 和 k=pi，因为 FFLO 状态可能在一般 k 处闭合能隙。

====================
七、gap 收敛和状态定义
====================

从 Nk=2048 与 Nk=4096 的差异估计 grid error。

至少对以下点执行 Nk doubling：

    全部小 gap 点
    全部小 Pfaffian-margin 点
    全部 topology change 邻域点
    随机抽取至少 5% 的普通点

定义：

    gap_error = abs(Eg_Nk - Eg_2Nk)

建议使用如下逻辑，具体 tolerance 放入 YAML 配置：

    trusted_gapped:
        Eg > max(10 * gap_error, gap_tol_abs)

    trusted_gapless:
        refined Eg <= gap_tol_abs
        and result stable under Nk doubling / local refinement

    gap_unresolved:
        介于上述两者之间，或结果未收敛

gap_tol_abs 必须相对于模型能量尺度定义，例如：

    gap_tol_abs = gap_tol_rel * E_scale

不要把 gapless 和 topology-unresolved 合并。

最终字段：

    spectral_status:
        gapped
        gapless
        gap_unresolved

====================
八、有限温度拓扑定义
====================

有限温度只通过该温度下的自洽参数进入：

    Delta_opt(kBT, J_A)
    q_opt(kBT, J_A)
    mu_opt 或固定 mu

计算：

    H_BdG(k; Delta_opt(T), q_opt(T), ...)

再对该有效 BdG Hamiltonian计算 Pfaffian Z2。

不得给 Pfaffian 额外乘：

    tanh(beta E / 2)

不得把本任务描述为 mixed-state density-matrix topology。

额外保存热保护指标：

    thermal_gap_ratio = Eg / kBT

对于 kBT = 0，保存为：

    inf 或显式 zero_temperature 标志

thermal_gap_ratio 不参与 Z2 标签，只作为保护强度诊断。

====================
九、标签规则
====================

normal 点：

    topology = not_applicable
    z2_value = not_defined

SC 且 spectral_status = gapped：

    pf_product < 0:
        topology = topological
        z2_value = 1

    pf_product > 0:
        topology = trivial
        z2_value = 0

SC 且 spectral_status = gapless：

    topology = gapless_SC
    z2_value = not_defined

数值未收敛：

    topology = topology_unresolved
    z2_value = not_defined

topology_unresolved 只是可靠性状态，不是物理相。

对于：

    rerun_required
    not trusted_exact
    q_unresolved
    delta_unresolved

可以计算 provisional topology diagnostic，但必须：

    topology_trusted = false

这些点不得作为最终 topo/trivial boundary 的正式支撑点。

====================
十、full pass 执行
====================

完成 pilot 并通过单元测试后，对全部 5567 个 SC 点运行。

使用 checkpoint：

    每 256 或 512 点写入一次

支持：

    resume
    deterministic rerun
    failed-point retry
    per-point exception logging

避免一次构造全部 k-grid Hamiltonian。

根据 VRAM 自动选择 batch size，建议最大 VRAM 使用率不超过 70%。

如果 GPU complex128 性能不理想，允许：

    Pfaffian on CPU
    gap scan on CPU vectorized
    或 CPU/GPU hybrid

以实际 benchmark 为准。

====================
十一、输出数据集
====================

生成派生数据集，不修改 dataset_iter035：

    dataset_iter035_topology_ground_v1.parquet

至少包含：

    point_id
    kBT
    J_A
    thermo_phase
    Delta_opt
    q_opt

    P0
    Ppi
    pf_product
    pfaffian_margin

    bulk_gap
    k_at_bulk_gap
    gap_grid_error
    spectral_status

    z2_value
    topology_label
    topology_trusted

    thermal_gap_ratio

    trusted_exact
    training_eligible_exact
    rerun_required
    q_unresolved
    delta_unresolved

    backend
    Nk_used
    local_refinement_used
    runtime_seconds
    failure_reason

另输出：

    topology_pass_config.yaml
    topology_pass_summary.json
    topology_benchmark_report.json
    topology_failed_points.csv
    topology_validation_report.md

====================
十二、诊断图
====================

生成以下初步图，不将插值轮廓称为最终边界：

1. 全部 SC 点的 topology scatter

       trivial
       topological
       gapless
       unresolved

2. FFLO-only topology scatter

3. uniform-SC-only topology scatter

4. bulk-gap heatmap/scatter

5. Pfaffian-margin map

6. Eg/kBT map

7. thermodynamic hard-risk overlay

8. sampling coverage map

对 trusted FFLO 点构造 Delaunay triangulation，标记：

    相邻点 z2 不同的边
    P0 异号边
    Ppi 异号边
    large circumradius triangles
    large nearest-neighbour-distance regions

这些只作为：

    candidate topology boundary seeds
    coverage-hole diagnostics

不得直接称作最终 topo/trivial phase boundary。

====================
十三、结果汇总
====================

summary 必须报告：

    total SC points
    processed points
    trusted topology points
    provisional topology points
    trivial uniform-SC count
    topological uniform-SC count
    gapless uniform-SC count
    trivial FFLO count
    topological FFLO count
    gapless FFLO count
    unresolved count

    P0 sign-change candidate edges
    Ppi sign-change candidate edges
    z2-change candidate edges

    number and size of sampling coverage holes
    minimum/median/p95 bulk gap
    minimum/median/p95 Pfaffian margin

    CPU benchmark
    GPU benchmark
    selected backend
    projected runtime
    actual runtime
    peak memory

====================
十四、最终决策报告
====================

根据结果，将下一步归入以下一种情况：

Case A:
    已有 trusted topo 和 trivial 点，并且存在稳定异号 bracket

    建议：
        Pfaffian-root contour continuation
        + 少量 targeted exact refinement

Case B:
    已有 topo 和 trivial 点，但边界附近采样明显稀疏

    建议：
        topology-aware acquisition pilot
        包含：
            topology uncertainty
            small Pfaffian margin
            small bulk gap
            coverage guard

Case C:
    现有点几乎全为同一 Z2，且存在较大 coverage holes

    不得得出“无拓扑区”的结论。

    建议：
        先执行 FFLO interior coverage pilot
        再决定是否存在拓扑区域

Case D:
    大量点 gapless 或 topology_unresolved

    建议：
        优先审计 Hamiltonian convention
        k-grid convergence
        q/Delta reliability
        而不是启动 topology AL

====================
十五、禁止事项
====================

本轮禁止：

    修改 dataset_iter035
    修改原 thermodynamic phase labels
    重新运行 full AL loop
    重新训练原三分类器
    新增大规模 exact Δ-q calculation
    把 hard-risk topology 当作 definitive label
    把 gapless 当作 trivial
    把 topology_unresolved 当作物理相
    仅凭 Pfaffian 符号而不检查 global gap
    对全部点进行 OBC
    宣布最终 topological phase boundary

最终目标只是：

    完成现有点的可靠 topology classification
    验证 Pfaffian + bulk-gap oracle
    判断现有 acquisition 对 topology 的覆盖情况
    为下一轮 topology-aware acquisition 提供可执行依据