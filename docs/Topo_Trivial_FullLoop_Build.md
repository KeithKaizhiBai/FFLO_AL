你正在接续现有 BdG active-learning 相图项目。请检查当前代码库、历史 full-loop 实现、exact oracle、topology pass 和既有 HPC 打包方式，然后自行制定具体实施计划、完成代码修改、测试并生成可提交的 HPC 包。

本 prompt 只规定科学目标、不可变约束和交付要求。具体模块划分、类设计、函数接口、调度方式和优化细节由你在审查现有代码后决定。优先复用现有可靠实现，不要平行重写核心物理代码。

==================================================
一、任务目标
==================================================

建立一次真正从头开始的 topology-aware full active-learning loop：

    initial random/spatial-filling exact points
    -> train surrogates
    -> acquisition scoring
    -> select new exact points
    -> update models
    -> progressively focus on all relevant phase boundaries
    -> joint convergence

该循环要同时学习：

1. normal / superconducting boundary；
2. uniform-SC / FFLO boundary；
3. trivial / topological gapped-SC boundary；
4. trivial / nodal boundary；
5. topological / nodal boundary。

新 run 不从 dataset_iter035 warm-start。

dataset_iter035 及其 topology pass 只能用于：

    unit tests
    regression checks
    numerical-tolerance reference
    post-run comparison

不得将其样本加入新 full loop 的初始训练集。

建议 run_id：

    active_phase_topology_from_scratch_full_loop_v1

==================================================
二、冻结的物理与数值原则
==================================================

保持 thermodynamic phase criterion 不变：

    任意 Delta > 0 且自由能低于 normal state
        -> superconducting

    superconducting state:
        q_opt 区分 uniform-SC 与 FFLO

    normal state:
        q_not_applicable，不是 q_unresolved

保持 production exact oracle：

    robust_incremental
    + near-zero Delta refinement
    + basin-level local refinement
    + rank_and_cap_k3

不得让 topology、bulk-gap 或 branch completeness 反向否定一个已经可靠的 thermodynamic phase label。

继续区分：

    thermodynamic phase reliability
    q-window / q-grid reliability
    spectral-gap reliability
    topology reliability
    response-side eta / Ic reliability

本任务不处理 eta、Ic 或其他 response-side 问题。

==================================================
三、初始设计
==================================================

使用：

    512 scrambled Sobol points

要求：

    512 = 2^9
    覆盖完整参数域
    使用可配置随机 seed
    保存 Sobol sequence metadata
    不使用 dataset_iter035 作为 initial design

可加入极小随机 jitter，但不能破坏参数域约束。

初始 512 个点全部运行完整 exact oracle，不允许使用 surrogate labels。

batch size 延续当前 production full-loop 的标准值，预期为：

    256 points per acquisition batch

但必须放入配置文件，不能散落硬编码。

==================================================
四、每个 exact point 的输出
==================================================

第一步始终运行 thermodynamic exact oracle，并输出：

    normal / uniform-SC / FFLO
    Delta_opt
    q_opt
    free-energy diagnostics
    exact-label reliability fields
    retained K3 basin information

对于 normal：

    topology = not_applicable
    spectral_status = not_applicable

对于 superconducting points：

1. 复用已经验证的 BdG Hamiltonian builder；
2. 计算 P0 和 Ppi；
3. 计算 full-Brillouin-zone bulk gap Eg；
4. 保存 k_at_minimum_gap；
5. 区分 gapped、nodal 和 unresolved；
6. 仅对 trusted gapped SC 定义 Z2。

Pfaffian 约定：

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

对于 trusted gapped SC：

    P0 * Ppi < 0:
        Z2 = 1
        topology = topological

    P0 * Ppi > 0:
        Z2 = 0
        topology = trivial

对于可靠 Eg <= gap tolerance：

    spectral_status = nodal
    Z2 = not_defined

对于 gap 或输入数值未收敛：

    spectral_status = unresolved
    Z2 = not_defined

nodal 是物理谱状态。

unresolved 是可靠性状态。

二者不得合并。

对保留的 K3 basins，在成本允许的情况下同步保存：

    basin P0
    basin Ppi
    basin bulk gap
    basin Z2
    delta_F_to_ground

这用于未来识别不同 topology basin 之间的一阶 ground-state switch。

==================================================
五、模型总体结构
==================================================

不要构造单一的五分类模型。

保留分层模型：

A. Thermodynamic phase model

    normal
    uniform-SC
    FFLO

B. Delta-related surrogate

    沿用当前 full acquisition 中已经验证的实现

C. Bulk-gap surrogate

    回归适当变换后的 Eg，例如 log-scaled Eg
    只使用 bulk-gap reliable 的 SC exact points训练

D. Pfaffian surrogates

    独立回归 P0
    独立回归 Ppi

第一版不强制加入独立的 Z2 classifier。

优先通过 P0、Ppi ensemble prediction 和 bulk-gap prediction
产生 topology/spectral acquisition。

具体模型类型、ensemble 实现和 uncertainty estimator
应尽量复用现有 active-learning 框架，由你审查代码后决定。

必须加入冷启动保护：

    样本不足时禁止训练不稳定模型
    对应 acquisition quota 自动转移到 coverage/exploration
    不得伪造第二类样本
    不得因单类训练失败而终止整个 loop

==================================================
六、Acquisition 的简单总体结构
==================================================

最终只保留三个顶层 channel：

    A_phase
    A_spectral
    A_coverage

不要建立大量彼此竞争的 acquisition channel。

--------------------------------------------------
6.1 A_phase
--------------------------------------------------

完整保留已经验证过的 full thermodynamic acquisition 语义，包括：

    normal/SC uncertainty
    uniform-SC/FFLO uncertainty
    B_delta gating
    active-pool narrowing
    sampling-power annealing
    exploration annealing
    R_obs
    R_batch
    high-confidence phase-interior handling

A_phase 负责：

    normal / SC boundary
    uniform-SC / FFLO boundary

normal/SC 边界仍由 thermodynamic phase、Delta 和自由能信息决定。

不能直接用 quasiparticle bulk gap 驱动 normal/SC acquisition，
因为 normal region 本身可能具有零能谱。

--------------------------------------------------
6.2 A_spectral
--------------------------------------------------

A_spectral 只在预测 superconducting、重点是 FFLO 的区域工作。

它由两个互补信息组成：

    bulk-gap boundary uncertainty
    Pfaffian-zero/sign uncertainty

Bulk-gap 部分寻找：

    gapped / nodal boundary

因此自动同时覆盖：

    trivial / nodal
    topological / nodal

具体属于哪一种边界，由 exact calculation 后 gapped 一侧的 Z2 决定。

Pfaffian 部分寻找：

    P0 = 0
    Ppi = 0

用于强化连续的 trivial / topological 临界线。

要求：

    分别建模 P0 和 Ppi
    不只建模 P0*Ppi
    不硬编码只有 P0 会变号

A_spectral 必须有软门控：

    predicted SC probability
    predicted Delta > 0
    predicted FFLO probability
    uniform/FFLO boundary halo

其语义必须保证：

    deep normal 被抑制
    Delta -> 0 的 SC/normal 边界主要交给 A_phase
    Delta > 0 的 FFLO interior 可以被 A_spectral 高分选中

原有的 high-confidence thermodynamic phase-interior penalty
不得直接压制 A_spectral。

因为 topo/trivial 边界可能位于高度可信的 FFLO interior。

对 bulk-gap score 和 Pfaffian score：

    先在各自有效候选池内归一化或 rank-normalize
    再进行简单组合

推荐保持“任意一个高即可入选，二者同时高有少量奖励”的语义。

具体公式由你根据现有 acquisition 框架实现，
但不要让 raw numerical scale 决定哪个 score 支配 batch。

--------------------------------------------------
6.3 A_coverage
--------------------------------------------------

始终保留 10% 左右的 coverage/exploration。

建议语义上拆为：

    global parameter-domain coverage
    predicted FFLO-interior coverage

目标：

    防止 phase classifier 早期错误门控
    防止遗漏未被 bracket 的 SC island
    防止遗漏未被 bracket 的 topology/nodal island

使用：

    kNN fill distance
    maximin distance
    或现有代码中等价、稳定的局部覆盖指标

不要使用未过滤的 full-convex-hull Delaunay circumradius
作为主要 acquisition score。

==================================================
七、Batch 配额
==================================================

初始 production 配额：

    A_phase       45%
    A_spectral    45%
    A_coverage    10%

当 thermodynamic map 已经连续若干轮稳定后，
允许一次简单的 late-stage 切换：

    A_phase       25%
    A_spectral    65%
    A_coverage    10%

阶段切换依据应来自已有 StopController 的 phase stability 指标，
而不是固定 iteration number。

具体整数 quota、rounding 和 backfill 规则由你实现。

各 channel：

    先生成超额候选
    去重
    应用 observation-distance control
    应用 batch-diversity control
    quota 不足时按明确规则 backfill

必须记录每个 selected point：

    primary acquisition channel
    component scores
    gates
    rank
    diversity penalty
    fallback/backfill reason

==================================================
八、Active pools
==================================================

必须使用彼此独立的候选池：

    phase_active_pool
    spectral_active_pool
    global_coverage_pool

phase_active_pool：

    保持现有 phase-boundary narrowing 逻辑

spectral_active_pool：

    predicted SC/FFLO support
    uniform/FFLO halo
    FFLO interior
    spectral coverage holes

global_coverage_pool：

    完整参数域中的少量空间填充候选

最终 batch 从三个 pool 的候选并集中选取。

不得先用 phase-boundary pool 裁剪全部候选，
再在剩余点上计算 topology score。

否则 FFLO interior topology boundary 会被系统性遗漏。

==================================================
九、一阶 topology transition 的简单保险
==================================================

第一版不要单独训练复杂的 branch-switch surrogate。

加入一个轻量 deterministic safeguard 即可：

若局部邻近的两个 trusted gapped FFLO points：

    Z2 不同
    两端 bulk gap 都明显大于 nodal tolerance
    参数空间距离处于合理局部邻接尺度

则将它们之间的 bracket point 或 midpoint
加入 spectral 高优先级候选队列。

该 safeguard 用于防止：

    两个有隙自洽 branches 发生一阶 ground-state switch
    导致 ground-state Z2 跳变
    但沿各自 branch 没有显式 bulk-gap closing

必须使用局部 kNN 邻接或经过过滤的局部几何关系，
不得使用可能跨越大空洞的原始 Delaunay 长边。

==================================================
十、训练标签资格
==================================================

Phase model：

    只使用 thermodynamic trusted labels

Bulk-gap model：

    只使用 SC 且 bulk-gap numerical result reliable 的点

Pfaffian models：

    使用输入参数和 Pfaffian 数值可靠的 SC 点
    保留相应 reliability weighting 或 filtering

Z2 只作为 derived label：

    trusted SC
    trusted gapped
    reliable Pfaffian signs

以下点不得进入 trusted Z2 training：

    nodal
    spectral unresolved
    topology unresolved
    thermodynamic untrusted

normal 永远不能被作为 trivial SC。

Topology failure 不得阻止一个可靠 thermodynamic point
进入 phase training dataset。

==================================================
十一、StopController
==================================================

复用现有 StopController 的基本结构与 trusted-surprise 语义。

full loop 结束应要求：

    thermodynamic_main_converged
    AND
    spectral_topology_main_converged

Thermodynamic side 继续监控：

    phase-map change
    normal/SC boundary shift
    uniform/FFLO boundary shift
    boundary coverage
    trusted phase surprise
    hard-risk frontier

Spectral/topology side至少监控：

    gapped/nodal map change
    nodal-boundary shift and coverage
    trusted gapped Z2 map change
    topo/trivial boundary shift and coverage
    trusted spectral/topology surprise
    connected-component stability
    unresolved/hard-risk frontier

缺失边界不得自动视为 shift = 0。

必须区分：

    boundary absent because physics has one phase
    boundary not yet discovered
    boundary temporarily missing from surrogate
    boundary unresolved because of numerical risk

正式 convergence 与 hard-risk frontier 继续分开。

少量 unresolved points 不应无限阻止主相图收敛，
但必须保留独立的 numerical-frontier audit。

具体阈值优先复用现有参数尺度并配置化，
不要在多个源文件中硬编码。

==================================================
十二、代码实现要求
==================================================

先审查当前 repository，再决定最小改动方案。

优先：

    复用现有 exact oracle
    复用现有 phase acquisition
    复用现有 checkpoint/resume
    复用现有 model-training infrastructure
    复用 topology_pass 中已验证的 Pfaffian 和 bulk-gap code

不要：

    复制一套独立 Hamiltonian
    改变原 full acquisition 的 production profile
    覆盖旧 run outputs
    将新逻辑偷偷混入旧 frozen run
    建立无法恢复的单体长任务

新代码应作为：

    独立 run profile
    独立配置
    独立 output directory
    独立 run_id

支持：

    deterministic seed
    checkpoint every iteration
    resume from latest valid checkpoint
    interrupted-task recovery
    per-point failure logging
    timeout/rerun accounting
    partial batch recovery
    clean stop
    dry run
    small smoke test

==================================================
十三、测试与验证
==================================================

由你制定具体测试计划，但至少覆盖：

1. Pfaffian regression test

    与现有 topology pass 结果比较
    检查 P0、Ppi 和 Z2

2. Bulk-gap regression test

    CPU/GPU 或不同 backend 数值一致性
    gap tolerance 和 nodal/unresolved 分类

3. Label hierarchy test

    normal -> topology not_applicable
    nodal -> Z2 not_defined
    unresolved 不被当作物理相
    topology failure 不影响 thermodynamic label append

4. Acquisition unit tests

    deep normal 不被 spectral score 主导
    FFLO interior topology candidate 可以获得高分
    gapped/nodal boundary 获得高 gap score
    Pfaffian zero 获得高 Pfaffian score
    coverage quota 始终存在
    phase-interior penalty 不误伤 spectral score

5. Cold-start behavior

    当 SC 或 FFLO 样本不足时不崩溃
    无法训练的 spectral quota 正确回流
    不伪造 topology labels

6. One-iteration integration smoke test

    小型 initial design
    train
    score
    select
    exact evaluation
    append
    checkpoint
    resume

7. HPC package validation

    shell syntax
    Python compile/import
    configuration parsing
    archive integrity
    node exclusion
    restart command

不要在本地启动正式 full loop。

只运行足以验证代码路径的小型 smoke test。

==================================================
十四、HPC 包
==================================================

生成一个可直接上传和提交的 HPC package。

建议包名：

    active_phase_topology_from_scratch_full_loop_v1_hpc.tar.gz

包内至少包含：

    source changes or self-contained source tree
    production configuration
    smoke-test configuration
    environment/dependency specification
    submit script
    resume script
    status/monitoring script
    result collection script
    README with exact commands
    manifest with SHA256 checksums

请先检查现有 HPC package 和 scheduler 脚本，
尽量保持项目既有目录结构和调度方式。

正式包应支持：

    preflight
    production submit
    resume
    checkpoint inspection
    failed-task inspection
    final artifact collection

不得自动提交任务。

只生成、验证并打包。

==================================================
十五、必须排除 gpuh01
==================================================

所有会在超算提交的 Slurm 脚本，包括：

    production
    resume
    smoke test
    auxiliary worker jobs
    postprocessing jobs if they request compute nodes

都必须显式包含：

    #SBATCH --exclude=gpuh01

README 中的 sbatch 示例也必须保留该排除条件。

另外加入运行时防护：

    检查 SLURMD_NODENAME 或 hostname
    如果节点为 gpuh01，则立即安全退出
    写出清晰错误信息
    不开始 exact calculation
    不损坏 checkpoint

不要把 gpuh01 放入任何允许节点列表、fallback 节点列表
或 host-specific optimization 配置。

如果集群不是 Slurm，则使用调度器等价的 node-exclusion 机制，
但仍保留运行时 hostname guard。

==================================================
十六、编码与跨平台问题
==================================================

HPC package 必须专门审计编码问题。

要求：

1. Python、YAML、JSON、Markdown：

       UTF-8
       no BOM
       LF line endings

2. shell 和 Slurm scripts：

       尽量只使用 ASCII
       LF line endings
       no BOM
       不使用全角标点、智能引号或 Unicode dash

3. 文件名、目录名、配置 key：

       只使用 ASCII
       不使用空格
       不使用 Delta、pi、mu 等 Unicode 字符作为机器字段名
       使用 delta、pi、mu 等 ASCII key

4. shell 环境中显式设置：

       export LANG=C.UTF-8
       export LC_ALL=C.UTF-8
       export PYTHONUTF8=1
       export PYTHONIOENCODING=utf-8

5. JSON machine outputs：

       必须保证严格可解析
       NaN/Infinity 的处理规则明确
       不依赖非标准编码

6. archive 前执行：

       Python compile/import validation
       bash -n validation
       UTF-8 validation
       BOM scan
       CRLF scan
       broken symlink scan
       tar listing validation
       checksum generation

7. 不在包中写入：

       Windows absolute paths
       local-user-specific paths
       非 ASCII 临时目录名
       无法在 compute node 解码的日志文件名

8. 若 Parquet 为必须格式：

       在 environment 中明确声明 pyarrow

   否则：

       提供 CSV/NPZ 等可靠 fallback
       不得因为 pyarrow 缺失让 full loop 失败

==================================================
十七、资源与性能原则
==================================================

Topology/Pfaffian/bulk-gap 后处理不是主要瓶颈。

主要计算成本仍是 thermodynamic exact oracle。

因此：

    不要为了 topology code 重新设计整个并行框架
    将 topology evaluation 放在 exact-point 后处理路径
    使用批量化和可靠 checkpoint
    优先优化 exact tasks 的调度、恢复和负载均衡

不要默认 GPU 一定更快。

复用现有 benchmark 逻辑或进行小型 backend preflight，
随后自动选择可靠实现。

所有数值关键计算保持：

    float64 / complex128

除非现有项目已验证其他精度模式。

==================================================
十八、交付物
==================================================

完成后请提供：

1. 实施总结

    采用了什么架构
    修改了哪些模块
    哪些旧代码被复用
    哪些配置为默认值
    仍有哪些风险

2. changed-files manifest

3. 测试报告

4. smoke-test 报告

5. acquisition score diagnostic plots

    至少证明：
        phase channel 聚焦 thermodynamic boundaries
        spectral channel 可进入 FFLO interior
        gap score 响应 nodal boundary
        Pfaffian score 响应 topology boundary
        coverage points 分布合理

6. HPC archive

7. archive SHA256

8. 精确提交命令

9. 精确 resume 命令

10. gpuh01 exclusion verification

11. encoding validation report

12. 尚未执行的正式运行步骤列表

==================================================
十九、验收条件
==================================================

只有满足以下条件才认为任务完成：

    new full loop is cold-start
    initial design is exactly 512 scrambled Sobol points
    dataset_iter035 is not used for training initialization
    original thermodynamic acquisition remains available and unchanged
    spectral acquisition can search FFLO interior
    normal metallic gaplessness cannot dominate spectral acquisition
    trivial/topological and both nodal boundaries are covered
    nodal and unresolved are separate
    coverage does not depend on raw Delaunay circumradius
    checkpoint/resume works
    small integration smoke test passes
    all HPC jobs exclude gpuh01
    runtime hostname guard rejects gpuh01
    archive contains only valid UTF-8/ASCII-safe files
    package passes syntax, import, archive and checksum checks
    no production job has been submitted automatically

开始时先输出一个简洁的 repository audit 和 implementation plan。
随后自行完成实现、测试、HPC 打包和最终报告。
不要等待用户逐步批准每个内部编码决定；只有遇到无法从代码库判断、
且会改变科学定义或破坏已有数据的关键歧义时才暂停。