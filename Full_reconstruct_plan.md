# 修正重写计划大纲

现在需要在重写 acquisition function 之前，先清理 active learning 流程中错误加入的 midpoint point selection / geometric bracket selection 逻辑。

背景：
之前为了边界细化，代码中加入了基于 exact dataset 中几何 bracket midpoint 的非 ML 选点策略。这个策略本质上是 bisection / midpoint refinement，不是 ML-guided active learning。它会绕过 acquisition function，使新 exact calls 不再完全由 ML uncertainty、boundary proximity、numerical risk 等 acquisition components 决定。

现在要求：
彻底禁用 midpoint quotas，并移除或旁路所有 midpoint candidate selection 逻辑。后续边界候选点只能由 acquisition function 产生。

核心原则：
1. 不再使用 geometric bracket midpoint 作为 active-learning candidate source。
2. 不再为 midpoint selection 保留 quota。
3. 不再把 midpoint candidates 与 acquisition candidates 混合排序。
4. 不再根据 exact boundary brackets 直接生成 new exact calls。
5. 如果未来需要 boundary refinement，也必须通过 ML-guided boundary proposal 实现，即由 acquisition score 中的 B_delta、B_q_SC、classifier uncertainty、gradient score 等项自然给出，而不是由 exact-data midpoint rule 给出。
6. exact oracle 仍然只负责计算 selected candidates，不负责决定 candidate 位置。
7. 所有最终 selected points 必须能追溯到 acquisition score。

请先只做清理和重构，不要同时重写新的 acquisition function。新的 acquisition function 会在下一步单独实现。

一、需要搜索并定位的关键词

请在项目中搜索以下关键词和相关变体：

- midpoint
- mid_point
- boundary_midpoint
- bracket
- brackets
- bisection
- bisect
- boundary_refine
- boundary_refinement
- midpoint_quota
- boundary_quota
- midpoint_candidates
- bracket_candidates
- selected_midpoints
- boundary_band_midpoint
- refine_midpoints

请列出所有命中的文件、函数、配置项和调用路径。

二、需要移除或禁用的内容

请移除或禁用以下逻辑：

1. midpoint candidate generation
   - 不再从 exact dataset 中寻找相邻异相点。
   - 不再根据两个 exact points 的几何中点生成候选点。
   - 不再根据 normal/SC bracket 或 uniform SC/FFLO bracket 生成 midpoint。
   - 不再根据 boundary-band endpoints 生成 midpoint。

2. midpoint quota
   - 删除或默认禁用所有 midpoint_quota / boundary_midpoint_quota / bracket_quota 配置。
   - 如果为了兼容旧配置必须保留字段，则字段必须默认设为 0，并在日志中明确说明：
     midpoint selection is disabled; this config field is ignored.

3. mixed candidate source
   - 不允许 selected batch 由 acquisition candidates + midpoint candidates 拼接而成。
   - 不允许 midpoint candidates 在 acquisition ranking 之后插入。
   - 不允许 midpoint candidates 绕过 acquisition score。
   - 最终 selected batch 只能来自 acquisition-ranked candidate grid。

4. boundary midpoint fallback
   - 删除所有“如果 acquisition 不足，则用 midpoint 补足 batch”的逻辑。
   - 如果 acquisition 没有足够可选点，应该明确返回不足数量并输出原因，而不是用 midpoint fallback。

三、允许保留的内容

以下内容可以保留，但不能参与选点：

1. boundary diagnostics
   - 可以继续提取 normal/SC boundary segments。
   - 可以继续提取 uniform SC/FFLO boundary segments。
   - 可以继续保存 boundary diagnostic report。
   - 但这些 diagnostic boundary segments 不能生成 exact-call candidates。

2. exact quality gate
   - 保留 q-window risk 检查。
   - 保留 delta boundary-band metadata。
   - 保留 rerun-required / trusted exact / boundary-band normal 分类。
   - 但 quality gate 不能反向生成 midpoint candidate。

3. visualization
   - 可以继续画 boundary brackets、boundary segments、diagnostic midpoints。
   - 但图中的 midpoint 或 bracket 不能进入 active learning selected batch。

四、需要建立新的候选点流向

清理后，candidate selection 的唯一合法数据流应该是：

candidate grid
    → ML predictions
    → compute A0_main(x)
    → apply M_unseen(x)
    → apply R_obs(x)
    → apply R_batch(x)
    → selected acquisition batch
    → exact BdG oracle

不允许出现：

exact dataset
    → geometric brackets
    → midpoint candidates
    → selected batch

如果代码中仍保留 boundary proposal 接口，请改名或改注释，明确它现在只能接收 ML acquisition map，而不能从 exact bracket midpoint 生成候选。

五、selected candidate 必须包含 acquisition trace

每个 selected candidate 输出时必须包含：

- grid index
- physical coordinate: kT, JA
- A0_main
- Ar_final 或 final_score
- A_phase
- A_numerical
- A_explore
- selection_rank
- selection_source

其中 selection_source 必须恒为：

"acquisition"

不允许出现：

"midpoint"
"bracket"
"bisection"
"boundary_midpoint"

如果代码中还有这些 source，需要删除或 raise error。

六、配置清理

请检查所有 config/dataclass/yaml/json 参数。

需要删除或禁用：

- midpoint_quota
- boundary_midpoint_quota
- bracket_quota
- use_midpoint_selection
- use_boundary_midpoints
- midpoint_radius
- midpoint_refinement_enabled
- bisection_enabled

如果为了兼容旧文件保留字段，请满足：

1. 默认值必须是 False 或 0。
2. 如果用户设置为 True 或正数，程序必须 raise ValueError：
   "Midpoint-based selection has been disabled. Use ML-guided acquisition instead."
3. 不允许静默忽略用户显式开启 midpoint 的配置。

七、日志要求

每轮 active learning 开始时打印：

- candidate grid size
- available candidates after masks
- selected batch size
- selection source: acquisition only
- midpoint selection: disabled

如果检测到旧 midpoint config 被设置，直接报错。

八、测试要求

请添加或更新测试，至少包含以下检查：

1. test_no_midpoint_candidates_generated
   - 构造一个 toy exact dataset，其中明显存在两个异相点 bracket。
   - 确认程序不会生成几何 midpoint candidate。

2. test_selected_source_is_acquisition_only
   - 运行一次 toy active learning selection。
   - 检查所有 selected candidates 的 selection_source 都是 "acquisition"。

3. test_midpoint_config_rejected
   - 如果配置中 use_midpoint_selection=True 或 midpoint_quota>0，必须 raise ValueError。

4. test_no_fallback_to_midpoints
   - 构造 acquisition 可选点不足的情况。
   - 程序应该返回少于 batch_size 的 selected list 或明确报错。
   - 不允许用 midpoint candidates 补足。

5. test_boundary_diagnostics_do_not_select
   - boundary diagnostics 可以输出 boundary segments。
   - 但 selected candidates 不得来自这些 segments 的 midpoint。

九、不要做的事情

- 不要重写 neural network model。
- 不要重写 exact BdG oracle。
- 不要重写 quality gate。
- 不要重新设计 A0_main。
- 不要加入新的 midpoint-like heuristic。
- 不要用 “boundary bracket midpoint with score” 的方式伪装成 acquisition。
- 不要把 exact-data geometric midpoint 加入 candidate pool 后再赋 acquisition score。
  这仍然是 midpoint candidate source，不允许。

十、完成后请输出报告

修改完成后，请输出：

1. 删除或禁用的 midpoint 相关函数/配置列表。
2. 当前唯一合法 selection pipeline。
3. selected candidate 的字段示例。
4. 测试结果。
5. 明确说明：
   "All selected points are now acquisition-guided. Geometric midpoint selection is disabled."

请先输出修改计划，不要直接改代码。
计划中需要说明：
- 哪些文件会被改
- 哪些函数会被删除或旁路
- 如何保证 midpoint 不再进入 selected batch
- 如何保持 boundary diagnostics 但不让 diagnostics 参与选点

# 原始Acquisition Score 重构计划

现在需要重构 active learning 中第七节的原始 acquisition function A0(x)。

背景：
当前报告中旧的 acquisition score 为：

S(x) =
    wc Uc(x)
  + wr Ur(x)
  + wΔ BΔ(x)
  + wq Bq(x)
  + wη Bη(x)
  + wg G(x)
  + wd D(x)
  + wedge Eq(x)
  + wref RΔ(x)
  + wext Eext(x)

旧版本存在几个问题：
1. 所有项直接线性相加，物理目标混在一起。
2. Bη 追踪 η = 0 response sign boundary，但它不是 thermodynamic phase boundary，不应进入主分数。
3. Diversity score D(x) 和后处理中的采样覆盖/排斥机制重复，应从 A0 中移除。
4. q-boundary 和 q-window risk 在 normal 区域可能没有物理意义，需要乘 superconducting probability gate。
5. regression ensemble uncertainty 不应该对 Δ, q, η, Ic+, Ic- 简单平均，需要拆分。
6. gradient score 需要拆分为 phase-gradient 和 response-gradient。
7. Δ-refinement risk 与 BΔ 高度重叠，当前版本先移除。
8. extrapolation risk 低权重保留，但必须和 uncertainty 相乘。

现在要实现新的 acquisition 设计。

一、总体结构

请将原来的单一 S(x) 改为分层 score：

A_phase(x):
    主相边界分数，用于 normal/SC 和 uniform SC/FFLO 边界学习。

A_numerical(x):
    数值风险分数，当前主要包含 q-window risk。

A_response(x):
    response-function 分数，当前不进入主分数，只单独输出给 quota 或诊断使用。

A_explore(x):
    外推探索分数，低权重，只在模型不确定时生效。

最终主分数定义为：

A0_main(x) =
    A_phase(x)
  + A_numerical(x)
  + A_explore(x)

注意：
- Bη 不进入 A0_main。
- Diversity D(x) 不进入 A0_main。
- Δ-refinement risk RΔ 不进入 A0_main。
- A_response 只输出，不参与主排序，除非上层 batch quota 显式要求 response queue。

最终被选点前，还会经过：

Ar(x) = A0_main(x) * M_unseen(x) * R_obs(x) * R_batch(x)

本次任务只负责 A0_main 及其 components 的计算，不要重写 M_unseen / R_obs / R_batch，除非需要对接接口。

二、输入和已有预测量

假设当前已有以下候选网格上的预测或中间量：

- phase_prob, shape = grid_shape + (n_phase,)
  phases = normal, uniform_SC, FFLO

- regression ensemble predictions:
  delta_pred_members
  q_pred_members
  eta_pred_members
  ic_plus_pred_members
  ic_minus_pred_members

或者已经有：
  delta_pred_mean, delta_pred_std
  q_pred_mean, q_pred_std
  eta_pred_mean, eta_pred_std
  ic_plus_pred_mean, ic_plus_pred_std
  ic_minus_pred_mean, ic_minus_pred_std

- coords_norm, shape = grid_shape + (dim,)

- optional:
  q_window_edge_score_raw
  extrapolation_score_raw

需要保证所有 component 的 shape 都等于 grid_shape。

三、superconducting probability gate

实现：

P_normal = phase_prob[..., normal_index]
P_uniform = phase_prob[..., uniform_SC_index]
P_FFLO = phase_prob[..., FFLO_index]

P_SC = 1 - P_normal
或
P_SC = P_uniform + P_FFLO

要求：
- P_SC clip 到 [0, 1]
- q-related terms 必须乘 P_SC
- η response terms 如果输出，也必须乘 P_SC

四、classifier uncertainty 改为 entropy + top-2 margin 混合

旧版本：
Uc = 1 - max_c p(c|x)

新版本：
同时计算 normalized entropy 和 top-2 margin uncertainty。

1. normalized entropy:

H = -sum_c p_c log(p_c)
H_norm = H / log(n_phase)

2. top-2 margin uncertainty:

取最大概率 p1 和第二大概率 p2：

margin = p1 - p2

U_margin = exp(-(margin / tau_margin)^2)

默认：
tau_margin = 0.2

3. 混合：

U_cls_mix =
    w_cls_entropy_inner * H_norm
  + w_cls_margin_inner  * U_margin

默认内部权重：
w_cls_entropy_inner = 0.4
w_cls_margin_inner  = 0.6

要求：
- H_norm, U_margin, U_cls_mix 都 clip 到 [0, 1]
- 输出 debug components:
  cls_entropy
  cls_margin_uncertainty
  cls_uncertainty_mix

五、regression ensemble uncertainty 拆分

不要再使用一个平均的 Ur。

分别计算：

U_delta = normalized_std(delta_pred)
U_q     = normalized_std(q_pred)
U_eta   = normalized_std(eta_pred)
U_icp   = normalized_std(ic_plus_pred)
U_icm   = normalized_std(ic_minus_pred)

其中 normalized_std 可以使用候选网格上的 min-max normalization：

U = (std - finite_min) / (finite_max - finite_min + eps)

或者已有项目中统一的 robust normalization。

要求：
- 每个 U 都归一化到 [0, 1]
- nan / inf 设为 0 或不可选，按现有策略处理，但不要静默传播 nan
- 主相图分数只使用 U_delta 和 U_q
- U_eta, U_icp, U_icm 只输出给 response diagnostics，不进入 A0_main

定义：

U_reg_phase =
    0.5 * U_delta
  + 0.5 * U_q

可选：

U_reg_response =
    0.6 * U_eta
  + 0.2 * U_icp
  + 0.2 * U_icm

但 U_reg_response 不进入 A0_main。

六、保留 Δ-boundary proximity

保留：

B_delta = exp(-abs(delta_pred_mean - delta_eps) / delta_width)

默认：
delta_eps = 1e-3
delta_width = 5e-3

要求：
- B_delta clip 到 [0, 1]
- B_delta 进入 A_phase
- 不要删除
- 不要用 R_delta_refine 替代它

七、q-boundary proximity 加 superconducting gate

旧版本：

B_q = exp(-abs(abs(q_pred) - q_eps) / q_width)

新版本：

B_q_raw = exp(-abs(abs(q_pred_mean) - q_eps) / q_width)

B_q_SC = P_SC * B_q_raw

默认：
q_eps = 1e-2
q_width = 2e-2

要求：
- B_q_raw 和 B_q_SC 都输出 debug
- A_phase 使用 B_q_SC，不使用 B_q_raw
- normal 区域中即使 q_pred 靠近 q_eps，也不应该获得高 q-boundary 分数

八、Bη 不进入主分数

旧版本：

B_eta = exp(-abs(eta_pred) / eta_width)

新版本：
仍可计算 B_eta_response 作为诊断或 response queue，但不得加入 A0_main。

定义：

B_eta_raw = exp(-abs(eta_pred_mean) / eta_width)
B_eta_response = P_SC * B_eta_raw

默认：
eta_width = 0.05

要求：
- B_eta_response 输出到 debug components
- A0_main 不包含 B_eta_response
- 代码中明确注释：
  eta = 0 is a response-function sign boundary, not a thermodynamic phase boundary

九、gradient score 拆分

旧版本 G(x) 混合 predicted η, Δ, q 的梯度。

新版本拆成：

1. G_phase
   用于主相图边界 refinement。

建议：

G_delta = normalized_gradient(delta_pred_mean)
G_q     = normalized_gradient(abs(q_pred_mean)) * P_SC
G_phase = 0.5 * G_delta + 0.5 * G_q

2. G_response
   只用于诊断或 response queue。

G_eta = normalized_gradient(eta_pred_mean) * P_SC
G_response = G_eta

要求：
- A_phase 使用 G_phase
- A0_main 不使用 G_response
- G_response 输出 debug
- gradient 计算必须在 normalized parameter grid 上进行
- 如果已有 gradient 工具函数，复用它
- gradient 结果归一化到 [0, 1]
- 不要让 η gradient 进入主相边界分数

十、移除 diversity score D(x)

旧版本 D(x) 是 distance to existing exact data 的加性 diversity score。

新版本：
D(x) 不再进入 A0_main。
历史点覆盖和重复控制由后处理处理：

Ar(x) = A0_main(x) * M_unseen(x) * R_obs(x) * R_batch(x)

要求：
- 删除或禁用 w_diversity 对 A0_main 的贡献
- 可以保留 D(x) debug 输出，但默认不参与主分数
- 不要在 A0_main 里再加入 distance_to_sampled_points

十一、q-window risk 归入 numerical-risk queue，并乘 P_SC

旧版本 Eq 进入总分。

新版本：

E_q_raw:
    原有 q-window risk score
    例如 q_pred 接近 q_min/q_max，或者 high-JA q-risk 区域

E_q_SC = P_SC * E_q_raw

A_numerical = w_q_edge_risk * E_q_SC

默认：
w_q_edge_risk = 0.4

注意：
旧报告中 q-window risk 默认权重较高。新版本中它仍然保留，但作为 numerical-risk 项，不应压过主相边界。

要求：
- E_q_raw, E_q_SC 输出 debug
- A_numerical 使用 E_q_SC
- normal 区域不应因为 q-window risk 得高分

十二、移除 Δ-refinement risk RΔ

旧版本 R_delta_refine 与 B_delta 重复。

新版本：
- A0_main 不包含 R_delta_refine
- 不计算或只保留为 debug
- 当前版本先移除，等以后有 ΔF+ 或 boundary-band metadata 后再重新定义

要求：
- 删除 w_delta_refine_risk 对主分数的贡献
- 注释说明：
  R_delta_refine is disabled because it duplicates B_delta until finite-resolution boundary-band metadata is modeled explicitly.

十三、extrapolation risk 低权重保留，且和 uncertainty 相乘

旧版本 E_ext 直接进入总分。

新版本：

E_ext_raw:
    原有 extrapolation risk score

E_ext_uncertain =
    E_ext_raw * max(U_cls_mix, U_reg_phase)

A_explore =
    w_extrapolation * E_ext_uncertain

默认：
w_extrapolation = 0.15

要求：
- E_ext_raw 和 E_ext_uncertain 输出 debug
- 不允许单纯因为“远离 warm-start coverage”就获得高分
- 只有 extrapolation + uncertainty 同时存在，才有探索分数

十四、新权重建议

请使用如下默认权重：

A_phase =
    1.0 * U_cls_mix
  + 0.6 * U_reg_phase
  + 1.0 * B_delta
  + 0.9 * B_q_SC
  + 0.5 * G_phase

A_numerical =
    0.4 * E_q_SC

A_explore =
    0.15 * E_ext_uncertain

A0_main =
    A_phase + A_numerical + A_explore

A_response =
    0.3 * B_eta_response
  + 0.3 * G_response
  + 0.3 * U_reg_response

但 A_response 不进入 A0_main。

要求：
- 所有 group score 都输出：
  A_phase
  A_numerical
  A_explore
  A_response
  A0_main

十五、归一化和数值安全

实现一个统一的 normalize_component()：

def normalize_component(arr, method="minmax", clip=True, eps=1e-12):
    ...

要求：
- 输入 arr shape = grid_shape
- 只使用 finite values 计算 min/max
- 如果 finite_max - finite_min 很小，则返回 zeros_like
- nan / inf 处理要明确
- 输出默认 clip 到 [0, 1]

对已经天然在 [0,1] 的项，例如 probabilities、entropy、exponential proximity，可以只 clip，不重复 minmax。

十六、函数结构建议

请新增或重构以下函数：

1. compute_classifier_uncertainty_components(phase_prob, tau_margin=0.2)

返回 dict:
- cls_entropy
- cls_margin_uncertainty
- cls_uncertainty_mix
- P_SC

2. compute_regression_uncertainty_components(pred_members or pred_std)

返回 dict:
- U_delta
- U_q
- U_eta
- U_ic_plus
- U_ic_minus
- U_reg_phase
- U_reg_response

3. compute_boundary_components(delta_mean, q_mean, eta_mean, P_SC, config)

返回 dict:
- B_delta
- B_q_raw
- B_q_SC
- B_eta_raw
- B_eta_response

4. compute_gradient_components(delta_mean, q_mean, eta_mean, P_SC, coords_norm)

返回 dict:
- G_delta
- G_q
- G_phase
- G_eta
- G_response

5. compute_numerical_risk_components(q_mean, P_SC, config)

返回 dict:
- E_q_raw
- E_q_SC

6. compute_exploration_components(E_ext_raw, U_cls_mix, U_reg_phase)

返回 dict:
- E_ext_raw
- E_ext_uncertain

7. compute_A0_main(components, weights)

返回:
- A0_main
- A_phase
- A_numerical
- A_explore
- A_response
- debug_components

十七、配置项

请建立或更新 AcquisitionConfig，包含：

- tau_margin = 0.2

- delta_eps = 1e-3
- delta_width = 5e-3

- q_eps = 1e-2
- q_width = 2e-2

- eta_width = 0.05

- weights:
    w_cls_mix = 1.0
    w_reg_phase = 0.6
    w_delta_boundary = 1.0
    w_q_boundary_sc = 0.9
    w_gradient_phase = 0.5
    w_q_edge_risk = 0.4
    w_extrapolation = 0.15

- response weights:
    w_eta_response = 0.3
    w_gradient_response = 0.3
    w_reg_response = 0.3

要求：
- 不要再使用旧的 w_diversity
- 不要再使用旧的 w_delta_refine_risk
- 旧配置可以保留兼容，但必须默认禁用，并在日志中说明

十八、输出 debug 表

每轮 acquisition 后，输出或保存以下统计：

- min/max/mean of A0_main
- min/max/mean of A_phase
- min/max/mean of A_numerical
- min/max/mean of A_explore
- min/max/mean of A_response
- top 20 candidates by A0_main with component breakdown
- top 20 candidates by A_response with component breakdown, but do not send them to main exact queue unless response quota is enabled

每个 selected candidate 输出字段：
- index
- physical coordinates: kT, JA
- A0_main
- A_phase
- A_numerical
- A_explore
- A_response
- cls_uncertainty_mix
- U_reg_phase
- B_delta
- B_q_SC
- G_phase
- E_q_SC
- E_ext_uncertain
- B_eta_response
- G_response

十九、测试要求

请添加 toy test：

- 构造 2D grid: n_ja=20, n_kt=30
- 构造 phase_prob，使某条人工边界附近 top-2 margin 变小
- 构造 delta_mean，使一条区域接近 delta_eps
- 构造 q_mean，使另一条区域接近 q_eps
- 构造 eta_mean，使大量区域接近 0
- 检查：
    1. B_eta_response 不进入 A0_main
    2. B_q_SC 在 P_SC 很小时接近 0
    3. E_q_SC 在 P_SC 很小时接近 0
    4. A0_main 中没有 diversity 项
    5. A0_main 中没有 delta_refine_risk 项
    6. E_ext_uncertain 只有在 E_ext_raw 和 uncertainty 同时大时才大
    7. 所有输出 shape 正确
    8. 没有 nan / inf

二十、请先输出修改计划，不要直接改代码

先输出：
1. 需要修改哪些文件
2. 每个文件新增或修改哪些函数
3. 新 A0_main 的数据流
4. 哪些旧项被移除
5. 哪些旧项被保留但改名或 gated
6. 新旧配置如何兼容
7. 如何测试

确认计划后再改代码。

# 原始Acquisition Score 后处理重构计划

现在需要重构 active learning 中从原始 acquisition score A0(x) 到修正后 score Ar(x) 的处理逻辑。

背景：
我已经有一个 neural-network-based active learning 模块，可以为每个候选参数点 x 计算原始 acquisition score：

A0(x)

现在需要把它修正为：

Ar(x) = A0(x) * M_unseen(x) * R_obs(x) * R_batch(x)

其中：

1. M_unseen(x)
   - 负责排除已经计算过、正在计算、无效、以及本轮已经选中的点。
   - 对固定 grid 候选点，必须优先使用整数 grid index mask，不能依赖浮点坐标判断重复。
   - 允许额外加入“极小半径重复判断”作为安全检查，用于处理外部导入的浮点坐标或 pending 点坐标。
   - 但这个极小半径只用于判断“是否同一个点”，不能作为正常采样排斥半径。
   - 默认 exact_duplicate_radius_norm = 1e-6，但需要检查它是否小于最小网格间距的 0.1 倍；如果不满足，需要报错，而不是静默运行。

2. R_obs(x)
   - 负责历史已采样点的软排斥。
   - 不允许再对历史已采样点使用固定半径硬屏蔽。
   - 使用增量最近距离数组 d_obs_min 来维护每个候选点到历史已采样点的最近距离。
   - 每轮新增一批真实计算完成的点后，只用新点增量更新 d_obs_min，不要每轮重新对所有历史点计算距离。
   - R_obs 的公式为：

     R_obs(x) = repulsion_floor + (1 - repulsion_floor) * (1 - exp(-(d_obs_min(x) / ell_obs)^2))

   - repulsion_floor 默认 0.05，防止历史点附近的 score 被完全压死。
   - ell_obs 在归一化参数空间中定义，默认可以设为 0.05。

3. R_batch(x)
   - 负责本轮 batch 内部的软排斥。
   - 只对本轮已经选中的点计算距离，不要和所有历史点重复计算。
   - 每选出一个点后，动态更新 R_batch 或动态更新当前 score。
   - R_batch 的公式为：

     R_batch(x; S_selected) = product over s in S_selected [
         batch_floor + (1 - batch_floor) * (1 - exp(-(d(x, s) / ell_batch)^2))
     ]

   - batch_floor 默认 0.01。
   - ell_batch 默认 0.03。
   - 本轮已经选中的同一个 grid index 必须硬排除，不能靠 R_batch 自动降权。

4. 所有距离必须在归一化参数空间中计算。
   - 需要实现或使用 coords_norm。
   - coords_norm 的 shape 应该是 grid_shape + (dim,)
   - 例如二维相图时，coords_norm.shape = (n_ja, n_kt, 2)
   - coords_norm[..., 0] = normalized kT
   - coords_norm[..., 1] = normalized JA
   - 归一化方式：
       kT_norm = (kT - kT_min) / (kT_max - kT_min)
       JA_norm = (JA - JA_min) / (JA_max - JA_min)
   - 如果将来扩展到高维参数空间，代码应该自然支持 dim > 2。

核心目标：
把原来的“固定半径硬屏蔽”改成：
- computed / pending / invalid / exact selected duplicate：硬排除
- 历史点：软排斥 R_obs
- 本轮已选点：软排斥 R_batch + exact index 硬排除

不要再因为历史点半径覆盖导致几十轮后无点可选。

请按下面结构修改或新增代码。

一、需要实现的数据结构

请建立一个 AcquisitionState 或等价的轻量数据结构，至少包含：

- computed_mask: bool array, shape = grid_shape
- pending_mask: bool array, shape = grid_shape
- invalid_mask: bool array, shape = grid_shape
- d_obs_min: float array, shape = grid_shape
- coords_norm: float array, shape = grid_shape + (dim,)
- grid_shape
- dim

要求：
- computed_mask / pending_mask / invalid_mask 必须和 A0.shape 完全一致。
- d_obs_min 初始化为 +inf。
- coords_norm 使用 float64。
- mask 使用 bool。
- 不要用复杂 class 封装过度；如果已有项目偏函数式，就使用 dataclass 或普通 dict 均可。

二、实现 build_coords_norm()

函数签名建议：

def build_coords_norm(param_grids: dict[str, np.ndarray]) -> np.ndarray:

二维时可以支持：
- param_grids["kT"] = kT_vec
- param_grids["JA"] = JA_vec

返回：
- coords_norm, shape = (n_ja, n_kt, 2)

注意：
- 输出顺序要和 score matrix 的 shape 对齐。
- 如果现有矩阵 shape 是 (n_ja, n_kt)，则 coords_norm[i_ja, i_kt] = [kT_norm, JA_norm]
- 不要搞反 JA/kT 轴。
- 写清楚注释。

三、实现 update_d_obs_min_incremental()

函数签名建议：

def update_d_obs_min_incremental(
    d_obs_min: np.ndarray,
    coords_norm: np.ndarray,
    new_indices: list[tuple[int, ...]],
) -> np.ndarray:

功能：
- 输入当前 d_obs_min。
- 输入本轮新完成真实计算的 grid indices。
- 对每个 new index，计算全 grid 到这个新点的归一化欧氏距离。
- d_obs_min = minimum(d_obs_min, d_new)
- 返回更新后的 d_obs_min。

要求：
- 只用 new_indices 更新，不要重新遍历所有历史采样点。
- 如果 new_indices 为空，直接返回原 d_obs_min。
- 注意不要原地修改导致外部状态混乱，除非函数名或注释明确说明 in-place。
- 对大 grid，可以支持 chunk，但第一版 numpy 全量计算即可；如果写 chunk，需要保持代码简单。

四、实现 compute_M_unseen()

函数签名建议：

def compute_M_unseen(
    computed_mask: np.ndarray,
    pending_mask: np.ndarray,
    invalid_mask: np.ndarray,
    selected_mask: np.ndarray | None = None,
) -> np.ndarray:

功能：
- 返回 bool mask，True 表示可以候选，False 表示必须排除。
- M_unseen = ~(computed_mask | pending_mask | invalid_mask | selected_mask)

要求：
- selected_mask 为空时忽略。
- 必须 assert 所有 mask shape 一致。
- 不允许用浮点坐标作为主要重复判断。

五、实现 exact_duplicate_radius_check()

函数签名建议：

def exact_duplicate_radius_check(
    candidate_coords_norm: np.ndarray,
    forbidden_coords_norm: np.ndarray,
    exact_duplicate_radius_norm: float,
    min_grid_spacing_norm: float,
) -> np.ndarray:

功能：
- 这是额外安全检查，不是主要去重机制。
- 用于当 pending points 或外部导入候选点是浮点坐标时，判断它们是否和候选 grid 点几乎相等。
- 返回 duplicate_mask，True 表示该点应该被视为重复点并排除。

要求：
- 如果 exact_duplicate_radius_norm >= 0.1 * min_grid_spacing_norm，直接 raise ValueError。
- 对固定 grid 内部重复，仍然应该用 index mask，而不是这个函数。
- 如果 forbidden_coords_norm 为空，返回全 False。
- 计算量要注意，不要默认用 N_grid * N_forbidden 的巨大 dense 矩阵。
- 第一版可以在 forbidden_coords 数量较小时直接循环 forbidden 点，逐个更新 duplicate_mask。
- 对每个 forbidden point，只计算全 grid 到该点距离，然后 distance < exact_duplicate_radius_norm 标记为重复。

六、实现 compute_R_obs()

函数签名建议：

def compute_R_obs(
    d_obs_min: np.ndarray,
    ell_obs: float = 0.05,
    repulsion_floor: float = 0.05,
) -> np.ndarray:

公式：

R_obs = repulsion_floor + (1 - repulsion_floor) * (1 - exp(-(d_obs_min / ell_obs)^2))

要求：
- d_obs_min = inf 的地方，R_obs 应该自然接近 1。
- R_obs 数值范围应该在 [repulsion_floor, 1]。
- 对数值误差做 clip。
- 不要出现 nan。
- 如果 ell_obs <= 0，raise ValueError。
- 如果 repulsion_floor 不在 [0,1)，raise ValueError。

七、实现 batch_select_with_Rbatch()

函数签名建议：

def batch_select_with_Rbatch(
    A0: np.ndarray,
    M_unseen: np.ndarray,
    R_obs: np.ndarray,
    coords_norm: np.ndarray,
    batch_size: int,
    ell_batch: float = 0.03,
    batch_floor: float = 0.01,
    exact_duplicate_radius_norm: float = 1e-6,
) -> list[dict]:

功能：
- 输入 A0、M_unseen、R_obs。
- 使用 greedy 方式选择本轮 batch。
- 每一步：
  1. 当前 score = A0 * M_unseen * R_obs * R_batch
  2. 选择 score 最大的未选点
  3. 保存该点 index、坐标、A0、R_obs、R_batch、final_score、selection_rank
  4. 将该点 exact index 硬排除
  5. 根据该点更新 R_batch
  6. 继续选下一个点
- 返回 selected list。

要求：
- M_unseen 是 bool，不能直接参与乘法前隐式转换混乱；可以先 score[~M_unseen] = -inf。
- 本轮已选点 exact index 必须硬排除。
- R_batch 只根据本轮已选点更新。
- 不要和所有历史点重新计算距离。
- 如果没有可选点，停止并返回当前 selected。
- 不要因为 score 全是 -inf 报错。
- 不要选择 nan 或 inf 异常点。
- A0 里如果有 nan，先视为不可选。
- selected 内部不得重复 index。

R_batch 更新方式：
- 初始化 R_batch = ones_like(A0)
- 每选一个点 s：
    dist = norm(coords_norm - coords_norm[s], axis=-1)
    factor = batch_floor + (1 - batch_floor) * (1 - exp(-(dist / ell_batch)^2))
    R_batch *= factor
- 然后将 selected_mask[s] = True
- M_unseen_current = M_unseen & ~selected_mask

注意：
- factor 在选中点自身处接近 batch_floor，但 selected_mask 还要硬排除该点。
- 这样可以防止本轮新点扎堆，但不会完全禁止附近点。

八、实现 compute_corrected_acquisition()

函数签名建议：

def compute_corrected_acquisition(
    A0: np.ndarray,
    state: AcquisitionState,
    selected_mask: np.ndarray | None = None,
    ell_obs: float = 0.05,
    repulsion_floor: float = 0.05,
) -> tuple[np.ndarray, dict]:

功能：
- 只计算 Ar_base = A0 * M_unseen * R_obs，不处理 R_batch。
- R_batch 是 batch selection 内动态变化的，不应该在这里一次性固定。
- 返回 Ar_base 和 debug_components。

debug_components 至少包含：
- M_unseen
- R_obs
- A0_clean
- Ar_base

要求：
- A0 nan 位置不可选。
- computed/pending/invalid 位置必须是 -inf。
- 不要在这里做 top-k。
- 不要在这里做本轮 batch 选择。

九、输出 candidate list

每个 selected candidate 保存字段：

- selection_rank
- index: list[int] 或 tuple[int, ...]
- physical coordinates:
    kT
    JA
  如果当前函数拿不到物理坐标，则至少保存 normalized coordinate。
- A0
- R_obs
- R_batch
- final_score
- phase_uncertainty
- boundary_score
- topology_uncertainty
- eta_gradient_score
- novelty_score
  如果这些 component 在当前函数没有传入，则设计接口允许 optional components dict 传入，并在输出中附加。

建议 batch_select_with_Rbatch 支持：

components: dict[str, np.ndarray] | None = None

这样输出 selected 点时可以把各项分数一起记录，方便调试。

十、自检要求

必须添加 validate_selected_batch()：

def validate_selected_batch(
    selected,
    computed_mask,
    pending_mask,
    invalid_mask,
) -> None:

检查：
- selected 内部 index 不重复。
- selected 不落在 computed_mask。
- selected 不落在 pending_mask。
- selected 不落在 invalid_mask。
- selected final_score 是有限值。
- 如果违反，直接 raise AssertionError 或 ValueError，不要静默修复。

十一、数值和性能要求

1. 不允许每轮用所有历史 sampled points 和所有 grid points 计算完整距离。
   禁止 O(N_grid * N_sampled) 的全历史距离重算。

2. R_obs 必须依赖 d_obs_min。
   d_obs_min 只用新增点增量更新。

3. R_batch 只和本轮 selected points 计算距离。
   复杂度是 O(N_grid * batch_size)，可以接受。

4. M_unseen 主要用整数 index mask。
   极小半径只作为外部浮点坐标重复判断补充。

5. 所有 score 输出前都要处理 nan：
   - A0 nan → 不可选
   - R_obs nan → raise ValueError
   - R_batch nan → raise ValueError

6. 输出 debug 信息：
   - total grid points
   - number of computed points
   - number of pending points
   - number of invalid points
   - number of available points after M_unseen
   - selected batch size
   - min/max of A0, R_obs, final selected score

十二、不要做的事情

- 不要重新设计 A0。
- 不要改动神经网络模型结构。
- 不要把 NN softmax 结果当最终物理标签。
- 不要在历史点附近继续使用固定半径硬屏蔽。
- 不要直接 top-k 选择。
- 不要用浮点坐标作为主要去重方式。
- 不要做复杂 OOP 封装。
- 不要加入启发式 fallback 去随便选点；如果确实没有可选点，应明确返回空列表并打印原因。

十三、最小测试

请添加一个小测试或脚本，构造一个 2D toy grid：

- n_ja = 20
- n_kt = 30
- 随机 A0
- 随机设置 computed_mask / pending_mask / invalid_mask
- 初始化 d_obs_min
- 增量加入几个 sampled indices 更新 d_obs_min
- 计算 R_obs
- 运行 batch_select_with_Rbatch(batch_size=8)

测试断言：
- selected 数量 <= 8
- selected 不重复
- selected 不在 computed/pending/invalid 中
- R_obs 范围在 [repulsion_floor, 1]
- d_obs_min 更新后没有错误 shape
- final_score 全部有限

十四、请先输出修改计划

在动代码前，先输出：
1. 你将修改哪些文件
2. 每个文件新增哪些函数
3. 数据流从 A0 到 selected candidates 是怎样的
4. 你会如何保证不重复选点
5. 你会如何避免 O(N_grid * N_sampled) 的计算量爆炸

确认逻辑后再改代码。

---

# StopController implementation plan - 2026-05-14

Goal:

Soft acquisition repulsion means the active-learning selector can usually keep
finding candidates. Therefore "no available candidates" is no longer the main
convergence criterion. It remains only an exceptional stop reason. The main
stop decision must be based on exact-oracle feedback and stable thermodynamic
phase boundaries.

Files to add or modify:

- Add `ml_phase/stop_controller.py`.
- Add `scripts/dev_check_stop_controller.py`.
- Modify `ml_phase/active_refine.py` to write
  `monitor_predictions_iterXXX.npz` and selected-point
  `predicted_phase_before_exact` metadata.
- Modify `hpc_active_loop.sh` to call `python -m ml_phase.stop_controller`
  after merge and append.
- Update `MODEL_SPEC.md`, `docs/NUMERICS_SPEC.md`, `docs/DECISIONS.md`,
  `docs/PROJECT_SUMMARY.md`, and report Q&A notes.

StopController inputs:

- `iterXXX/monitor_predictions_iterXXX.npz` from candidate generation:
  fixed monitor grid, predicted phase map, and acquisition score fields.
- `iterXXX/selected_points_by_pool.csv` from candidate generation:
  selected coordinates, grid index, `A0_main`, and predicted phase before
  exact evaluation.
- `iterXXX/exact_merged_iterXXX.npz` from the exact oracle merge:
  exact observables, q-window flags, and quality-gate flags.
- `dataset_iterYYY.npz` after append:
  cumulative exact data used for boundary coverage.
- `stop_state.json` and `stop_metrics_history.json` from previous iterations.

Metrics:

1. `phase_map_change`: fraction of fixed monitor-grid cells whose predicted
   phase changed from the previous iteration.
2. `boundary_shift_normal_sc`: symmetric nearest-neighbor p95 distance between
   current and previous predicted normal/SC boundary points in normalized
   coordinates.
3. `boundary_shift_uniform_fflo`: same metric for the uniform_SC/FFLO
   boundary.
4. `label_surprise_rate`: fraction of selected points where
   `predicted_phase_before_exact` disagrees with the exact phase inferred from
   exact `Delta_opt` and `q_opt`.
5. `selected_A0_ratio`: current selected `A0_main` mean divided by the baseline
   mean from the first `warmup_reference_iters` iterations.
6. `q_edge_trigger_rate`: fraction of exact merged points with q expansion,
   q unresolved, or q-edge hit flags.
7. `rerun_required_rate`: fraction marked as `needs_rerun_exact`, or not
   `training_eligible_exact` if that field is the available quality-gate flag.
8. `boundary_coverage_p95`: p95 normalized distance from predicted main
   boundary points to the nearest cumulative exact data point. Only normal/SC
   and uniform_SC/FFLO boundaries are used. Eta-zero response boundaries and
   topology are excluded.

Stop logic:

- Hard stop if completed iterations reach `max_iterations`.
- Hard stop if `max_exact_calls` is configured and cumulative exact count
  reaches it.
- After `min_iterations`, compute C1 through C7.
- `convergence_pass` requires at least 5 of C1..C7 to be true.
- Additionally, C6 and C7 are mandatory gates because high q-edge/rerun rates
  or poor main-boundary coverage mean the exact calculation is not converged.
- Stop after `convergence_pass` holds for `patience` consecutive iterations.
- If a boundary is unavailable, record it as unavailable and count that
  condition as false unless `allow_missing_boundary=True`.

Outputs:

- `iterXXX/stop_metrics_iterXXX.json`
- `stop_state.json`
- `stop_metrics_history.json`
- HPC log lines listing metric values, passed conditions, mandatory-gate
  status, and patience counter.

Tests:

- Stable phase map and main boundaries with low surprise stop after patience.
- Existing selectable candidates do not prevent convergence stop.
- Eta-only response changes do not block stop.
- High q-edge trigger rate prevents stop.
- Poor boundary coverage prevents stop.
