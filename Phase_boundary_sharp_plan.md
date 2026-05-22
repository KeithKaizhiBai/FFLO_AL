现在需要一次性综合修改 discovery-mode active learning 的选点策略。不要再分阶段实验；这次直接把 B_delta gating、active-pool 收窄、interior filter、sampling annealing、exploration annealing、自适应 batch size 全部接入。

任务目标：
让 discovery active learning 从“弱聚焦 stochastic coverage”变成“boundary-focused stochastic acquisition”。

不要修改：
- BdG exact oracle
- NN architecture
- phase label 定义
- warm-start/refinement 逻辑
- hidden ground truth evaluator
- StopController 的主收敛逻辑
- midpoint selection
- finite_t_band_width / prior_band 逻辑

只修改：
- acquisition component construction
- active_pool gating
- stochastic sampling schedule
- high-confidence interior penalty/filter
- adaptive batch-size selection
- diagnostics/report

背景诊断：

最新 report 显示：

- run_mode = discovery
- candidate_domain_mode = full
- selection_mode = stochastic
- active_pool_quantile = 0.9
- active_pool_rel_to_p95 = 0.3
- finite_t_band_width disabled
- completed iterations = 50
- stop reason = max_iterations
- convergence pass = no

最新 selection diagnostics 显示：

unseen candidates:
- normal_interior fraction ≈ 0.618
- SC interior fraction ≈ 0.346
- boundary_band fraction ≈ 0.036

active_pool:
- normal_interior fraction ≈ 0.818
- SC interior fraction ≈ 0.134
- boundary_band fraction ≈ 0.048

selected points:
- normal_interior fraction ≈ 0.852
- SC interior fraction ≈ 0.078
- boundary_band fraction ≈ 0.070

component attribution:
selected normal_interior:
- A0_main ≈ 1.026
- A_phase ≈ 1.021
- A_numerical ≈ 0
- A_explore ≈ 0.0047
- U_cls ≈ 0.00135
- U_reg_phase ≈ 0.093
- B_delta ≈ 0.961
- B_q_SC ≈ 0
- E_ext ≈ 0.031

结论：
normal interior 被大量选中，主要不是 R_obs 或 E_ext，而是 B_delta 在 predicted normal interior 中异常高，导致 A_phase / A0_main 偏高。active_pool gate 又太宽，把大量 normal interior 放进 pool。stochastic sampler 只是继承了这个偏置。

一、修正 B_delta：加入 normal/SC competition gate

当前问题：
B_delta_raw = exp(-abs(delta_pred - delta_eps) / delta_width)

这个定义会在 deep normal 区域也变大，因为 deep normal 中 delta_pred≈0，而 delta_eps 很小。

新增：

P_normal = phase_prob[..., normal_index]
P_uniform = phase_prob[..., uniform_SC_index]
P_FFLO = phase_prob[..., FFLO_index]
P_SC = P_uniform + P_FFLO

U_NS = 4 * P_normal * P_SC
U_NS = clip(U_NS, 0, 1)

含义：
- deep normal: P_normal≈1, P_SC≈0, U_NS≈0
- deep SC: P_normal≈0, P_SC≈1, U_NS≈0
- normal/SC boundary: P_normal≈0.5, P_SC≈0.5, U_NS≈1

改成：

B_delta_raw = exp(-abs(delta_pred_mean - delta_eps) / delta_width)
B_delta_gated = B_delta_raw * U_NS

A_phase 中必须使用 B_delta_gated，不再使用 B_delta_raw。

要求：
- debug 同时保存 B_delta_raw 和 B_delta_gated
- report 表格显示 B_delta_raw, U_NS, B_delta_gated
- 如果 B_delta_raw 在 normal interior 高，但 U_NS 低，report 要能看出来
- 旧字段名 B_delta 如果必须保留，应指向 gated version，避免混淆

二、增加 uniform/FFLO competition gate，可先用于 diagnostics，必要时用于 B_q

新增：

U_UF = 4 * P_uniform * P_FFLO
U_UF = clip(U_UF, 0, 1)

含义：
- uniform/FFLO 边界附近大
- deep uniform、deep FFLO、deep normal 中小

当前 B_q_SC = P_SC * B_q_raw 可以保留作为默认。

但请新增配置：

q_boundary_gate_mode = "psc" | "uf_competition"

默认先用：
q_boundary_gate_mode = "psc"

如果 q_boundary_gate_mode="uf_competition":
B_q_gated = B_q_raw * U_UF

无论哪种模式，都保存：
- B_q_raw
- P_SC
- U_UF
- B_q_gated

三、重构 active_pool gate：从宽松 OR 改为严格 max-threshold + annealing

当前问题：
active_pool = A0 >= q90 OR A0 >= 0.3*p95

这个 OR 太宽，导致 active_pool 占 unseen candidates 的 75% 以上。

新逻辑：

用 A0_for_pool 构造 active_pool。

A0_for_pool = A0_main_after_gating_and_interior_penalty

计算 hard-unseen candidates 中的：
q_thr = quantile(A0_for_pool, q_current)
p95 = percentile(A0_for_pool, 95)
rel_thr = active_pool_rel_to_p95 * p95

threshold = max(q_thr, rel_thr)

active_pool = hard_unseen & (A0_for_pool >= threshold)

默认参数：
active_pool_rule = "max_threshold"
active_pool_quantile_schedule = "piecewise"
active_pool_quantile_start = 0.90
active_pool_quantile_mid = 0.95
active_pool_quantile_end = 0.98
active_pool_quantile_mid_iter = 10
active_pool_quantile_end_iter = 30
active_pool_rel_to_p95 = 0.7

piecewise schedule:
if iter < 10:
    q_current = 0.90
elif iter < 30:
    q_current = 0.95
else:
    q_current = 0.98

额外增加 active_pool_fraction cap：

active_pool_max_fraction_schedule = "piecewise"
active_pool_max_fraction_start = 0.20
active_pool_max_fraction_end = 0.05
active_pool_max_fraction_end_iter = 30

如果 active_pool_fraction > max_fraction:
    自动提高 quantile threshold，直到 active_pool_fraction <= max_fraction
    或 quantile 达到 0.995

要求：
- 不要使用 region quota
- 不按 JA/kT 分区固定比例
- active_pool 完全由 score + hard mask + interior filter 决定
- 每轮输出：
  active_pool_count
  unseen_count
  active_pool_fraction
  q_current
  q_thr
  rel_thr
  final_threshold
  whether fraction cap tightened threshold

四、加入 high-confidence low-information interior penalty/filter

目的：
过滤或降权高置信相内部点，尤其当前大量 normal interior。

定义：

P_max = max(P_normal, P_uniform, P_FFLO)

main_boundary_gate = max(U_NS, U_UF)

high_confidence_interior if:
    P_max > P_conf_threshold
    and U_NS < U_NS_low
    and U_UF < U_UF_low
    and G_phase < G_phase_low
    and E_q_SC < E_q_low
    and E_ext_uncertain < E_ext_low

默认：
P_conf_threshold = 0.98
U_NS_low = 0.05
U_UF_low = 0.05
G_phase_low = 0.05
E_q_low = 0.05
E_ext_low = 0.05

采用 soft penalty，不要第一版 hard remove：

if high_confidence_interior:
    A0_for_pool = A0_main * interior_penalty
else:
    A0_for_pool = A0_main

默认：
interior_filter_mode = "soft_penalty"
interior_penalty_start_iter = 10
interior_penalty_early = 0.5
interior_penalty_late = 0.1

schedule:
if iter < 10:
    penalty = 0.5
else:
    penalty = 0.1

要求：
- A0_main 原始值仍保存
- A0_for_pool 保存
- interior_penalty_mask 保存
- high_confidence_interior 不直接删除点，除非配置 interior_filter_mode="hard_exclude"
- 默认不要 hard_exclude，避免早期漏相
- filter 对 normal、uniform、FFLO 都适用，不要只针对 normal

五、sampling_power 退火

当前 gamma=2。修正 B_delta 和 active_pool 后，可以提高采样尖锐度。

新增：

sampling_power_schedule = "piecewise"
sampling_power_start = 1.5
sampling_power_mid = 2.5
sampling_power_end = 4.0
sampling_power_mid_iter = 10
sampling_power_end_iter = 30

schedule:
if iter < 10:
    gamma = 1.5
elif iter < 30:
    gamma = 2.5
else:
    gamma = 4.0

采样概率：

p_i ∝ max(Aselect_i, 0)^gamma

其中：

Aselect = A0_for_pool * R_obs * R_batch

注意：
- active_pool 用 A0_for_pool 筛
- sampling 用 Aselect
- 不要直接用 A0_main raw
- 每选一个点后更新 R_batch 并重新归一化

每个 selected point 保存：
- A0_main_raw
- A0_for_pool
- interior_penalty_applied
- R_obs
- R_batch
- Aselect
- sampling_power
- sampling_probability_before_pick

六、exploration weight 退火

当前 w_ext=0.15。后期 exploration 不应继续大面积吸引 interior。

新增：

w_ext_schedule = "piecewise"
w_ext_start = 0.15
w_ext_mid = 0.08
w_ext_end = 0.03
w_ext_mid_iter = 10
w_ext_end_iter = 30

schedule:
if iter < 10:
    w_ext_current = 0.15
elif iter < 30:
    w_ext_current = 0.08
else:
    w_ext_current = 0.03

A_explore = w_ext_current * E_ext_uncertain

要求：
- report 中显示当前 iteration 的 w_ext_current
- diagnostics 保存每轮 w_ext_current
- 不要删除 exploration，只是后期衰减

七、自适应 batch size：不要强行补满 256

当前报告显示每轮仍然 selected=256，说明实际仍近似强行补满。

新逻辑：

batch_size_max = 256
batch_size_min_before_min_iter = 64
batch_size_min_after_min_iter = 0 或 16

如果 iter < STOP_MIN_ITERATIONS:
    如果 active_pool 中可选点不足 batch_size_min_before_min_iter:
        逐步放宽 active_pool threshold：
            q_current -= 0.02
        直到选够 min batch 或 q_current <= 0.70
    目的：早期不要因为模型未稳定而过早停。

如果 iter >= STOP_MIN_ITERATIONS:
    不强行补满。
    selected_size = min(batch_size_max, active_pool_available_count)
    如果 active_pool_available_count < batch_size_max:
        只选 active_pool_available_count 个。
    如果 active_pool_available_count 很少或为 0:
        本轮可返回少量或空 batch，由 StopController 判断是否停止。

要求：
- 不要从 active_pool 外补低质量点
- 不要为了凑满 256 去 normal interior 补点
- 每轮输出：
  requested_batch_size
  selected_batch_size
  active_pool_available_count
  batch_was_underfilled
  underfill_reason
  threshold_relaxed
  final_quantile_used

八、R_obs 保持 mild，不要强化

当前 R_obs:
obs_repulsion_length = 0.02
obs_repulsion_floor = 0.5

这个不是当前主因。请保持 mild，不要变强。

要求：
- R_obs 只做历史点温和降权
- 不要让 R_obs 变成 coverage sampler
- 不要用 R_obs 决定 active_pool 是否进入
- active_pool 由 A0_for_pool 决定

九、报告和 diagnostics 更新

报告新增/更新以下字段：

1. Acquisition component section:
   - B_delta_raw
   - U_NS
   - B_delta_gated
   - U_UF
   - B_q_raw
   - B_q_gated
   - current w_ext
   - current gamma
   - current active_pool quantile

2. Selection diagnostics:
   - active_pool_fraction
   - active_pool_count / unseen_count
   - selected_normal_interior_fraction
   - selected_sc_interior_fraction
   - selected_boundary_band_fraction
   - random_baseline_boundary_band_fraction
   - selected_A0_for_pool / random_A0_for_pool
   - N_eff / active_pool_size

3. Interior filter diagnostics:
   - high_confidence_interior_fraction among unseen
   - high_confidence_interior_fraction among active_pool before penalty
   - high_confidence_interior_fraction among selected
   - selected points with interior_penalty_applied

4. Component attribution:
   Compare regions:
   - selected normal_interior
   - selected sc_interior
   - selected boundary_band
   - active_pool normal_interior
   - active_pool boundary_band

   columns:
   - A0_main_raw
   - A0_for_pool
   - A_phase
   - A_numerical
   - A_explore
   - B_delta_raw
   - U_NS
   - B_delta_gated
   - U_cls_mix
   - U_reg_phase
   - B_q_gated
   - G_phase
   - E_q_SC
   - E_ext_uncertain
   - R_obs
   - Aselect

5. Add automatic interpretation:
   - If selected_normal_interior_fraction remains high, warn.
   - If B_delta_raw high but U_NS low in normal interior, say B_delta gate is suppressing deep-normal false boundary score.
   - If active_pool_fraction remains > 0.3 after iter 30, warn pool is still too wide.
   - If selected_boundary_band_fraction improves, note stronger boundary focusing.

十、配置 defaults

Update default discovery config:

B_delta_gate_mode = "normal_sc_competition"
q_boundary_gate_mode = "psc"

active_pool_rule = "max_threshold"
active_pool_quantile_schedule = "piecewise"
active_pool_quantile_start = 0.90
active_pool_quantile_mid = 0.95
active_pool_quantile_end = 0.98
active_pool_quantile_mid_iter = 10
active_pool_quantile_end_iter = 30
active_pool_rel_to_p95 = 0.7
active_pool_max_fraction_start = 0.20
active_pool_max_fraction_end = 0.05
active_pool_max_fraction_end_iter = 30

interior_filter_mode = "soft_penalty"
interior_penalty_start_iter = 10
interior_penalty_early = 0.5
interior_penalty_late = 0.1
P_conf_threshold = 0.98
U_NS_low = 0.05
U_UF_low = 0.05
G_phase_low = 0.05
E_q_low = 0.05
E_ext_low = 0.05

sampling_power_schedule = "piecewise"
sampling_power_start = 1.5
sampling_power_mid = 2.5
sampling_power_end = 4.0
sampling_power_mid_iter = 10
sampling_power_end_iter = 30

w_ext_schedule = "piecewise"
w_ext_start = 0.15
w_ext_mid = 0.08
w_ext_end = 0.03
w_ext_mid_iter = 10
w_ext_end_iter = 30

batch_size_max = 256
batch_size_min_before_min_iter = 64
batch_size_min_after_min_iter = 0
allow_underfilled_batch_after_min_iter = true

十一、测试要求

Add or update tests:

1. test_bdelta_gated_suppresses_deep_normal
   Input:
   P_normal=0.99, P_SC=0.01, B_delta_raw≈1
   Expected:
   U_NS≈0.04
   B_delta_gated≈0.04, not ≈1

2. test_bdelta_gated_preserves_normal_sc_boundary
   Input:
   P_normal=0.5, P_SC=0.5, B_delta_raw high
   Expected:
   U_NS≈1
   B_delta_gated≈B_delta_raw

3. test_active_pool_max_threshold_not_or
   Ensure active_pool threshold = max(q_thr, rel_thr), not OR loose union.

4. test_active_pool_fraction_cap
   If active_pool_fraction > cap, quantile threshold tightens.

5. test_high_confidence_interior_penalty
   High-confidence interior point gets A0_for_pool = A0_main * penalty.
   It is not hard removed by default.

6. test_boundary_candidate_not_penalized
   If U_NS high or U_UF high, no interior penalty even if P_max moderately high.

7. test_sampling_power_schedule
   iter < 10 -> gamma=1.5
   10<=iter<30 -> gamma=2.5
   iter>=30 -> gamma=4.0

8. test_w_ext_schedule
   correct w_ext per iteration.

9. test_underfilled_batch_after_min_iter
   If active_pool has 80 candidates and batch_size_max=256, iter>=min_iter selects <=80, not 256.

10. test_no_region_quota
    Ensure no fixed JA/kT quota or phase quota is introduced.

十二、不要做的事

- Do not reintroduce midpoint selection.
- Do not use finite_t_band_width in discovery mode.
- Do not switch to prior_band.
- Do not use region quota.
- Do not switch to deterministic top-k.
- Do not strengthen R_obs.
- Do not use hidden ground truth for online acquisition or stop.
- Do not modify exact BdG oracle.
- Do not modify NN architecture.

十三、请先输出修改计划

Before editing code, output a plan covering:

1. Which functions compute B_delta and A_phase.
2. Where phase probabilities P_normal/P_uniform/P_FFLO are available.
3. How to add U_NS and B_delta_gated.
4. Where active_pool is constructed and how to replace OR with max_threshold.
5. Where stochastic sampling_power is applied and how to anneal it.
6. Where A_explore / w_ext is computed and how to anneal it.
7. Where batch_size is enforced and how to allow underfilled batch.
8. How to save new diagnostics.
9. How to update report tables/plots.
10. Which tests will be added.

After plan confirmation, implement all changes in one pass.