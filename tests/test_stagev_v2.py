import numpy as np
import pandas as pd

from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC
from ml_phase.stagev_v2 import (
    BOUNDARY_NAMES,
    StageV2Config,
    combine_multihead_scores,
    compute_per_boundary_rewards,
    fit_multihead_value_models,
    rank_normalize,
    select_micro_batch_v2,
    source_density_correction,
    topology_status_from_labels,
    update_boundary_alphas,
    update_multihead_lambdas,
)


def _score_frame(n=128):
    rng = np.random.default_rng(123)
    n_topo = min(8, n)
    sources = ["global_sobol"] * (n - n_topo) + ["p0_surface"] * n_topo
    frame = pd.DataFrame(
        {
            "kT": rng.random(n) * 0.56,
            "JA": rng.random(n) * 2.12,
            "mu": -0.5 + 2.0 * rng.random(n),
            "candidate_source": sources,
            "p_normal": rng.random(n),
            "p_SC": rng.random(n),
            "p_uniform_SC": rng.random(n),
            "p_FFLO": rng.random(n),
            "nearest_exact_distance": rng.random(n) * 0.1 + 0.01,
            "exact_repulsion": np.ones(n),
            "selection_probability": np.full(n, 1.0 / n),
            "m_NS": rng.normal(size=n),
            "m_UF": rng.normal(size=n),
            "m_P0": rng.normal(size=n),
            "m_Ppi": rng.normal(size=n),
            "m_gap": rng.normal(size=n),
            "pf_product_pred": rng.normal(size=n),
            "pfaffian_margin_pred": rng.random(n),
        }
    )
    for suffix in ["normal_sc", "uniform_fflo", "p0_topology", "ppi_topology", "gap_nodal"]:
        frame[f"A_{suffix}"] = rng.random(n)
        frame[f"B_{suffix}"] = rng.random(n)
        frame[f"U_{suffix}"] = rng.random(n)
        frame[f"H_{suffix}"] = rng.random(n)
        frame[f"support_distance_{suffix}"] = rng.random(n) * 0.1
    return frame


def test_per_boundary_reward_keeps_topology_independent_from_ns():
    selected = _score_frame(6)
    selected["B_normal_sc"] = [1, 1, 1, 1, 1, 1]
    selected["B_p0_topology"] = [0, 0, 1, 1, 0, 1]
    selected["created_topology_bracket"] = [0, 0, 1, 0, 0, 1]
    rewards = compute_per_boundary_rewards(selected)
    assert "reward_ns_normalized" in rewards
    assert "reward_p0_normalized" in rewards
    assert rewards.loc[[2, 5], "reward_p0_raw"].mean() > rewards.loc[[0, 1], "reward_p0_raw"].mean()
    assert rewards["reward_ns_raw"].std() < rewards["reward_p0_raw"].std()


def test_multihead_lambdas_are_independent():
    cfg = StageV2Config(learned_min_reward_samples=8, learned_validation_margin=0.02)
    validation = {"reward_sample_count": 16}
    for name in BOUNDARY_NAMES:
        validation[f"reward_sample_count_{name}"] = 16
        validation[f"rank_correlation_delta_vs_a0_{name}"] = -0.2
    validation["rank_correlation_delta_vs_a0_ns"] = 0.4
    lambdas = update_multihead_lambdas(None, validation, cfg)
    assert lambdas["ns"] > 0.0
    assert lambdas["p0"] == 0.0
    assert lambdas["ppi"] == 0.0


def test_multihead_model_fallback_for_sparse_topology():
    cfg = StageV2Config(learned_min_reward_samples=64)
    history = _score_frame(12)
    rewards = compute_per_boundary_rewards(history)
    history = pd.concat([history, rewards], axis=1)
    models = fit_multihead_value_models(history, cfg)
    assert all(models[name]["status"] in {"insufficient_data", "trained"} for name in BOUNDARY_NAMES)
    lambdas = update_multihead_lambdas(None, {"reward_sample_count": 12}, cfg)
    assert lambdas["p0"] == 0.0
    assert lambdas["ppi"] == 0.0


def test_rank_normalization_prevents_raw_scale_suppression():
    small = np.linspace(0.0, 1.0e-9, 10)
    large = np.linspace(0.0, 1.0e9, 10)
    assert np.allclose(rank_normalize(small), rank_normalize(large))
    assert rank_normalize(small)[-1] == 1.0


def test_alpha_update_increases_topology_and_lowers_stable_ns():
    cfg = StageV2Config(alpha_learning_rate=0.5)
    alphas = update_boundary_alphas(
        None,
        {
            "ns": {"convergence_success": 1.0},
            "p0": {"coverage_deficit": 1.0, "boundary_state": "insufficient_support"},
            "ppi": {"coverage_deficit": 1.0, "boundary_state": "missing_boundary"},
        },
        cfg,
    )
    assert alphas["ns"] < 0.0
    assert alphas["p0"] > 0.0
    assert alphas["ppi"] > 0.0


def test_source_density_correction_penalizes_huge_global_pool():
    source = np.array(["global_sobol"] * 1000 + ["p0_surface"] * 10 + ["bracket_midpoint"] * 10)
    correction = source_density_correction(source, strength=0.5)
    global_mean = correction[source == "global_sobol"].mean()
    p0_mean = correction[source == "p0_surface"].mean()
    assert global_mean < p0_mean


def test_stagev2_selection_records_propensity_and_avoids_duplicates():
    cfg = StageV2Config(micro_batch_size=16, diversity_radius=0.02)
    scored, _ = combine_multihead_scores(_score_frame(96), cfg)
    selected, meta, summary = select_micro_batch_v2(scored, cfg, rng=np.random.default_rng(7))
    assert selected.shape == (16, 3)
    assert meta["selection_probability"].between(0.0, 1.0).all()
    assert meta["final_rank"].is_monotonic_increasing
    assert np.unique(np.round(selected, 10), axis=0).shape[0] == selected.shape[0]
    assert summary["selection_score"] == "A_total_v2"


def test_label_hierarchy_normal_not_trivial_and_nodal_undefined():
    assert topology_status_from_labels(PHASE_NORMAL, "gapped", "trivial") == "not_applicable_normal"
    assert topology_status_from_labels(PHASE_UNIFORM_SC, "gapless", "trivial") == "z2_undefined_nodal"
    assert topology_status_from_labels(PHASE_FFLO, "gapped", "topological") == "z2_defined"
    assert topology_status_from_labels(PHASE_FFLO, "unresolved", "unresolved") == "topology_unresolved"


def test_no_data_leakage_defaults_are_cold_start():
    cfg = StageV2Config()
    assert cfg.run_id == "stagev_v2_multihead_boundary_learning_3d_v1"
    assert cfg.initial_seed_size == 1024
    assert cfg.mu_min == -0.5
    assert cfg.mu_max == 1.5
