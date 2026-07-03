import numpy as np
import pandas as pd

from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC
from ml_phase.stagev_acqv2 import (
    StageVConfig,
    build_boundary_support_sets,
    compute_point_rewards,
    generate_stagev_candidate_pool,
    score_stagev_a0,
    select_micro_batch,
    sobol_points_3d,
    train_linear_value_model,
    predict_linear_value_model,
    update_lambda_t,
)
from ml_phase.dataset_builder import FlatDataset


def _synthetic_stagev_dataset(n=360, seed=11):
    rng = np.random.default_rng(seed)
    kT = rng.random(n) * 0.56
    mu = -0.5 + 2.0 * rng.random(n)
    boundary = 0.34 + 0.7 * np.exp(-4.0 * kT) + 0.12 * np.sin(np.pi * mu)
    JA = rng.random(n) * 2.12
    sc = JA < boundary
    fflo = sc & (JA > 0.45 + 0.12 * np.cos(2.0 * np.pi * kT))
    phase = np.full(n, PHASE_NORMAL, dtype=np.int64)
    phase[sc] = PHASE_UNIFORM_SC
    phase[fflo] = PHASE_FFLO
    delta = np.where(sc, np.maximum(boundary - JA, 0.01), 0.0)
    q = np.where(fflo, 0.08 + 0.12 * np.abs(mu), 0.0)
    p0 = (mu - 0.35) * (JA - 0.58)
    ppi = mu + 2.2
    topo = np.full(n, -1, dtype=np.int64)
    topo[sc] = (p0[sc] < 0).astype(np.int64)
    records = {
        "kT": kT,
        "JA": JA,
        "mu": mu,
        "delta_opt": delta,
        "q_opt": q,
        "phase_label": phase,
        "trusted_exact": np.ones(n, dtype=np.int8),
        "training_eligible_exact": np.ones(n, dtype=np.int8),
        "needs_rerun_exact": np.zeros(n, dtype=np.int8),
        "q_unresolved": np.zeros(n, dtype=np.int8),
        "delta_unresolved": np.zeros(n, dtype=np.int8),
        "free_energy_gap_to_normal": JA - boundary,
        "topology_trusted": sc.astype(np.int8),
        "topology_label_code": topo,
        "topology_spectral_status_code": np.where(sc, 0, -1),
        "topology_p0": p0,
        "topology_ppi": ppi,
        "topology_bulk_gap": np.where(sc, np.abs(p0) + 1e-3, np.nan),
    }
    return FlatDataset(
        x=np.column_stack([kT, JA, mu]),
        y_reg=np.column_stack([delta, q, np.zeros(n), np.zeros(n), np.zeros(n)]),
        y_phase=phase,
        y_eta_sign=np.zeros(n, dtype=np.int64),
        y_strong_diode=np.zeros(n, dtype=np.int64),
        records=records,
    )


def _candidate_features(points, cfg):
    kT, JA, mu = points[:, 0], points[:, 1], points[:, 2]
    boundary = 0.34 + 0.7 * np.exp(-4.0 * kT) + 0.12 * np.sin(np.pi * mu)
    m_ns = JA - boundary
    m_uf = JA - (0.45 + 0.12 * np.cos(2.0 * np.pi * kT))
    p0 = (mu - 0.35) * (JA - 0.58)
    ppi = mu + 2.2
    p_sc = 1.0 / (1.0 + np.exp(8.0 * m_ns))
    return pd.DataFrame(
        {
            "kT": kT,
            "JA": JA,
            "mu": mu,
            "p_normal": 1.0 - p_sc,
            "p_uniform_SC": 0.35 * p_sc,
            "p_FFLO": 0.65 * p_sc,
            "p_SC": p_sc,
            "m_NS": m_ns,
            "sigma_NS": np.full_like(m_ns, 0.03),
            "m_UF": m_uf,
            "sigma_UF": np.full_like(m_uf, 0.04),
            "m_P0": p0,
            "sigma_P0": np.full_like(p0, 0.04),
            "m_Ppi": ppi,
            "sigma_Ppi": np.full_like(ppi, 0.04),
            "m_gap": np.log(np.maximum(np.abs(p0), 1e-6) / cfg.gap_tol),
            "sigma_gap": np.full_like(p0, 0.2),
            "pf_product_pred": p0 * ppi,
            "pfaffian_margin_pred": np.minimum(np.abs(p0), np.abs(ppi)),
        }
    )


def test_stagev_sobol_seed_is_3d_and_within_bounds():
    cfg = StageVConfig(initial_seed_size=128, random_seed=123)
    points = sobol_points_3d(cfg.initial_seed_size, cfg)
    lo, hi = cfg.bounds()
    assert points.shape == (128, 3)
    assert np.all(points >= lo)
    assert np.all(points <= hi)


def test_stagev_boundary_support_sets_do_not_use_global_long_edges():
    cfg = StageVConfig(bracket_neighbor_k=8, bracket_max_distance=0.18)
    dataset = _synthetic_stagev_dataset()
    support = build_boundary_support_sets(dataset, cfg)
    assert support["normal_sc"].shape[0] > 0
    assert support["uniform_fflo"].shape[0] > 0
    assert support["p0_topology"].shape[0] > 0
    assert support["ppi_topology"].shape[0] == 0
    assert support["trusted_exact"].shape[1] == 3


def test_stagev_a0_prefers_boundary_support_over_deep_interior():
    cfg = StageVConfig(candidate_pool_size=1024, micro_batch_size=16)
    dataset = _synthetic_stagev_dataset()
    support = build_boundary_support_sets(dataset, cfg)
    candidates, meta, _ = generate_stagev_candidate_pool(dataset, cfg, iteration=1)
    features = _candidate_features(candidates, cfg)
    features["candidate_source"] = meta["candidate_source"].to_numpy()
    scored, summary = score_stagev_a0(features, support, cfg)
    near = scored["B_normal_sc"] >= scored["B_normal_sc"].quantile(0.90)
    far = scored["B_normal_sc"] <= scored["B_normal_sc"].quantile(0.10)
    assert summary["support_counts"]["normal_sc"] > 0
    assert scored.loc[near, "A0"].median() > scored.loc[far, "A0"].median()


def test_stagev_micro_batch_logs_propensity_and_diversifies():
    cfg = StageVConfig(candidate_pool_size=1024, micro_batch_size=24, diversity_radius=0.04)
    dataset = _synthetic_stagev_dataset()
    support = build_boundary_support_sets(dataset, cfg)
    candidates, meta, _ = generate_stagev_candidate_pool(dataset, cfg, iteration=2)
    features = _candidate_features(candidates, cfg)
    features["candidate_source"] = meta["candidate_source"].to_numpy()
    scored, _ = score_stagev_a0(features, support, cfg)
    selected, selected_meta, summary = select_micro_batch(scored, cfg, rng=np.random.default_rng(5))
    assert selected.shape == (24, 3)
    assert selected_meta["selection_probability"].between(0.0, 1.0).all()
    assert summary["selected_batch_size"] == 24
    assert selected_meta["selection_rank"].is_monotonic_increasing


def test_stagev_learned_residual_is_shadow_until_supported():
    cfg = StageVConfig(learned_min_reward_samples=32)
    assert update_lambda_t(0.0, {"reward_sample_count": 31, "rank_correlation_delta_vs_a0": 1.0}, cfg) == 0.0
    assert update_lambda_t(0.0, {"reward_sample_count": 32, "rank_correlation_delta_vs_a0": 0.1}, cfg) > 0.0
    assert update_lambda_t(0.4, {"reward_sample_count": 64, "rank_correlation_delta_vs_a0": -0.5}, cfg) < 0.4


def test_stagev_reward_model_trains_on_logged_features():
    cfg = StageVConfig(candidate_pool_size=1024, micro_batch_size=48)
    dataset = _synthetic_stagev_dataset()
    support = build_boundary_support_sets(dataset, cfg)
    candidates, meta, _ = generate_stagev_candidate_pool(dataset, cfg, iteration=3)
    features = _candidate_features(candidates, cfg)
    features["candidate_source"] = meta["candidate_source"].to_numpy()
    scored, _ = score_stagev_a0(features, support, cfg)
    _, selected_meta, _ = select_micro_batch(scored, cfg, rng=np.random.default_rng(9))
    rewards = compute_point_rewards(selected_meta)
    model = train_linear_value_model(
        selected_meta.assign(**rewards),
        rewards["reward_scalar"].to_numpy(float),
        ["A0", "B_normal_sc", "B_p0_topology", "H_normal_sc", "H_p0_topology", "nearest_exact_distance"],
    )
    assert model["status"] == "trained"
    assert model["sample_count"] == len(selected_meta)


def test_stagev_reward_model_prediction_tolerates_preselection_missing_features():
    features = pd.DataFrame(
        {
            "A0": np.linspace(0.1, 1.0, 12),
            "selection_probability": np.linspace(0.01, 0.02, 12),
        }
    )
    rewards = np.linspace(0.0, 1.0, 12)
    model = train_linear_value_model(features, rewards, ["A0", "selection_probability"])
    candidates = pd.DataFrame({"A0": np.linspace(0.2, 0.8, 5)})
    pred = predict_linear_value_model(model, candidates)
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
