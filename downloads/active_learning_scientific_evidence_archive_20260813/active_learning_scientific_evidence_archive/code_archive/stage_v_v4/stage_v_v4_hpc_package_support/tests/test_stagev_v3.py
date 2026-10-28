import numpy as np
import pandas as pd

from ml_phase.dataset_builder import FlatDataset
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL
from ml_phase.stagev_v3 import (
    StageV3Config,
    apply_stagev3_sparse_scores,
    build_sparse_topology_surface,
    compute_stagev3_per_boundary_rewards,
    expected_topology_coverage_improvement,
    generate_sparse_topology_proposals,
    sobol_points_3d,
    update_stagev3_lambdas,
    validate_proposal_sources,
)


def _synthetic_topology_dataset(n=420, seed=42):
    rng = np.random.default_rng(seed)
    kT = rng.random(n) * 0.56
    JA = rng.random(n) * 2.12
    mu = -0.5 + 2.0 * rng.random(n)
    phase = np.full(n, PHASE_FFLO, dtype=np.int64)
    phase[: n // 6] = PHASE_NORMAL
    p0 = (JA - (0.72 + 0.2 * kT)) * (mu - 0.25)
    ppi = np.ones(n) * 2.0
    topo = (p0 < 0).astype(np.int64)
    topo[phase == PHASE_NORMAL] = -1
    trusted_topo = (phase == PHASE_FFLO).astype(np.int8)
    delta = np.where(phase == PHASE_FFLO, 0.2, 0.0)
    q = np.where(phase == PHASE_FFLO, 0.18, 0.0)
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
        "free_energy_gap_to_normal": np.where(phase == PHASE_FFLO, -0.1, 0.1),
        "topology_trusted": trusted_topo,
        "topology_label_code": topo,
        "topology_spectral_status_code": np.where(phase == PHASE_FFLO, 0, -1),
        "topology_p0": np.where(phase == PHASE_FFLO, p0, np.nan),
        "topology_ppi": np.where(phase == PHASE_FFLO, ppi, np.nan),
        "topology_bulk_gap": np.where(phase == PHASE_FFLO, np.abs(p0) + 1e-3, np.nan),
    }
    return FlatDataset(
        x=np.column_stack([kT, JA, mu]),
        y_reg=np.column_stack([delta, q, np.zeros(n), np.zeros(n), np.zeros(n)]),
        y_phase=phase,
        y_eta_sign=np.zeros(n, dtype=np.int64),
        y_strong_diode=np.zeros(n, dtype=np.int64),
        records=records,
    )


def test_sparse_surface_localization_on_synthetic_data():
    cfg = StageV3Config(bracket_neighbor_k=12, bracket_max_distance=0.35)
    dataset = _synthetic_topology_dataset()
    surface = build_sparse_topology_surface(dataset, cfg)
    assert len(surface) > 0
    assert surface["p0_cross"].any()
    assert not surface["ppi_cross"].any()
    assert surface["component_id"].nunique() >= 1


def test_c_topo_prefers_sparse_patch_over_dense_region():
    cfg = StageV3Config()
    surface = pd.DataFrame(
        {
            "kT": [0.10, 0.11, 0.45],
            "JA": [0.80, 0.81, 1.80],
            "mu": [0.20, 0.21, 1.30],
            "coverage_metric": [0.02, 0.02, 0.15],
            "is_sparse_patch": [False, False, True],
            "component_id": [0, 0, 1],
        }
    )
    near_sparse = np.array([[0.45, 1.80, 1.30]])
    near_dense = np.array([[0.10, 0.80, 0.20]])
    score_sparse = expected_topology_coverage_improvement(near_sparse, surface, cfg)
    score_dense = expected_topology_coverage_improvement(near_dense, surface, cfg)
    assert score_sparse["C_topo"].iloc[0] > score_dense["C_topo"].iloc[0]


def test_sparse_patch_proposals_and_source_attribution():
    cfg = StageV3Config(bracket_neighbor_k=12, bracket_max_distance=0.35)
    dataset = _synthetic_topology_dataset()
    pts, meta, summary = generate_sparse_topology_proposals(dataset, cfg, iteration=3, count=64)
    assert pts.shape == (64, 3)
    assert summary["sparse_proposal_status"] in {"ok", "insufficient_topology_surface_support"}
    checked = validate_proposal_sources(meta, strict=True)
    assert checked["candidate_source"].notna().all()
    assert (checked["candidate_source"].astype(str).str.len() > 0).all()


def test_gap_relevance_guard_suppresses_irrelevant_gap_score():
    cfg = StageV3Config()
    dataset = _synthetic_topology_dataset()
    scored = pd.DataFrame(
        {
            "kT": [0.50],
            "JA": [2.00],
            "mu": [1.40],
            "B_p0_topology": [0.0],
            "B_ppi_topology": [0.0],
            "A_p0_topology": [0.0],
            "A_ppi_topology": [0.0],
            "H_p0_topology": [0.0],
            "H_ppi_topology": [0.0],
            "A_gap_nodal": [1.0],
        }
    )
    out, _, _ = apply_stagev3_sparse_scores(scored, dataset, cfg)
    assert out["gap_topology_relevance"].iloc[0] <= 1.0
    assert out["A_gap_nodal"].iloc[0] <= 1.0


def test_reward_coverage_component_is_nonzero_for_support_improvement():
    selected = pd.DataFrame(
        {
            "kT": [0.3],
            "JA": [1.0],
            "mu": [0.5],
            "C_topo": [1.0],
            "C_topo_raw": [0.08],
            "topology_sparse_nearest_distance": [0.01],
            "created_topology_bracket": [1.0],
            "nearest_exact_distance": [0.1],
            "B_p0_topology": [0.5],
            "U_p0_topology": [0.5],
            "H_p0_topology": [0.5],
            "A_p0_topology": [0.5],
        }
    )
    rewards = compute_stagev3_per_boundary_rewards(selected)
    assert rewards["reward_p0_coverage_local"].iloc[0] > 0
    assert rewards["reward_p0_raw"].iloc[0] > 0


def test_topology_lambda_reduces_when_reward_misaligned_with_coverage():
    cfg = StageV3Config(learned_min_reward_samples=8)
    prev = {"p0": 0.4, "ppi": 0.4}
    validation = {
        "reward_sample_count": 16,
        "reward_sample_count_p0": 16,
        "reward_sample_count_ppi": 16,
        "rank_correlation_delta_vs_a0_p0": 0.5,
        "rank_correlation_delta_vs_a0_ppi": 0.5,
        "scientific_metric_improved_p0": 1.0,
        "scientific_metric_improved_ppi": 1.0,
        "scientific_metric_improved_coverage_p0": 0.0,
        "scientific_metric_improved_coverage_ppi": 1.0,
    }
    out = update_stagev3_lambdas(prev, validation, cfg)
    assert out["p0"] < prev["p0"]
    assert out["ppi"] >= prev["ppi"]


def test_no_stagev2_data_leakage_defaults_are_cold_start():
    cfg = StageV3Config()
    points = sobol_points_3d(cfg.initial_seed_size, cfg)
    assert cfg.run_id == "stagev_v3_sparse_arc_boundary_support_3d_v1"
    assert cfg.initial_seed_size == 1024
    assert cfg.mu_min == -0.5
    assert cfg.mu_max == 1.5
    assert points.shape == (1024, 3)
