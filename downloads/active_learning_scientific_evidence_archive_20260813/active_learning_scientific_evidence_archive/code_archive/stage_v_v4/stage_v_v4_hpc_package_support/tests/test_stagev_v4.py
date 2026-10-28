from types import SimpleNamespace

import numpy as np
import pandas as pd
import json

from ml_phase.dataset_builder import FlatDataset
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC
from ml_phase.stagev_v4 import (
    ALLOWED_ACTION_FAMILIES,
    StageV4Config,
    attach_graph_context_scores,
    build_observation_descriptors,
    build_phase_structure_graph,
    compute_stagev4_rewards,
    generate_graph_candidate_proposals,
    generate_stagev4_candidate_pool,
    graph_uncertainty_summary,
    stagev4_leakage_guard_summary,
    validate_stagev4_proposal_sources,
)
from scripts.stagev_v4_select import select_seed


def _synthetic_graph_dataset(n=460, seed=7):
    rng = np.random.default_rng(seed)
    kT = rng.random(n) * 0.56
    JA = rng.random(n) * 2.12
    mu = -0.5 + 2.0 * rng.random(n)
    phase = np.full(n, PHASE_FFLO, dtype=np.int64)
    phase[: n // 5] = PHASE_NORMAL
    phase[n // 5 : n // 4] = PHASE_UNIFORM_SC
    p0 = (JA - (0.72 + 0.18 * kT)) * (mu - 0.20)
    ppi = 1.0 + 0.08 * mu
    topo = (p0 < 0).astype(np.int64)
    topo[phase != PHASE_FFLO] = -1
    anomaly = (kT > 0.15) & (kT < 0.32) & (JA > 1.45) & (mu > 0.30) & (phase == PHASE_FFLO)
    q = np.where(phase == PHASE_FFLO, 0.16, 0.0)
    q = np.where(anomaly, 0.55, q)
    hard_risk = (kT < 0.05) & (JA > 1.55) & (phase == PHASE_FFLO)
    trusted = np.ones(n, dtype=np.int8)
    trusted[hard_risk] = 0
    records = {
        "kT": kT,
        "JA": JA,
        "mu": mu,
        "trusted_exact": trusted,
        "training_eligible_exact": np.ones(n, dtype=np.int8),
        "needs_rerun_exact": hard_risk.astype(np.int8),
        "q_unresolved": np.zeros(n, dtype=np.int8),
        "delta_unresolved": np.zeros(n, dtype=np.int8),
        "topology_trusted": ((phase == PHASE_FFLO) & (trusted == 1)).astype(np.int8),
        "topology_label_code": topo,
        "topology_spectral_status_code": np.where(phase == PHASE_FFLO, 0, -1),
        "topology_p0": np.where(phase == PHASE_FFLO, p0, np.nan),
        "topology_ppi": np.where(phase == PHASE_FFLO, ppi, np.nan),
        "topology_bulk_gap": np.where(phase == PHASE_FFLO, np.abs(p0) + 1e-3, np.nan),
        "free_energy_gap_to_normal": np.where(phase == PHASE_NORMAL, 0.1, -0.1),
        "deltaF_to_second_basin": np.where(anomaly, 0.004, 0.08),
        "basin_count": np.where(anomaly, 2, 1),
    }
    return FlatDataset(
        x=np.column_stack([kT, JA, mu]),
        y_reg=np.column_stack([np.where(phase == PHASE_NORMAL, 0.0, 0.2), q, np.zeros(n), np.zeros(n), np.zeros(n)]),
        y_phase=phase,
        y_eta_sign=np.zeros(n, dtype=np.int64),
        y_strong_diode=np.zeros(n, dtype=np.int64),
        records=records,
    )


def test_descriptor_builder_and_graph_miner_on_synthetic_data():
    cfg = StageV4Config(candidate_pool_size=512, micro_batch_size=8)
    dataset = _synthetic_graph_dataset()
    desc = build_observation_descriptors(dataset, cfg)
    graph = build_phase_structure_graph(dataset, cfg)
    assert len(desc) == dataset.x.shape[0]
    assert {"cFFLO_trivial", "tFFLO_topological"}.issubset(set(desc["node_label"].astype(str)))
    assert graph["summary"]["known_node_count"] > 0
    assert graph["summary"]["edge_count"] > 0


def test_hard_risk_points_do_not_define_candidate_nodes():
    cfg = StageV4Config(candidate_phase_min_points=8)
    dataset = _synthetic_graph_dataset()
    graph = build_phase_structure_graph(dataset, cfg)
    nodes = graph["nodes"]
    if "hard_risk_fraction" in nodes and len(nodes):
        assert float(nodes["hard_risk_fraction"].max()) < 0.75


def test_graph_uncertainty_and_context_scores_are_positive():
    cfg = StageV4Config()
    dataset = _synthetic_graph_dataset()
    graph = build_phase_structure_graph(dataset, cfg)
    summary = graph_uncertainty_summary(graph["nodes"], graph["edges"], graph["descriptors"])
    assert summary["graph_uncertainty_total"] > 0
    pts, meta, _ = generate_graph_candidate_proposals(dataset, graph, cfg, iteration=2, count=32)
    meta = validate_stagev4_proposal_sources(meta, strict=True)
    frame = pd.DataFrame({"kT": pts[:, 0], "JA": pts[:, 1], "mu": pts[:, 2], "nearest_exact_distance": 0.05})
    frame["candidate_source"] = meta["candidate_source"].to_numpy()
    frame["proposal_source"] = meta["proposal_source"].to_numpy()
    frame["action_family"] = meta["action_family"].to_numpy()
    scored, _ = attach_graph_context_scores(frame, graph, cfg)
    assert (scored["graph_uncertainty_score"] >= 0).all()
    assert scored["graph_uncertainty_score"].max() > 0


def test_graph_reward_is_causal_and_nonzero_for_informative_point():
    selected = pd.DataFrame(
        {
            "kT": [0.2],
            "JA": [1.1],
            "mu": [0.3],
            "action_family": ["candidate_phase_validation"],
            "candidate_source": ["descriptor_OOD"],
            "proposal_source": ["descriptor_OOD"],
            "graph_uncertainty_score": [0.8],
            "node_existence_uncertainty": [0.7],
            "edge_boundary_uncertainty": [0.5],
            "candidate_phase_score": [0.6],
            "oracle_anomaly_uncertainty": [0.2],
            "nearest_exact_distance": [0.06],
        }
    )
    rewards = compute_stagev4_rewards(selected)
    assert rewards["reward_graph_raw"].iloc[0] > 0
    assert rewards["reward_graph_normalized"].iloc[0] > 0


def test_stagev4_candidate_pool_has_no_blank_sources_or_action_families():
    cfg = StageV4Config(candidate_pool_size=512, micro_batch_size=8)
    dataset = _synthetic_graph_dataset()
    pts, meta, summary, _ = generate_stagev4_candidate_pool(dataset, cfg, iteration=4)
    meta = validate_stagev4_proposal_sources(meta, strict=True)
    assert pts.shape[0] == len(meta)
    assert meta["candidate_source"].astype(str).str.len().gt(0).all()
    assert meta["proposal_source"].astype(str).str.len().gt(0).all()
    assert meta["action_family"].isin(ALLOWED_ACTION_FAMILIES).all()
    assert summary["stagev4_candidate_count"] == pts.shape[0]


def test_cold_start_seed_selection_writes_no_prior_data_flags(tmp_path):
    cfg = StageV4Config(initial_seed_size=32, micro_batch_size=8, output_root=str(tmp_path), run_id="stagev4_test")
    args = SimpleNamespace(iteration=0, world_size=4, partition_strategy="round_robin")
    out_run_dir = tmp_path / "active_runs" / "stagev4_test"
    select_seed(args, cfg, out_run_dir)
    meta = pd.read_csv(out_run_dir / "iter000" / "selected_points_metadata.csv")
    summary = json.loads((out_run_dir / "iter000" / "stagev_selection_summary.json").read_text(encoding="utf-8"))
    assert len(meta) == 32
    assert meta["action_family"].eq("global_exploration").all()
    assert bool(summary["cold_start"])
    assert not bool(summary["stagev3_data_used_for_training"])


def test_leakage_guard_defaults_are_false_for_prior_data():
    guard = stagev4_leakage_guard_summary()
    assert guard["cold_start"] is True
    for key, value in guard.items():
        if key.endswith("_data_used_for_training"):
            assert value is False
