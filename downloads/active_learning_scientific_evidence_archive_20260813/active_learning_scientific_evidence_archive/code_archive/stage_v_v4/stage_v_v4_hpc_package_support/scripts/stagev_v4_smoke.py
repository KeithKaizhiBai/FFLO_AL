from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.dataset_builder import FlatDataset
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC
from ml_phase.stagev_v4 import (
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def synthetic_dataset(n: int = 520, seed: int = 20260708) -> FlatDataset:
    rng = np.random.default_rng(seed)
    kT = rng.random(n) * 0.56
    JA = rng.random(n) * 2.12
    mu = -0.5 + 2.0 * rng.random(n)
    phase = np.full(n, PHASE_FFLO, dtype=np.int64)
    phase[: n // 5] = PHASE_NORMAL
    phase[n // 5 : n // 4] = PHASE_UNIFORM_SC
    p0 = (JA - (0.72 + 0.18 * kT)) * (mu - 0.22)
    ppi = 1.0 + 0.1 * mu
    topo = (p0 < 0).astype(np.int64)
    topo[phase != PHASE_FFLO] = -1
    delta = np.where(phase == PHASE_NORMAL, 0.0, 0.2)
    q = np.where(phase == PHASE_FFLO, 0.16, 0.0)

    # A compact trusted FFLO descriptor anomaly. It should become only a
    # provisional candidate node because persistence is not yet established.
    anomaly = (kT > 0.18) & (kT < 0.30) & (JA > 1.52) & (JA < 1.78) & (mu > 0.48) & (mu < 0.82)
    p0 = np.where(anomaly, p0 + 0.9, p0)
    q = np.where(anomaly & (phase == PHASE_FFLO), 0.52, q)

    hard_risk = (kT < 0.06) & (JA > 1.6) & (phase == PHASE_FFLO)
    trusted = np.ones(n, dtype=np.int8)
    eligible = np.ones(n, dtype=np.int8)
    trusted[hard_risk] = 0

    records = {
        "kT": kT,
        "JA": JA,
        "mu": mu,
        "trusted_exact": trusted,
        "training_eligible_exact": eligible,
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
        "deltaF_to_second_basin": np.where(anomaly, 0.005, 0.08),
        "basin_count": np.where(anomaly, 2, 1),
    }
    return FlatDataset(
        x=np.column_stack([kT, JA, mu]),
        y_reg=np.column_stack([delta, q, np.zeros(n), np.zeros(n), np.zeros(n)]),
        y_phase=phase,
        y_eta_sign=np.zeros(n, dtype=np.int64),
        y_strong_diode=np.zeros(n, dtype=np.int64),
        records=records,
    )


def run_smoke(output_dir: Path) -> dict[str, Any]:
    cfg = StageV4Config(candidate_pool_size=2048, micro_batch_size=16, initial_seed_size=128, reg_epochs=4, cls_epochs=4)
    dataset = synthetic_dataset()
    desc = build_observation_descriptors(dataset, cfg)
    graph = build_phase_structure_graph(dataset, cfg)
    proposals, meta, proposal_summary = generate_graph_candidate_proposals(dataset, graph, cfg, iteration=2, count=64)
    meta = validate_stagev4_proposal_sources(meta, strict=True)
    scored = pd.DataFrame(
        {
            "kT": proposals[:, 0],
            "JA": proposals[:, 1],
            "mu": proposals[:, 2],
            "proposal_source": meta["proposal_source"],
            "candidate_source": meta["candidate_source"],
            "action_family": meta["action_family"],
            "A_total_v2": np.ones(len(proposals)),
            "nearest_exact_distance": 0.05,
        }
    )
    scored, graph_context_summary = attach_graph_context_scores(scored, graph, cfg)
    rewards = compute_stagev4_rewards(scored.head(16))
    candidates, candidate_meta, pool_summary, _ = generate_stagev4_candidate_pool(dataset, cfg, iteration=3)
    candidate_meta = validate_stagev4_proposal_sources(candidate_meta, strict=True)
    uncertainty = graph_uncertainty_summary(graph["nodes"], graph["edges"], graph["descriptors"])
    checks = {
        "status": "pass",
        "run_id": cfg.run_id,
        "cold_start": True,
        "descriptor_rows": int(desc.shape[0]),
        "node_count": int(graph["summary"].get("known_node_count", 0)) + int(graph["summary"].get("candidate_node_count", 0)),
        "edge_count": int(graph["summary"].get("edge_count", 0)),
        "candidate_node_count": int(graph["summary"].get("candidate_node_count", 0)),
        "proposal_count": int(proposals.shape[0]),
        "candidate_pool_count": int(candidates.shape[0]),
        "action_family_valid": bool(candidate_meta["action_family"].isin({"global_exploration", "known_boundary_refinement", "candidate_phase_validation", "oracle_anomaly_audit"}).all()),
        "proposal_sources_valid": bool(candidate_meta["proposal_source"].astype(str).str.len().gt(0).all()),
        "graph_context_nonzero": bool((scored["graph_uncertainty_score"] > 0).any()),
        "graph_reward_present": "reward_graph_normalized" in rewards,
        "graph_reward_nonzero": bool((rewards["reward_graph_raw"] > 0).any()),
        "hard_risk_not_candidate_dominant": bool(graph["nodes"].get("hard_risk_fraction", pd.Series(dtype=float)).fillna(0).max() < 0.75),
        "excluded_nodes_required": ["gpuh01"],
        **stagev4_leakage_guard_summary(),
        **proposal_summary,
        **pool_summary,
        **uncertainty,
        **graph_context_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stagev_v4_smoke_summary.json", checks)
    desc.to_csv(output_dir / "stagev_v4_smoke_descriptors.csv", index=False)
    graph["nodes"].to_csv(output_dir / "stagev_v4_smoke_nodes.csv", index=False)
    graph["edges"].to_csv(output_dir / "stagev_v4_smoke_edges.csv", index=False)
    scored.head(256).to_csv(output_dir / "stagev_v4_smoke_scored_candidates.csv", index=False)
    if (
        checks["node_count"] <= 0
        or checks["proposal_count"] <= 0
        or not checks["proposal_sources_valid"]
        or not checks["action_family_valid"]
        or not checks["graph_context_nonzero"]
    ):
        checks["status"] = "fail"
        write_json(output_dir / "stagev_v4_smoke_summary.json", checks)
        raise SystemExit(1)
    return checks


def main() -> None:
    p = argparse.ArgumentParser(description="Run Stage V-v4 phase-structure-graph smoke checks.")
    p.add_argument("--output-dir", type=Path, default=Path("reports/stagev_v4_graph_acq_smoke"))
    args = p.parse_args()
    print(json.dumps(run_smoke(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
