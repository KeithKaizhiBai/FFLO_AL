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
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL
from ml_phase.stagev_v3 import (
    StageV3Config,
    apply_stagev3_sparse_scores,
    build_sparse_topology_surface,
    compute_stagev3_per_boundary_rewards,
    generate_sparse_topology_proposals,
    validate_proposal_sources,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def synthetic_dataset(n: int = 384, seed: int = 20260706) -> FlatDataset:
    rng = np.random.default_rng(seed)
    kT = rng.random(n) * 0.56
    JA = rng.random(n) * 2.12
    mu = -0.5 + 2.0 * rng.random(n)
    phase = np.full(n, PHASE_FFLO, dtype=np.int64)
    phase[: n // 5] = PHASE_NORMAL
    p0 = (JA - (0.74 + 0.16 * kT)) * (mu - 0.22)
    ppi = 1.0 + 0.2 * mu
    topo = (p0 < 0).astype(np.int64)
    topo[phase == PHASE_NORMAL] = -1
    delta = np.where(phase == PHASE_FFLO, 0.2, 0.0)
    q = np.where(phase == PHASE_FFLO, 0.16, 0.0)
    records = {
        "kT": kT,
        "JA": JA,
        "mu": mu,
        "trusted_exact": np.ones(n, dtype=np.int8),
        "training_eligible_exact": np.ones(n, dtype=np.int8),
        "needs_rerun_exact": np.zeros(n, dtype=np.int8),
        "q_unresolved": np.zeros(n, dtype=np.int8),
        "delta_unresolved": np.zeros(n, dtype=np.int8),
        "topology_trusted": (phase == PHASE_FFLO).astype(np.int8),
        "topology_label_code": topo,
        "topology_spectral_status_code": np.where(phase == PHASE_FFLO, 0, -1),
        "topology_p0": np.where(phase == PHASE_FFLO, p0, np.nan),
        "topology_ppi": np.where(phase == PHASE_FFLO, ppi, np.nan),
        "topology_bulk_gap": np.where(phase == PHASE_FFLO, np.abs(p0) + 1e-3, np.nan),
        "free_energy_gap_to_normal": np.where(phase == PHASE_FFLO, -0.1, 0.1),
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
    cfg = StageV3Config(candidate_pool_size=2048, micro_batch_size=16, initial_seed_size=128, reg_epochs=4, cls_epochs=4)
    dataset = synthetic_dataset()
    surface = build_sparse_topology_surface(dataset, cfg)
    proposals, meta, proposal_summary = generate_sparse_topology_proposals(dataset, cfg, iteration=1, count=64)
    meta = validate_proposal_sources(meta, strict=True)
    scored = pd.DataFrame(
        {
            "kT": proposals[:, 0],
            "JA": proposals[:, 1],
            "mu": proposals[:, 2],
            "B_p0_topology": np.ones(len(proposals)) * 0.5,
            "B_ppi_topology": np.zeros(len(proposals)),
            "A_p0_topology": np.ones(len(proposals)) * 0.25,
            "A_ppi_topology": np.zeros(len(proposals)),
            "H_p0_topology": np.zeros(len(proposals)),
            "H_ppi_topology": np.zeros(len(proposals)),
            "A_gap_nodal": np.ones(len(proposals)),
        }
    )
    scored, sparse_summary, _ = apply_stagev3_sparse_scores(scored, dataset, cfg)
    rewards = compute_stagev3_per_boundary_rewards(scored.head(16))
    checks = {
        "status": "pass",
        "run_id": cfg.run_id,
        "cold_start": True,
        "stagev2_data_used_for_training": False,
        "stagev2_tail_data_used_for_training": False,
        "surface_samples": int(surface.shape[0]),
        "proposal_count": int(proposals.shape[0]),
        "proposal_sources_valid": bool(meta["candidate_source"].notna().all()),
        "c_topo_nonzero": bool((scored["C_topo"] > 0).any()),
        "gap_relevance_present": "gap_topology_relevance" in scored,
        "coverage_reward_present": "reward_p0_coverage_local" in rewards,
        "excluded_nodes_required": ["gpuh01"],
        **proposal_summary,
        **sparse_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stagev_v3_smoke_summary.json", checks)
    surface.to_csv(output_dir / "stagev_v3_smoke_sparse_surface.csv", index=False)
    scored.head(256).to_csv(output_dir / "stagev_v3_smoke_scored_candidates.csv", index=False)
    if checks["surface_samples"] <= 0 or not checks["proposal_sources_valid"] or not checks["c_topo_nonzero"]:
        checks["status"] = "fail"
        write_json(output_dir / "stagev_v3_smoke_summary.json", checks)
        raise SystemExit(1)
    return checks


def main() -> None:
    p = argparse.ArgumentParser(description="Run Stage V-v3 sparse-arc smoke checks.")
    p.add_argument("--output-dir", type=Path, default=Path("reports/stagev_v3_sparse_arc_smoke"))
    args = p.parse_args()
    print(json.dumps(run_smoke(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
