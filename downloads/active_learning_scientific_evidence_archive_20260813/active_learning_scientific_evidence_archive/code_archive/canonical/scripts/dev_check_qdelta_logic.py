from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eta_phase_diagram_cuda import EtaPhaseConfig
from ml_phase.exact_oracle import (
    DELTA_REFINED_CONFIRMED,
    PHASE_NORMAL,
    PointOracleResult,
    Q_NOT_APPLICABLE,
    _delta_boundary_ambiguous,
    _phase_candidate,
    expand_q_config,
    refine_delta_config,
)
from ml_phase.hpc import merge_exact_shards


def _fake_result(delta_opt: float, q_opt: float, q_min: float = -1.0, q_max: float = 0.5) -> PointOracleResult:
    return PointOracleResult(
        kT=0.1,
        JA=1.0,
        eta=0.0,
        q_opt=q_opt,
        delta_opt=delta_opt,
        ic_plus=0.0,
        ic_minus=0.0,
        omega_global=-1e-3,
        q_min=q_min,
        q_max=q_max,
        n_q=400,
        q_index=0,
        q_edge_distance=0.0,
        q_edge_hit_raw=1,
        delta_min=0.0,
        delta_max=0.6,
        n_delta=400,
    )


def check_normal_q_not_applicable_logic() -> None:
    r = _fake_result(delta_opt=0.0, q_opt=-1.0)
    delta_amb = _delta_boundary_ambiguous(r, delta_eps=1e-3, delta_boundary_margin=1e-4, free_energy_ambiguity_tol=1e-9)
    assert not delta_amb
    stable_normal = _delta_boundary_ambiguous(
        r,
        delta_eps=1e-3,
        delta_boundary_margin=1e-4,
        free_energy_ambiguity_tol=1e-9,
        positive_delta_gap=1e-5,
    )
    boundary_normal = _delta_boundary_ambiguous(
        r,
        delta_eps=1e-3,
        delta_boundary_margin=1e-4,
        free_energy_ambiguity_tol=1e-9,
        positive_delta_gap=1e-10,
    )
    assert not stable_normal
    assert boundary_normal
    assert _phase_candidate(r, delta_eps=1e-3, delta_ambiguous=delta_amb) == PHASE_NORMAL
    q_status = Q_NOT_APPLICABLE if r.delta_opt < 1e-3 else 1
    q_edge_hit = 0 if q_status == Q_NOT_APPLICABLE else r.q_edge_hit_raw
    assert q_status == Q_NOT_APPLICABLE
    assert q_edge_hit == 0


def check_q_expansion_config() -> None:
    cfg = EtaPhaseConfig(q_min=-1.0, q_max=0.5, n_q=400)
    r = _fake_result(delta_opt=0.1, q_opt=-1.0)
    expanded = expand_q_config(cfg, r, expand_factor=1.5, pad_steps=50, q_max_abs=np.pi)
    assert expanded.q_min < cfg.q_min
    assert expanded.q_max == cfg.q_max
    assert expanded.n_q > cfg.n_q


def check_delta_refinement_config() -> None:
    cfg = EtaPhaseConfig(delta_min=0.0, delta_max=0.6, n_delta=400)
    r = _fake_result(delta_opt=0.001, q_opt=-0.2)
    refined = refine_delta_config(cfg, r, delta_eps=1e-3, half_width=0.03, n_delta_refined=300, refinement_level=1)
    assert refined.delta_min == 0.0
    assert 0.0 < refined.delta_max <= cfg.delta_max
    assert refined.n_delta == 300


def check_hpc_trusted_split() -> None:
    root = Path("ML_Phase/tmp_dev_check_qdelta")
    if root.exists():
        shutil.rmtree(root)
    iter_dir = root / "iter000"
    iter_dir.mkdir(parents=True)

    common = {
        "kT": np.asarray([0.1, 0.2]),
        "JA": np.asarray([1.0, 1.5]),
        "eta": np.asarray([0.0, 0.1]),
        "q_opt": np.asarray([-1.0, -1.2]),
        "delta_opt": np.asarray([0.0, 0.1]),
        "ic_plus": np.asarray([0.0, 0.2]),
        "ic_minus": np.asarray([0.0, -0.2]),
        "phase_candidate": np.asarray([0, 1], dtype=np.int8),
        "q_status": np.asarray([0, 4], dtype=np.int8),
        "delta_status": np.asarray([0, DELTA_REFINED_CONFIRMED], dtype=np.int8),
        "exact_status_code": np.asarray([0, 1], dtype=np.int64),
        "trusted_exact": np.asarray([1, 0], dtype=np.int8),
    }
    np.savez(iter_dir / "exact_shard_rank000_of001.npz", **common)
    merge_exact_shards(root, iteration=0, world_size=1)
    with np.load(iter_dir / "exact_trusted_iter000.npz", allow_pickle=False) as z:
        assert z["kT"].shape[0] == 1
        assert float(z["kT"][0]) == 0.1
    rerun = (iter_dir / "rerun_points.csv").read_text(encoding="utf-8")
    assert "q_edge_unresolved" in rerun
    shutil.rmtree(root)


def main() -> None:
    check_normal_q_not_applicable_logic()
    check_q_expansion_config()
    check_delta_refinement_config()
    check_hpc_trusted_split()
    print("q/Delta logic checks passed.")


if __name__ == "__main__":
    main()
