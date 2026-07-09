from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_phase.hpc import _add_boundary_band_metadata


ROOT = Path("hpc_upload_robust_oracle_acq_compare_20260525_12h/ML_Phase_512_20260601")
RUNS = {
    "full": "active_boundary_discovery_robust_oracle_full_acq_v1",
    "simple_phase": "active_boundary_discovery_robust_oracle_simple_acq_v1",
}
DELTA_EPS = 1e-3
POSITIVE_DELTA_GAP_TOL = 1e-8


def _load_iter_shards(run_id: str, iteration: int) -> dict[str, np.ndarray]:
    iter_dir = ROOT / "active_runs" / run_id / f"iter{iteration:03d}"
    arrays: dict[str, list[np.ndarray]] = {}
    for path in sorted(iter_dir.glob("exact_shard_rank*_of*.npz")):
        with np.load(path, allow_pickle=True) as z:
            for key in z.files:
                arr = z[key]
                if arr.ndim >= 1 and arr.shape[0] != 1:
                    arrays.setdefault(key, []).append(arr.copy())
    if not arrays:
        raise FileNotFoundError(f"No shard arrays found under {iter_dir}")
    return {key: np.concatenate(vals, axis=0) for key, vals in arrays.items()}


def _copy_payload(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(val).copy() for key, val in payload.items()}


def _validate_existing_shards() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile, run_id in RUNS.items():
        for iteration in (31, 32):
            payload = _load_iter_shards(run_id, iteration)
            patched = _copy_payload(payload)
            _add_boundary_band_metadata(
                patched,
                delta_eps=DELTA_EPS,
                positive_delta_gap_tol=POSITIVE_DELTA_GAP_TOL,
            )
            delta = np.asarray(patched["delta_opt"], dtype=float)
            gap = np.asarray(patched["positive_delta_gap"], dtype=float)
            q_status = np.asarray(patched["q_status"], dtype=int)
            q_unresolved = np.asarray(patched["q_unresolved"], dtype=bool)
            eligible = np.asarray(patched["training_eligible_exact"], dtype=bool)
            stable_normal = (delta < DELTA_EPS) & np.isfinite(gap) & (gap > POSITIVE_DELTA_GAP_TOL)
            normal_q_ok = (q_status[stable_normal] == 0).all() and (~q_unresolved[stable_normal]).all()
            stable_normal_eligible = eligible[stable_normal].all()
            if not normal_q_ok:
                raise AssertionError(f"{profile} iter{iteration}: stable normal q metadata invalid")
            if not stable_normal_eligible:
                raise AssertionError(f"{profile} iter{iteration}: stable normal points not eligible")
            rows.append(
                {
                    "profile": profile,
                    "iteration": iteration,
                    "points": int(delta.shape[0]),
                    "stable_normal_count": int(stable_normal.sum()),
                    "training_eligible_count_after_patch": int(eligible.sum()),
                    "q_unresolved_count": int(q_unresolved.sum()),
                    "stable_normal_q_not_applicable_ok": bool(normal_q_ok),
                    "stable_normal_training_eligible_ok": bool(stable_normal_eligible),
                }
            )
    return rows


def _validate_synthetic_cases() -> list[dict[str, object]]:
    payload = {
        "kT": np.array([0.5, 0.1, 0.05, 0.2], dtype=float),
        "JA": np.array([2.0, 1.2, 0.4, 1.5], dtype=float),
        "delta_opt": np.array([0.0, 0.0, 0.2, 0.2], dtype=float),
        "positive_delta_gap": np.array([5.0e-7, 5.0e-9, -1.0e-4, -1.0e-4], dtype=float),
        "trusted_exact": np.array([0, 0, 1, 1], dtype=np.int8),
        "exact_status_code": np.array([0, 0, 0, 1], dtype=np.int64),
        "delta_unresolved": np.array([0, 0, 0, 0], dtype=np.int8),
        "delta_boundary_ambiguous": np.array([1, 1, 0, 0], dtype=np.int8),
        "boundary_ambiguous": np.array([1, 1, 0, 0], dtype=np.int8),
        "q_unresolved": np.array([0, 0, 0, 1], dtype=np.int8),
    }
    _add_boundary_band_metadata(payload, delta_eps=DELTA_EPS, positive_delta_gap_tol=POSITIVE_DELTA_GAP_TOL)
    eligible = payload["training_eligible_exact"].astype(bool)
    expected = np.array([True, True, True, False])
    if not np.array_equal(eligible, expected):
        raise AssertionError(f"Synthetic eligibility mismatch: got {eligible}, expected {expected}")
    return [
        {"case": "stable_normal_gap_5e-7", "training_eligible": bool(eligible[0]), "expected": True},
        {"case": "boundary_band_normal_gap_5e-9", "training_eligible": bool(eligible[1]), "expected": True},
        {"case": "clean_sc", "training_eligible": bool(eligible[2]), "expected": True},
        {"case": "solver_failed_sc", "training_eligible": bool(eligible[3]), "expected": False},
    ]


def main() -> None:
    existing = _validate_existing_shards()
    synthetic = _validate_synthetic_cases()
    out_dir = Path("reports/robust_oracle_label_closure_validation/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "existing_shard_validation.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")
    (out_dir / "synthetic_case_validation.json").write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    pd.DataFrame(existing).to_csv(out_dir / "existing_shard_validation.csv", index=False)
    pd.DataFrame(synthetic).to_csv(out_dir / "synthetic_case_validation.csv", index=False)
    print(json.dumps({"existing_shards": existing, "synthetic_cases": synthetic}, indent=2))


if __name__ == "__main__":
    main()
