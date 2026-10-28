from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import ActiveLearningConfig, ensure_output_dirs
from .labels import PHASE_NAMES, eta_sign_label, label_name_map, phase_label, strong_diode_label


@dataclass
class FlatDataset:
    x: np.ndarray
    y_reg: np.ndarray
    y_phase: np.ndarray
    y_eta_sign: np.ndarray
    y_strong_diode: np.ndarray
    records: Dict[str, np.ndarray]


OPTIONAL_RECORD_DEFAULTS: Dict[str, float | int] = {
    "mu": 0.55,
    "q_min": np.nan,
    "q_max": np.nan,
    "n_q": 0,
    "q_index": -1,
    "q_edge_distance": np.nan,
    "q_edge_hit": 0,
    "q_refinement_level": 0,
    "q_status": 0,
    "q_expanded": 0,
    "q_unresolved": 0,
    "delta_min": np.nan,
    "delta_max": np.nan,
    "n_delta": 0,
    "n_delta_refined": 0,
    "delta_refinement_level": 0,
    "delta_status": 0,
    "delta_boundary_ambiguous": 0,
    "delta_boundary_band_normal": 0,
    "delta_refined": 0,
    "delta_unresolved": 0,
    "free_energy_gap_to_normal": np.nan,
    "positive_delta_gap": np.nan,
    "positive_delta_checked": 0,
    "phase_candidate": 0,
    "exact_status_code": 0,
    "trusted_exact": 1,
    "training_eligible_exact": 1,
    "needs_rerun_exact": 0,
    "topology_enabled": 0,
    "topology_applicable": 0,
    "topology_pending": 1,
    "topology_label_code": -1,
    "topology_z2": -1,
    "topology_spectral_status_code": -1,
    "topology_trusted": 0,
    "topology_p0": np.nan,
    "topology_ppi": np.nan,
    "topology_pf_product": np.nan,
    "topology_pfaffian_margin": np.nan,
    "topology_bulk_gap": np.nan,
    "topology_k_at_bulk_gap": np.nan,
    "topology_gap_tol": np.nan,
    "topology_gap_nk": 0,
    "topology_gap_backend_code": -1,
    "topology_runtime_sec": np.nan,
    "topology_error_code": 0,
}


def _with_optional_record_defaults(records: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
    out = dict(records)
    for key, default in OPTIONAL_RECORD_DEFAULTS.items():
        if key not in out:
            dtype = np.float64 if isinstance(default, float) and np.isnan(default) else np.asarray(default).dtype
            out[key] = np.full(n, default, dtype=dtype)
    return out


def _write_records_csv(path: Path, records: Dict[str, np.ndarray]) -> None:
    keys = list(records.keys())
    n = len(next(iter(records.values()))) if records else 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([records[k][i] for k in keys])


def load_warm_start_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        required = {
            "kT_vec",
            "JA_vec",
            "eta_matrix",
            "q_opt_matrix",
            "delta_opt_matrix",
            "ic_plus_matrix",
            "ic_minus_matrix",
        }
        missing = required - keys
        if missing:
            raise KeyError(f"Missing required arrays: {sorted(missing)}")
        return {k: z[k].copy() for k in z.files}


def _flatten_grid(npz_data: Dict[str, np.ndarray], cfg: ActiveLearningConfig) -> FlatDataset:
    kT = np.asarray(npz_data["kT_vec"], dtype=np.float64)
    JA = np.asarray(npz_data["JA_vec"], dtype=np.float64)
    eta = np.asarray(npz_data["eta_matrix"], dtype=np.float64)
    q_opt = np.asarray(npz_data["q_opt_matrix"], dtype=np.float64)
    delta_opt = np.asarray(npz_data["delta_opt_matrix"], dtype=np.float64)
    ic_plus = np.asarray(npz_data["ic_plus_matrix"], dtype=np.float64)
    ic_minus = np.asarray(npz_data["ic_minus_matrix"], dtype=np.float64)

    if eta.shape != (JA.size, kT.size):
        raise ValueError(
            f"Shape mismatch: eta_matrix shape={eta.shape}, expected ({JA.size}, {kT.size}) from JA/kT vectors."
        )

    kt_mesh, ja_mesh = np.meshgrid(kT, JA, indexing="xy")
    x = np.stack([kt_mesh.ravel(), ja_mesh.ravel()], axis=1)
    y_reg = np.stack([delta_opt.ravel(), q_opt.ravel(), eta.ravel(), ic_plus.ravel(), ic_minus.ravel()], axis=1)

    phase = phase_label(delta_opt.ravel(), q_opt.ravel(), cfg.delta_eps, cfg.q_eps)
    eta_sign = eta_sign_label(eta.ravel())
    strong = strong_diode_label(eta.ravel(), cfg.eta_strong)

    records = {
            "kT": x[:, 0],
            "JA": x[:, 1],
            "mu": x[:, 2] if x.shape[1] >= 3 else np.full(x.shape[0], OPTIONAL_RECORD_DEFAULTS["mu"], dtype=np.float64),
            "delta_opt": y_reg[:, 0],
            "q_opt": y_reg[:, 1],
            "eta": y_reg[:, 2],
            "ic_plus": y_reg[:, 3],
            "ic_minus": y_reg[:, 4],
            "phase_label": phase,
            "phase_name": label_name_map(phase, PHASE_NAMES),
            "eta_sign_label": eta_sign,
            "strong_diode_label": strong,
        }
    records = _with_optional_record_defaults(records, x.shape[0])

    return FlatDataset(
        x=x.astype(np.float64),
        y_reg=y_reg.astype(np.float64),
        y_phase=phase.astype(np.int64),
        y_eta_sign=eta_sign.astype(np.int64),
        y_strong_diode=strong.astype(np.int64),
        records=records,
    )


def build_warm_start_dataset(
    warm_start_npz: Path,
    cfg: ActiveLearningConfig,
    output_root: Path | None = None,
) -> Tuple[FlatDataset, Path, Path, Path]:
    npz_data = load_warm_start_npz(warm_start_npz)
    flat = _flatten_grid(npz_data, cfg)

    if output_root is None:
        output_root = cfg.output_root_path
    output_root.mkdir(parents=True, exist_ok=True)
    datasets_dir = output_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    base = datasets_dir / cfg.warm_start_output_name
    out_npz = base.with_suffix(".npz")
    out_csv = base.with_suffix(".csv")
    out_meta = base.with_suffix(".json")

    np.savez(
        out_npz,
        x=flat.x,
        y_reg=flat.y_reg,
        y_phase=flat.y_phase,
        y_eta_sign=flat.y_eta_sign,
        y_strong_diode=flat.y_strong_diode,
        **{k: v for k, v in flat.records.items() if k in OPTIONAL_RECORD_DEFAULTS},
    )
    _write_records_csv(out_csv, flat.records)
    out_meta.write_text(
        json.dumps(
            {
                "warm_start_npz": str(warm_start_npz),
                "n_samples": int(flat.x.shape[0]),
                "n_features": int(flat.x.shape[1]),
                "n_regression_targets": int(flat.y_reg.shape[1]),
                "config": cfg.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return flat, out_npz, out_csv, out_meta


def load_flat_dataset(npz_path: Path) -> FlatDataset:
    with np.load(npz_path, allow_pickle=False) as z:
        x = z["x"].astype(np.float64)
        y_reg = z["y_reg"].astype(np.float64)
        y_phase = z["y_phase"].astype(np.int64)
        y_eta_sign = z["y_eta_sign"].astype(np.int64)
        y_strong_diode = z["y_strong_diode"].astype(np.int64)
        optional = {k: z[k].copy() for k in OPTIONAL_RECORD_DEFAULTS if k in z.files}

    records = {
            "kT": x[:, 0],
            "JA": x[:, 1],
            "mu": x[:, 2] if x.shape[1] >= 3 else np.full(x.shape[0], OPTIONAL_RECORD_DEFAULTS["mu"], dtype=np.float64),
            "delta_opt": y_reg[:, 0],
            "q_opt": y_reg[:, 1],
            "eta": y_reg[:, 2],
            "ic_plus": y_reg[:, 3],
            "ic_minus": y_reg[:, 4],
            "phase_label": y_phase,
            "phase_name": label_name_map(y_phase, PHASE_NAMES),
            "eta_sign_label": y_eta_sign,
            "strong_diode_label": y_strong_diode,
        }
    records.update(optional)
    records = _with_optional_record_defaults(records, x.shape[0])
    return FlatDataset(
        x=x,
        y_reg=y_reg,
        y_phase=y_phase,
        y_eta_sign=y_eta_sign,
        y_strong_diode=y_strong_diode,
        records=records,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build warm-start ML dataset from eta phase-diagram npz.")
    p.add_argument("--input", required=True, type=Path, help="Path to eta_phase_diagram_*.npz.")
    p.add_argument("--output-root", type=Path, default=Path("ML_Phase"), help="Output root directory.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ActiveLearningConfig(output_root=str(args.output_root))
    ensure_output_dirs(cfg)
    flat, out_npz, out_csv, out_meta = build_warm_start_dataset(args.input, cfg, args.output_root)
    print(f"Built dataset with {flat.x.shape[0]} samples.")
    print(f"Wrote {out_npz}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()
