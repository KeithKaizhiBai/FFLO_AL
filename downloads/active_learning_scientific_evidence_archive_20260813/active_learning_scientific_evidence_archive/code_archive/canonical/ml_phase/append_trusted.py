from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .active_refine import _dataset_from_result
from .config import ActiveLearningConfig
from .dataset_builder import FlatDataset, load_flat_dataset


def _filter_point_arrays(payload: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    n = int(mask.shape[0])
    out: Dict[str, np.ndarray] = {}
    for key, val in payload.items():
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def _save_flat_dataset(dataset: FlatDataset, npz_path: Path, csv_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        x=dataset.x,
        y_reg=dataset.y_reg,
        y_phase=dataset.y_phase,
        y_eta_sign=dataset.y_eta_sign,
        y_strong_diode=dataset.y_strong_diode,
        **{
            k: v
            for k, v in dataset.records.items()
            if k
            not in {
                "kT",
                "JA",
                "delta_opt",
                "q_opt",
                "eta",
                "ic_plus",
                "ic_minus",
                "phase_label",
                "phase_name",
                "eta_sign_label",
                "strong_diode_label",
            }
        },
    )
    pd.DataFrame(dataset.records).to_csv(csv_path, index=False)


def append_trusted_exact(
    dataset_path: Path,
    trusted_exact_path: Path,
    output_npz: Path,
    output_csv: Path,
    cfg: ActiveLearningConfig,
) -> dict:
    dataset = load_flat_dataset(dataset_path)
    input_samples = int(dataset.x.shape[0])
    with np.load(trusted_exact_path, allow_pickle=False) as z:
        exact = {k: z[k].copy() for k in z.files}

    if "kT" not in exact or "JA" not in exact:
        raise KeyError(f"Trusted exact file is missing kT/JA arrays: {trusted_exact_path}")

    n_exact = int(np.asarray(exact["kT"]).shape[0])
    trusted_mask = np.asarray(exact.get("trusted_exact", np.ones(n_exact, dtype=np.int8))).astype(bool)
    status = np.asarray(exact.get("exact_status_code", np.zeros(n_exact, dtype=np.int64))).astype(np.int64)
    clean_trusted_mask = trusted_mask & (status == 0)
    if "training_eligible_exact" in exact:
        append_mask = np.asarray(exact["training_eligible_exact"]).astype(bool)
    else:
        append_mask = clean_trusted_mask
    boundary_band_mask = np.asarray(exact.get("delta_boundary_band_normal", np.zeros(n_exact, dtype=np.int8))).astype(bool)
    trusted = _filter_point_arrays(exact, append_mask)

    if int(np.sum(append_mask)) > 0:
        dataset = _dataset_from_result(dataset, trusted, cfg)

    output_samples = int(dataset.x.shape[0])
    _save_flat_dataset(dataset, output_npz, output_csv)
    summary = {
        "input_dataset": str(dataset_path),
        "trusted_exact": str(trusted_exact_path),
        "output_npz": str(output_npz),
        "output_csv": str(output_csv),
        "exact_points": n_exact,
        "clean_trusted_points": int(np.sum(clean_trusted_mask)),
        "boundary_band_points_appended": int(np.sum(append_mask & boundary_band_mask)),
        "training_eligible_points_appended": int(np.sum(append_mask)),
        "trusted_points_appended": int(np.sum(append_mask)),
        "input_samples": input_samples,
        "output_samples": output_samples,
        "new_unique_samples_added": max(0, output_samples - input_samples),
    }
    output_npz.with_suffix(".append.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append trusted H100 exact points into the active-learning dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="Input dataset_iterXXX.npz.")
    p.add_argument("--trusted-exact", type=Path, required=True, help="exact_trusted_iterXXX.npz from shard merge.")
    p.add_argument("--output-npz", type=Path, required=True, help="Output dataset_iterYYY.npz.")
    p.add_argument("--output-csv", type=Path, required=True, help="Output dataset_iterYYY.csv.")
    p.add_argument("--output-root", type=Path, default=Path("ML_Phase"), help="Output root for config defaults.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ActiveLearningConfig(output_root=str(args.output_root))
    summary = append_trusted_exact(
        dataset_path=args.dataset,
        trusted_exact_path=args.trusted_exact,
        output_npz=args.output_npz,
        output_csv=args.output_csv,
        cfg=cfg,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
