from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eta_phase_diagram_cuda import EtaPhaseConfig
from ml_phase.exact_oracle import evaluate_points


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-check robust exact oracle on a small point set.")
    p.add_argument("--points-csv", type=Path, required=True, help="Input CSV with at least columns kT, JA.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison outputs.")
    p.add_argument("--limit", type=int, default=12, help="Maximum number of points to test.")
    p.add_argument("--device", type=str, default="cuda:0", help="Torch device for exact solver.")
    p.add_argument("--world-size", type=int, default=1, help="Metadata only.")
    p.add_argument("--q-min", type=float, default=-1.0)
    p.add_argument("--q-max", type=float, default=0.5)
    p.add_argument("--n-q", type=int, default=300)
    p.add_argument("--delta-min", type=float, default=0.0)
    p.add_argument("--delta-max", type=float, default=0.6)
    p.add_argument("--n-delta", type=int, default=260)
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> EtaPhaseConfig:
    return EtaPhaseConfig(
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        n_q=int(args.n_q),
        delta_min=float(args.delta_min),
        delta_max=float(args.delta_max),
        n_delta=int(args.n_delta),
    )


def to_frame(res: dict[str, np.ndarray], mode: str) -> pd.DataFrame:
    keep = [
        "kT",
        "JA",
        "phase_candidate",
        "q_opt",
        "delta_opt",
        "free_energy_gap_to_normal",
        "trusted_exact",
        "q_edge_hit",
        "q_expanded",
        "q_unresolved",
        "delta_boundary_ambiguous",
        "delta_unresolved",
        "q_expansion_count",
        "expanded_window_found_lower_branch",
        "phase_changed_after_q_expansion",
        "boundary_ambiguous",
        "exact_status_code",
        "exact_status_name",
        "unresolved_reason",
        "oracle_mode",
    ]
    out: dict[str, np.ndarray] = {}
    for key in keep:
        if key in res:
            out[key] = np.asarray(res[key])
        else:
            n = int(np.asarray(res["kT"]).shape[0])
            if key in {"exact_status_name", "unresolved_reason", "oracle_mode"}:
                out[key] = np.asarray(["N/A"] * n)
            else:
                out[key] = np.full(n, np.nan)
    df = pd.DataFrame(out)
    df["mode"] = mode
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.points_csv)
    if "kT" not in df.columns or "JA" not in df.columns:
        raise ValueError("points-csv must include columns: kT, JA")
    points = df[["kT", "JA"]].to_numpy(dtype=np.float64)
    if int(args.limit) > 0:
        points = points[: int(args.limit)]
    if points.shape[0] == 0:
        raise ValueError("No points selected for smoke check.")

    cfg = build_cfg(args)
    legacy = evaluate_points(
        points=points,
        cfg=cfg,
        device=args.device,
        output_file=args.output_dir / "legacy_partial.npz",
        save_every=1,
        oracle_mode="legacy",
        enable_q_expansion=True,
        enable_delta_refinement=True,
    ).to_dict()
    robust = evaluate_points(
        points=points,
        cfg=cfg,
        device=args.device,
        output_file=args.output_dir / "robust_partial.npz",
        save_every=1,
        oracle_mode="robust_al",
        enable_q_expansion=True,
        enable_delta_refinement=True,
        branch_dir=args.output_dir / "branch_candidates",
    ).to_dict()

    legacy_df = to_frame(legacy, mode="legacy")
    robust_df = to_frame(robust, mode="robust_al")
    merged = legacy_df.merge(
        robust_df,
        on=["kT", "JA"],
        suffixes=("_legacy", "_robust"),
        how="inner",
    )
    merged["phase_changed"] = merged["phase_candidate_legacy"] != merged["phase_candidate_robust"]
    merged["deltaf_changed"] = merged["free_energy_gap_to_normal_robust"] - merged["free_energy_gap_to_normal_legacy"]
    merged["qopt_changed"] = merged["q_opt_robust"] - merged["q_opt_legacy"]
    merged.to_csv(args.output_dir / "smoke_compare.csv", index=False)

    summary = {
        "n_points": int(points.shape[0]),
        "phase_changed_count": int(merged["phase_changed"].sum()),
        "robust_q_expanded_count": int(np.nansum(np.asarray(merged["q_expanded_robust"], dtype=np.float64))),
        "robust_q_unresolved_count": int(np.nansum(np.asarray(merged["q_unresolved_robust"], dtype=np.float64))),
        "robust_delta_ambiguous_count": int(np.nansum(np.asarray(merged["delta_boundary_ambiguous_robust"], dtype=np.float64))),
        "robust_trusted_count": int(np.nansum(np.asarray(merged["trusted_exact_robust"], dtype=np.float64))),
        "device": str(args.device),
    }
    (args.output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {args.output_dir / 'smoke_compare.csv'}")


if __name__ == "__main__":
    main()
