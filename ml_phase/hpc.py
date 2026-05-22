from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def partition_points(points: np.ndarray, world_size: int, strategy: str = "round_robin") -> List[np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be shape (n, 2) with columns [kT, JA]")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    strategy = strategy.lower().strip()
    shards: List[np.ndarray] = []
    if strategy == "round_robin":
        for r in range(world_size):
            shards.append(points[r::world_size].copy())
        return shards
    if strategy == "contiguous":
        idx = np.array_split(np.arange(points.shape[0]), world_size)
        for chunk in idx:
            shards.append(points[chunk].copy())
        return shards
    if strategy == "cost_aware":
        order = np.argsort(points[:, 0] + 0.25 * points[:, 1])[::-1]
        buckets: List[list[np.ndarray]] = [[] for _ in range(world_size)]
        loads = np.zeros(world_size, dtype=np.float64)
        for idx in order:
            cost = points[idx, 0] + 0.25 * points[idx, 1]
            r = int(np.argmin(loads))
            buckets[r].append(points[idx])
            loads[r] += cost
        for b in buckets:
            shards.append(np.array(b, dtype=np.float64) if b else np.empty((0, 2), dtype=np.float64))
        return shards
    raise ValueError(f"Unsupported partition strategy: {strategy}")


def write_point_shards(
    run_dir: Path,
    iteration: int,
    points: np.ndarray,
    world_size: int,
    strategy: str,
) -> list[Path]:
    iter_dir = run_dir / f"iter{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float64)
    all_csv = iter_dir / "selected_points.csv"
    pd.DataFrame(points, columns=["kT", "JA"]).to_csv(all_csv, index=False)

    shards = partition_points(points, world_size=world_size, strategy=strategy)
    shard_paths: list[Path] = []
    for rank, shard in enumerate(shards):
        p = iter_dir / f"selected_points_rank{rank:03d}_of{world_size:03d}.csv"
        pd.DataFrame(shard, columns=["kT", "JA"]).to_csv(p, index=False)
        shard_paths.append(p)

    (iter_dir / "partition_metadata.json").write_text(
        json.dumps(
            {
                "iteration": iteration,
                "world_size": world_size,
                "strategy": strategy,
                "n_selected_points": int(points.shape[0]),
                "rank_sizes": [int(s.shape[0]) for s in shards],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return shard_paths


def merge_exact_shards(
    run_dir: Path,
    iteration: int,
    world_size: int,
    delta_eps: float = 1e-3,
    positive_delta_gap_tol: float = 1e-8,
) -> Path:
    iter_dir = run_dir / f"iter{iteration:03d}"
    shard_paths = [iter_dir / f"exact_shard_rank{r:03d}_of{world_size:03d}.npz" for r in range(world_size)]
    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard files:\n" + "\n".join(missing))

    arrays: dict[str, list[np.ndarray]] = {}
    for p in shard_paths:
        with np.load(p, allow_pickle=False) as z:
            for key in z.files:
                arrays.setdefault(key, []).append(z[key].copy())

    merged: dict[str, np.ndarray] = {}
    for key, vals in arrays.items():
        sample = vals[0]
        if sample.ndim == 0:
            merged[key] = sample
        else:
            merged[key] = np.concatenate(vals, axis=0)

    _add_boundary_band_metadata(
        merged,
        delta_eps=float(delta_eps),
        positive_delta_gap_tol=float(positive_delta_gap_tol),
    )

    out_path = iter_dir / f"exact_merged_iter{iteration:03d}.npz"
    np.savez(out_path, **merged)
    _write_trusted_npz(iter_dir, iteration, merged)
    _write_rerun_points(iter_dir, merged)
    _write_merge_summary(iter_dir, iteration, merged)
    return out_path


def _add_boundary_band_metadata(
    merged: dict[str, np.ndarray],
    delta_eps: float,
    positive_delta_gap_tol: float,
) -> None:
    if "kT" not in merged:
        return
    n = int(np.asarray(merged["kT"]).shape[0])
    if n == 0:
        return

    trusted = np.asarray(merged.get("trusted_exact", np.ones(n, dtype=np.int8))).astype(bool)
    status = np.asarray(merged.get("exact_status_code", np.zeros(n, dtype=np.int64))).astype(np.int64)
    delta_opt = np.asarray(merged.get("delta_opt", np.full(n, np.nan)), dtype=np.float64)
    positive_delta_gap = np.asarray(merged.get("positive_delta_gap", np.full(n, np.nan)), dtype=np.float64)
    delta_unresolved = np.asarray(merged.get("delta_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)

    trusted_clean = trusted & (status == 0)
    finite_gap = np.isfinite(positive_delta_gap)
    boundary_band_normal = (
        delta_unresolved
        & (delta_opt < float(delta_eps))
        & finite_gap
        & (positive_delta_gap >= 0.0)
        & (positive_delta_gap <= float(positive_delta_gap_tol))
    )
    training_eligible = trusted_clean | boundary_band_normal
    needs_rerun = ~(training_eligible)

    merged["delta_boundary_band_normal"] = boundary_band_normal.astype(np.int8)
    merged["training_eligible_exact"] = training_eligible.astype(np.int8)
    merged["needs_rerun_exact"] = needs_rerun.astype(np.int8)


def _per_point_mask(merged: dict[str, np.ndarray], n: int) -> np.ndarray:
    if "training_eligible_exact" in merged:
        return np.asarray(merged["training_eligible_exact"]).astype(bool)
    trusted = np.asarray(merged.get("trusted_exact", np.ones(n, dtype=np.int8))).astype(bool)
    status = np.asarray(merged.get("exact_status_code", np.zeros(n, dtype=np.int64))).astype(np.int64)
    return trusted & (status == 0)


def _write_trusted_npz(iter_dir: Path, iteration: int, merged: dict[str, np.ndarray]) -> Path | None:
    if "kT" not in merged or "JA" not in merged:
        return None
    n = int(np.asarray(merged["kT"]).shape[0])
    trusted_mask = _per_point_mask(merged, n)
    trusted: dict[str, np.ndarray] = {}
    for key, val in merged.items():
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == n:
            trusted[key] = arr[trusted_mask]
        else:
            trusted[key] = arr
    out = iter_dir / f"exact_trusted_iter{iteration:03d}.npz"
    np.savez(out, **trusted)
    training_out = iter_dir / f"exact_training_iter{iteration:03d}.npz"
    np.savez(training_out, **trusted)
    return out


def _status_reason_and_action(i: int, merged: dict[str, np.ndarray], n: int) -> tuple[list[str], list[str]]:
    status = np.asarray(merged.get("exact_status_code", np.zeros(n, dtype=np.int64))).astype(np.int64)
    q_status = np.asarray(merged.get("q_status", np.zeros(n, dtype=np.int64))).astype(np.int64)
    delta_status = np.asarray(merged.get("delta_status", np.zeros(n, dtype=np.int64))).astype(np.int64)
    q_unresolved = np.asarray(merged.get("q_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    delta_unresolved = np.asarray(merged.get("delta_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)

    reasons: list[str] = []
    actions: list[str] = []
    if q_unresolved[i] or (status[i] & 1):
        reasons.append("q_edge_unresolved")
        actions.append("expand q-window and rerun exact oracle")
    if delta_unresolved[i] or (status[i] & 2):
        reasons.append("delta_boundary_unresolved")
        actions.append("increase/refine Delta grid near the normal-SC boundary")
    if status[i] & 4:
        reasons.append("nonfinite_output")
        actions.append("rerun with stricter diagnostics")
    if status[i] & 8:
        reasons.append("max_q_refinement_reached")
        actions.append("increase q_max_abs or inspect physical q range")
    if status[i] & 16:
        reasons.append("max_delta_refinement_reached")
        actions.append("increase max_delta_refinements or n_delta_refined")
    if q_status[i] == 0 and status[i] == 0:
        reasons.append("normal_q_not_applicable")
        actions.append("no q rerun needed if Delta status is stable")
    if delta_status[i] == 1:
        reasons.append("delta_boundary_ambiguous")
        if "increase/refine Delta grid near the normal-SC boundary" not in actions:
            actions.append("increase/refine Delta grid near the normal-SC boundary")
    return reasons, actions


def _write_rerun_points(iter_dir: Path, merged: dict[str, np.ndarray]) -> Path | None:
    if "kT" not in merged or "JA" not in merged:
        return None
    n = int(np.asarray(merged["kT"]).shape[0])
    if n == 0:
        return None

    trusted_mask = _per_point_mask(merged, n)
    status = np.asarray(merged.get("exact_status_code", np.zeros(n, dtype=np.int64))).astype(np.int64)
    boundary_band = np.asarray(merged.get("delta_boundary_band_normal", np.zeros(n, dtype=np.int8))).astype(bool)
    mask = ~(trusted_mask | boundary_band)

    out = iter_dir / "rerun_points.csv"
    if not np.any(mask):
        pd.DataFrame(
            columns=[
                "kT",
                "JA",
                "q_opt",
                "delta_opt",
                "phase_candidate",
                "q_status",
                "delta_status",
                "exact_status_code",
                "delta_boundary_band_normal",
                "training_eligible_exact",
                "reason",
                "recommended_action",
            ]
        ).to_csv(out, index=False)
        return out

    rows: list[dict[str, object]] = []
    for i in np.where(mask)[0]:
        reasons, actions = _status_reason_and_action(i, merged, n)
        rows.append(
            {
                "kT": float(np.asarray(merged["kT"])[i]),
                "JA": float(np.asarray(merged["JA"])[i]),
                "q_opt": float(np.asarray(merged.get("q_opt", np.full(n, np.nan)))[i]),
                "delta_opt": float(np.asarray(merged.get("delta_opt", np.full(n, np.nan)))[i]),
                "phase_candidate": int(np.asarray(merged.get("phase_candidate", np.full(n, -1)))[i]),
                "q_status": int(np.asarray(merged.get("q_status", np.full(n, -1)))[i]),
                "delta_status": int(np.asarray(merged.get("delta_status", np.full(n, -1)))[i]),
                "exact_status_code": int(status[i]),
                "delta_boundary_band_normal": int(boundary_band[i]),
                "training_eligible_exact": int(trusted_mask[i]),
                "reason": ";".join(reasons),
                "recommended_action": ";".join(actions),
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _write_merge_summary(iter_dir: Path, iteration: int, merged: dict[str, np.ndarray]) -> Path | None:
    if "kT" not in merged:
        return None
    n = int(np.asarray(merged["kT"]).shape[0])
    status = np.asarray(merged.get("exact_status_code", np.zeros(n, dtype=np.int64))).astype(np.int64)
    clean_trusted = np.asarray(merged.get("trusted_exact", np.zeros(n, dtype=np.int8))).astype(bool) & (status == 0)
    boundary_band = np.asarray(merged.get("delta_boundary_band_normal", np.zeros(n, dtype=np.int8))).astype(bool)
    training_eligible = np.asarray(merged.get("training_eligible_exact", clean_trusted.astype(np.int8))).astype(bool)
    delta_unresolved = np.asarray(merged.get("delta_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    q_unresolved = np.asarray(merged.get("q_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    summary = {
        "iteration": int(iteration),
        "merged_points": n,
        "clean_trusted_points": int(np.sum(clean_trusted)),
        "boundary_band_normal_points": int(np.sum(boundary_band)),
        "training_eligible_points": int(np.sum(training_eligible)),
        "rerun_required_points": int(n - np.sum(training_eligible)),
        "q_unresolved_points": int(np.sum(q_unresolved)),
        "delta_unresolved_points": int(np.sum(delta_unresolved)),
        "delta_unresolved_requiring_rerun": int(np.sum(delta_unresolved & ~boundary_band)),
        "q_expanded_points": int(np.sum(np.asarray(merged.get("q_expanded", np.zeros(n, dtype=np.int8))).astype(bool))),
        "delta_refined_points": int(np.sum(np.asarray(merged.get("delta_refined", np.zeros(n, dtype=np.int8))).astype(bool))),
    }
    out = iter_dir / f"merge_summary_iter{iteration:03d}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HPC helpers for active-learning point partition and shard merge.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory under ML_Phase/active_runs.")
    p.add_argument("--iteration", type=int, required=True, help="Iteration index.")
    p.add_argument("--world-size", type=int, default=1, help="Number of shards/ranks.")
    p.add_argument("--delta-eps", type=float, default=1e-3, help="Delta threshold for boundary-band metadata.")
    p.add_argument(
        "--positive-delta-gap-tol",
        type=float,
        default=1e-8,
        help="Sub-tolerance positive-Delta gap treated as normal/SC boundary band.",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--partition", action="store_true", help="Partition selected points.")
    mode.add_argument("--merge", action="store_true", help="Merge exact shard npz files.")

    p.add_argument("--points-file", type=Path, default=None, help="CSV file with columns kT, JA for partition mode.")
    p.add_argument("--strategy", type=str, default="round_robin", help="Partition strategy.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.partition:
        if args.points_file is None:
            raise ValueError("--points-file is required for --partition")
        df = pd.read_csv(args.points_file)
        points = df[["kT", "JA"]].to_numpy(dtype=np.float64)
        paths = write_point_shards(args.run_dir, args.iteration, points, args.world_size, args.strategy)
        print(f"Wrote {len(paths)} shard files under {args.run_dir / f'iter{args.iteration:03d}'}")
        return

    merged = merge_exact_shards(
        args.run_dir,
        args.iteration,
        args.world_size,
        delta_eps=float(args.delta_eps),
        positive_delta_gap_tol=float(args.positive_delta_gap_tol),
    )
    print(f"Wrote merged shard file: {merged}")


if __name__ == "__main__":
    main()
