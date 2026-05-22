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


def merge_exact_shards(run_dir: Path, iteration: int, world_size: int) -> Path:
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

    out_path = iter_dir / f"exact_merged_iter{iteration:03d}.npz"
    np.savez(out_path, **merged)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HPC helpers for active-learning point partition and shard merge.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory under ML_Phase/active_runs.")
    p.add_argument("--iteration", type=int, required=True, help="Iteration index.")
    p.add_argument("--world-size", type=int, default=1, help="Number of shards/ranks.")

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

    merged = merge_exact_shards(args.run_dir, args.iteration, args.world_size)
    print(f"Wrote merged shard file: {merged}")


if __name__ == "__main__":
    main()

