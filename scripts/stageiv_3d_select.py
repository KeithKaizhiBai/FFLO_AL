from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.dataset_builder import load_flat_dataset
from ml_phase.stageiv_3d import (
    STAGEIV_OUTPUT_ROOT,
    STAGEIV_RUN_ID,
    StageIV3DConfig,
    generate_stageiv_candidate_pool,
    score_stageiv_candidates,
    select_stageiv_batch,
    sobol_points_3d,
    write_empty_stageiv_dataset,
    write_stageiv_selection,
)


def _load_config(path: Path | None) -> StageIV3DConfig:
    if path is None:
        return StageIV3DConfig()
    return StageIV3DConfig.from_json(path)


def _run_dir(cfg: StageIV3DConfig, output_root: str | None, run_id: str | None) -> Path:
    root = Path(output_root or cfg.output_root)
    rid = run_id or cfg.run_id
    return root / "active_runs" / rid


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def select_seed(args: argparse.Namespace, cfg: StageIV3DConfig, run_dir: Path) -> None:
    if args.iteration != 0:
        raise ValueError("Seed mode must use --iteration 0.")
    write_empty_stageiv_dataset(run_dir, iteration=0)
    points = sobol_points_3d(int(cfg.initial_seed_size), cfg, seed_offset=0)
    meta = pd.DataFrame(
        {
            "selection_rank": np.arange(1, points.shape[0] + 1, dtype=np.int64),
            "selection_source": "stageiv_3d_scrambled_sobol_seed",
            "kT": points[:, 0],
            "JA": points[:, 1],
            "mu": points[:, 2],
        }
    )
    summary = {
        "mode": "seed",
        "run_id": cfg.run_id,
        "iteration": int(args.iteration),
        "selected_batch_size": int(points.shape[0]),
        "initial_seed_size": int(cfg.initial_seed_size),
        "world_size": int(args.world_size),
        "partition_strategy": str(args.partition_strategy),
        "cold_start": True,
    }
    write_stageiv_selection(
        run_dir,
        int(args.iteration),
        points,
        meta,
        cfg,
        world_size=int(args.world_size),
        partition_strategy=str(args.partition_strategy),
        summary=summary,
    )
    _write_json(run_dir / "run_config.json", cfg.to_dict())


def select_acquisition(args: argparse.Namespace, cfg: StageIV3DConfig, run_dir: Path) -> None:
    dataset_path = args.dataset or (run_dir / f"dataset_iter{int(args.iteration):03d}.npz")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing cumulative dataset for acquisition selection: {dataset_path}")
    dataset = load_flat_dataset(dataset_path)
    if dataset.x.shape[1] != 3:
        raise ValueError(f"Stage IV acquisition requires 3D dataset coordinates, got x.shape={dataset.x.shape}.")
    if dataset.x.shape[0] < 8:
        raise ValueError("Stage IV acquisition needs at least 8 exact points; run seed exact first.")

    candidates, candidate_meta, candidate_summary = generate_stageiv_candidate_pool(dataset, cfg, int(args.iteration))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores, score_summary = score_stageiv_candidates(dataset, candidates, cfg, int(args.iteration), device=device)
    selected, meta, select_summary = select_stageiv_batch(
        candidates,
        scores,
        cfg,
        int(args.iteration),
        candidate_metadata=candidate_meta,
    )
    summary = {
        "mode": "acquisition",
        "run_id": cfg.run_id,
        "iteration": int(args.iteration),
        "dataset_path": str(dataset_path),
        "dataset_samples": int(dataset.x.shape[0]),
        "world_size": int(args.world_size),
        "partition_strategy": str(args.partition_strategy),
        "device": str(device),
        **candidate_summary,
        **score_summary,
        **select_summary,
    }
    iter_dir = run_dir / f"iter{int(args.iteration):03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    candidate_points_frame = pd.DataFrame(candidates, columns=["kT", "JA", "mu"])
    if "candidate_source" in candidate_meta:
        candidate_points_frame["candidate_source"] = candidate_meta["candidate_source"].to_numpy()
    candidate_points_frame.to_csv(iter_dir / "candidate_points.csv", index=False)
    score_frame = pd.DataFrame({"kT": candidates[:, 0], "JA": candidates[:, 1], "mu": candidates[:, 2]})
    if "candidate_source" in candidate_meta:
        score_frame["candidate_source"] = candidate_meta["candidate_source"].to_numpy()
    for key, value in scores.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.shape[0] == candidates.shape[0]:
            score_frame[key] = arr
    score_frame.to_csv(iter_dir / "candidate_scores.csv", index=False)
    write_stageiv_selection(
        run_dir,
        int(args.iteration),
        selected,
        meta,
        cfg,
        world_size=int(args.world_size),
        partition_strategy=str(args.partition_strategy),
        summary=summary,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage IV 3D cold-start topology-aware batch selector.")
    p.add_argument("--config", type=Path, default=None, help="Stage IV JSON config. Defaults to production values.")
    p.add_argument("--mode", choices=["seed", "acquisition"], required=True, help="Selection mode.")
    p.add_argument("--iteration", type=int, required=True, help="Iteration index to select.")
    p.add_argument("--output-root", type=str, default=STAGEIV_OUTPUT_ROOT, help="Output root.")
    p.add_argument("--run-id", type=str, default=STAGEIV_RUN_ID, help="Run id.")
    p.add_argument("--dataset", type=Path, default=None, help="Optional cumulative dataset path for acquisition mode.")
    p.add_argument("--world-size", type=int, default=8, help="Number of exact-oracle shards.")
    p.add_argument("--partition-strategy", type=str, default="cost_aware", choices=["round_robin", "cost_aware"])
    p.add_argument("--device", type=str, default="cpu", help="Training device for acquisition scoring.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    run_dir = _run_dir(cfg, args.output_root, args.run_id)
    if args.mode == "seed":
        select_seed(args, cfg, run_dir)
    else:
        select_acquisition(args, cfg, run_dir)
    print(f"Wrote Stage IV selection for {args.mode} iteration {int(args.iteration):03d}: {run_dir}")


if __name__ == "__main__":
    main()
