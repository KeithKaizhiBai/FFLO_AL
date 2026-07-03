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

from ml_phase.stagev_acqv2 import (
    STAGEV_OUTPUT_ROOT,
    STAGEV_RUN_ID,
    StageVConfig,
    build_boundary_support_sets,
    generate_stagev_candidate_pool,
    predict_linear_value_model,
    predict_stagev_fields,
    score_stagev_a0,
    select_micro_batch,
    sobol_points_3d,
    write_empty_stagev_dataset,
    write_stagev_selection,
)
from ml_phase.dataset_builder import load_flat_dataset


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_config(path: Path | None) -> StageVConfig:
    return StageVConfig() if path is None else StageVConfig.from_json(path)


def run_dir(cfg: StageVConfig, output_root: str | None, run_id: str | None) -> Path:
    return Path(output_root or cfg.output_root) / "active_runs" / (run_id or cfg.run_id)


def select_seed(args: argparse.Namespace, cfg: StageVConfig, out_run_dir: Path) -> None:
    if int(args.iteration) != 0:
        raise ValueError("seed mode requires --iteration 0")
    write_empty_stagev_dataset(out_run_dir, iteration=0)
    points = sobol_points_3d(int(cfg.initial_seed_size), cfg, seed_offset=0)
    meta = pd.DataFrame(
        {
            "selection_rank": np.arange(1, points.shape[0] + 1, dtype=np.int64),
            "selection_source": "stagev_scrambled_sobol_seed",
            "candidate_source": "initial_sobol",
            "kT": points[:, 0],
            "JA": points[:, 1],
            "mu": points[:, 2],
            "A0": np.nan,
            "g_theta": 0.0,
            "lambda_t": 0.0,
            "final_A": np.nan,
            "selection_probability": 1.0 / max(points.shape[0], 1),
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
        "stageiv_data_used_for_training": False,
    }
    write_stagev_selection(out_run_dir, int(args.iteration), points, meta, cfg, int(args.world_size), str(args.partition_strategy), summary)
    write_json(out_run_dir / "run_config.json", cfg.to_dict())


def select_acquisition(args: argparse.Namespace, cfg: StageVConfig, out_run_dir: Path) -> None:
    dataset_path = args.dataset or (out_run_dir / f"dataset_iter{int(args.iteration):03d}.npz")
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing Stage V cumulative dataset: {dataset_path}")
    dataset = load_flat_dataset(dataset_path)
    if dataset.x.shape[1] != 3:
        raise ValueError(f"Stage V requires 3D dataset coordinates, got x.shape={dataset.x.shape}")
    if dataset.x.shape[0] < 8:
        raise ValueError("Stage V acquisition requires at least 8 exact points")
    candidates, candidate_meta, candidate_summary = generate_stagev_candidate_pool(dataset, cfg, int(args.iteration))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, field_summary = predict_stagev_fields(dataset, candidates, cfg, device=device)
    support_sets = build_boundary_support_sets(dataset, cfg)
    scored, a0_summary = score_stagev_a0(features, support_sets, cfg)
    if "candidate_source" in candidate_meta:
        scored["candidate_source"] = candidate_meta["candidate_source"].to_numpy()
    reward_model = read_json(out_run_dir / "stagev_reward_model.json")
    g = predict_linear_value_model(reward_model, scored)
    lambda_t = float(read_json(out_run_dir / "stagev_lambda_state.json").get("lambda_t", 0.0))
    selected, meta, select_summary = select_micro_batch(
        scored,
        cfg,
        rng=np.random.default_rng(int(cfg.random_seed) + int(args.iteration) * 100003),
        learned_values=g,
        lambda_t=lambda_t,
    )
    iter_dir = out_run_dir / f"iter{int(args.iteration):03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates, columns=["kT", "JA", "mu"]).assign(candidate_source=scored.get("candidate_source", "unknown")).to_csv(
        iter_dir / "candidate_points.csv",
        index=False,
    )
    top_k = int(args.log_top_k)
    scored.assign(g_theta=g, lambda_t=lambda_t).nlargest(min(top_k, len(scored)), "A0").to_csv(iter_dir / "candidate_score_topk.csv", index=False)
    summary = {
        "mode": "acquisition",
        "run_id": cfg.run_id,
        "iteration": int(args.iteration),
        "dataset_path": str(dataset_path),
        "dataset_samples": int(dataset.x.shape[0]),
        "world_size": int(args.world_size),
        "partition_strategy": str(args.partition_strategy),
        "device": str(device),
        "lambda_t": lambda_t,
        "stageiv_data_used_for_training": False,
        **candidate_summary,
        **field_summary,
        **a0_summary,
        **select_summary,
    }
    write_stagev_selection(out_run_dir, int(args.iteration), selected, meta, cfg, int(args.world_size), str(args.partition_strategy), summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage V acquisition-v2 selector.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--mode", choices=["seed", "acquisition"], required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--output-root", default=STAGEV_OUTPUT_ROOT)
    p.add_argument("--run-id", default=STAGEV_RUN_ID)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--partition-strategy", choices=["round_robin", "cost_aware"], default="cost_aware")
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-top-k", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_run_dir = run_dir(cfg, args.output_root, args.run_id)
    if args.mode == "seed":
        select_seed(args, cfg, out_run_dir)
    else:
        select_acquisition(args, cfg, out_run_dir)
    print(f"Wrote Stage V selection for {args.mode} iteration {int(args.iteration):03d}: {out_run_dir}")


if __name__ == "__main__":
    main()
