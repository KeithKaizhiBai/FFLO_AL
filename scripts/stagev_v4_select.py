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
from ml_phase.stagev_v4 import (
    ACTION_FAMILY_BY_SOURCE,
    STAGEV4_OUTPUT_ROOT,
    STAGEV4_RUN_ID,
    StageV4Config,
    apply_graph_value_model,
    apply_stagev3_sparse_scores,
    attach_graph_context_scores,
    build_boundary_support_sets,
    combine_multihead_scores,
    generate_stagev4_candidate_pool,
    load_stagev4_state,
    predict_stagev_fields,
    score_stagev_a0,
    select_micro_batch_v2,
    sobol_points_3d,
    stagev4_leakage_guard_summary,
    validate_stagev4_proposal_sources,
    write_empty_stagev_dataset,
    write_stagev_selection,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_config(path: Path | None) -> StageV4Config:
    return StageV4Config() if path is None else StageV4Config.from_json(path)


def run_dir(cfg: StageV4Config, output_root: str | None, run_id: str | None) -> Path:
    return Path(output_root or cfg.output_root) / "active_runs" / (run_id or cfg.run_id)


def select_seed(args: argparse.Namespace, cfg: StageV4Config, out_run_dir: Path) -> None:
    if int(args.iteration) != 0:
        raise ValueError("seed mode requires --iteration 0")
    write_empty_stagev_dataset(out_run_dir, iteration=0)
    points = sobol_points_3d(int(cfg.initial_seed_size), cfg, seed_offset=0)
    meta = pd.DataFrame(
        {
            "selection_rank": np.arange(1, points.shape[0] + 1, dtype=np.int64),
            "final_rank": np.arange(1, points.shape[0] + 1, dtype=np.int64),
            "selection_source": "stagev4_scrambled_sobol_seed",
            "candidate_source": "initial_sobol",
            "proposal_source": "initial_sobol",
            "action_family": "global_exploration",
            "dominant_boundary": "seed",
            "kT": points[:, 0],
            "JA": points[:, 1],
            "mu": points[:, 2],
            "A_total_v2": np.nan,
            "A_total_v3": np.nan,
            "A_total_v4": np.nan,
            "graph_value_prior": 0.0,
            "graph_uncertainty_score": 0.0,
            "selection_probability": 1.0 / max(points.shape[0], 1),
            "stagev4_schema_version": cfg.stagev2_schema_version,
        }
    )
    meta = validate_stagev4_proposal_sources(meta, strict=True)
    summary = {
        "mode": "seed",
        "run_id": cfg.run_id,
        "iteration": int(args.iteration),
        "selected_batch_size": int(points.shape[0]),
        "initial_seed_size": int(cfg.initial_seed_size),
        "world_size": int(args.world_size),
        "partition_strategy": str(args.partition_strategy),
        "cold_start": True,
        "graph_learning_enabled": not bool(cfg.known_boundary_baseline_only),
        "candidate_phase_shadow_mode": bool(cfg.candidate_phase_shadow_mode),
        "graph_value_model_shadow_mode": bool(cfg.graph_shadow_mode),
        "stagev4_schema_version": cfg.stagev2_schema_version,
        **stagev4_leakage_guard_summary(),
    }
    write_stagev_selection(out_run_dir, int(args.iteration), points, meta, cfg, int(args.world_size), str(args.partition_strategy), summary)
    write_json(out_run_dir / "run_config.json", cfg.to_dict())


def _assign_sources(scored: pd.DataFrame, candidate_meta: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    if "candidate_source" in candidate_meta:
        frame["candidate_source"] = candidate_meta["candidate_source"].astype(str).to_numpy()
    if "proposal_source" in candidate_meta:
        frame["proposal_source"] = candidate_meta["proposal_source"].astype(str).to_numpy()
    else:
        frame["proposal_source"] = frame.get("candidate_source", "unknown_explicit")
    if "action_family" in candidate_meta:
        frame["action_family"] = candidate_meta["action_family"].astype(str).to_numpy()
    else:
        frame["action_family"] = [ACTION_FAMILY_BY_SOURCE.get(str(s), "global_exploration") for s in frame["proposal_source"]]
    return validate_stagev4_proposal_sources(frame, strict=True)


def select_acquisition(args: argparse.Namespace, cfg: StageV4Config, out_run_dir: Path) -> None:
    iteration = int(args.iteration)
    dataset_path = args.dataset or (out_run_dir / f"dataset_iter{iteration:03d}.npz")
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing Stage V-v4 cumulative dataset: {dataset_path}")
    dataset = load_flat_dataset(dataset_path)
    if dataset.x.shape[1] != 3:
        raise ValueError(f"Stage V-v4 requires 3D dataset coordinates, got x.shape={dataset.x.shape}")
    if dataset.x.shape[0] < 8:
        raise ValueError("Stage V-v4 acquisition requires at least 8 exact points")

    candidates, candidate_meta, candidate_summary, graph = generate_stagev4_candidate_pool(dataset, cfg, iteration)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, field_summary = predict_stagev_fields(dataset, candidates, cfg, device=device)
    support_sets = build_boundary_support_sets(dataset, cfg)
    scored, a0_summary = score_stagev_a0(features, support_sets, cfg)
    scored = _assign_sources(scored, candidate_meta)
    scored, sparse_summary, sparse_surface = apply_stagev3_sparse_scores(scored, dataset, cfg)
    scored = _assign_sources(scored, candidate_meta)
    scored, graph_context_summary = attach_graph_context_scores(scored, graph, cfg)

    state = load_stagev4_state(out_run_dir, cfg)
    scored, combine_summary = combine_multihead_scores(
        scored,
        cfg,
        models=state.get("models", {}),
        lambda_state=state.get("lambda_state", {}),
        alpha_state=state.get("alpha_state", {}),
    )
    scored, graph_value_summary = apply_graph_value_model(scored, state, cfg)
    scored["A_total_v2_base"] = scored["A_total_v2"].to_numpy(dtype=np.float64)
    scored["A_total_v2"] = scored["A_total_v4"].to_numpy(dtype=np.float64)
    selected, meta, select_summary = select_micro_batch_v2(
        scored,
        cfg,
        rng=np.random.default_rng(int(cfg.random_seed) + iteration * 100003),
    )
    meta = validate_stagev4_proposal_sources(meta, strict=bool(cfg.proposal_source_validation_strict))

    iter_dir = out_run_dir / f"iter{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates, columns=["kT", "JA", "mu"]).assign(
        candidate_source=candidate_meta.get("candidate_source", "unknown_explicit"),
        action_family=candidate_meta.get("action_family", "global_exploration"),
    ).to_csv(iter_dir / "candidate_points.csv", index=False)
    top_k = int(args.log_top_k)
    scored.nlargest(min(top_k, len(scored)), "A_total_v4").to_csv(iter_dir / "candidate_score_topk.csv", index=False)
    sparse_surface.to_csv(iter_dir / "stagev4_sparse_surface_samples.csv", index=False)
    graph["descriptors"].to_csv(iter_dir / "stagev4_graph_descriptors.csv", index=False)
    graph["nodes"].to_csv(iter_dir / "stagev4_graph_nodes.csv", index=False)
    graph["edges"].to_csv(iter_dir / "stagev4_graph_edges.csv", index=False)
    write_json(iter_dir / "stagev4_graph_summary.json", graph.get("summary", {}))

    summary = {
        "mode": "acquisition",
        "run_id": cfg.run_id,
        "iteration": iteration,
        "dataset_path": str(dataset_path),
        "dataset_samples": int(dataset.x.shape[0]),
        "world_size": int(args.world_size),
        "partition_strategy": str(args.partition_strategy),
        "device": str(device),
        "graph_learning_enabled": not bool(cfg.known_boundary_baseline_only),
        "candidate_phase_shadow_mode": bool(cfg.candidate_phase_shadow_mode),
        "graph_value_model_shadow_mode": bool(cfg.graph_shadow_mode),
        "stagev4_schema_version": cfg.stagev2_schema_version,
        "lambda_state": state.get("lambda_state", {}),
        "alpha_state": state.get("alpha_state", {}),
        "graph_lambda": float(state.get("graph_lambda", 0.0)),
        **stagev4_leakage_guard_summary(),
        **candidate_summary,
        **field_summary,
        **a0_summary,
        **sparse_summary,
        **graph_context_summary,
        **combine_summary,
        **graph_value_summary,
        **select_summary,
    }
    write_stagev_selection(out_run_dir, iteration, selected, meta, cfg, int(args.world_size), str(args.partition_strategy), summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage V-v4 phase-structure-graph acquisition selector.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--mode", choices=["seed", "acquisition"], required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--output-root", default=STAGEV4_OUTPUT_ROOT)
    p.add_argument("--run-id", default=STAGEV4_RUN_ID)
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
    print(f"Wrote Stage V-v4 selection for {args.mode} iteration {int(args.iteration):03d}: {out_run_dir}")


if __name__ == "__main__":
    main()
