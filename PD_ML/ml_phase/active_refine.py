from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .acquisition import build_candidate_grid, compute_acquisition_scores, select_top_diverse
from .config import ActiveLearningConfig, ensure_output_dirs
from .dataset_builder import FlatDataset, build_warm_start_dataset
from .evaluate import evaluate_predictions
from .exact_oracle import evaluate_points
from .hpc import write_point_shards
from .labels import PHASE_NAMES, eta_sign_label, phase_label, strong_diode_label
from .models import ModelBundle, predict_models, train_models
from .plot_active_learning import write_iteration_figures, write_learning_curve


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Active-learning phase-boundary refinement.")
    p.add_argument("--warm-start", type=Path, required=True, help="Path to eta_phase_diagram_*.npz")
    p.add_argument("--run-id", type=str, required=True, help="Run identifier")
    p.add_argument("--iterations", type=int, default=5, help="Number of active-learning iterations")
    p.add_argument("--points-per-iter", type=int, default=64, help="Selected exact points per iteration")
    p.add_argument("--mode", type=str, default="local", choices=["local", "hpc"], help="Execution mode")
    p.add_argument("--world-size", type=int, default=1, help="Number of H100 ranks/tasks for hpc mode")
    p.add_argument("--partition-strategy", type=str, default="round_robin", help="round_robin|contiguous|cost_aware")
    p.add_argument("--dry-run", action="store_true", help="Select points but skip exact oracle")
    p.add_argument("--device", type=str, default=None, help="Torch device for local exact oracle, e.g. cuda:0")
    p.add_argument("--output-root", type=Path, default=Path("ML_Phase"), help="Output root")
    p.add_argument("--n-ensemble", type=int, default=5, help="Model ensemble size")
    p.add_argument("--reg-epochs", type=int, default=240, help="Regression epochs per ensemble member")
    p.add_argument("--cls-epochs", type=int, default=240, help="Classification epochs per ensemble member")
    p.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    p.add_argument("--submit", action="store_true", help="In hpc mode, submit slurm array job")
    p.add_argument(
        "--slurm-script",
        type=Path,
        default=Path("scripts/slurm_exact_oracle_array.sh"),
        help="SLURM array script path for --mode hpc --submit",
    )
    return p.parse_args()


def _dataset_from_result(dataset: FlatDataset, result: Dict[str, np.ndarray], cfg: ActiveLearningConfig) -> FlatDataset:
    new_x = np.stack([result["kT"], result["JA"]], axis=1).astype(np.float64)
    new_y_reg = np.stack(
        [result["delta_opt"], result["q_opt"], result["eta"], result["ic_plus"], result["ic_minus"]],
        axis=1,
    ).astype(np.float64)

    x_all = np.vstack([dataset.x, new_x])
    y_reg_all = np.vstack([dataset.y_reg, new_y_reg])

    # Deduplicate by (kT, JA), keep latest entry
    keys = np.round(x_all, decimals=12)
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    x_all = x_all[uniq_idx]
    y_reg_all = y_reg_all[uniq_idx]

    y_phase = phase_label(y_reg_all[:, 0], y_reg_all[:, 1], cfg.delta_eps, cfg.q_eps)
    y_eta_sign = eta_sign_label(y_reg_all[:, 2])
    y_strong = strong_diode_label(y_reg_all[:, 2], cfg.eta_strong)

    df = pd.DataFrame(
        {
            "kT": x_all[:, 0],
            "JA": x_all[:, 1],
            "delta_opt": y_reg_all[:, 0],
            "q_opt": y_reg_all[:, 1],
            "eta": y_reg_all[:, 2],
            "ic_plus": y_reg_all[:, 3],
            "ic_minus": y_reg_all[:, 4],
            "phase_label": y_phase,
            "phase_name": np.vectorize(PHASE_NAMES.get)(y_phase),
            "eta_sign_label": y_eta_sign,
            "strong_diode_label": y_strong,
        }
    )

    return FlatDataset(
        x=x_all,
        y_reg=y_reg_all,
        y_phase=y_phase.astype(np.int64),
        y_eta_sign=y_eta_sign.astype(np.int64),
        y_strong_diode=y_strong.astype(np.int64),
        df=df,
    )


def _save_dataset(iter_dir: Path, iteration: int, dataset: FlatDataset) -> Tuple[Path, Path]:
    npz_path = iter_dir.parent / f"dataset_iter{iteration:03d}.npz"
    csv_path = iter_dir.parent / f"dataset_iter{iteration:03d}.csv"
    np.savez(
        npz_path,
        x=dataset.x,
        y_reg=dataset.y_reg,
        y_phase=dataset.y_phase,
        y_eta_sign=dataset.y_eta_sign,
        y_strong_diode=dataset.y_strong_diode,
    )
    dataset.df.to_csv(csv_path, index=False)
    return npz_path, csv_path


def _save_candidate_csv(iter_dir: Path, grid_points: np.ndarray, scores: Dict[str, np.ndarray]) -> Path:
    df = pd.DataFrame(
        {
            "kT": grid_points[:, 0],
            "JA": grid_points[:, 1],
            "score": scores["score"],
            "cls_uncertainty": scores["cls_uncertainty"],
            "reg_uncertainty": scores["reg_uncertainty"],
            "delta_boundary_score": scores["delta_boundary_score"],
            "q_boundary_score": scores["q_boundary_score"],
            "eta_zero_score": scores["eta_zero_score"],
            "gradient_score": scores["gradient_score"],
            "diversity_score": scores["diversity_score"],
        }
    )
    out = iter_dir / "candidate_scores.csv"
    df.to_csv(out, index=False)
    return out


def _evaluate_on_validation(bundle: ModelBundle, dataset: FlatDataset, n_exact_calls: int, cfg: ActiveLearningConfig) -> dict:
    preds = predict_models(bundle, dataset.x)
    metrics = evaluate_predictions(
        x=dataset.x,
        y_reg_true=dataset.y_reg,
        y_phase_true=dataset.y_phase,
        y_reg_pred=preds["reg_mean"],
        y_phase_pred=preds["phase_pred"],
        n_exact_calls=n_exact_calls,
        dense_grid_points=cfg.n_kt_candidates * cfg.n_ja_candidates,
    )
    return metrics.to_dict()


def _maybe_submit_slurm(script: Path, run_id: str, iteration: int) -> None:
    cmd = ["sbatch", str(script)]
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["ITER"] = str(iteration)
    print("Submitting SLURM array job:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _jsonable_args(ns: argparse.Namespace) -> dict:
    out: dict = {}
    for k, v in vars(ns).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def run_active_refinement(args: argparse.Namespace) -> None:
    cfg = ActiveLearningConfig(
        iterations=args.iterations,
        points_per_iter=args.points_per_iter,
        dry_run=bool(args.dry_run),
        mode=args.mode,
        world_size=args.world_size,
        partition_strategy=args.partition_strategy,
        output_root=str(args.output_root),
        n_ensemble=args.n_ensemble,
        reg_epochs=args.reg_epochs,
        cls_epochs=args.cls_epochs,
        batch_size=args.batch_size,
    )
    ensure_output_dirs(cfg)

    # Build warm-start flat dataset under ML_Phase/datasets.
    flat, warm_npz, warm_csv, _ = build_warm_start_dataset(args.warm_start, cfg, output_root=args.output_root)
    print(f"Warm-start dataset: {warm_npz}")
    print(f"Warm-start csv: {warm_csv}")

    run_dir = cfg.active_runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": _jsonable_args(args),
                "active_learning_config": cfg.to_dict(),
                "python": sys.version,
                "torch": torch.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = flat
    metrics_history: list[dict[str, float]] = []
    n_exact_calls = int(dataset.x.shape[0])

    for iteration in range(cfg.iterations):
        iter_dir = run_dir / f"iter{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        bundle = train_models(dataset.x, dataset.y_reg, dataset.y_phase, cfg)
        grid = build_candidate_grid(cfg)
        pred_grid = predict_models(bundle, grid.points)
        score_pack = compute_acquisition_scores(cfg, grid, pred_grid, existing_points=dataset.x)
        selected_idx = select_top_diverse(
            points=grid.points,
            scores=score_pack["score"],
            k=cfg.points_per_iter,
            min_dist=cfg.diversity_min_dist,
            kt_range=(cfg.kt_min, cfg.kt_max),
            ja_range=(cfg.ja_min, cfg.ja_max),
        )
        selected_points = grid.points[selected_idx]

        _save_candidate_csv(iter_dir, grid.points, score_pack)
        pd.DataFrame(selected_points, columns=["kT", "JA"]).to_csv(iter_dir / "selected_points.csv", index=False)

        # diagnostics and learning-curve metrics against current exact data
        current_metrics = _evaluate_on_validation(bundle, dataset, n_exact_calls=n_exact_calls, cfg=cfg)
        metrics_history.append(current_metrics)
        fig_paths = write_iteration_figures(
            figures_dir=cfg.figures_dir,
            run_id=args.run_id,
            iteration=iteration,
            grid=grid,
            predictions=pred_grid,
            scores=score_pack,
            selected_points=selected_points,
            existing_points=dataset.x,
        )
        (iter_dir / "metrics.json").write_text(json.dumps(current_metrics, indent=2), encoding="utf-8")
        (iter_dir / "figures.json").write_text(
            json.dumps({k: str(v) for k, v in fig_paths.items()}, indent=2),
            encoding="utf-8",
        )

        if args.mode == "hpc":
            shard_paths = write_point_shards(
                run_dir=run_dir,
                iteration=iteration,
                points=selected_points,
                world_size=max(1, args.world_size),
                strategy=args.partition_strategy,
            )
            (iter_dir / "hpc_instructions.txt").write_text(
                "\n".join(
                    [
                        f"world_size={args.world_size}",
                        f"partition_strategy={args.partition_strategy}",
                        f"slurm_script={args.slurm_script}",
                        f"selected_points={len(selected_points)}",
                        "Run SLURM array for exact oracle shards, then merge with ml_phase.hpc --merge.",
                    ]
                ),
                encoding="utf-8",
            )
            print(f"HPC mode: wrote {len(shard_paths)} point-shard files.")
            print(f"Iteration artifacts: {iter_dir}")
            if args.submit:
                _maybe_submit_slurm(args.slurm_script, args.run_id, iteration)
            _save_dataset(iter_dir, iteration, dataset)
            # HPC mode exits after shard generation for this iteration.
            break

        if args.dry_run:
            print(f"Dry-run iteration {iteration}: selected {len(selected_points)} points; exact oracle skipped.")
            _save_dataset(iter_dir, iteration, dataset)
            continue

        # local exact evaluation
        oracle = evaluate_points(
            points=selected_points,
            device=args.device,
            output_file=iter_dir / "exact_local_partial.npz",
            save_every=1,
        )
        result = oracle.to_dict()
        np.savez(iter_dir / "exact_local_iter.npz", **result)
        dataset = _dataset_from_result(dataset, result, cfg)
        n_exact_calls += selected_points.shape[0]
        _save_dataset(iter_dir, iteration, dataset)

    lc = write_learning_curve(cfg.figures_dir, args.run_id, metrics_history)
    (run_dir / "metrics_history.json").write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")
    print(f"Learning curve saved: {lc}")
    print(f"Run complete: {run_dir}")


def main() -> None:
    args = _parse_args()
    run_active_refinement(args)


if __name__ == "__main__":
    main()
