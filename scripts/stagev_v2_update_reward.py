from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.stagev_v2 import (
    BOUNDARY_NAMES,
    STAGEV2_OUTPUT_ROOT,
    STAGEV2_RUN_ID,
    StageV2Config,
    boundary_metric_proxy,
    compute_per_boundary_rewards,
    fit_multihead_value_models,
    initial_alpha_state,
    initial_lambda_state,
    load_stagev2_state,
    update_boundary_alphas,
    update_multihead_lambdas,
    validation_summary_by_boundary,
    write_stagev2_state,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["kT", "JA", "mu"]:
        if col not in out:
            out[col] = 0.0
        out[f"key_{col}"] = np.round(out[col].astype(float), 10)
    return out


def load_exact_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with np.load(path, allow_pickle=False) as z:
        payload = {k: z[k].copy() for k in z.files}
    n = int(np.asarray(payload.get("kT", [])).shape[0])
    rows: dict[str, Any] = {}
    for key, val in payload.items():
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == n:
            rows[key] = arr
    return pd.DataFrame(rows)


def build_reward_rows(selected: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    selected_keyed = _key_frame(selected)
    if exact.empty:
        matched = selected_keyed.copy()
        exact_frame = pd.DataFrame(index=matched.index)
    else:
        exact_keyed = _key_frame(exact)
        keep_cols = [c for c in exact_keyed.columns if c.startswith("key_") or c in {
            "trusted_exact",
            "training_eligible_exact",
            "needs_rerun_exact",
            "rerun_required",
            "q_unresolved",
            "delta_unresolved",
            "exact_status_code",
            "phase_label",
            "topology_trusted",
            "topology_label_code",
            "topology_spectral_status_code",
            "topology_p0",
            "topology_ppi",
            "topology_bulk_gap",
        }]
        matched = selected_keyed.merge(
            exact_keyed[keep_cols],
            on=["key_kT", "key_JA", "key_mu"],
            how="left",
            suffixes=("", "_exact"),
        )
        exact_cols = [c for c in matched.columns if c.endswith("_exact") or c in keep_cols]
        exact_frame = matched[exact_cols].copy()
    rewards = compute_per_boundary_rewards(matched, exact_frame)
    return pd.concat([matched.reset_index(drop=True), rewards.reset_index(drop=True)], axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update Stage V-v2 multi-head reward models.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output-root", default=STAGEV2_OUTPUT_ROOT)
    p.add_argument("--run-id", default=STAGEV2_RUN_ID)
    p.add_argument("--iteration", type=int, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StageV2Config() if args.config is None else StageV2Config.from_json(args.config)
    run_dir = Path(args.output_root) / "active_runs" / args.run_id
    iter_dir = run_dir / f"iter{int(args.iteration):03d}"
    selected_path = iter_dir / "selected_points_metadata.csv"
    exact_path = iter_dir / f"exact_merged_iter{int(args.iteration):03d}.npz"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    selected = pd.read_csv(selected_path)
    exact = load_exact_frame(exact_path)
    new_rows = build_reward_rows(selected, exact)
    history_path = run_dir / "stagev2_reward_history.csv"
    if history_path.exists():
        history = pd.concat([pd.read_csv(history_path), new_rows], ignore_index=True)
    else:
        history = new_rows
    history.to_csv(history_path, index=False)
    models = fit_multihead_value_models(history, cfg)
    validation = validation_summary_by_boundary(history, models)
    previous = load_stagev2_state(run_dir, cfg)
    lambda_state = update_multihead_lambdas(previous.get("lambda_state", initial_lambda_state(cfg)), validation, cfg)
    alpha_metrics = boundary_metric_proxy(selected)
    alpha_state = update_boundary_alphas(previous.get("alpha_state", initial_alpha_state(cfg)), alpha_metrics, cfg)
    state = {
        "models": models,
        "lambda_state": lambda_state,
        "alpha_state": alpha_state,
        "validation": validation,
        "alpha_metrics": alpha_metrics,
        "stagev2_schema_version": cfg.stagev2_schema_version,
        "stageiv_data_used_for_training": False,
        "stagev1_data_used_for_training": False,
    }
    write_stagev2_state(run_dir, state)
    model_path = run_dir / "stagev2_multihead_models.json"
    write_json(model_path, models)
    summary = {
        "iteration": int(args.iteration),
        "new_reward_rows": int(new_rows.shape[0]),
        "reward_history_rows": int(history.shape[0]),
        "model_status_by_boundary": {name: models.get(name, {}).get("status", "missing") for name in BOUNDARY_NAMES},
        "lambda_state": lambda_state,
        "alpha_state": alpha_state,
        "validation": validation,
        "stageiv_data_used_for_training": False,
        "stagev1_data_used_for_training": False,
    }
    write_json(iter_dir / "stagev2_reward_update_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
