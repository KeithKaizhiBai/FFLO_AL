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

from ml_phase.stagev_acqv2 import (
    STAGEV_OUTPUT_ROOT,
    STAGEV_RUN_ID,
    StageVConfig,
    compute_point_rewards,
    train_linear_value_model,
    update_lambda_t,
)


FEATURE_COLUMNS = [
    "A0",
    "B_normal_sc",
    "B_uniform_fflo",
    "B_p0_topology",
    "B_ppi_topology",
    "B_gap_nodal",
    "H_normal_sc",
    "H_uniform_fflo",
    "H_p0_topology",
    "H_ppi_topology",
    "H_gap_nodal",
    "nearest_exact_distance",
    "exact_repulsion",
    "selection_probability",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["kT", "JA", "mu"]:
        if col not in out:
            out[col] = 0.55 if col == "mu" else np.nan
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
    else:
        exact_keyed = _key_frame(exact)
        keep_cols = [c for c in exact_keyed.columns if c.startswith("key_") or c in {
            "trusted_exact",
            "training_eligible_exact",
            "needs_rerun_exact",
            "q_unresolved",
            "delta_unresolved",
            "exact_status_code",
            "phase_label",
            "topology_trusted",
            "topology_label_code",
            "topology_spectral_status_code",
        }]
        matched = selected_keyed.merge(
            exact_keyed[keep_cols],
            on=["key_kT", "key_JA", "key_mu"],
            how="left",
            suffixes=("", "_exact"),
        )
    exact_cols = [c for c in matched.columns if c.endswith("_exact") or c in {
        "trusted_exact",
        "training_eligible_exact",
        "needs_rerun_exact",
        "q_unresolved",
        "delta_unresolved",
        "exact_status_code",
    }]
    exact_frame = matched[exact_cols].copy()
    rewards = compute_point_rewards(matched, exact_frame)
    for col in FEATURE_COLUMNS:
        if col not in matched:
            matched[col] = 0.0
    return pd.concat([matched.reset_index(drop=True), rewards.reset_index(drop=True)], axis=1)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 4:
        return 0.0
    ra = pd.Series(a[valid]).rank(method="average").to_numpy(float)
    rb = pd.Series(b[valid]).rank(method="average").to_numpy(float)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser(description="Update Stage V learned residual acquisition reward model.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output-root", default=STAGEV_OUTPUT_ROOT)
    p.add_argument("--run-id", default=STAGEV_RUN_ID)
    p.add_argument("--iteration", type=int, required=True)
    args = p.parse_args()
    cfg = StageVConfig() if args.config is None else StageVConfig.from_json(args.config)
    run_dir = Path(args.output_root) / "active_runs" / args.run_id
    iter_dir = run_dir / f"iter{int(args.iteration):03d}"
    selected_path = iter_dir / "selected_points_metadata.csv"
    exact_path = iter_dir / f"exact_merged_iter{int(args.iteration):03d}.npz"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    selected = pd.read_csv(selected_path)
    exact = load_exact_frame(exact_path)
    new_rows = build_reward_rows(selected, exact)
    history_path = run_dir / "stagev_reward_history.csv"
    if history_path.exists():
        history = pd.concat([pd.read_csv(history_path), new_rows], ignore_index=True)
    else:
        history = new_rows
    history.to_csv(history_path, index=False)
    feature_cols = [c for c in FEATURE_COLUMNS if c in history.columns]
    model = train_linear_value_model(history, history["reward_scalar"].to_numpy(float), feature_cols)
    write_json(run_dir / "stagev_reward_model.json", model)
    corr_a0 = rank_corr(history["A0"].to_numpy(float), history["reward_scalar"].to_numpy(float))
    if model.get("status") == "trained":
        pred = float(model["intercept"]) + (
            (history[model["feature_columns"]].to_numpy(float) - np.asarray(model["mean"], dtype=float))
            / np.maximum(np.asarray(model["std"], dtype=float), 1e-12)
        ) @ np.asarray(model["coef"], dtype=float)
        corr_model = rank_corr(pred, history["reward_scalar"].to_numpy(float))
    else:
        corr_model = 0.0
    prev_lambda = float(read_json(run_dir / "stagev_lambda_state.json").get("lambda_t", 0.0))
    validation = {
        "reward_sample_count": int(history.shape[0]),
        "rank_correlation_a0": float(corr_a0),
        "rank_correlation_model": float(corr_model),
        "rank_correlation_delta_vs_a0": float(corr_model - corr_a0),
        "bad_sampling_detected": 0.0,
    }
    lambda_t = update_lambda_t(prev_lambda, validation, cfg)
    write_json(run_dir / "stagev_lambda_state.json", {"lambda_t": float(lambda_t), **validation})
    summary = {
        "iteration": int(args.iteration),
        "new_reward_rows": int(new_rows.shape[0]),
        "reward_history_rows": int(history.shape[0]),
        "reward_model_status": model.get("status", "unknown"),
        "lambda_t": float(lambda_t),
        **validation,
    }
    write_json(iter_dir / "stagev_reward_update_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
