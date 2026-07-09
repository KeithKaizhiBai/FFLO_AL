from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import PHASE_NORMAL
from .stagev_acqv2 import (
    StageVConfig,
    build_boundary_support_sets,
    generate_stagev_candidate_pool,
    predict_linear_value_model,
    predict_stagev_fields,
    score_stagev_a0,
    select_micro_batch,
    sobol_points_3d,
    train_linear_value_model,
    write_empty_stagev_dataset,
    write_stagev_selection,
)


STAGEV2_RUN_ID = "stagev_v2_multihead_boundary_learning_3d_v1"
STAGEV2_OUTPUT_ROOT = "ML_Phase_StageV_V2_Multihead"

BOUNDARY_NAMES = ("ns", "uf", "p0", "ppi", "gap")

BOUNDARY_SCORE_SUFFIX = {
    "ns": "normal_sc",
    "uf": "uniform_fflo",
    "p0": "p0_topology",
    "ppi": "ppi_topology",
    "gap": "gap_nodal",
}

BOUNDARY_MARGIN_COLUMNS = {
    "ns": "m_NS",
    "uf": "m_UF",
    "p0": "m_P0",
    "ppi": "m_Ppi",
    "gap": "m_gap",
}


@dataclass(frozen=True)
class StageV2Config(StageVConfig):
    run_id: str = STAGEV2_RUN_ID
    output_root: str = STAGEV2_OUTPUT_ROOT
    candidate_pool_size: int = 65536
    max_micro_batches: int = 96
    learned_min_reward_samples: int = 128
    learned_initial_lambda: float = 0.1
    learned_lambda_max: float = 0.7
    learned_validation_margin: float = 0.02
    alpha_learning_rate: float = 0.25
    alpha_min: float = -3.0
    alpha_max: float = 3.0
    source_density_correction_strength: float = 0.5
    source_quantile_bins: int = 64
    topology_channel_guard_alpha: float = 0.8
    topology_channel_guard_contribution: float = 0.08
    stagev2_schema_version: str = "stagev_v2_multihead_acq_1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "StageV2Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})


def initial_lambda_state(cfg: StageV2Config) -> dict[str, float]:
    return {name: 0.0 for name in BOUNDARY_NAMES}


def initial_alpha_state(cfg: StageV2Config) -> dict[str, float]:
    return {name: 0.0 for name in BOUNDARY_NAMES}


def stagev2_feature_columns(boundary: str) -> list[str]:
    suffix = BOUNDARY_SCORE_SUFFIX[boundary]
    common = [
        "nearest_exact_distance",
        "exact_repulsion",
        "selection_probability",
        "p_normal",
        "p_SC",
        "p_uniform_SC",
        "p_FFLO",
    ]
    per_boundary = [
        f"A_{suffix}",
        f"B_{suffix}",
        f"U_{suffix}",
        f"H_{suffix}",
        f"support_distance_{suffix}",
        BOUNDARY_MARGIN_COLUMNS[boundary],
    ]
    if boundary in {"p0", "ppi", "gap"}:
        per_boundary.extend(["pf_product_pred", "pfaffian_margin_pred"])
    return per_boundary + common


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out:
            out[col] = 0.0
    return out


def rank_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    out = np.zeros(arr.shape[0], dtype=np.float64)
    valid = np.isfinite(arr)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return out
    if n_valid == 1:
        out[valid] = 1.0
        return out
    ranks = pd.Series(arr[valid]).rank(method="average").to_numpy(dtype=np.float64)
    out[valid] = (ranks - 1.0) / max(float(n_valid - 1), 1.0)
    return np.clip(out, 0.0, 1.0)


def source_density_correction(source: pd.Series | np.ndarray, strength: float = 0.5) -> np.ndarray:
    s = pd.Series(np.asarray(source, dtype=object)).fillna("unknown").astype(str)
    counts = s.value_counts()
    if counts.empty:
        return np.zeros(len(s), dtype=np.float64)
    median_count = float(np.median(counts.to_numpy(dtype=np.float64)))
    offsets = {
        key: -float(strength) * math.log(max(float(count), 1.0) / max(median_count, 1.0))
        for key, count in counts.items()
    }
    arr = s.map(offsets).to_numpy(dtype=np.float64).copy()
    arr -= float(np.mean(arr)) if arr.size else 0.0
    return arr


def combine_multihead_scores(
    scored: pd.DataFrame,
    cfg: StageV2Config,
    models: dict[str, dict[str, Any]] | None = None,
    lambda_state: dict[str, float] | None = None,
    alpha_state: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = scored.copy().reset_index(drop=True)
    models = models or {}
    lambda_state = {**initial_lambda_state(cfg), **(lambda_state or {})}
    alpha_state = {**initial_alpha_state(cfg), **(alpha_state or {})}
    contributions: list[np.ndarray] = []
    boundary_summary: dict[str, Any] = {}
    for boundary in BOUNDARY_NAMES:
        suffix = BOUNDARY_SCORE_SUFFIX[boundary]
        a_col = f"A_{suffix}"
        if a_col not in frame:
            frame[a_col] = 0.0
        model = models.get(boundary, {})
        g = predict_linear_value_model(model, frame)
        lam = float(lambda_state.get(boundary, 0.0))
        alpha = float(alpha_state.get(boundary, 0.0))
        log_score = np.log(np.maximum(frame[a_col].to_numpy(dtype=np.float64), 1.0e-300))
        learned_log_score = log_score + lam * np.clip(g, -5.0, 5.0)
        rank = rank_normalize(learned_log_score)
        contribution = alpha + rank
        frame[f"g_{boundary}"] = g
        frame[f"lambda_{boundary}"] = lam
        frame[f"alpha_{boundary}"] = alpha
        frame[f"ranknorm_{boundary}"] = rank
        frame[f"contribution_{boundary}"] = contribution
        contributions.append(contribution)
        boundary_summary[boundary] = {
            "lambda": lam,
            "alpha": alpha,
            "raw_score_mean": float(np.nanmean(frame[a_col].to_numpy(dtype=np.float64))) if len(frame) else 0.0,
            "ranknorm_mean": float(np.nanmean(rank)) if len(frame) else 0.0,
        }
    stack = np.vstack(contributions) if contributions else np.zeros((1, len(frame)), dtype=np.float64)
    max_term = np.max(stack, axis=0)
    logsum = max_term + np.log(np.maximum(np.sum(np.exp(stack - max_term), axis=0), 1.0e-300))
    source = frame["candidate_source"] if "candidate_source" in frame else pd.Series(["unknown"] * len(frame))
    correction = source_density_correction(source, cfg.source_density_correction_strength)
    frame["source_density_correction"] = correction
    frame["stagev2_log_score"] = logsum + correction
    frame["A_total_v2"] = np.exp(frame["stagev2_log_score"].to_numpy(dtype=np.float64) - np.nanmax(frame["stagev2_log_score"].to_numpy(dtype=np.float64)))
    frame["A_total_v2"] *= frame.get("exact_repulsion", pd.Series(np.ones(len(frame)))).to_numpy(dtype=np.float64)
    dom_idx = np.argmax(stack, axis=0) if stack.shape[1] else np.array([], dtype=int)
    frame["dominant_boundary"] = [BOUNDARY_NAMES[int(i)] for i in dom_idx]
    summary = {
        "boundary_summary": boundary_summary,
        "source_correction_by_source": pd.DataFrame({"source": source, "correction": correction})
        .groupby("source")["correction"]
        .mean()
        .to_dict(),
        "stagev2_score_max": float(np.nanmax(frame["A_total_v2"])) if len(frame) else 0.0,
    }
    return frame, summary


def select_micro_batch_v2(
    scored: pd.DataFrame,
    cfg: StageV2Config,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    if "A_total_v2" not in scored:
        raise ValueError("select_micro_batch_v2 requires A_total_v2 from combine_multihead_scores")
    frame = scored.copy()
    frame["A0"] = frame["A_total_v2"].to_numpy(dtype=np.float64)
    selected, meta, summary = select_micro_batch(frame, cfg, rng=rng, learned_values=np.zeros(len(frame)), lambda_t=0.0)
    if not meta.empty:
        meta["stagev2_schema_version"] = cfg.stagev2_schema_version
        meta["final_rank"] = meta["selection_rank"].astype(int)
        if "dominant_boundary" not in meta:
            meta["dominant_boundary"] = "unknown"
    summary = {**summary, "selection_score": "A_total_v2", "stagev2_schema_version": cfg.stagev2_schema_version}
    return selected, meta, summary


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def compute_per_boundary_rewards(selected: pd.DataFrame, exact: pd.DataFrame | None = None) -> pd.DataFrame:
    selected = selected.reset_index(drop=True).copy()
    exact = exact.reset_index(drop=True).copy() if exact is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for i, row in selected.iterrows():
        exact_row = exact.iloc[i] if i < len(exact) else pd.Series(dtype=object)
        trusted = _truthy(exact_row.get("trusted_exact", True))
        eligible = _truthy(exact_row.get("training_eligible_exact", True))
        rerun = _truthy(exact_row.get("needs_rerun_exact", exact_row.get("rerun_required", False)))
        penalty_untrusted = 1.0 if not (trusted and eligible and not rerun) else 0.0
        penalty_redundant = 1.0 if float(row.get("nearest_exact_distance", np.inf)) < 0.01 else 0.0
        values: dict[str, Any] = {
            "penalty_untrusted": penalty_untrusted,
            "penalty_redundant": penalty_redundant,
        }
        for boundary in BOUNDARY_NAMES:
            suffix = BOUNDARY_SCORE_SUFFIX[boundary]
            b = float(row.get(f"B_{suffix}", 0.0))
            u = float(row.get(f"U_{suffix}", 0.0))
            h = float(row.get(f"H_{suffix}", 0.0))
            a = float(row.get(f"A_{suffix}", 0.0))
            distance = float(row.get(f"support_distance_{suffix}", np.inf))
            support_gain = h
            margin_reward = b
            uncertainty_reward = u
            bracket_reward = float(row.get(f"created_{boundary}_bracket", 0.0))
            if boundary in {"p0", "ppi"}:
                bracket_reward = max(bracket_reward, float(row.get("created_topology_bracket", 0.0)))
            if np.isfinite(distance):
                support_gain = max(support_gain, 1.0 - math.exp(-1.0 / max(distance + 1.0e-6, 1.0e-6)))
            raw = (
                0.30 * bracket_reward
                + 0.25 * support_gain
                + 0.25 * margin_reward
                + 0.15 * uncertainty_reward
                + 0.05 * min(max(a, 0.0), 1.0)
                - 0.25 * penalty_redundant
                - 0.60 * penalty_untrusted
            )
            values[f"reward_{boundary}_raw"] = float(raw)
            values[f"reward_{boundary}_bracket"] = float(bracket_reward)
            values[f"reward_{boundary}_support"] = float(support_gain)
            values[f"reward_{boundary}_margin"] = float(margin_reward)
        rows.append(values)
    out = pd.DataFrame(rows)
    for boundary in BOUNDARY_NAMES:
        col = f"reward_{boundary}_raw"
        vals = out[col].to_numpy(dtype=np.float64)
        med = float(np.nanmedian(vals)) if vals.size else 0.0
        q75 = float(np.nanpercentile(vals, 75)) if vals.size else 1.0
        q25 = float(np.nanpercentile(vals, 25)) if vals.size else 0.0
        scale = max(q75 - q25, float(np.nanstd(vals)) if vals.size else 0.0, 1.0e-6)
        out[f"reward_{boundary}_normalized"] = (vals - med) / scale
    return out


def fit_multihead_value_models(history: pd.DataFrame, cfg: StageV2Config) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for boundary in BOUNDARY_NAMES:
        cols = stagev2_feature_columns(boundary)
        frame = _ensure_columns(history, cols)
        reward_col = f"reward_{boundary}_normalized"
        if reward_col not in frame:
            frame[reward_col] = 0.0
        model = train_linear_value_model(frame, frame[reward_col].to_numpy(dtype=np.float64), cols)
        model["boundary"] = boundary
        model["reward_column"] = reward_col
        models[boundary] = model
    return models


def predict_multihead_values(models: dict[str, dict[str, Any]], features: pd.DataFrame) -> dict[str, np.ndarray]:
    return {boundary: predict_linear_value_model(models.get(boundary, {}), features) for boundary in BOUNDARY_NAMES}


def update_multihead_lambdas(
    previous: dict[str, float] | None,
    validation: dict[str, Any],
    cfg: StageV2Config,
) -> dict[str, float]:
    prev = {**initial_lambda_state(cfg), **(previous or {})}
    out: dict[str, float] = {}
    for boundary in BOUNDARY_NAMES:
        n = int(validation.get(f"reward_sample_count_{boundary}", validation.get("reward_sample_count", 0)))
        delta = float(validation.get(f"rank_correlation_delta_vs_a0_{boundary}", 0.0))
        improved = float(validation.get(f"scientific_metric_improved_{boundary}", 1.0))
        unstable = bool(validation.get(f"bad_sampling_detected_{boundary}", False))
        value = float(prev.get(boundary, 0.0))
        if n < int(cfg.learned_min_reward_samples):
            out[boundary] = 0.0
        elif unstable or improved < 0.0 or delta < -float(cfg.learned_validation_margin):
            out[boundary] = max(0.0, 0.5 * value)
        elif delta > float(cfg.learned_validation_margin) and improved >= 0.0:
            out[boundary] = min(float(cfg.learned_lambda_max), max(value, float(cfg.learned_initial_lambda)) + 0.05)
        else:
            out[boundary] = min(float(cfg.learned_lambda_max), max(0.0, value))
    return out


def update_boundary_alphas(
    previous: dict[str, float] | None,
    metrics: dict[str, dict[str, Any]],
    cfg: StageV2Config,
) -> dict[str, float]:
    prev = {**initial_alpha_state(cfg), **(previous or {})}
    out: dict[str, float] = {}
    for boundary in BOUNDARY_NAMES:
        m = metrics.get(boundary, {})
        state = str(m.get("boundary_state", "supported"))
        if state == "physically_absent":
            delta = -0.5
        else:
            delta = (
                float(m.get("surprise_deficit", 0.0))
                + float(m.get("coverage_deficit", 0.0))
                + float(m.get("shift_deficit", 0.0))
                + float(m.get("component_instability", 0.0))
                + float(m.get("support_deficit", 0.0))
                - float(m.get("convergence_success", 0.0))
            )
            if state in {"missing_boundary", "insufficient_support", "not_yet_discovered"}:
                delta += 0.5
        value = float(prev.get(boundary, 0.0)) + float(cfg.alpha_learning_rate) * delta
        out[boundary] = float(np.clip(value, float(cfg.alpha_min), float(cfg.alpha_max)))
    return out


def validation_summary_by_boundary(history: pd.DataFrame, models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"reward_sample_count": int(history.shape[0])}
    for boundary in BOUNDARY_NAMES:
        suffix = BOUNDARY_SCORE_SUFFIX[boundary]
        reward_col = f"reward_{boundary}_normalized"
        a_col = f"A_{suffix}"
        if reward_col not in history:
            continue
        y = history[reward_col].to_numpy(dtype=np.float64)
        a0 = history[a_col].to_numpy(dtype=np.float64) if a_col in history else np.zeros(len(history))
        pred = predict_linear_value_model(models.get(boundary, {}), history)
        summary[f"reward_sample_count_{boundary}"] = int(np.sum(np.isfinite(y)))
        summary[f"rank_correlation_a0_{boundary}"] = _rank_corr(a0, y)
        summary[f"rank_correlation_model_{boundary}"] = _rank_corr(pred, y)
        summary[f"rank_correlation_delta_vs_a0_{boundary}"] = (
            summary[f"rank_correlation_model_{boundary}"] - summary[f"rank_correlation_a0_{boundary}"]
        )
        summary[f"scientific_metric_improved_{boundary}"] = 1.0
    return summary


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 4:
        return 0.0
    ra = pd.Series(a[valid]).rank(method="average").to_numpy(dtype=np.float64)
    rb = pd.Series(b[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(ra) < 1.0e-12 or np.std(rb) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def boundary_metric_proxy(selected: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for boundary in BOUNDARY_NAMES:
        suffix = BOUNDARY_SCORE_SUFFIX[boundary]
        dist_col = f"support_distance_{suffix}"
        if dist_col in selected and len(selected):
            finite = selected[dist_col].replace([np.inf, -np.inf], np.nan).dropna()
            support_deficit = float(finite.quantile(0.75)) if len(finite) else 1.0
        else:
            support_deficit = 1.0
        metrics[boundary] = {
            "support_deficit": support_deficit,
            "coverage_deficit": max(0.0, support_deficit - 0.06),
            "convergence_success": 0.25 if support_deficit < 0.06 else 0.0,
            "boundary_state": "supported",
        }
    return metrics


def topology_channel_diagnostics(selected: pd.DataFrame, alpha_state: dict[str, float], cfg: StageV2Config) -> dict[str, Any]:
    if selected.empty:
        return {"topology_channel_suppressed": False}
    rows: dict[str, Any] = {}
    for boundary in ("p0", "ppi"):
        contrib = selected.get(f"contribution_{boundary}", pd.Series(np.zeros(len(selected)))).to_numpy(dtype=np.float64)
        alpha = float(alpha_state.get(boundary, 0.0))
        high_contribution_fraction = float(np.mean(contrib >= np.nanquantile(contrib, 0.75))) if contrib.size else 0.0
        rows[f"mean_contribution_{boundary}"] = float(np.nanmean(contrib)) if contrib.size else 0.0
        rows[f"alpha_{boundary}"] = alpha
        rows[f"high_contribution_fraction_{boundary}"] = high_contribution_fraction
    suppressed = any(
        float(alpha_state.get(boundary, 0.0)) >= float(cfg.topology_channel_guard_alpha)
        and float(rows.get(f"high_contribution_fraction_{boundary}", 0.0)) < float(cfg.topology_channel_guard_contribution)
        for boundary in ("p0", "ppi")
    )
    rows["topology_channel_suppressed"] = bool(suppressed)
    return rows


def topology_status_from_labels(phase: int, spectral_status: str | int | None, topology_label: str | int | None) -> str:
    if int(phase) == PHASE_NORMAL:
        return "not_applicable_normal"
    spectral = str(spectral_status).lower()
    label = str(topology_label).lower()
    if "nodal" in spectral or "gapless" in spectral or label in {"2", "gapless", "gapless_sc"}:
        return "z2_undefined_nodal"
    if "unresolved" in spectral or "unresolved" in label or label in {"-1", "nan", "none"}:
        return "topology_unresolved"
    if label in {"0", "1", "trivial", "topological", "cfflo", "tfflo"}:
        return "z2_defined"
    return "topology_unresolved"


def load_stagev2_state(run_dir: Path, cfg: StageV2Config) -> dict[str, Any]:
    state_path = run_dir / "stagev2_acquisition_state.json"
    if not state_path.exists():
        return {
            "models": {},
            "lambda_state": initial_lambda_state(cfg),
            "alpha_state": initial_alpha_state(cfg),
            "validation": {},
        }
    return json.loads(state_path.read_text(encoding="utf-8"))


def write_stagev2_state(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stagev2_acquisition_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "BOUNDARY_NAMES",
    "BOUNDARY_SCORE_SUFFIX",
    "STAGEV2_OUTPUT_ROOT",
    "STAGEV2_RUN_ID",
    "StageV2Config",
    "boundary_metric_proxy",
    "build_boundary_support_sets",
    "combine_multihead_scores",
    "compute_per_boundary_rewards",
    "fit_multihead_value_models",
    "generate_stagev_candidate_pool",
    "initial_alpha_state",
    "initial_lambda_state",
    "load_stagev2_state",
    "predict_stagev_fields",
    "rank_normalize",
    "score_stagev_a0",
    "select_micro_batch_v2",
    "sobol_points_3d",
    "source_density_correction",
    "stagev2_feature_columns",
    "topology_channel_diagnostics",
    "topology_status_from_labels",
    "update_boundary_alphas",
    "update_multihead_lambdas",
    "validation_summary_by_boundary",
    "write_empty_stagev_dataset",
    "write_stagev2_state",
    "write_stagev_selection",
]
