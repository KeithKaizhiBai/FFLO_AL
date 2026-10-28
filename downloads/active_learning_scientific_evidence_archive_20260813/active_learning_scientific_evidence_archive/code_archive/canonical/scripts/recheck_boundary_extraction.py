from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_phase.config import ActiveLearningConfig
from ml_phase.extract_phase_boundaries import extract_phase_boundaries
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC


RUN_ROOT = Path("hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42")
ACTIVE_RUN = RUN_ROOT / "active_runs/active_boundary_loop_v1"
BOUNDARY_ROOT = RUN_ROOT / "boundaries"
DEFAULT_RECHECK = BOUNDARY_ROOT / "recheck_iter042_default"
SENSITIVITY_ROOT = BOUNDARY_ROOT / "recheck_iter042_sensitivity"
AUDIT_ROOT = BOUNDARY_ROOT / "recheck_iter042_audit"

DATASET_FOR_BOUNDARIES = ACTIVE_RUN / "dataset_iter042.npz"
DATASET_PROVENANCE = [
    ACTIVE_RUN / "dataset_iter039.npz",
    ACTIVE_RUN / "dataset_iter040.npz",
    ACTIVE_RUN / "dataset_iter041.npz",
    ACTIVE_RUN / "dataset_iter042.npz",
]

DELTA_EPS = 1.0e-3
Q_EPS = 1.0e-2
ETA_STRONG = 0.5
EXISTING_MIN_DIST = 0.015


def array_content_hash(path: Path) -> tuple[str, int, list[str]]:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=True) as data:
        keys = sorted(data.files)
        n_samples = int(data["x"].shape[0])
        for key in keys:
            arr = np.asarray(data[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(arr.shape).encode("utf-8"))
            digest.update(str(arr.dtype).encode("utf-8"))
            digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest(), n_samples, keys


def load_dataset_frame(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as data:
        x = np.asarray(data["x"])
        y_reg = np.asarray(data["y_reg"])
        df = pd.DataFrame(
            {
                "kT": x[:, 0],
                "JA": x[:, 1],
                "delta_opt": y_reg[:, 0],
                "q_opt": y_reg[:, 1],
                "eta": y_reg[:, 2],
                "ic_plus": y_reg[:, 3],
                "ic_minus": y_reg[:, 4],
                "phase_label": np.asarray(data["y_phase"], dtype=int),
                "trusted_exact": np.asarray(data.get("trusted_exact", np.ones(x.shape[0])), dtype=bool),
                "training_eligible_exact": np.asarray(
                    data.get("training_eligible_exact", np.ones(x.shape[0])), dtype=bool
                ),
                "needs_rerun_exact": np.asarray(data.get("needs_rerun_exact", np.zeros(x.shape[0])), dtype=bool),
                "q_unresolved": np.asarray(data.get("q_unresolved", np.zeros(x.shape[0])), dtype=bool),
                "q_expanded": np.asarray(data.get("q_expanded", np.zeros(x.shape[0])), dtype=bool),
                "delta_unresolved": np.asarray(data.get("delta_unresolved", np.zeros(x.shape[0])), dtype=bool),
                "delta_refined": np.asarray(data.get("delta_refined", np.zeros(x.shape[0])), dtype=bool),
                "delta_boundary_band_normal": np.asarray(
                    data.get("delta_boundary_band_normal", np.zeros(x.shape[0])), dtype=bool
                ),
                "positive_delta_gap": np.asarray(data.get("positive_delta_gap", np.full(x.shape[0], np.nan))),
            }
        )
    return df


def phase_from_thresholds(delta: np.ndarray, q: np.ndarray) -> np.ndarray:
    phase = np.full(delta.shape[0], PHASE_NORMAL, dtype=int)
    sc = delta >= DELTA_EPS
    phase[sc & (np.abs(q) < Q_EPS)] = PHASE_UNIFORM_SC
    phase[sc & (np.abs(q) >= Q_EPS)] = PHASE_FFLO
    return phase


def normalized_min_distance(points: pd.DataFrame, exact_df: pd.DataFrame, cfg: ActiveLearningConfig) -> np.ndarray:
    if points.empty:
        return np.array([], dtype=float)
    exact = exact_df[["kT", "JA"]].to_numpy(dtype=float)
    pts = points[["kT", "JA"]].to_numpy(dtype=float)
    scale = np.array([cfg.kt_max - cfg.kt_min, cfg.ja_max - cfg.ja_min], dtype=float)
    scale = np.maximum(scale, 1.0e-12)
    out = np.empty(pts.shape[0], dtype=float)
    chunk = 512
    for start in range(0, pts.shape[0], chunk):
        p = pts[start : start + chunk]
        d = (p[:, None, :] - exact[None, :, :]) / scale[None, None, :]
        out[start : start + chunk] = np.sqrt(np.sum(d * d, axis=2)).min(axis=1)
    return out


def provenance_report() -> dict:
    rows = []
    for path in DATASET_PROVENANCE:
        content_hash, n_samples, keys = array_content_hash(path)
        rows.append(
            {
                "dataset": str(path),
                "file_size_bytes": int(path.stat().st_size),
                "n_samples": n_samples,
                "content_sha256": content_hash,
                "n_arrays": len(keys),
            }
        )
    reference = rows[-1]["content_sha256"]
    for row in rows:
        row["same_content_as_dataset_iter042"] = row["content_sha256"] == reference
    return {"datasets": rows}


def dataset_consistency(df: pd.DataFrame) -> dict:
    phase_expected = phase_from_thresholds(df["delta_opt"].to_numpy(), df["q_opt"].to_numpy())
    phase_mismatch = df["phase_label"].to_numpy(dtype=int) != phase_expected
    coord4 = df[["kT", "JA"]].round(4)
    coord8 = df[["kT", "JA"]].round(8)
    finite_cols = ["kT", "JA", "delta_opt", "q_opt", "eta", "ic_plus", "ic_minus"]
    nonfinite_by_col = {col: int((~np.isfinite(df[col].to_numpy(dtype=float))).sum()) for col in finite_cols}
    return {
        "n_points": int(df.shape[0]),
        "kT_negative_points": int((df["kT"] < 0.0).sum()),
        "nonfinite_by_column": nonfinite_by_col,
        "phase_counts": {str(int(k)): int(v) for k, v in df["phase_label"].value_counts().sort_index().items()},
        "phase_threshold_mismatches": int(phase_mismatch.sum()),
        "trusted_exact_false": int((~df["trusted_exact"]).sum()),
        "training_eligible_false": int((~df["training_eligible_exact"]).sum()),
        "needs_rerun_exact_true": int(df["needs_rerun_exact"].sum()),
        "q_unresolved_true": int(df["q_unresolved"].sum()),
        "q_expanded_true": int(df["q_expanded"].sum()),
        "delta_unresolved_true": int(df["delta_unresolved"].sum()),
        "delta_refined_true": int(df["delta_refined"].sum()),
        "delta_boundary_band_normal_true": int(df["delta_boundary_band_normal"].sum()),
        "duplicate_coordinates_round4": int(df.shape[0] - coord4.drop_duplicates().shape[0]),
        "duplicate_coordinates_round8": int(df.shape[0] - coord8.drop_duplicates().shape[0]),
    }


def audit_boundaries(boundaries: pd.DataFrame) -> dict:
    failures = {
        "normal_sc_predicate_failures": 0,
        "uniform_fflo_predicate_failures": 0,
        "eta_zero_predicate_failures": 0,
        "strong_diode_predicate_failures": 0,
        "interpolation_outside_segment": 0,
        "low_confidence_without_severe_reason": 0,
        "high_confidence_with_risk_reason": 0,
    }
    severe_reasons = ("needs_rerun_exact", "q_unresolved", "not_training_eligible", "large_local_spacing")

    for row in boundaries.itertuples(index=False):
        boundary_type = str(row.boundary_type)
        phase_left = int(row.phase_left)
        phase_right = int(row.phase_right)
        v0 = float(row.value_left)
        v1 = float(row.value_right)
        if boundary_type == "normal_sc":
            if (phase_left == PHASE_NORMAL) == (phase_right == PHASE_NORMAL):
                failures["normal_sc_predicate_failures"] += 1
        elif boundary_type == "uniform_fflo":
            left_sc = phase_left != PHASE_NORMAL
            right_sc = phase_right != PHASE_NORMAL
            if not (left_sc and right_sc and ((phase_left == PHASE_FFLO) != (phase_right == PHASE_FFLO))):
                failures["uniform_fflo_predicate_failures"] += 1
        elif boundary_type == "eta_zero":
            if not ((v0 <= 0.0 <= v1) or (v1 <= 0.0 <= v0)):
                failures["eta_zero_predicate_failures"] += 1
        elif boundary_type == "strong_diode":
            if not ((v0 <= 0.0 <= v1) or (v1 <= 0.0 <= v0)):
                failures["strong_diode_predicate_failures"] += 1

        kt_min = min(float(row.kT_left), float(row.kT_right)) - 1.0e-12
        kt_max = max(float(row.kT_left), float(row.kT_right)) + 1.0e-12
        ja_min = min(float(row.JA_left), float(row.JA_right)) - 1.0e-12
        ja_max = max(float(row.JA_left), float(row.JA_right)) + 1.0e-12
        if not (kt_min <= float(row.kT_boundary) <= kt_max and ja_min <= float(row.JA_boundary) <= ja_max):
            failures["interpolation_outside_segment"] += 1

        reason = str(row.risk_reason)
        if row.confidence == "low" and not any(r in reason for r in severe_reasons):
            failures["low_confidence_without_severe_reason"] += 1
        if row.confidence == "high" and reason != "clean_bracket":
            failures["high_confidence_with_risk_reason"] += 1

    by_type = {str(k): int(v) for k, v in boundaries["boundary_type"].value_counts().sort_index().items()}
    by_conf = {str(k): int(v) for k, v in boundaries["confidence"].value_counts().sort_index().items()}
    return {"boundary_counts_by_type": by_type, "boundary_counts_by_confidence": by_conf, "failures": failures}


def run_sensitivity() -> pd.DataFrame:
    configs = [
        ("a_kt00025_spacing0035", 0.0025, 0.035),
        ("b_kt00050_spacing0035", 0.0050, 0.035),
        ("c_kt00100_spacing0035", 0.0100, 0.035),
        ("d_kt00050_spacing0020", 0.0050, 0.020),
        ("e_kt00050_spacing0050", 0.0050, 0.050),
    ]
    rows = []
    for name, kt_bin_width, max_spacing in configs:
        out_dir = SENSITIVITY_ROOT / name
        args = Namespace(
            dataset=DATASET_FOR_BOUNDARIES,
            output_dir=out_dir,
            kt_bin_width=kt_bin_width,
            max_local_spacing=max_spacing,
            max_refinement_points=512,
            output_root=Path("ML_Phase"),
        )
        summary = extract_phase_boundaries(args)
        row = {
            "case": name,
            "kt_bin_width": kt_bin_width,
            "max_local_spacing": max_spacing,
            "n_boundary_segments": int(summary["n_boundary_segments"]),
            "n_targeted_refinement_points": int(summary["n_targeted_refinement_points"]),
        }
        for boundary_type, count in summary["boundary_segments_by_type"].items():
            row[f"{boundary_type}_segments"] = int(count)
        for confidence, count in summary["boundary_segments_by_confidence"].items():
            row[f"{confidence}_confidence_segments"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def make_prioritized_targets(boundaries: pd.DataFrame, exact_df: pd.DataFrame) -> dict:
    cfg = ActiveLearningConfig()
    quotas = {
        "normal_sc": 192,
        "uniform_fflo": 128,
        "strong_diode": 64,
        "eta_zero": 128,
    }
    confidence_priority = {"low": 0, "medium": 1, "high": 2}
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    combined = []
    combined_basic = []
    counts = {}
    for boundary_type, quota in quotas.items():
        group = boundaries[boundaries["boundary_type"] == boundary_type].copy()
        if group.empty:
            counts[boundary_type] = {"available_boundaries": 0, "recommended_points": 0}
            continue
        group["priority"] = group["confidence"].map(confidence_priority).fillna(3)
        group = group.sort_values(["priority", "local_spacing_normalized"], ascending=[True, False])
        rows = []
        seen = set()
        for row in group.itertuples(index=False):
            if row.confidence == "high" and float(row.local_spacing_normalized) <= 0.02:
                continue
            kt = 0.5 * (float(row.kT_left) + float(row.kT_right))
            ja = 0.5 * (float(row.JA_left) + float(row.JA_right))
            key = (round(kt, 6), round(ja, 6), boundary_type)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "kT": kt,
                    "JA": ja,
                    "boundary_type": boundary_type,
                    "source_confidence": row.confidence,
                    "source_reason": row.risk_reason,
                    "source_spacing_normalized": float(row.local_spacing_normalized),
                    "kT_boundary_estimate": float(row.kT_boundary),
                    "JA_boundary_estimate": float(row.JA_boundary),
                }
            )
        targets = pd.DataFrame(rows)
        if not targets.empty:
            targets["min_distance_to_existing_exact"] = normalized_min_distance(targets, exact_df, cfg)
            targets["within_existing_min_dist"] = targets["min_distance_to_existing_exact"] < EXISTING_MIN_DIST
            radius_checked = targets[~targets["within_existing_min_dist"]].head(quota).copy()
        else:
            targets["min_distance_to_existing_exact"] = []
            targets["within_existing_min_dist"] = []
            radius_checked = targets.copy()
        targets.to_csv(AUDIT_ROOT / f"target_{boundary_type}_all_candidates.csv", index=False)
        radius_checked.to_csv(AUDIT_ROOT / f"target_{boundary_type}_radius_checked.csv", index=False)
        combined.append(radius_checked)
        combined_basic.append(targets.head(quota).copy())
        counts[boundary_type] = {
            "available_boundaries": int(group.shape[0]),
            "candidate_midpoints_after_basic_filters": int(targets.shape[0]),
            "excluded_by_existing_min_dist": int(targets["within_existing_min_dist"].sum()) if not targets.empty else 0,
            "recommended_points": int(radius_checked.shape[0]),
            "quota": int(quota),
        }

    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    if combined_basic:
        combined_basic_df = pd.concat(combined_basic, ignore_index=True)
    else:
        combined_basic_df = pd.DataFrame()
    combined_df.to_csv(AUDIT_ROOT / "targeted_refinement_prioritized_radius_checked.csv", index=False)
    combined_basic_df.to_csv(AUDIT_ROOT / "targeted_refinement_prioritized_basic_filtered.csv", index=False)
    return {
        "target_counts": counts,
        "combined_recommended_points": int(combined_df.shape[0]),
        "combined_basic_filtered_points": int(combined_basic_df.shape[0]),
        "radius_check_note": (
            "existing_min_dist is strict for bracket midpoints; use the basic-filtered list "
            "for boundary refinement only after accepting this exception explicitly."
        ),
    }


def write_report(summary: dict, sensitivity: pd.DataFrame) -> None:
    report_path = AUDIT_ROOT / "boundary_extraction_recheck_report.md"
    sensitivity_buffer = StringIO()
    sensitivity.to_csv(sensitivity_buffer, index=False)
    lines = [
        "# Boundary Extraction Recheck Report",
        "",
        "Date: 2026-05-12",
        "",
        "## Input Provenance",
        "",
    ]
    for row in summary["provenance"]["datasets"]:
        lines.append(
            f"- {Path(row['dataset']).name}: samples={row['n_samples']}, "
            f"size={row['file_size_bytes']}, same_as_iter042={row['same_content_as_dataset_iter042']}"
        )
    lines += [
        "",
        "## Default Reproduction",
        "",
        f"- boundary segments: {summary['default_boundary_audit']['boundary_counts_by_type']}",
        f"- confidence counts: {summary['default_boundary_audit']['boundary_counts_by_confidence']}",
        f"- predicate failures: {summary['default_boundary_audit']['failures']}",
        "",
        "## Dataset Consistency",
        "",
    ]
    for key, value in summary["dataset_consistency"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Parameter Sensitivity",
        "",
        "```csv",
        sensitivity_buffer.getvalue().strip(),
        "```",
        "",
        "## Prioritized Refinement Outputs",
        "",
    ]
    for key, value in summary["prioritized_targets"]["target_counts"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        f"Combined radius-checked recommended points: {summary['prioritized_targets']['combined_recommended_points']}",
        f"Combined basic-filtered points before hard radius exclusion: {summary['prioritized_targets']['combined_basic_filtered_points']}",
        f"Radius-check note: {summary['prioritized_targets']['radius_check_note']}",
        "",
        "## Interpretation",
        "",
        "- The default boundary extraction is considered reproducible only if the counts match the archived result.",
        "- Boundary predicates must have zero failures before the CSVs are used for physics interpretation.",
        "- The radius check is intentionally strict and may reject closely bracketed midpoint targets.",
        "- If eta_zero dominates the unfiltered candidates, use the per-type prioritized CSVs instead of the archived combined target list.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_RECHECK.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_ROOT.mkdir(parents=True, exist_ok=True)

    default_args = Namespace(
        dataset=DATASET_FOR_BOUNDARIES,
        output_dir=DEFAULT_RECHECK,
        kt_bin_width=0.005,
        max_local_spacing=0.035,
        max_refinement_points=512,
        output_root=Path("ML_Phase"),
    )
    default_summary = extract_phase_boundaries(default_args)
    df = load_dataset_frame(DATASET_FOR_BOUNDARIES)
    boundaries = pd.read_csv(DEFAULT_RECHECK / "all_boundary_segments.csv")

    sensitivity = run_sensitivity()
    sensitivity.to_csv(AUDIT_ROOT / "boundary_parameter_sensitivity.csv", index=False)

    summary = {
        "default_summary": default_summary,
        "provenance": provenance_report(),
        "dataset_consistency": dataset_consistency(df),
        "default_boundary_audit": audit_boundaries(boundaries),
        "prioritized_targets": make_prioritized_targets(boundaries, df),
        "sensitivity_csv": str(AUDIT_ROOT / "boundary_parameter_sensitivity.csv"),
    }
    (AUDIT_ROOT / "boundary_recheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, sensitivity)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
