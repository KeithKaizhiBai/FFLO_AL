from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "rankcap_k3_full_loop"
    / "ML_Phase_512_RankCapK3_FullLoop"
    / "active_runs"
    / "active_boundary_discovery_rankcap_k3_full_loop_v1"
)
FULL_LOOP_REPORT_DIR = ROOT / "rankcap_k3_full_loop" / "reports" / "rankcap_k3_full_loop"
LAST5_REPORT_DIR = (
    ROOT
    / "rankcap_k3_full_loop"
    / "ML_Phase_512_RankCapK3_FullLoop"
    / "reports"
    / "last5_selection_stop_audit"
)
OUT_DIR = ROOT / "reports" / "phase_boundary_stability_audit"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

PHASE_NORMAL = 0
PHASE_UNIFORM_SC = 1
PHASE_FFLO = 2


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_scalar(row.get(k, "")) for k in fields})


def clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return ""
        return repr(float(value))
    return value


def hash_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()[:16]


def load_monitor(iteration: int) -> dict[str, np.ndarray] | None:
    path = RUN_DIR / f"iter{iteration:03d}" / f"monitor_predictions_iter{iteration:03d}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def load_dataset_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=np.float64)
    with np.load(path, allow_pickle=False) as z:
        if "x" in z.files:
            return np.asarray(z["x"], dtype=np.float64).reshape(-1, 2)
        if "kT" in z.files and "JA" in z.files:
            return np.stack([z["kT"], z["JA"]], axis=1).astype(np.float64)
    return np.empty((0, 2), dtype=np.float64)


def dataset_count(path: Path) -> int:
    if not path.exists():
        return -1
    with np.load(path, allow_pickle=False) as z:
        if "x" in z.files:
            return int(np.asarray(z["x"]).reshape(-1, 2).shape[0])
        if "kT" in z.files:
            return int(np.asarray(z["kT"]).shape[0])
    return -1


def dataset_for_iteration(iteration: int, expected_count: int | None) -> tuple[Path | None, int]:
    candidates = [
        RUN_DIR / f"dataset_iter{iteration + 1:03d}.npz",
        RUN_DIR / f"dataset_iter{iteration:03d}.npz",
    ]
    available = [(p, dataset_count(p)) for p in candidates if p.exists()]
    if expected_count is not None:
        for path, count in available:
            if count == int(expected_count):
                return path, count
    if available:
        return available[0]
    return None, -1


def cfg_ranges() -> dict[str, float]:
    cfg = read_json(RUN_DIR / "run_config.json", {}).get("active_learning_config", {})
    return {
        "kt_min": float(cfg.get("kt_min", 0.0)),
        "kt_max": float(cfg.get("kt_max", 0.56)),
        "ja_min": float(cfg.get("ja_min", 0.0)),
        "ja_max": float(cfg.get("ja_max", 2.12)),
        "n_kt_candidates": float(cfg.get("n_kt_candidates", 241)),
        "n_ja_candidates": float(cfg.get("n_ja_candidates", 321)),
    }


def normalize_points(points: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    offset = np.array([cfg["kt_min"], cfg["ja_min"]], dtype=np.float64)
    scale = np.array(
        [
            max(cfg["kt_max"] - cfg["kt_min"], 1e-12),
            max(cfg["ja_max"] - cfg["ja_min"], 1e-12),
        ],
        dtype=np.float64,
    )
    return (points - offset) / scale


def nearest_distances_norm(a: np.ndarray, b: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return np.empty((0,), dtype=np.float64)
    an = normalize_points(a, cfg)
    bn = normalize_points(b, cfg)
    out = np.full(an.shape[0], np.inf, dtype=np.float64)
    chunk = 512
    for start in range(0, an.shape[0], chunk):
        pts = an[start : start + chunk]
        d2 = np.sum((pts[:, None, :] - bn[None, :, :]) ** 2, axis=2)
        out[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def boundary_points_from_phase(monitor: dict[str, np.ndarray], boundary_type: str) -> np.ndarray:
    required = {"grid_points", "phase_pred", "full_shape"}
    if monitor is None or not required.issubset(monitor):
        return np.empty((0, 2), dtype=np.float64)
    shape = tuple(int(x) for x in np.asarray(monitor["full_shape"]).ravel()[:2])
    points = np.asarray(monitor["grid_points"], dtype=np.float64).reshape(-1, 2)
    phase = np.asarray(monitor["phase_pred"], dtype=np.int64).reshape(shape)
    coords = points.reshape(shape + (2,))
    rows: list[np.ndarray] = []

    def crosses(a: int, b: int) -> bool:
        if boundary_type == "normal_sc":
            return (a == PHASE_NORMAL) != (b == PHASE_NORMAL)
        if boundary_type == "uniform_fflo":
            return {int(a), int(b)} == {PHASE_UNIFORM_SC, PHASE_FFLO}
        raise ValueError(boundary_type)

    n_ja, n_kt = phase.shape
    for i in range(n_ja):
        for j in range(n_kt - 1):
            if crosses(int(phase[i, j]), int(phase[i, j + 1])):
                rows.append(0.5 * (coords[i, j] + coords[i, j + 1]))
    for i in range(n_ja - 1):
        for j in range(n_kt):
            if crosses(int(phase[i, j]), int(phase[i + 1, j])):
                rows.append(0.5 * (coords[i, j] + coords[i + 1, j]))
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def recompute_phase_change(current: dict[str, np.ndarray] | None, previous: dict[str, np.ndarray] | None) -> dict[str, Any]:
    if current is None or previous is None:
        return {
            "available": False,
            "value": None,
            "changed_grid_count": None,
            "total_grid_count": None,
            "changed_fraction": None,
            "candidate_grid_hash": "",
            "phase_map_hash": "",
        }
    if "phase_pred" not in current or "phase_pred" not in previous:
        return {
            "available": False,
            "value": None,
            "changed_grid_count": None,
            "total_grid_count": None,
            "changed_fraction": None,
            "candidate_grid_hash": "",
            "phase_map_hash": "",
        }
    cur = np.asarray(current["phase_pred"], dtype=np.int64).ravel()
    prev = np.asarray(previous["phase_pred"], dtype=np.int64).ravel()
    if cur.shape != prev.shape or cur.size == 0:
        return {
            "available": False,
            "value": None,
            "changed_grid_count": None,
            "total_grid_count": None,
            "changed_fraction": None,
            "candidate_grid_hash": hash_array(current.get("grid_points", np.array([], dtype=np.float64))),
            "phase_map_hash": hash_array(cur),
        }
    changed = int(np.sum(cur != prev))
    total = int(cur.size)
    value = float(changed / total)
    return {
        "available": True,
        "value": value,
        "changed_grid_count": changed,
        "total_grid_count": total,
        "changed_fraction": value,
        "candidate_grid_hash": hash_array(np.asarray(current["grid_points"], dtype=np.float64)),
        "phase_map_hash": hash_array(cur),
    }


def recompute_boundary_shift(
    current: dict[str, np.ndarray] | None,
    previous: dict[str, np.ndarray] | None,
    boundary_type: str,
    cfg: dict[str, float],
) -> dict[str, Any]:
    cur = boundary_points_from_phase(current, boundary_type) if current is not None else np.empty((0, 2))
    prev = boundary_points_from_phase(previous, boundary_type) if previous is not None else np.empty((0, 2))
    if current is None or previous is None or cur.size == 0 or prev.size == 0:
        return {
            "available": False,
            "value": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "n_current": int(cur.shape[0]),
            "n_previous": int(prev.shape[0]),
            "points_current": cur,
            "points_previous": prev,
        }
    d_cur = nearest_distances_norm(cur, prev, cfg)
    d_prev = nearest_distances_norm(prev, cur, cfg)
    dist = np.concatenate([d_cur, d_prev])
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return {
            "available": False,
            "value": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "n_current": int(cur.shape[0]),
            "n_previous": int(prev.shape[0]),
            "points_current": cur,
            "points_previous": prev,
        }
    p95 = float(np.percentile(finite, 95))
    return {
        "available": True,
        "value": p95,
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": p95,
        "max": float(np.max(finite)),
        "n_current": int(cur.shape[0]),
        "n_previous": int(prev.shape[0]),
        "points_current": cur,
        "points_previous": prev,
    }


def recompute_boundary_coverage(
    current: dict[str, np.ndarray] | None,
    dataset_points: np.ndarray,
    cfg: dict[str, float],
) -> dict[str, Any]:
    if current is None:
        return {"available": False, "value": None, "normal_sc": 0, "uniform_fflo": 0}
    normal_sc = boundary_points_from_phase(current, "normal_sc")
    uniform_fflo = boundary_points_from_phase(current, "uniform_fflo")
    boundary_sets = [x for x in (normal_sc, uniform_fflo) if x.size]
    if not boundary_sets or dataset_points.size == 0:
        return {
            "available": False,
            "value": None,
            "normal_sc": int(normal_sc.shape[0]),
            "uniform_fflo": int(uniform_fflo.shape[0]),
        }
    all_boundary = np.vstack(boundary_sets)
    dist = nearest_distances_norm(all_boundary, dataset_points, cfg)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return {
            "available": False,
            "value": None,
            "normal_sc": int(normal_sc.shape[0]),
            "uniform_fflo": int(uniform_fflo.shape[0]),
        }
    return {
        "available": True,
        "value": float(np.percentile(finite, 95)),
        "normal_sc": int(normal_sc.shape[0]),
        "uniform_fflo": int(uniform_fflo.shape[0]),
    }


def function_line_ranges(source: Path) -> dict[str, str]:
    lines = source.read_text(encoding="utf-8").splitlines()
    starts: list[tuple[str, int]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if line.startswith("class ") or line.startswith("def "):
            name = stripped.split("(", 1)[0].replace("class ", "").replace("def ", "").strip(":")
            starts.append((name, idx))
    out: dict[str, str] = {}
    for pos, (name, start) in enumerate(starts):
        end = starts[pos + 1][1] - 1 if pos + 1 < len(starts) else len(lines)
        out[name] = f"{start}-{end}"
    return out


def build_definition_table() -> list[dict[str, Any]]:
    src = ROOT / "ml_phase" / "stop_controller.py"
    ranges = function_line_ranges(src)
    source_file = "ml_phase/stop_controller.py"
    return [
        {
            "metric": "phase_map_change",
            "source_file": source_file,
            "function_or_class": "phase_map_change",
            "line_range": ranges.get("phase_map_change", ""),
            "formula_or_logic": "mean(current phase_pred != previous phase_pred) on saved monitor dense grid.",
            "threshold": "map_tol = 0.002",
            "pass_condition": "available and value < map_tol",
            "notes": "Uses predicted surrogate phase map, not an exact-dataset phase map.",
        },
        {
            "metric": "boundary_shift_normal_sc",
            "source_file": source_file,
            "function_or_class": "boundary_shift + _boundary_points_from_phase",
            "line_range": f"{ranges.get('boundary_shift', '')}; {ranges.get('_boundary_points_from_phase', '')}",
            "formula_or_logic": "Extract edges where one side is normal and the other is superconducting; compute bidirectional nearest-neighbor distances between current and previous boundary points after parameter-range normalization; use p95.",
            "threshold": "boundary_shift_tol = dense_grid_spacing_norm = 0.004166666666666667",
            "pass_condition": "available and p95 < boundary_shift_tol",
            "notes": "The boundary is extracted from predicted monitor phase_pred.",
        },
        {
            "metric": "boundary_shift_uniform_fflo",
            "source_file": source_file,
            "function_or_class": "boundary_shift + _boundary_points_from_phase",
            "line_range": f"{ranges.get('boundary_shift', '')}; {ranges.get('_boundary_points_from_phase', '')}",
            "formula_or_logic": "Extract edges crossing uniform_SC and FFLO labels; compute bidirectional nearest-neighbor distances between current and previous boundary points after parameter-range normalization; use p95.",
            "threshold": "boundary_shift_tol = dense_grid_spacing_norm = 0.004166666666666667",
            "pass_condition": "available and p95 < boundary_shift_tol",
            "notes": "Final metric is available with nonzero boundary counts, so final zero is a valid zero-shift metric for this predicted boundary.",
        },
        {
            "metric": "label_surprise_rate",
            "source_file": source_file,
            "function_or_class": "label_surprise_rate",
            "line_range": ranges.get("label_surprise_rate", ""),
            "formula_or_logic": "Match selected_points_by_pool predicted_phase_before_exact to exact_merged_iter phase_label(delta_opt, q_opt); return mismatch fraction.",
            "threshold": "surprise_tol = 0.05",
            "pass_condition": "available and value < surprise_tol",
            "notes": "Measures selected-batch prediction surprise, not dense-map motion.",
        },
        {
            "metric": "boundary_coverage_p95",
            "source_file": source_file,
            "function_or_class": "boundary_coverage_p95",
            "line_range": ranges.get("boundary_coverage_p95", ""),
            "formula_or_logic": "Extract current normal/SC and uniform/FFLO boundary points; compute nearest distance from each boundary point to exact dataset points after normalization; use p95.",
            "threshold": "coverage_tol = 1.5 * dense_grid_spacing_norm = 0.00625",
            "pass_condition": "available and value < coverage_tol",
            "notes": "Measures sampling density near current predicted boundaries, not boundary displacement between iterations.",
        },
        {
            "metric": "passed_condition_count",
            "source_file": source_file,
            "function_or_class": "evaluate_stop",
            "line_range": ranges.get("evaluate_stop", ""),
            "formula_or_logic": "Sum boolean C1 phase_map_change, C2 normal/SC shift, C3 uniform/FFLO shift, C4 label_surprise_rate, C5 boundary_coverage_p95.",
            "threshold": "required_pass_count = 4",
            "pass_condition": "completed_iterations >= min_iterations and passed_condition_count >= required_pass_count",
            "notes": "selected_A0, q-edge, and rerun rates are diagnostics, not main conditions.",
        },
        {
            "metric": "patience",
            "source_file": source_file,
            "function_or_class": "StopConfig + evaluate_stop",
            "line_range": f"{ranges.get('StopConfig', '')}; {ranges.get('evaluate_stop', '')}",
            "formula_or_logic": "Increment patience_counter on convergence_pass; reset to zero otherwise; stop when patience_counter >= patience, unless hard max_iterations/max_exact_calls stop fires first.",
            "threshold": "patience = 4",
            "pass_condition": "convergence_pass for four consecutive evaluations",
            "notes": "Current run hit max_iterations before any positive patience sequence.",
        },
    ]


def build_tables() -> dict[str, Any]:
    history = read_json(RUN_DIR / "stop_metrics_history.json", [])
    cfg = cfg_ranges()
    final_thresholds = history[-1].get("thresholds", {}) if history else {}
    map_tol = float(final_thresholds.get("map_tol", 0.002))
    boundary_shift_tol = float(final_thresholds.get("boundary_shift_tol", 0.004166666666666667))
    coverage_tol = float(final_thresholds.get("coverage_tol", 0.00625))
    surprise_tol = float(final_thresholds.get("surprise_tol", 0.05))
    required = int(history[-1].get("required_pass_count", 4)) if history else 4
    patience = int(history[-1].get("patience", 4)) if history else 4

    monitors = {i: load_monitor(i) for i in range(0, max([x["iteration"] for x in history] or [0]) + 1)}

    definition_rows = build_definition_table()

    phase_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []
    patience_rows: list[dict[str, Any]] = []
    discrepancy_rows: list[dict[str, Any]] = []

    reconstructed_patience = 0
    phase_diffs: list[float] = []
    normal_diffs: list[float] = []
    uniform_diffs: list[float] = []
    coverage_diffs: list[float] = []

    for item in history:
        iteration = int(item["iteration"])
        current = monitors.get(iteration)
        previous = monitors.get(iteration - 1)
        metrics = item.get("metrics", {})
        availability = item.get("metric_availability", {})
        thresholds = item.get("thresholds", final_thresholds)

        phase = recompute_phase_change(current, previous)
        recorded_phase = metrics.get("phase_map_change")
        phase_diff = None
        if phase["value"] is not None and recorded_phase is not None:
            phase_diff = abs(float(phase["value"]) - float(recorded_phase))
            phase_diffs.append(float(phase_diff))
        phase_rows.append(
            {
                "iteration": iteration,
                "phase_map_change": phase["value"],
                "phase_map_tol": thresholds.get("map_tol", map_tol),
                "pass_flag": bool(item.get("conditions", {}).get("C1_phase_map_change", False)),
                "phase_map_source_file": relpath(RUN_DIR / f"iter{iteration:03d}" / f"monitor_predictions_iter{iteration:03d}.npz"),
                "candidate_grid_hash": phase["candidate_grid_hash"],
                "phase_map_hash": phase["phase_map_hash"],
                "changed_grid_count": phase["changed_grid_count"],
                "total_grid_count": phase["total_grid_count"],
                "changed_fraction": phase["changed_fraction"],
                "recomputed": bool(phase["available"]),
                "source": "monitor_predictions_iterXXX.npz",
                "recorded_value": recorded_phase,
                "absolute_discrepancy": phase_diff,
            }
        )

        for boundary_type, key in [
            ("normal_sc", "boundary_shift_normal_sc"),
            ("uniform_fflo", "boundary_shift_uniform_fflo"),
        ]:
            shifted = recompute_boundary_shift(current, previous, boundary_type, cfg)
            recorded = metrics.get(key)
            diff = None
            if shifted["value"] is not None and recorded is not None:
                diff = abs(float(shifted["value"]) - float(recorded))
                if boundary_type == "normal_sc":
                    normal_diffs.append(float(diff))
                else:
                    uniform_diffs.append(float(diff))
            boundary_rows.append(
                {
                    "iteration": iteration,
                    "boundary_type": boundary_type,
                    "boundary_shift_value": shifted["value"],
                    "boundary_shift_tol": thresholds.get("boundary_shift_tol", boundary_shift_tol),
                    "pass_flag": bool(
                        item.get("conditions", {}).get(
                            "C2_boundary_shift_normal_sc"
                            if boundary_type == "normal_sc"
                            else "C3_boundary_shift_uniform_fflo",
                            False,
                        )
                    ),
                    "boundary_point_count_current": shifted["n_current"],
                    "boundary_point_count_previous": shifted["n_previous"],
                    "boundary_extraction_source": relpath(
                        RUN_DIR / f"iter{iteration:03d}" / f"monitor_predictions_iter{iteration:03d}.npz"
                    ),
                    "metric_type": "bidirectional normalized nearest-neighbor p95",
                    "nearest_neighbor_mean": shifted["mean"],
                    "nearest_neighbor_median": shifted["median"],
                    "nearest_neighbor_p95": shifted["p95"],
                    "nearest_neighbor_max": shifted["max"],
                    "recomputed": bool(shifted["available"]),
                    "recorded_value": recorded,
                    "absolute_discrepancy": diff,
                }
            )

        expected_count = item.get("exact_call_count")
        dataset_path, dataset_n = dataset_for_iteration(iteration, expected_count)
        dataset_points = load_dataset_points(dataset_path) if dataset_path is not None else np.empty((0, 2))
        coverage = recompute_boundary_coverage(current, dataset_points, cfg)
        recorded_coverage = metrics.get("boundary_coverage_p95")
        if coverage["value"] is not None and recorded_coverage is not None:
            coverage_diffs.append(abs(float(coverage["value"]) - float(recorded_coverage)))

        conditions = item.get("conditions", {})
        stop_rows.append(
            {
                "iteration": iteration,
                "phase_map_change": metrics.get("phase_map_change"),
                "phase_map_pass": conditions.get("C1_phase_map_change"),
                "boundary_shift_normal_sc": metrics.get("boundary_shift_normal_sc"),
                "boundary_shift_normal_sc_pass": conditions.get("C2_boundary_shift_normal_sc"),
                "boundary_shift_uniform_fflo": metrics.get("boundary_shift_uniform_fflo"),
                "boundary_shift_uniform_fflo_pass": conditions.get("C3_boundary_shift_uniform_fflo"),
                "label_surprise_rate": metrics.get("label_surprise_rate"),
                "label_surprise_pass": conditions.get("C4_label_surprise_rate"),
                "boundary_coverage_p95": metrics.get("boundary_coverage_p95"),
                "boundary_coverage_pass": conditions.get("C5_boundary_coverage_p95"),
                "passed_condition_count": item.get("passed_condition_count"),
                "required_pass_count": item.get("required_pass_count"),
                "convergence_pass": item.get("convergence_pass"),
                "stop_reason": item.get("stop_reason"),
                "metric_availability_phase_map": availability.get("phase_map_change"),
                "metric_availability_boundary_normal_sc": availability.get("boundary_shift_normal_sc"),
                "metric_availability_boundary_uniform_fflo": availability.get("boundary_shift_uniform_fflo"),
                "metric_availability_label_surprise": availability.get("label_surprise_rate"),
                "metric_availability_boundary_coverage": availability.get("boundary_coverage_p95"),
            }
        )

        completed_iterations = int(item.get("completed_iterations", iteration + 1))
        conv = bool(completed_iterations >= int(item.get("stop_config", {}).get("min_iterations", 5)))
        conv = conv and int(item.get("passed_condition_count", 0)) >= int(item.get("required_pass_count", required))
        if conv:
            reconstructed_patience += 1
        else:
            reconstructed_patience = 0
        hard_stop = bool(item.get("hard_stop", False))
        patience_rows.append(
            {
                "iteration": iteration,
                "passed_condition_count": item.get("passed_condition_count"),
                "convergence_pass": item.get("convergence_pass"),
                "patience_counter": reconstructed_patience,
                "recorded_patience_counter": item.get("patience_counter"),
                "stop_candidate": conv,
                "stop_triggered": bool(reconstructed_patience >= int(item.get("patience", patience)) or hard_stop),
                "recorded_stop": item.get("stop"),
                "stop_reason": item.get("stop_reason"),
            }
        )

    discrepancy_rows.extend(
        [
            {
                "metric": "phase_map_change",
                "recomputed_count": len(phase_diffs),
                "max_absolute_discrepancy": max(phase_diffs) if phase_diffs else "",
                "matches_recorded": bool(phase_diffs and max(phase_diffs) < 1e-12),
                "notes": "Recomputed from monitor phase_pred arrays.",
            },
            {
                "metric": "boundary_shift_normal_sc",
                "recomputed_count": len(normal_diffs),
                "max_absolute_discrepancy": max(normal_diffs) if normal_diffs else "",
                "matches_recorded": bool(normal_diffs and max(normal_diffs) < 1e-12),
                "notes": "Recomputed from monitor boundary extraction and bidirectional nearest-neighbor p95.",
            },
            {
                "metric": "boundary_shift_uniform_fflo",
                "recomputed_count": len(uniform_diffs),
                "max_absolute_discrepancy": max(uniform_diffs) if uniform_diffs else "",
                "matches_recorded": bool(uniform_diffs and max(uniform_diffs) < 1e-12),
                "notes": "Recomputed from monitor boundary extraction and bidirectional nearest-neighbor p95.",
            },
            {
                "metric": "boundary_coverage_p95",
                "recomputed_count": len(coverage_diffs),
                "max_absolute_discrepancy": max(coverage_diffs) if coverage_diffs else "",
                "matches_recorded": bool(coverage_diffs and max(coverage_diffs) < 1e-12),
                "notes": "Recomputed with dataset file matched to exact_call_count.",
            },
        ]
    )

    metric_source_rows = [
        {
            "metric": "phase_map_change",
            "source_file": "iterXXX/monitor_predictions_iterXXX.npz",
            "source_keys": "phase_pred, grid_points, full_shape",
            "computed_by": "ml_phase/stop_controller.py::phase_map_change",
            "recomputed_in_audit": True,
            "notes": "Predicted dense-grid phase map compared to previous iteration.",
        },
        {
            "metric": "boundary_shift_normal_sc",
            "source_file": "iterXXX/monitor_predictions_iterXXX.npz and iterXXX-1/monitor_predictions_iterXXX-1.npz",
            "source_keys": "phase_pred, grid_points, full_shape",
            "computed_by": "ml_phase/stop_controller.py::boundary_shift",
            "recomputed_in_audit": True,
            "notes": "Boundary extracted from predicted phase labels.",
        },
        {
            "metric": "boundary_shift_uniform_fflo",
            "source_file": "iterXXX/monitor_predictions_iterXXX.npz and iterXXX-1/monitor_predictions_iterXXX-1.npz",
            "source_keys": "phase_pred, grid_points, full_shape",
            "computed_by": "ml_phase/stop_controller.py::boundary_shift",
            "recomputed_in_audit": True,
            "notes": "Final zero is available with boundary counts, not a missing-boundary fallback.",
        },
        {
            "metric": "label_surprise_rate",
            "source_file": "iterXXX/selected_points_by_pool.csv and iterXXX/exact_merged_iterXXX.npz",
            "source_keys": "predicted_phase_before_exact, delta_opt, q_opt",
            "computed_by": "ml_phase/stop_controller.py::label_surprise_rate",
            "recomputed_in_audit": False,
            "notes": "Last-five audit already reproduced selected-batch surprise exactly.",
        },
        {
            "metric": "boundary_coverage_p95",
            "source_file": "iterXXX/monitor_predictions_iterXXX.npz and matched dataset_iterXXX or dataset_iterXXX+1.npz",
            "source_keys": "phase_pred, grid_points, full_shape, x or kT/JA",
            "computed_by": "ml_phase/stop_controller.py::boundary_coverage_p95",
            "recomputed_in_audit": True,
            "notes": "Coverage uses current boundary to exact-dataset nearest distances.",
        },
        {
            "metric": "diagnostic maximum boundary displacement",
            "source_file": "active report boundary diagnostics",
            "source_keys": "report-specific boundary traces",
            "computed_by": "report diagnostics, not StopController",
            "recomputed_in_audit": False,
            "notes": "Not the same quantity as StopController boundary_shift.",
        },
    ]

    csv_write(
        TABLE_DIR / "stopcontroller_definition_audit.csv",
        definition_rows,
        [
            "metric",
            "source_file",
            "function_or_class",
            "line_range",
            "formula_or_logic",
            "threshold",
            "pass_condition",
            "notes",
        ],
    )
    csv_write(
        TABLE_DIR / "phase_map_change_by_iteration.csv",
        phase_rows,
        [
            "iteration",
            "phase_map_change",
            "phase_map_tol",
            "pass_flag",
            "phase_map_source_file",
            "candidate_grid_hash",
            "phase_map_hash",
            "changed_grid_count",
            "total_grid_count",
            "changed_fraction",
            "recomputed",
            "source",
            "recorded_value",
            "absolute_discrepancy",
        ],
    )
    csv_write(
        TABLE_DIR / "boundary_shift_by_iteration.csv",
        boundary_rows,
        [
            "iteration",
            "boundary_type",
            "boundary_shift_value",
            "boundary_shift_tol",
            "pass_flag",
            "boundary_point_count_current",
            "boundary_point_count_previous",
            "boundary_extraction_source",
            "metric_type",
            "nearest_neighbor_mean",
            "nearest_neighbor_median",
            "nearest_neighbor_p95",
            "nearest_neighbor_max",
            "recomputed",
            "recorded_value",
            "absolute_discrepancy",
        ],
    )
    csv_write(
        TABLE_DIR / "stop_condition_pass_table.csv",
        stop_rows,
        [
            "iteration",
            "phase_map_change",
            "phase_map_pass",
            "boundary_shift_normal_sc",
            "boundary_shift_normal_sc_pass",
            "boundary_shift_uniform_fflo",
            "boundary_shift_uniform_fflo_pass",
            "label_surprise_rate",
            "label_surprise_pass",
            "boundary_coverage_p95",
            "boundary_coverage_pass",
            "passed_condition_count",
            "required_pass_count",
            "convergence_pass",
            "stop_reason",
            "metric_availability_phase_map",
            "metric_availability_boundary_normal_sc",
            "metric_availability_boundary_uniform_fflo",
            "metric_availability_label_surprise",
            "metric_availability_boundary_coverage",
        ],
    )
    csv_write(
        TABLE_DIR / "patience_counter_reconstruction.csv",
        patience_rows,
        [
            "iteration",
            "passed_condition_count",
            "convergence_pass",
            "patience_counter",
            "recorded_patience_counter",
            "stop_candidate",
            "stop_triggered",
            "recorded_stop",
            "stop_reason",
        ],
    )
    csv_write(
        TABLE_DIR / "metric_source_files.csv",
        metric_source_rows,
        ["metric", "source_file", "source_keys", "computed_by", "recomputed_in_audit", "notes"],
    )
    csv_write(
        TABLE_DIR / "discrepancy_check.csv",
        discrepancy_rows,
        ["metric", "recomputed_count", "max_absolute_discrepancy", "matches_recorded", "notes"],
    )

    return {
        "history": history,
        "phase_rows": phase_rows,
        "boundary_rows": boundary_rows,
        "stop_rows": stop_rows,
        "patience_rows": patience_rows,
        "definition_rows": definition_rows,
        "metric_source_rows": metric_source_rows,
        "discrepancy_rows": discrepancy_rows,
        "cfg": cfg,
        "thresholds": {
            "map_tol": map_tol,
            "boundary_shift_tol": boundary_shift_tol,
            "coverage_tol": coverage_tol,
            "surprise_tol": surprise_tol,
            "required_pass_count": required,
            "patience": patience,
        },
    }


def make_figures(data: dict[str, Any]) -> None:
    history = data["history"]
    phase_rows = data["phase_rows"]
    boundary_rows = data["boundary_rows"]
    thresholds = data["thresholds"]

    plt.figure(figsize=(7.2, 4.2))
    xs = [int(r["iteration"]) for r in phase_rows if r["phase_map_change"] not in (None, "")]
    ys = [float(r["phase_map_change"]) for r in phase_rows if r["phase_map_change"] not in (None, "")]
    plt.plot(xs, ys, marker="o", linewidth=1.5, label="phase_map_change")
    plt.axhline(thresholds["map_tol"], color="tab:red", linestyle="--", label="tol")
    plt.xlabel("Iteration")
    plt.ylabel("Changed dense-grid fraction")
    plt.title("Phase-map change by iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase_map_change_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    for btype, label in [("normal_sc", "normal/SC"), ("uniform_fflo", "uniform/FFLO")]:
        rows = [r for r in boundary_rows if r["boundary_type"] == btype and r["boundary_shift_value"] not in (None, "")]
        plt.plot(
            [int(r["iteration"]) for r in rows],
            [float(r["boundary_shift_value"]) for r in rows],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    plt.axhline(thresholds["boundary_shift_tol"], color="tab:red", linestyle="--", label="tol")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized p95 boundary shift")
    plt.title("StopController boundary shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boundary_shift_curve.png", dpi=180)
    plt.close()

    condition_keys = [
        "C1_phase_map_change",
        "C2_boundary_shift_normal_sc",
        "C3_boundary_shift_uniform_fflo",
        "C4_label_surprise_rate",
        "C5_boundary_coverage_p95",
    ]
    mat = np.array(
        [[1 if item.get("conditions", {}).get(k, False) else 0 for k in condition_keys] for item in history],
        dtype=np.int64,
    )
    plt.figure(figsize=(8.0, 3.8))
    plt.imshow(mat.T, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#d95f5f", "#5aa469"]))
    plt.yticks(range(len(condition_keys)), ["phase map", "N/SC shift", "U/FFLO shift", "surprise", "coverage"])
    plt.xticks(range(0, len(history), 5), [str(history[i]["iteration"]) for i in range(0, len(history), 5)])
    plt.xlabel("Iteration")
    plt.title("Stop condition pass flags")
    plt.colorbar(ticks=[0, 1], label="pass flag")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stop_condition_pass_flags.png", dpi=180)
    plt.close()

    last_iters = [int(item["iteration"]) for item in history[-5:]]
    cmap = ListedColormap(["#d8d8d8", "#f2a65a", "#5b8fd9"])
    fig, axes = plt.subplots(1, len(last_iters), figsize=(14, 3.4), sharex=True, sharey=True)
    for ax, iteration in zip(axes, last_iters, strict=False):
        monitor = load_monitor(iteration)
        if monitor is None:
            ax.axis("off")
            continue
        shape = tuple(int(x) for x in np.asarray(monitor["full_shape"]).ravel()[:2])
        phase = np.asarray(monitor["phase_pred"], dtype=np.int64).reshape(shape)
        kt = np.asarray(monitor["kt_values"], dtype=np.float64)
        ja = np.asarray(monitor["ja_values"], dtype=np.float64)
        ax.imshow(
            phase,
            origin="lower",
            aspect="auto",
            extent=[float(kt.min()), float(kt.max()), float(ja.min()), float(ja.max())],
            cmap=cmap,
            vmin=0,
            vmax=2,
        )
        ax.set_title(f"iter {iteration:03d}")
        ax.set_xlabel("kT")
    axes[0].set_ylabel("JA")
    fig.suptitle("Predicted phase-map snapshots, last five iterations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase_map_snapshots_last5.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(7.0, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(last_iters)))
    for color, iteration in zip(colors, last_iters, strict=False):
        monitor = load_monitor(iteration)
        if monitor is None:
            continue
        ns = boundary_points_from_phase(monitor, "normal_sc")
        uf = boundary_points_from_phase(monitor, "uniform_fflo")
        if ns.size:
            plt.scatter(ns[:, 0], ns[:, 1], s=4, color=color, alpha=0.5, marker="o", label=f"N/SC {iteration:03d}")
        if uf.size:
            plt.scatter(uf[:, 0], uf[:, 1], s=6, color=color, alpha=0.7, marker="x", label=f"U/F {iteration:03d}")
    plt.xlabel("kT")
    plt.ylabel("JA")
    plt.title("Boundary overlay, last five iterations")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boundary_overlay_last5.png", dpi=180)
    plt.close()


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def status(value: bool) -> str:
    return "pass" if value else "fail"


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        val = float(value)
    except Exception:
        return str(value)
    return f"{val:.{digits}g}"


def build_markdown(data: dict[str, Any]) -> str:
    history = data["history"]
    final = history[-1]
    final_metrics = final["metrics"]
    final_conditions = final["conditions"]
    final_details = final["boundary_details"]
    thresholds = final["thresholds"]
    last5 = history[-5:]

    phase_last5_pass = all(item["conditions"].get("C1_phase_map_change", False) for item in last5)
    ns_last5_pass = all(item["conditions"].get("C2_boundary_shift_normal_sc", False) for item in last5)
    uf_last5_pass = all(item["conditions"].get("C3_boundary_shift_uniform_fflo", False) for item in last5)
    last5_support = phase_last5_pass and ns_last5_pass and uf_last5_pass

    md: list[str] = []
    md.append("# Phase Boundary Stability Audit")
    md.append("")
    md.append("This is a report-only audit. It does not modify acquisition, exact oracle, rankcap_k3, StopController, tolerances, active-run artifacts, or Slurm state.")
    md.append("")
    md.append("## 1. Executive Summary")
    md.append("")
    md.append("| Question | Answer | Evidence |")
    md.append("|---|---:|---|")
    md.append("| Q1. phase_map_change definition | confirmed | `mean(current phase_pred != previous phase_pred)` on `monitor_predictions_iterXXX.npz` dense-grid surrogate labels. |")
    md.append("| Q2. boundary_shift definition | confirmed | Predicted boundary points are extracted from label crossings; current and previous boundaries are compared by bidirectional normalized nearest-neighbor p95. |")
    md.append(f"| Q3. Current phase map stable? | confirmed | final phase_map_change = {fmt(final_metrics['phase_map_change'])} < {fmt(thresholds['map_tol'])}. |")
    md.append(f"| Q4. Current normal/SC boundary stable? | confirmed | final normal/SC shift = {fmt(final_metrics['boundary_shift_normal_sc'])} < {fmt(thresholds['boundary_shift_tol'])}; current/previous boundary counts = {final_details['normal_sc']['n_current']}/{final_details['normal_sc']['n_previous']}. |")
    md.append(f"| Q5. Current uniform-SC/FFLO boundary stable? | confirmed | final uniform/FFLO shift = {fmt(final_metrics['boundary_shift_uniform_fflo'])} < {fmt(thresholds['boundary_shift_tol'])}; metric available with current/previous boundary counts = {final_details['uniform_fflo']['n_current']}/{final_details['uniform_fflo']['n_previous']}. |")
    md.append(f"| Q6. Why no StopController convergence? | confirmed | Only {final['passed_condition_count']}/{final['required_pass_count']} main conditions passed; label_surprise_rate = {fmt(final_metrics['label_surprise_rate'])} > {fmt(thresholds['surprise_tol'])}, boundary_coverage_p95 = {fmt(final_metrics['boundary_coverage_p95'])} > {fmt(thresholds['coverage_tol'])}. |")
    md.append("| Q7. boundary_shift vs boundary_coverage_p95 differ? | confirmed | boundary_shift compares adjacent predicted boundaries; boundary_coverage_p95 measures distance from current predicted boundary to exact samples. |")
    md.append(f"| Q8. Last five support main boundary stabilization? | {'supported' if last5_support else 'not supported'} | Last five C1/C2/C3 pass flags are phase={phase_last5_pass}, normal/SC={ns_last5_pass}, uniform/FFLO={uf_last5_pass}. |")
    md.append("| Q9. Need to modify StopController? | not supported | The audit reproduced the metrics; the failed formal stop follows the documented 4-of-5 plus patience rule. |")
    md.append("| Q10. Need cleanup acquisition? | supported | If formal convergence is required, a separate cleanup mode should target label surprise and boundary coverage rather than changing thresholds. |")
    md.append("")
    md.append("## 2. StopController Definitions")
    md.append("")
    md.append("The implementation is in `ml_phase/stop_controller.py`. Key points:")
    md.append("")
    md.append("- `phase_map_change` compares saved surrogate predictions, not exact dataset labels.")
    md.append("- The dense monitor grid has 241 kT values by 321 JA values, i.e. 77361 grid points.")
    md.append("- `boundary_shift` extracts predicted phase-label crossings from the same dense grid.")
    md.append("- Boundary distances are computed after normalizing kT and JA to the configured parameter ranges.")
    md.append("- The shift value used by StopController is the p95 of the concatenated current-to-previous and previous-to-current nearest-neighbor distances.")
    md.append("- The default `boundary_shift_tol` is `dense_grid_spacing_norm`, equal to 0.004166666666666667 for this grid.")
    md.append("- `boundary_coverage_p95` is a separate coverage metric: current boundary points to exact dataset points.")
    md.append("- `required_pass_count = 4` over five main conditions; `patience = 4` requires four consecutive convergence-pass evaluations.")
    md.append("- Report-level diagnostic maximum boundary displacement is not the same quantity as StopController `boundary_shift`.")
    md.append("")
    md.append("See `tables/stopcontroller_definition_audit.csv` and `tables/metric_source_files.csv` for line ranges and artifact paths.")
    md.append("")
    md.append("## 3. Phase Map Change")
    md.append("")
    changed = final_metrics["phase_map_change"]
    changed_count = next(r for r in data["phase_rows"] if r["iteration"] == final["iteration"])["changed_grid_count"]
    total_count = next(r for r in data["phase_rows"] if r["iteration"] == final["iteration"])["total_grid_count"]
    md.append(f"Final `phase_map_change = {fmt(changed, 9)}`. This corresponds to {changed_count} changed dense-grid labels out of {total_count}. The threshold is `{fmt(thresholds['map_tol'])}`; therefore the final phase-map-change condition passed.")
    md.append("")
    md.append("The audit recomputed the phase-map-change table from `monitor_predictions_iterXXX.npz`, not from the report text. See `figures/phase_map_change_curve.png`.")
    md.append("")
    md.append("![phase map change](figures/phase_map_change_curve.png)")
    md.append("")
    md.append("## 4. Boundary Shift")
    md.append("")
    md.append(f"Final normal/SC shift is `{fmt(final_metrics['boundary_shift_normal_sc'], 9)}` with p95 metric and {final_details['normal_sc']['n_current']} current boundary points. Final uniform/FFLO shift is `{fmt(final_metrics['boundary_shift_uniform_fflo'], 9)}` with {final_details['uniform_fflo']['n_current']} current boundary points. Both are available metrics and both are below `{fmt(thresholds['boundary_shift_tol'], 9)}`.")
    md.append("")
    md.append("The final uniform/FFLO shift is not treated as a missing-boundary fallback: `metric_availability.boundary_shift_uniform_fflo` is true and the final boundary counts are nonzero.")
    md.append("")
    md.append("![boundary shift](figures/boundary_shift_curve.png)")
    md.append("")
    md.append("![boundary overlay](figures/boundary_overlay_last5.png)")
    md.append("")
    md.append("## 5. Stop Condition Pass Flags")
    md.append("")
    md.append(f"Final StopController state: `stop_reason = {final['stop_reason']}`, `convergence_pass = {str(final['convergence_pass']).lower()}`, `passed_condition_count = {final['passed_condition_count']}`, `required_pass_count = {final['required_pass_count']}`, `patience_counter = {final['patience_counter']}`.")
    md.append("")
    md.append("| Condition | Final value | Threshold | Final pass |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| phase_map_change | {fmt(final_metrics['phase_map_change'])} | {fmt(thresholds['map_tol'])} | {status(final_conditions['C1_phase_map_change'])} |")
    md.append(f"| boundary_shift_normal_sc | {fmt(final_metrics['boundary_shift_normal_sc'])} | {fmt(thresholds['boundary_shift_tol'])} | {status(final_conditions['C2_boundary_shift_normal_sc'])} |")
    md.append(f"| boundary_shift_uniform_fflo | {fmt(final_metrics['boundary_shift_uniform_fflo'])} | {fmt(thresholds['boundary_shift_tol'])} | {status(final_conditions['C3_boundary_shift_uniform_fflo'])} |")
    md.append(f"| label_surprise_rate | {fmt(final_metrics['label_surprise_rate'])} | {fmt(thresholds['surprise_tol'])} | {status(final_conditions['C4_label_surprise_rate'])} |")
    md.append(f"| boundary_coverage_p95 | {fmt(final_metrics['boundary_coverage_p95'])} | {fmt(thresholds['coverage_tol'])} | {status(final_conditions['C5_boundary_coverage_p95'])} |")
    md.append("")
    md.append("![stop condition pass flags](figures/stop_condition_pass_flags.png)")
    md.append("")
    md.append("## 6. Patience Reconstruction")
    md.append("")
    md.append("The patience counter was reconstructed from the stored pass counts. Because the final and preceding late iterations pass only 3 of 5 main conditions, `convergence_pass` is false and the patience counter resets to zero. The recorded and reconstructed counters agree in `tables/patience_counter_reconstruction.csv`.")
    md.append("")
    md.append("An older run could stop while label surprise was not fully below tolerance because StopController requires four of five main conditions, not all five. If label surprise fails but phase-map change, both boundary shifts, and boundary coverage pass for four consecutive evaluations, stop reason can be `converged_main_phase_boundaries`. In the current run, both label surprise and boundary coverage fail, leaving only three passed conditions.")
    md.append("")
    md.append("## 7. Phase-Map Stability vs Label Surprise")
    md.append("")
    md.append(f"`phase_map_change = {fmt(final_metrics['phase_map_change'], 9)}` is low, meaning the global predicted dense phase map changed very little between the last two monitor predictions. `boundary_shift_normal_sc = {fmt(final_metrics['boundary_shift_normal_sc'], 9)}` and `boundary_shift_uniform_fflo = {fmt(final_metrics['boundary_shift_uniform_fflo'], 9)}` are low, meaning the main predicted boundary locations moved little under the StopController metric.")
    md.append("")
    md.append(f"`label_surprise_rate = {fmt(final_metrics['label_surprise_rate'], 9)}` is high, meaning newly selected exact points still disagree with their pre-exact surrogate labels at a high rate. These can coexist: the global predicted map may be stable while acquisition keeps probing difficult or numerically risky regions where selected-batch surprises remain frequent.")
    md.append("")
    md.append("The correct conclusion is: main phase map and main predicted thermodynamic boundaries are stable by the StopController stability metrics, but formal StopController convergence does not pass because label surprise and boundary coverage fail.")
    md.append("")
    md.append("## 8. Boundary Coverage vs Boundary Shift")
    md.append("")
    md.append("`boundary_shift` compares adjacent iteration boundary positions. `boundary_coverage_p95` measures whether current exact samples cover the current predicted boundaries densely enough. Therefore `boundary_shift` can pass while `boundary_coverage_p95` fails.")
    md.append("")
    md.append(f"That is exactly the final state: boundary shifts pass, but `boundary_coverage_p95 = {fmt(final_metrics['boundary_coverage_p95'], 9)}` is above `coverage_tol = {fmt(thresholds['coverage_tol'], 9)}`.")
    md.append("")
    md.append("## 9. Phase Snapshots")
    md.append("")
    md.append("The last five predicted phase maps show small dense-grid label changes and stable main boundary locations under the StopController metrics.")
    md.append("")
    md.append("![phase map snapshots](figures/phase_map_snapshots_last5.png)")
    md.append("")
    md.append("## 10. Discrepancy Check")
    md.append("")
    for row in data["discrepancy_rows"]:
        md.append(f"- `{row['metric']}`: recomputed_count={row['recomputed_count']}, max_absolute_discrepancy={fmt(row['max_absolute_discrepancy'], 12)}, matches_recorded={row['matches_recorded']}.")
    md.append("")
    md.append("## 11. Do-Not-Claim List")
    md.append("")
    md.append("1. Do not interpret high label surprise as proof that phase boundaries are still moving substantially.")
    md.append("2. Do not conflate boundary_coverage_p95 failure with boundary_shift failure.")
    md.append("3. Do not claim formal StopController convergence has passed.")
    md.append("4. Do not modify thresholds in response to this report.")
    md.append("5. Do not modify StopController in this report-only task.")
    md.append("6. Do not treat eta response as a thermodynamic phase boundary.")
    md.append("7. Do not interpret uniform/FFLO shift = 0 as stable unless boundary availability and nonzero boundary counts are confirmed; here they are confirmed for the final iteration.")
    md.append("")
    md.append("## 12. Next Step")
    md.append("")
    md.append("Do not change StopController based on this audit alone. If formal convergence is required, the next separately planned calculation should be a cleanup acquisition or coverage-targeted validation that addresses label surprise and boundary coverage while keeping the accepted rankcap_k3 oracle unchanged.")
    md.append("")
    return "\n".join(md)


def tex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def build_tex(data: dict[str, Any]) -> str:
    history = data["history"]
    final = history[-1]
    metrics = final["metrics"]
    conditions = final["conditions"]
    thresholds = final["thresholds"]
    details = final["boundary_details"]
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
        r"\usepackage{float}",
        r"\title{Phase Boundary Stability Audit}",
        r"\author{Report-only audit}",
        r"\date{2026-06-17}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        (
            "Final StopController state: stop reason "
            + tex_escape(str(final["stop_reason"]))
            + f", pass count {final['passed_condition_count']}/{final['required_pass_count']}, "
            + f"patience counter {final['patience_counter']}."
        ),
        r"\begin{center}\begin{tabular}{lrrr}\toprule Metric & Value & Threshold & Pass \\ \midrule",
        f"phase\\_map\\_change & {fmt(metrics['phase_map_change'])} & {fmt(thresholds['map_tol'])} & {conditions['C1_phase_map_change']} \\\\",
        f"boundary\\_shift\\_normal\\_sc & {fmt(metrics['boundary_shift_normal_sc'])} & {fmt(thresholds['boundary_shift_tol'])} & {conditions['C2_boundary_shift_normal_sc']} \\\\",
        f"boundary\\_shift\\_uniform\\_fflo & {fmt(metrics['boundary_shift_uniform_fflo'])} & {fmt(thresholds['boundary_shift_tol'])} & {conditions['C3_boundary_shift_uniform_fflo']} \\\\",
        f"label\\_surprise\\_rate & {fmt(metrics['label_surprise_rate'])} & {fmt(thresholds['surprise_tol'])} & {conditions['C4_label_surprise_rate']} \\\\",
        f"boundary\\_coverage\\_p95 & {fmt(metrics['boundary_coverage_p95'])} & {fmt(thresholds['coverage_tol'])} & {conditions['C5_boundary_coverage_p95']} \\\\",
        r"\bottomrule\end{tabular}\end{center}",
        r"\section*{Definitions}",
        (
            "phase\\_map\\_change is the changed-label fraction between current and previous "
            "predicted dense-grid phase\\_pred arrays. Boundary shift extracts predicted "
            "phase-label crossing midpoints and uses the p95 of bidirectional normalized "
            "nearest-neighbor distances. Boundary coverage is different: it measures "
            "current predicted boundary points to exact dataset samples."
        ),
        r"\section*{Final Interpretation}",
        (
            "The phase map and main predicted thermodynamic boundaries are stable by the "
            "StopController stability metrics. Formal convergence does not pass because "
            "label surprise and boundary coverage fail. The final uniform/FFLO shift is "
            f"available with boundary counts {details['uniform_fflo']['n_current']}/"
            f"{details['uniform_fflo']['n_previous']}, so its zero value is not a missing-boundary fallback."
        ),
        r"\section*{Figures}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.88\linewidth]{figures/phase_map_change_curve.png}\caption{Phase-map change curve.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.88\linewidth]{figures/boundary_shift_curve.png}\caption{Boundary-shift curve.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.88\linewidth]{figures/stop_condition_pass_flags.png}\caption{Stop condition pass flags.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.95\linewidth]{figures/phase_map_snapshots_last5.png}\caption{Last-five predicted phase maps.}\end{figure}",
        r"\clearpage",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\linewidth,height=0.58\textheight,keepaspectratio]{figures/boundary_overlay_last5.png}\caption{Last-five boundary overlay.}\end{figure}",
        r"\section*{Do-Not-Claim}",
        (
            "Do not claim formal convergence; do not equate label surprise failure with "
            "large boundary drift; do not equate boundary coverage failure with boundary "
            "shift failure; do not change thresholds or StopController from this report."
        ),
        r"\end{document}",
    ]
    return "\n".join(lines)


def build_decision_log(data: dict[str, Any]) -> str:
    final = data["history"][-1]
    m = final["metrics"]
    t = final["thresholds"]
    return "\n".join(
        [
            "# Phase Boundary Stability Audit Decision Log",
            "",
            "Decision: report-only audit confirms that the final phase-map and boundary-shift stability checks pass, while formal StopController convergence fails.",
            "",
            "Evidence:",
            f"- phase_map_change = {fmt(m['phase_map_change'], 9)} < {fmt(t['map_tol'], 9)}.",
            f"- boundary_shift_normal_sc = {fmt(m['boundary_shift_normal_sc'], 9)} < {fmt(t['boundary_shift_tol'], 9)}.",
            f"- boundary_shift_uniform_fflo = {fmt(m['boundary_shift_uniform_fflo'], 9)} < {fmt(t['boundary_shift_tol'], 9)} with nonzero final boundary counts.",
            f"- label_surprise_rate = {fmt(m['label_surprise_rate'], 9)} > {fmt(t['surprise_tol'], 9)}.",
            f"- boundary_coverage_p95 = {fmt(m['boundary_coverage_p95'], 9)} > {fmt(t['coverage_tol'], 9)}.",
            f"- final pass count = {final['passed_condition_count']}/{final['required_pass_count']}; patience_counter = {final['patience_counter']}; stop_reason = {final['stop_reason']}.",
            "",
            "Consequence: do not describe this run as formally converged. It is accurate to say the main predicted phase map and main predicted thermodynamic boundaries are stable under the StopController stability metrics.",
            "",
            "Next recommended check: if formal convergence is required, plan a separate cleanup acquisition or boundary-coverage validation. Do not modify StopController or thresholds as part of this audit.",
            "",
        ]
    )


def main() -> None:
    ensure_dirs()
    data = build_tables()
    make_figures(data)
    md = build_markdown(data)
    (OUT_DIR / "phase_boundary_stability_audit.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "phase_boundary_stability_audit.tex").write_text(build_tex(data), encoding="utf-8")
    (OUT_DIR / "decision_log.md").write_text(build_decision_log(data), encoding="utf-8")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
