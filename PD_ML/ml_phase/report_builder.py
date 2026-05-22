from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np


def _load_json(path: Path, default: dict | list | None = None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float | int | str | None) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "N/A"
        return f"{float(v):.6g}"
    return str(v)


def _tex_escape(s: str) -> str:
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
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _latest_iter_dir(run_dir: Path) -> Path | None:
    iters = sorted([p for p in run_dir.glob("iter*") if p.is_dir()])
    return iters[-1] if iters else None


def _copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    try:
        from PIL import Image

        with Image.open(dst) as im:
            if im.mode not in {"RGB", "L"}:
                im.convert("RGB").save(dst)
    except Exception:
        pass
    return dst.name


def _placeholder_png(dst: Path) -> str:
    """Write a tiny valid PNG so LaTeX can compile even when an optional plot is absent."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    png_8x8_rgb = (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFUlEQVR4nGP8"
        "//8/AzbAhFV00EoAAFbUAw037MyjAAAAAElFTkSuQmCC"
    )
    dst.write_bytes(base64.b64decode(png_8x8_rgb))
    return dst.name


def build_report(run_dir: Path, template_path: Path, output_tex: Path) -> Path:
    run_cfg = _load_json(run_dir / "run_config.json", default={})
    metrics_hist = _load_json(run_dir / "metrics_history.json", default=[])
    latest_metrics = metrics_hist[-1] if metrics_hist else {}
    latest_iter = _latest_iter_dir(run_dir)

    warm_npz = run_cfg.get("args", {}).get("warm_start")
    n_warm = "N/A"
    if warm_npz:
        p = Path(warm_npz)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                n_warm = int(z["eta_matrix"].shape[0] * z["eta_matrix"].shape[1])

    n_exact = "N/A"
    if latest_iter and (run_dir / f"dataset_iter{latest_iter.name[-3:]}.npz").exists():
        with np.load(run_dir / f"dataset_iter{latest_iter.name[-3:]}.npz", allow_pickle=False) as z:
            n_exact = int(z["x"].shape[0])

    fig_phase = ""
    fig_uncert = ""
    fig_acq = ""
    fig_selected = ""
    fig_lc = ""

    output_root = run_dir.parent.parent
    figures_dir = output_root / "figures"
    run_id = run_cfg.get("args", {}).get("run_id", run_dir.name)
    if latest_iter:
        i = int(latest_iter.name[-3:])
        fig_phase = _copy_if_exists(
            figures_dir / f"{run_id}_iter{i:03d}_phase_prediction.png",
            Path("report/figures") / f"{run_id}_iter{i:03d}_phase_prediction.png",
        )
        fig_uncert = _copy_if_exists(
            figures_dir / f"{run_id}_iter{i:03d}_uncertainty.png",
            Path("report/figures") / f"{run_id}_iter{i:03d}_uncertainty.png",
        )
        fig_acq = _copy_if_exists(
            figures_dir / f"{run_id}_iter{i:03d}_acquisition.png",
            Path("report/figures") / f"{run_id}_iter{i:03d}_acquisition.png",
        )
        fig_selected = _copy_if_exists(
            figures_dir / f"{run_id}_iter{i:03d}_selected_points.png",
            Path("report/figures") / f"{run_id}_iter{i:03d}_selected_points.png",
        )
    fig_lc = _copy_if_exists(
        figures_dir / f"{run_id}_learning_curve.png",
        Path("report/figures") / f"{run_id}_learning_curve.png",
    )

    replacements: Dict[str, str] = {
        "{{RUN_ID}}": _tex_escape(str(run_id)),
        "{{N_WARM_START}}": _tex_escape(_fmt(n_warm)),
        "{{N_EXACT}}": _tex_escape(_fmt(n_exact)),
        "{{N_ITERS}}": _tex_escape(_fmt(len(metrics_hist))),
        "{{REG_MODEL}}": _tex_escape("Torch MLP Ensemble"),
        "{{CLS_MODEL}}": _tex_escape("Torch MLP Ensemble"),
        "{{N_ENSEMBLE}}": _tex_escape(_fmt(run_cfg.get("active_learning_config", {}).get("n_ensemble"))),
        "{{N_FEATURES}}": _tex_escape("2"),
        "{{N_REG_TARGETS}}": _tex_escape("5"),
        "{{VAL_FRACTION}}": _tex_escape(_fmt(run_cfg.get("active_learning_config", {}).get("val_fraction"))),
        "{{MODE}}": _tex_escape(_fmt(run_cfg.get("args", {}).get("mode"))),
        "{{WORLD_SIZE}}": _tex_escape(_fmt(run_cfg.get("args", {}).get("world_size"))),
        "{{PARTITION_STRATEGY}}": _tex_escape(_fmt(run_cfg.get("args", {}).get("partition_strategy"))),
        "{{POINTS_PER_ITER}}": _tex_escape(_fmt(run_cfg.get("args", {}).get("points_per_iter"))),
        "{{DELTA_RMSE}}": _tex_escape(_fmt(latest_metrics.get("delta_rmse"))),
        "{{Q_RMSE}}": _tex_escape(_fmt(latest_metrics.get("q_rmse"))),
        "{{ETA_RMSE}}": _tex_escape(_fmt(latest_metrics.get("eta_rmse"))),
        "{{ICP_RMSE}}": _tex_escape(_fmt(latest_metrics.get("ic_plus_rmse"))),
        "{{ICM_RMSE}}": _tex_escape(_fmt(latest_metrics.get("ic_minus_rmse"))),
        "{{PHASE_ACC}}": _tex_escape(_fmt(latest_metrics.get("phase_accuracy"))),
        "{{BOUNDARY_F1}}": _tex_escape(_fmt(latest_metrics.get("boundary_f1"))),
        "{{EXACT_REDUCTION}}": _tex_escape(_fmt(latest_metrics.get("estimated_reduction"))),
        "__FIG_PHASE__": fig_phase or _placeholder_png(Path("report/figures/missing_phase.png")),
        "__FIG_UNCERT__": fig_uncert or _placeholder_png(Path("report/figures/missing_uncertainty.png")),
        "__FIG_ACQ__": fig_acq or _placeholder_png(Path("report/figures/missing_acquisition.png")),
        "__FIG_SELECTED__": fig_selected or _placeholder_png(Path("report/figures/missing_selected.png")),
        "__FIG_LC__": fig_lc or _placeholder_png(Path("report/figures/missing_learning_curve.png")),
    }

    tex = template_path.read_text(encoding="utf-8")
    for k, v in replacements.items():
        tex = tex.replace(k, v)

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(tex, encoding="utf-8")
    return output_tex


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LaTeX report from active-learning outputs.")
    p.add_argument("--run-id", type=str, required=True, help="Run id under ML_Phase/active_runs")
    p.add_argument("--run-root", type=Path, default=Path("ML_Phase/active_runs"), help="Active runs root")
    p.add_argument(
        "--template",
        type=Path,
        default=Path("report/active_learning_phase_boundary_report.tex"),
        help="LaTeX template path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("ML_Phase/reports/active_learning_phase_boundary_report.tex"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_root / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    out = build_report(run_dir, args.template, args.output)
    print(f"Wrote report tex: {out}")


if __name__ == "__main__":
    main()
