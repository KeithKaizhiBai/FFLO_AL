from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _load_json(path: Path, default: dict | list):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _latest_iter(run_dir: Path) -> int | None:
    vals = []
    for p in run_dir.glob("iter*"):
        try:
            vals.append(int(p.name.replace("iter", "")))
        except Exception:
            continue
    return max(vals) if vals else None


def _load_selection_diag(run_dir: Path, it: int) -> dict:
    return _load_json(run_dir / f"iter{it:03d}" / "selection_diagnostics.json", default={})


def _phase_counts(run_dir: Path, it: int) -> dict[str, int]:
    ds = run_dir / f"dataset_iter{it:03d}.npz"
    if not ds.exists():
        return {"normal": 0, "uniform_sc": 0, "fflo": 0}
    raw = np.load(ds, allow_pickle=True)
    phase = np.asarray(raw.get("y_phase", []))
    if phase.size == 0:
        return {"normal": 0, "uniform_sc": 0, "fflo": 0}
    return {
        "normal": int(np.sum(phase == 0)),
        "uniform_sc": int(np.sum(phase == 1)),
        "fflo": int(np.sum(phase == 2)),
    }


def _run_summary(run_dir: Path) -> dict:
    latest = _latest_iter(run_dir)
    if latest is None:
        return {"run_dir": str(run_dir), "status": "missing"}
    sel = _load_selection_diag(run_dir, latest)
    stop_hist = _load_json(run_dir / "stop_metrics_history.json", default=[])
    merge = _load_json(run_dir / f"iter{latest:03d}" / f"merge_summary_iter{latest:03d}.json", default={})
    counts = _phase_counts(run_dir, latest + 1)
    return {
        "run_dir": str(run_dir),
        "latest_iteration": int(latest),
        "selected_boundary_band_fraction": sel.get("selected_boundary_band_fraction"),
        "selected_high_confidence_interior_fraction": sel.get("selected_high_confidence_interior_fraction"),
        "active_pool_fraction": sel.get("active_pool_fraction"),
        "neff_over_pool": sel.get("N_eff_over_active_pool_size"),
        "q_expanded_points_latest": merge.get("q_expanded_points"),
        "delta_refined_points_latest": merge.get("delta_refined_points"),
        "trusted_points_latest": merge.get("trusted_points"),
        "stop_records": len(stop_hist) if isinstance(stop_hist, list) else 0,
        **counts,
        "status": "ok",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare full vs simple acquisition run summaries.")
    p.add_argument("--full-run", type=Path, required=True)
    p.add_argument("--simple-run", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("ML_Phase/reports/acquisition_profile_comparison"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full = _run_summary(args.full_run)
    simple = _run_summary(args.simple_run)
    df = pd.DataFrame([{"profile": "full", **full}, {"profile": "simple_phase", **simple}])
    df.to_csv(args.out_dir / "summary_table.csv", index=False)

    md = [
        "# Acquisition Profile Comparison",
        "",
        "This report is a run-level diagnostic comparison only.",
        "Do not claim physical superiority from single-seed metrics.",
        "",
        "## Summary Table",
        "",
        df.to_markdown(index=False),
        "",
        "## Do-Not-Claim",
        "",
        "- Single-seed comparison is insufficient for final acquisition ranking.",
        "- Boundary-band fraction alone is not a proof of better physics.",
        "- If oracle metadata differ, verify exact-oracle behavior before attributing differences to acquisition.",
    ]
    (args.out_dir / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    tex_lines = [
        "\\documentclass{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{booktabs}",
        "\\begin{document}",
        "\\section*{Acquisition Profile Comparison}",
        "Diagnostic run-level comparison only. Single-seed results are not final evidence.",
        "\\begin{center}",
        df.to_latex(index=False, escape=True),
        "\\end{center}",
        "\\paragraph{Do-Not-Claim}",
        "Single-seed comparison is insufficient for final acquisition ranking. Boundary-band fraction alone does not prove better physics.",
        "\\end{document}",
    ]
    tex_path = args.out_dir / "comparison.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(args.out_dir), str(tex_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    (args.out_dir / "decision_log.md").write_text(
        "# Decision Log\n\n- Output generated from existing run artifacts.\n- No acquisition, oracle, or dataset mutation performed.\n",
        encoding="utf-8",
    )
    print(str(args.out_dir / "summary_table.csv"))
    print(str(args.out_dir / "comparison.md"))
    print(str(args.out_dir / "comparison.tex"))


if __name__ == "__main__":
    main()
