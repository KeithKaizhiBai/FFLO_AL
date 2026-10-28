from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_json(path: Path, default):
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


def main() -> None:
    p = argparse.ArgumentParser(description="Export lightweight md+csv run summary.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-root", type=Path, default=Path("ML_Phase/active_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("ML_Phase/reports"))
    args = p.parse_args()

    run_dir = args.run_root / args.run_id
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_iter(run_dir)
    if latest is None:
        raise SystemExit(f"no iteration directories found under {run_dir}")

    run_cfg = _load_json(run_dir / "run_config.json", {})
    sel = _load_json(run_dir / f"iter{latest:03d}" / "selection_diagnostics.json", {})
    merge = _load_json(run_dir / f"iter{latest:03d}" / f"merge_summary_iter{latest:03d}.json", {})
    stop = _load_json(run_dir / f"iter{latest:03d}" / f"stop_metrics_iter{latest:03d}.json", {})

    row = {
        "run_id": args.run_id,
        "latest_iteration": latest,
        "acquisition_profile": run_cfg.get("active_learning_config", {}).get("acquisition_profile", "N/A"),
        "oracle_mode": run_cfg.get("active_learning_config", {}).get("oracle_mode", "N/A"),
        "selected_boundary_band_fraction": sel.get("selected_boundary_band_fraction"),
        "selected_high_confidence_interior_fraction": sel.get("selected_high_confidence_interior_fraction"),
        "active_pool_fraction": sel.get("active_pool_fraction"),
        "N_eff_over_active_pool_size": sel.get("N_eff_over_active_pool_size"),
        "trusted_points_latest": merge.get("trusted_points"),
        "q_expanded_points_latest": merge.get("q_expanded_points"),
        "delta_refined_points_latest": merge.get("delta_refined_points"),
        "stop_reason": stop.get("stop_reason"),
    }
    df = pd.DataFrame([row])
    csv_path = args.out_dir / f"{args.run_id}_summary.csv"
    md_path = args.out_dir / f"{args.run_id}_summary.md"
    df.to_csv(csv_path, index=False)

    md = [
        f"# Run Summary: {args.run_id}",
        "",
        df.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- This is a lightweight machine-readable summary.",
        "- Full physics report should be read together with the TeX/PDF output.",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(str(csv_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
