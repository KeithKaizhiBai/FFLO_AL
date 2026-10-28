#!/usr/bin/env python3
"""Read-only descriptive summaries for the scientific-depth manuscript branch.

This script never changes labels, reconstructs boundaries, or evaluates the
BdG model.  It only summarizes already frozen exact-sample tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    fig2_path = ROOT / "figures/fig2_strict_data/canonical/fig2_final_exact_samples.csv"
    fig3_path = ROOT / "figures/fig3_strict_data/canonical/fig3_exact_topology_samples.csv"
    fig4_path = ROOT / "figures/fig4_strict_data/canonical/fig4_exact_3d_samples.csv"
    sources = [fig2_path, fig3_path, fig4_path]
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError([str(path) for path in sources if not path.is_file()])

    fig2 = pd.read_csv(fig2_path)
    bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.56]
    fig2["temperature_bin"] = pd.cut(fig2["kT"], bins, include_lowest=True)
    rows2: list[dict[str, object]] = []
    for interval, group in fig2.groupby("temperature_bin", observed=True):
        for phase, phase_group in group.groupby("phase_name", observed=True):
            rows2.append({
                "temperature_bin": str(interval),
                "phase": str(phase),
                "sample_count": int(len(phase_group)),
                "JA_min": float(phase_group["JA"].min()),
                "JA_q10": float(phase_group["JA"].quantile(0.10)),
                "JA_median": float(phase_group["JA"].median()),
                "JA_q90": float(phase_group["JA"].quantile(0.90)),
                "JA_max": float(phase_group["JA"].max()),
            })
    pd.DataFrame(rows2).to_csv(out / "fig2_exact_support_by_temperature.csv", index=False)

    fig3 = pd.read_csv(fig3_path)
    rows3: list[dict[str, object]] = []
    for topology, group in fig3[fig3["topology_name"].isin(["cFFLO", "tFFLO"])].groupby("topology_name"):
        rows3.append({
            "topology": topology,
            "sample_count": int(len(group)),
            "kT_min": float(group["kT"].min()),
            "kT_q10": float(group["kT"].quantile(0.10)),
            "kT_median": float(group["kT"].median()),
            "kT_q90": float(group["kT"].quantile(0.90)),
            "kT_max": float(group["kT"].max()),
            "JA_min": float(group["JA"].min()),
            "JA_q10": float(group["JA"].quantile(0.10)),
            "JA_median": float(group["JA"].median()),
            "JA_q90": float(group["JA"].quantile(0.90)),
            "JA_max": float(group["JA"].max()),
        })
    pd.DataFrame(rows3).to_csv(out / "fig3_topology_exact_support.csv", index=False)

    fig4 = pd.read_csv(fig4_path)
    rows4: list[dict[str, object]] = []
    centers = [-0.45, 0.15, 0.85, 1.45]
    half_width = 0.025
    for center in centers:
        selected = fig4[(fig4["mu"] - center).abs() <= half_width + 1e-12]
        for phase, group in selected.groupby("phase_name", observed=True):
            rows4.append({
                "mu_center": center,
                "slice_half_width": half_width,
                "category_type": "thermodynamic_phase",
                "category": phase,
                "sample_count": int(len(group)),
                "kT_min": float(group["kT"].min()),
                "kT_max": float(group["kT"].max()),
                "JA_min": float(group["JA"].min()),
                "JA_max": float(group["JA"].max()),
            })
        trusted = selected[selected["trusted_fflo"] == 1]
        for topology, group in trusted.groupby("topology_name", observed=True):
            rows4.append({
                "mu_center": center,
                "slice_half_width": half_width,
                "category_type": "trusted_FFLO_topology",
                "category": topology,
                "sample_count": int(len(group)),
                "kT_min": float(group["kT"].min()),
                "kT_max": float(group["kT"].max()),
                "JA_min": float(group["JA"].min()),
                "JA_max": float(group["JA"].max()),
            })
    pd.DataFrame(rows4).to_csv(out / "fig4_fixed_mu_exact_support.csv", index=False)

    manifest = {
        "schema_version": 1,
        "read_only": True,
        "physics_recomputed": False,
        "labels_changed": False,
        "boundary_reconstruction": False,
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in sources],
        "selection_rules": {
            "fig2": "group frozen exact rows by declared phase and fixed kBT bins; report observed JA support and quantiles",
            "fig3": "select topology_name in {cFFLO,tFFLO}; report exact observed coordinate support and quantiles",
            "fig4": "select abs(mu-mu_center)<=0.025 at four registered centers; group frozen exact rows by phase and trusted FFLO topology",
        },
        "computation_script": str(Path(__file__).resolve()),
        "claim_scope": "descriptive support only; not phase probability, fitted boundary, continuous surface, or causal mechanism",
        "outputs": [
            "fig2_exact_support_by_temperature.csv",
            "fig3_topology_exact_support.csv",
            "fig4_fixed_mu_exact_support.csv",
        ],
    }
    (out / "descriptive_statistics_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
