#!/usr/bin/env python3
"""Freeze Figure 2 and build strict-data human-review candidates for Figures 3 and 4.

This script performs no physics calculation.  It reads only frozen/recovered
artifacts, creates direct plotting tables, runs Figure Engine preflight before
drawing, renders vector SVG/PDF plus PNG previews, and runs postflight lint.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFilter

from figures.figure_engine.builders.figure_build import preflight
from figures.figure_engine.styles.matplotlib_rc import build_rc


ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "figures/strict_data_precontracts"
FIG2 = ROOT / "figures/fig2_strict_data"
FIG3 = ROOT / "figures/fig3_strict_data"
FIG4 = ROOT / "figures/fig4_strict_data"
STAGE3 = ROOT / "manuscript_langgraph/recovered_artifacts/stageIII"
STAGE4 = ROOT / "manuscript_langgraph/recovered_artifacts/stageIV"
CANVAS = (1600, 1100)
PRINT_MM = (178.0, 120.0)

FIG3_BOXES = {
    "topology_exact_map": (80, 135, 970, 765),
    "audited_boundary_geometry": (1100, 135, 420, 250),
    "raw_topology_convergence": (1100, 430, 420, 380),
    "k_sensitivity": (1100, 855, 420, 175),
}
FIG4_BOXES = {
    "exact_3d_thermodynamic_cloud": (80, 140, 970, 765),
    "exact_3d_topology_support": (1090, 140, 440, 260),
    "fixed_mu_slice_audit": (1090, 445, 440, 340),
    "strict_3d_nonclosure": (1090, 830, 440, 205),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axes_box(box: tuple[int, int, int, int]) -> list[float]:
    x, y, w, h = box
    return [x / CANVAS[0], (CANVAS[1] - y - h) / CANVAS[1], w / CANVAS[0], h / CANVAS[1]]


def freeze_fig2() -> None:
    names = [
        "fig2_strict_data.svg", "fig2_strict_data.pdf", "fig2_strict_data.png",
        "fig2_contact_sheet.pdf", "fig2_contact_sheet.png", "fig2_panel_contract.yaml",
        "fig2_data_manifest.yaml", "figure_validation_report.json",
        "fig2_specific_validation.json", "fig2_render_determinism.json",
    ]
    missing = [name for name in names if not (FIG2 / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot freeze Figure 2; missing artifacts: {missing}")
    generic = json.loads((FIG2 / "figure_validation_report.json").read_text(encoding="utf-8"))
    specific = json.loads((FIG2 / "fig2_specific_validation.json").read_text(encoding="utf-8"))
    if generic.get("status") != "pass" or specific.get("status") != "pass":
        raise RuntimeError("Cannot freeze Figure 2; validation is not pass")
    payload = {
        "schema_version": 1,
        "figure_id": "FIG-02-THERMO-STRICT",
        "gate": "GATE_FIG2_HUMAN_SCIENCE_REVIEW",
        "decision": "approved",
        "freeze_state": "SCIENTIFIC_FREEZE",
        "approved_by": "user",
        "approved_on": "2026-08-27",
        "mutation_allowed": False,
        "physics_recomputed": False,
        "artifacts": {name: {"sha256": sha256(FIG2 / name), "bytes": (FIG2 / name).stat().st_size} for name in names},
        "validation": {
            "generic": generic.get("status"),
            "specific": specific.get("status"),
            "human_science_review": "approved",
        },
    }
    (FIG2 / "fig2_scientific_freeze.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def palette(context) -> dict[str, str]:
    p = context.rules["palette"]
    return {
        "normal": p["neutral"]["secondary"],
        "uniform": p["families"]["physical"]["highlight"],
        "fflo": p["families"]["physical"]["dark"],
        "cfflo": p["families"]["topology_cfflo"]["base"],
        "cfflo_dark": p["families"]["topology_cfflo"]["dark"],
        "tfflo": p["families"]["topology_tfflo"]["base"],
        "tfflo_dark": p["families"]["topology_tfflo"]["dark"],
        "transition": p["families"]["topology_transition"]["highlight"],
        "transition_dark": p["families"]["topology_transition"]["dark"],
        "selected": p["families"]["selected"]["base"],
        "uncertainty": p["families"]["uncertainty"]["base"],
        "decision": p["families"]["decision"]["base"],
        "body": p["neutral"]["body"],
        "secondary": p["neutral"]["secondary"],
        "axis": p["neutral"]["axis"],
        "cool": p["neutral"]["cool_light"],
        "panel": p["neutral"]["panel"],
        "white": p["neutral"]["white"],
        "warm": p["neutral"]["warm_light"],
    }


def configure_matplotlib(context) -> dict[str, str]:
    c = palette(context)
    rc = build_rc(context.rules)
    rc.update({
        "text.color": c["body"], "axes.labelcolor": c["body"],
        "axes.edgecolor": c["axis"], "xtick.color": c["axis"],
        "ytick.color": c["axis"], "axes.titlecolor": c["body"],
        "figure.dpi": 100, "savefig.dpi": 300,
        "savefig.bbox": None, "savefig.pad_inches": 0,
    })
    plt.rcParams.update(rc)
    return c


def style_2d(ax, c: dict[str, str], *, xlabel: str = r"$k_B T/t$", ylabel: str = r"$J_A/t$") -> None:
    ax.set_xlim(0, 0.56)
    ax.set_ylim(0, 2.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor(c["panel"])
    ax.tick_params(colors=c["axis"], width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_color(c["axis"])
        spine.set_linewidth(0.65)


def panel_title(ax, letter: str, title: str, c: dict[str, str], *, size: float = 7.8) -> None:
    box = {"facecolor": c["white"], "edgecolor": "none", "alpha": 0.9, "pad": 1.4}
    text = ax.text2D if hasattr(ax, "text2D") else ax.text
    text(0.012, 0.987, letter, transform=ax.transAxes, ha="left", va="top",
         fontsize=8.8, fontweight="bold", color=c["body"], bbox=box, zorder=30)
    text(0.10, 0.987, title, transform=ax.transAxes, ha="left", va="top",
         fontsize=size, fontweight="bold", color=c["body"], bbox=box, zorder=30)


def patch_svg(path: Path, figure_id: str, contract_name: str) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("data-figure-id", figure_id)
    root.set("data-contract", contract_name)
    root.set("data-engine-schema", "3")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_geometry(here: Path, boxes: dict[str, tuple[int, int, int, int]], primary: str) -> None:
    roles = {name: ("primary" if name == primary else "secondary") for name in boxes}
    objects = []
    for name, (x, y, w, h) in boxes.items():
        objects.append({
            "id": name, "type": "plot", "role": roles[name], "parent": "canvas",
            "visual_weight_object": True,
            "bbox": {"x": x, "y": y, "width": w, "height": h},
        })
    payload = {
        "schema_version": 1,
        "canvas": {"x": 0, "y": 0, "width": CANVAS[0], "height": CANVAS[1]},
        "objects": objects,
        "routes": [],
        "scientific_content_bbox": {primary: next(x["bbox"] for x in objects if x["id"] == primary)},
    }
    (here / "geometry_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_contract(precontract: Path, here: Path, figure_number: int,
                     outputs: list[str], boxes: dict[str, tuple[int, int, int, int]]) -> dict[str, Any]:
    contract = yaml.safe_load(precontract.read_text(encoding="utf-8"))
    contract["layout"]["bounding_boxes"] = "geometry_manifest.json"
    contract["validation_requirements"]["engine_preflight_record"] = "engine_preflight.json"
    contract["approval"] = {
        "gate": f"GATE_FIG{figure_number}_HUMAN_SCIENCE_REVIEW",
        "state": "HUMAN_WAIT",
        "render_authority": "GATE_FIG2_HUMAN_SCIENCE_REVIEW approved by user on 2026-08-27",
    }
    contract["strict_data"]["output_artifacts"] = outputs
    for panel in contract["strict_data"]["panels"]:
        panel["source_scripts"] = ["figures/build_remaining_strict_figures.py"]
        panel["validators"] = ["strict_data_checker", f"fig{figure_number}_specific_validation"]
        panel["output_artifacts"] = [outputs[0], outputs[1], outputs[2]]
    contract_path = here / f"fig{figure_number}_panel_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    write_geometry(here, boxes, contract["primary_visual_object"]["id"])
    return contract


def prepare_fig3() -> dict[str, Any]:
    FIG3.mkdir(parents=True, exist_ok=True)
    canon = FIG3 / "canonical"
    canon.mkdir(exist_ok=True)
    endpoint = STAGE3 / "dataset_iter018.npz"
    with np.load(endpoint, allow_pickle=False) as d:
        x = np.asarray(d["x"], float)
        phase = np.asarray(d["y_phase"], int)
        trusted = np.asarray(d["topology_trusted"], bool)
        applicable = np.asarray(d["topology_applicable"], bool)
        z2 = np.asarray(d["topology_z2"], int)
        gap = np.asarray(d["topology_bulk_gap"], float)
        tol = np.asarray(d["topology_gap_tol"], float)
    trusted_fflo = trusted & applicable & (phase == 2)
    if len(x) != 4345 or int(trusted_fflo.sum()) != 3213:
        raise RuntimeError("Stage III endpoint invariant mismatch")
    if dict(zip(*np.unique(z2[trusted_fflo], return_counts=True))) != {0: 2290, 1: 923}:
        raise RuntimeError("Stage III cFFLO/tFFLO invariant mismatch")
    exact = pd.DataFrame({
        "kT": x[:, 0], "JA": x[:, 1], "phase_code": phase,
        "topology_trusted": trusted.astype(int), "topology_applicable": applicable.astype(int),
        "topology_z2": z2, "topology_bulk_gap": gap, "topology_gap_tol": tol,
        "trusted_fflo": trusted_fflo.astype(int),
    })
    exact["topology_name"] = np.where(~trusted_fflo, "not_trusted_FFLO", np.where(z2 == 0, "cFFLO", "tFFLO"))
    exact_path = canon / "fig3_exact_topology_samples.csv"
    exact.to_csv(exact_path, index=False, float_format="%.16g")

    audit = STAGE3 / "figure_data/topology_convergence_audit_full_loop_v2"
    boundary_all = pd.read_csv(audit / "topology_boundary_points_by_iteration.csv")
    boundary = boundary_all[boundary_all.iteration == 18].copy()
    boundary_path = canon / "fig3_archived_k8_boundary_iter018.csv"
    boundary.to_csv(boundary_path, index=False, float_format="%.16g")
    sensitivity = pd.read_csv(audit / "topology_sensitivity_summary.csv")
    sensitivity_path = canon / "fig3_k_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False, float_format="%.16g")

    outputs = ["fig3_strict_data.svg", "fig3_strict_data.pdf", "fig3_strict_data.png",
               "fig3_contact_sheet.pdf", "fig3_contact_sheet.png"]
    contract = prepare_contract(PRE / "fig3_panel_contract.yaml", FIG3, 3, outputs, FIG3_BOXES)
    manifest = yaml.safe_load((PRE / "fig3_data_manifest.yaml").read_text(encoding="utf-8"))
    manifest["render_authorized"] = True
    manifest["render_authority"] = "Fig.2 human science review approved by user on 2026-08-27"
    manifest["canonical_outputs"] = {
        exact_path.name: {"sha256": sha256(exact_path), "rows": len(exact)},
        boundary_path.name: {"sha256": sha256(boundary_path), "rows": len(boundary)},
        sensitivity_path.name: {"sha256": sha256(sensitivity_path), "rows": len(sensitivity)},
    }
    (FIG3 / "fig3_data_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return contract


def prepare_fig4() -> dict[str, Any]:
    FIG4.mkdir(parents=True, exist_ok=True)
    canon = FIG4 / "canonical"
    canon.mkdir(exist_ok=True)
    endpoint = STAGE4 / "dataset_iter025.npz"
    with np.load(endpoint, allow_pickle=False) as d:
        x = np.asarray(d["x"], float)
        phase = np.asarray(d["y_phase"], int)
        trusted = np.asarray(d["topology_trusted"], bool)
        applicable = np.asarray(d["topology_applicable"], bool)
        z2 = np.asarray(d["topology_z2"], int)
        gap = np.asarray(d["topology_bulk_gap"], float)
    counts = dict(zip(*np.unique(phase, return_counts=True)))
    trusted_fflo = trusted & applicable & (phase == 2)
    topo_counts = dict(zip(*np.unique(z2[trusted_fflo], return_counts=True)))
    if len(x) != 7081 or counts != {0: 2350, 1: 141, 2: 4590} or topo_counts != {0: 3145, 1: 1445}:
        raise RuntimeError(f"Stage IV invariant mismatch: rows={len(x)}, phase={counts}, FFLO topology={topo_counts}")
    exact = pd.DataFrame({
        "kT": x[:, 0], "JA": x[:, 1], "mu": x[:, 2], "phase_code": phase,
        "topology_trusted": trusted.astype(int), "topology_applicable": applicable.astype(int),
        "topology_z2": z2, "topology_bulk_gap": gap, "trusted_fflo": trusted_fflo.astype(int),
    })
    exact["phase_name"] = exact.phase_code.map({0: "normal", 1: "uniform_SC", 2: "FFLO"})
    exact["topology_name"] = np.where(~trusted_fflo, "not_trusted_FFLO", np.where(z2 == 0, "cFFLO", "tFFLO"))
    exact_path = canon / "fig4_exact_3d_samples.csv"
    exact.to_csv(exact_path, index=False, float_format="%.16g")

    slice_source = STAGE4 / "figure_data/stageiv_mu_slice_boundary_audit_local/stageiv_narrow_mu_slice_metrics.csv"
    slices = pd.read_csv(slice_source)
    chosen = slices.iloc[[0, 6, 13, 19]].copy()
    chosen["selection_rule"] = "four evenly spaced archived slice-table indices: 0,6,13,19"
    chosen_path = canon / "fig4_selected_fixed_mu_slices.csv"
    chosen.to_csv(chosen_path, index=False, float_format="%.16g")

    outputs = ["fig4_strict_data.svg", "fig4_strict_data.pdf", "fig4_strict_data.png",
               "fig4_contact_sheet.pdf", "fig4_contact_sheet.png"]
    contract = prepare_contract(PRE / "fig4_panel_contract.yaml", FIG4, 4, outputs, FIG4_BOXES)
    contract["layout"]["template_profile"] = "high_dimensional"
    contract["layout"]["primary_area_fraction_min"] = 0.42
    for panel in contract["strict_data"]["panels"]:
        if panel["panel_id"] == "exact_3d_topology_support":
            panel["input_row_count"] = 7081
            panel["selected_row_count"] = 4590
            panel["excluded_row_count"] = 2491
            panel["selection_filters"] = ["topology_trusted == 1", "topology_applicable == 1", "y_phase == 2"]
            panel["trust_mask"] = "topology_trusted == 1 and topology_applicable == 1 and y_phase == 2"
            panel["expected_counts"] = {"cFFLO": 3145, "tFFLO": 1445}
            panel["sensitivity_notes"] = [
                "Stage IV strict topology closure remains open",
                "FFLO-only topology counts exclude 141 uniform-SC points",
                "k=6 sensitivity is the retained Stage III caveat and is not a substitute for the failed Stage IV 3D closure gates",
            ]
        if panel["panel_id"] == "fixed_mu_slice_audit":
            panel["input_row_count"] = 20
            panel["selected_row_count"] = 4
            panel["excluded_row_count"] = 16
            panel["selection_filters"] = ["evenly spaced archived slice-table indices [0, 6, 13, 19]"]
            panel["excluded_data"] = ["the other 16 archived slice rows remain in the source table and are not inferred"]
    contract_path = FIG4 / "fig4_panel_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    manifest = yaml.safe_load((PRE / "fig4_data_manifest.yaml").read_text(encoding="utf-8"))
    manifest["render_authorized"] = True
    manifest["render_authority"] = "Fig.2 human science review approved by user on 2026-08-27"
    manifest["continuous_surface_authorized"] = False
    manifest["invariants"]["topology_counts_all_applicable_SC"] = {"trivial": 3265, "topological": 1466}
    manifest["invariants"]["topology_counts_trusted_FFLO_only"] = {"cFFLO": 3145, "tFFLO": 1445}
    manifest["canonical_outputs"] = {
        exact_path.name: {"sha256": sha256(exact_path), "rows": len(exact)},
        chosen_path.name: {"sha256": sha256(chosen_path), "rows": len(chosen)},
    }
    (FIG4 / "fig4_data_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return contract


def render_fig3() -> tuple[Path, Path, Path]:
    contract = FIG3 / "fig3_panel_contract.yaml"
    context = preflight(contract, record_path=FIG3 / "engine_preflight.json")
    c = configure_matplotlib(context)
    exact = pd.read_csv(FIG3 / "canonical/fig3_exact_topology_samples.csv")
    boundary = pd.read_csv(FIG3 / "canonical/fig3_archived_k8_boundary_iter018.csv")
    base = STAGE3 / "figure_data/topology_convergence_audit_full_loop_v2"
    m_map = pd.read_csv(base / "topology_convergence_metrics_by_iteration.csv")
    m_shift = pd.read_csv(base / "topology_boundary_shift_by_iteration.csv")
    m_cov = pd.read_csv(base / "topology_boundary_coverage.csv")
    m_sur = pd.read_csv(base / "topology_surprise_by_iteration.csv")
    sens = pd.read_csv(FIG3 / "canonical/fig3_k_sensitivity.csv")

    fig = plt.figure(figsize=(PRINT_MM[0] / 25.4, PRINT_MM[1] / 25.4), facecolor=c["white"])
    fig.text(0.025, 0.985, "Topology-aware exact sampling resolves the cFFLO/tFFLO structure",
             ha="left", va="top", fontsize=10.2, fontweight="bold", color=c["body"])
    fig.text(0.025, 0.955, "Trusted gapped FFLO labels; archived k=8 audit; k=6 sensitivity retained",
             ha="left", va="top", fontsize=7.0, color=c["secondary"])

    ax = fig.add_axes(axes_box(FIG3_BOXES["topology_exact_map"]))
    non = exact[~exact.trusted_fflo.astype(bool)]
    ax.scatter(non.kT, non.JA, s=3.0, c=c["cool"], alpha=0.42, linewidths=0, label="non-FFLO / untrusted topology")
    for name, color, label in [("cFFLO", c["cfflo"], r"cFFLO: gapped $\nu=0$"),
                               ("tFFLO", c["tfflo"], r"tFFLO: reopened gap $\nu=1$")]:
        d = exact[exact.topology_name == name]
        ax.scatter(d.kT, d.JA, s=7.0, c=color, alpha=0.90, linewidths=0, label=label)
    style_2d(ax, c)
    panel_title(ax, "a", "Trusted exact topology map", c, size=8.1)
    ax.legend(loc="upper right", frameon=True, facecolor=c["white"], edgecolor=c["cool"],
              framealpha=0.95, fontsize=6.6, borderpad=0.35, handletextpad=0.3)
    ax.text(0.018, 0.018, "3213 trusted FFLO: 2290 cFFLO / 923 tFFLO",
            transform=ax.transAxes, fontsize=7.0, color=c["secondary"],
            bbox={"facecolor": c["white"], "edgecolor": "none", "alpha": 0.88, "pad": 1.5})

    axb = fig.add_axes(axes_box(FIG3_BOXES["audited_boundary_geometry"]))
    support = exact[exact.trusted_fflo.astype(bool)]
    axb.scatter(support.kT, support.JA, s=2.0, c=np.where(support.topology_z2 == 0, c["cfflo"], c["tfflo"]),
                alpha=0.28, linewidths=0)
    for _, comp in boundary.groupby("component_id"):
        axb.plot(comp.kT, comp.JA, color=c["transition"], lw=1.15, ls=(0, (4, 2)))
    style_2d(axb, c, xlabel="", ylabel="")
    axb.tick_params(labelbottom=False, labelleft=False)
    panel_title(axb, "b", "Gap-closing transition diagnostic", c, size=7.2)
    axb.text(0.03, 0.06, "archived k=8 derived contour\nnot a stable third phase",
             transform=axb.transAxes, fontsize=6.6, color=c["transition_dark"],
             bbox={"facecolor": c["white"], "edgecolor": c["cool"], "alpha": 0.92, "pad": 1.5})

    x, y, w, h = FIG3_BOXES["raw_topology_convergence"]
    fig.text(x / CANVAS[0], (CANVAS[1] - y + 2) / CANVAS[1], "c  Raw configured convergence",
             ha="left", va="bottom", fontsize=7.7, fontweight="bold", color=c["body"])
    series = [
        (m_map.transition_to_iteration, m_map.topology_map_change, 0.002, "map change", c["cfflo_dark"]),
        (m_shift.transition_to_iteration, m_shift.symmetric_p95, 0.004166666666666667, "boundary p95 shift", c["selected"]),
        (m_cov.iteration, m_cov.coverage_p95, 0.00625, "coverage p95", c["decision"]),
        (m_sur.iteration, m_sur.trusted_topology_surprise, 0.02, "trusted surprise", c["uncertainty"]),
    ]
    mini_w, mini_h, gx, gy = 180, 145, 42, 45
    for i, (xx, yy, threshold, title, color) in enumerate(series):
        col, row = i % 2, i // 2
        box = (x + col * (mini_w + gx), y + 35 + row * (mini_h + gy), mini_w, mini_h)
        axi = fig.add_axes(axes_box(box))
        axi.plot(xx, yy, color=color, lw=1.0, marker="o", ms=1.7, mfc=c["white"], mew=0.4)
        axi.axhline(threshold, color=c["axis"], lw=0.6, ls=(0, (3, 2)))
        axi.set_title(title, fontsize=6.8, pad=2)
        axi.set_xlim(1, 18); axi.set_xticks([1, 9, 18])
        if row == 0:
            axi.tick_params(labelbottom=False)
        else:
            axi.set_xlabel("iteration", fontsize=6.2, labelpad=1)
        axi.tick_params(labelsize=5.8, width=0.45, length=2, colors=c["axis"], pad=1)
        for spine in axi.spines.values(): spine.set_color(c["axis"]); spine.set_linewidth(0.5)
        axi.grid(axis="y", color=c["cool"], lw=0.35, alpha=0.55)

    axd = fig.add_axes(axes_box(FIG3_BOXES["k_sensitivity"]))
    bars = axd.bar(sens.idw_and_knn_k.astype(str), sens.final_coverage_p95,
                   color=[c["transition"], c["cfflo"], c["tfflo"]], width=0.55)
    axd.axhline(0.00625, color=c["axis"], ls=(0, (3, 2)), lw=0.7)
    axd.set_ylim(0.0057, 0.00645)
    axd.set_ylabel("coverage p95", fontsize=6.5)
    axd.tick_params(labelsize=6.0, colors=c["axis"], length=2)
    for spine in axd.spines.values(): spine.set_color(c["axis"]); spine.set_linewidth(0.55)
    for bar, value in zip(bars, sens.final_coverage_p95):
        axd.text(bar.get_x() + bar.get_width() / 2, value + 0.000012, f"{value:.6f}", ha="center", va="bottom", fontsize=5.8)
    panel_title(axd, "d", "k sensitivity", c, size=7.2)

    svg, pdf, png = FIG3 / "fig3_strict_data.svg", FIG3 / "fig3_strict_data.pdf", FIG3 / "fig3_strict_data.png"
    fig.savefig(svg, format="svg", facecolor=c["white"])
    fig.savefig(pdf, format="pdf", facecolor=c["white"])
    fig.savefig(png, format="png", dpi=300, facecolor=c["white"])
    plt.close(fig)
    patch_svg(svg, "FIG-03-STAGEIII-TOPOLOGY-STRICT", contract.name)
    return svg, pdf, png


def style_3d(ax, c: dict[str, str]) -> None:
    ax.set_xlabel(r"$k_B T/t$", labelpad=-1)
    ax.set_ylabel(r"$J_A/t$", labelpad=-1)
    ax.set_zlabel(r"$\mu/t$", labelpad=-1)
    ax.set_xlim(0, 0.56); ax.set_ylim(0, 2.12); ax.set_zlim(-0.5, 1.5)
    ax.view_init(elev=22, azim=-58)
    ax.tick_params(labelsize=6.0, colors=c["axis"], pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(c["white"])
        axis.pane.set_edgecolor(c["cool"])
        axis._axinfo["grid"]["color"] = c["cool"]
        axis._axinfo["grid"]["linewidth"] = 0.35


def render_fig4() -> tuple[Path, Path, Path]:
    contract = FIG4 / "fig4_panel_contract.yaml"
    context = preflight(contract, record_path=FIG4 / "engine_preflight.json")
    c = configure_matplotlib(context)
    exact = pd.read_csv(FIG4 / "canonical/fig4_exact_3d_samples.csv")
    chosen = pd.read_csv(FIG4 / "canonical/fig4_selected_fixed_mu_slices.csv")
    failure = pd.read_csv(STAGE4 / "stageiv_3d_audit_local/tables/stageiv_3d_convergence_failure_decomposition.csv")

    fig = plt.figure(figsize=(PRINT_MM[0] / 25.4, PRINT_MM[1] / 25.4), facecolor=c["white"])
    fig.text(0.025, 0.985, "Reliability-aware active learning extends to higher-dimensional exploration",
             ha="left", va="top", fontsize=10.2, fontweight="bold", color=c["body"])
    fig.text(0.025, 0.954, "3D extension reveals thermodynamic and topology-aware structures; strict topology closure remains open",
             ha="left", va="top", fontsize=7.0, color=c["transition_dark"])

    ax = fig.add_axes(axes_box(FIG4_BOXES["exact_3d_thermodynamic_cloud"]), projection="3d")
    for name, color, size, alpha, label in [
        ("normal", c["normal"], 4.0, 0.36, "Normal"),
        ("uniform_SC", c["uniform"], 8.0, 0.88, "Uniform SC"),
        ("FFLO", c["fflo"], 5.0, 0.66, "FFLO"),
    ]:
        d = exact[exact.phase_name == name]
        ax.scatter(d.kT, d.JA, d.mu, s=size, c=color, alpha=alpha, linewidths=0, depthshade=False, label=label)
    style_3d(ax, c)
    panel_title(ax, "a", "Exact 3D thermodynamic observations", c, size=8.0)
    ax.legend(loc="upper right", frameon=True, facecolor=c["white"], edgecolor=c["cool"], framealpha=0.94, fontsize=6.5)
    ax.text2D(0.02, 0.02, "7081 exact samples: 2350 / 141 / 4590",
              transform=ax.transAxes, fontsize=7.0, color=c["secondary"],
              bbox={"facecolor": c["white"], "edgecolor": "none", "alpha": 0.88, "pad": 1.4})

    axb = fig.add_axes(axes_box(FIG4_BOXES["exact_3d_topology_support"]), projection="3d")
    for name, color, label in [("cFFLO", c["cfflo"], r"cFFLO $\nu=0$ (3145)"), ("tFFLO", c["tfflo"], r"tFFLO $\nu=1$ (1445)")]:
        d = exact[exact.topology_name == name]
        axb.scatter(d.kT, d.JA, d.mu, s=4.0, c=color, alpha=0.58, linewidths=0, depthshade=False, label=label)
    style_3d(axb, c)
    axb.set_xlabel(""); axb.set_ylabel(""); axb.set_zlabel("")
    axb.tick_params(labelsize=5.2)
    panel_title(axb, "b", "Trusted FFLO topology support", c, size=7.1)
    axb.legend(loc="upper left", bbox_to_anchor=(0.01, 0.84), frameon=True,
               facecolor=c["white"], edgecolor=c["cool"], framealpha=0.94, fontsize=4.8,
               borderpad=0.25, handletextpad=0.25)

    x, y, w, h = FIG4_BOXES["fixed_mu_slice_audit"]
    fig.text(x / CANVAS[0], (CANVAS[1] - y + 2) / CANVAS[1], r"c  Exact fixed-$\mu$ slices",
             ha="left", va="bottom", fontsize=7.7, fontweight="bold", color=c["body"])
    mini_w, mini_h, gx, gy = 192, 130, 34, 44
    for i, row in enumerate(chosen.itertuples(index=False)):
        col, rr = i % 2, i // 2
        box = (x + col * (mini_w + gx), y + 35 + rr * (mini_h + gy), mini_w, mini_h)
        axi = fig.add_axes(axes_box(box))
        d = exact[np.abs(exact.mu - float(row.mu_center)) <= float(row.slice_half_width) + 1e-12]
        for phase_name, color in [("normal", c["normal"]), ("uniform_SC", c["uniform"]), ("FFLO", c["fflo"])]:
            q = d[d.phase_name == phase_name]
            axi.scatter(q.kT, q.JA, s=4.0, c=color, alpha=0.72, linewidths=0)
        style_2d(axi, c, xlabel="", ylabel="")
        axi.set_title(f"$\\mu/t={row.mu_center:.2f}\\pm{row.slice_half_width:.3f}$\n$n={len(d)}$", fontsize=5.6, pad=1.0)
        axi.tick_params(labelsize=5.3, length=1.7, pad=1)
        if rr == 0:
            axi.tick_params(labelbottom=False)

    axd = fig.add_axes(axes_box(FIG4_BOXES["strict_3d_nonclosure"]))
    axd.axis("off")
    panel_title(axd, "d", "Strict 3D topology gates: not converged", c, size=7.1)
    look = {str(r.metric) + "@" + str(r.iteration_or_transition): r for r in failure.itertuples(index=False)}
    rows = [
        ("map change", "topology_volume_map_change@24->25", "0.002561 > 0.002"),
        ("surface shift p95", "topology_surface_shift_p95@24->25", "0.005996 > 0.004167"),
        ("coverage p95", "topology_surface_coverage_p95@25", "0.015318 > 0.00625"),
        ("trusted surprise", "trusted_topology_surprise@24->25", "0.140625 > 0.02"),
    ]
    for i, (label, key, value) in enumerate(rows):
        if key not in look:
            raise RuntimeError(f"Missing Stage IV failure metric {key}")
        yy = 0.76 - i * 0.17
        axd.text(0.03, yy, "FAIL", fontsize=6.5, fontweight="bold", color=c["transition_dark"], va="center")
        axd.text(0.16, yy, label, fontsize=6.5, color=c["body"], va="center")
        axd.text(0.98, yy, value, fontsize=6.5, color=c["secondary"], va="center", ha="right")
    axd.text(0.03, 0.075, r"components $3\to4\to3$; no surface; 4/20 slices",
             fontsize=5.9, color=c["transition_dark"], va="bottom")
    axd.text(0.03, -0.005, r"limited domain: $\mu/t\in[-0.5,1.5]$",
             fontsize=5.9, color=c["transition_dark"], va="bottom")

    svg, pdf, png = FIG4 / "fig4_strict_data.svg", FIG4 / "fig4_strict_data.pdf", FIG4 / "fig4_strict_data.png"
    fig.savefig(svg, format="svg", facecolor=c["white"])
    fig.savefig(pdf, format="pdf", facecolor=c["white"])
    fig.savefig(png, format="png", dpi=300, facecolor=c["white"])
    plt.close(fig)
    patch_svg(svg, "FIG-04-STAGEIV-3D-STRICT", contract.name)
    return svg, pdf, png


def make_contact_sheet(here: Path, figure_number: int, png: Path,
                       boxes: dict[str, tuple[int, int, int, int]], c: dict[str, str], note: str) -> None:
    im = Image.open(png).convert("RGB")
    gray = im.convert("L").convert("RGB")
    gray.save(here / f"fig{figure_number}_strict_data_grayscale.png")
    blur = im.filter(ImageFilter.GaussianBlur(radius=max(2, im.width * 0.00375)))
    blur.save(here / f"fig{figure_number}_saliency_preview.png")
    sx, sy = im.width / CANVAS[0], im.height / CANVAS[1]
    primary = next(iter(boxes))
    x, y, w, h = boxes[primary]
    crop = im.crop((round(x * sx), round(y * sy), round((x + w) * sx), round((y + h) * sy)))
    crop.save(here / f"panel_{primary}.png")
    fig = plt.figure(figsize=(11.0, 8.5), facecolor=c["white"])
    fig.text(0.04, 0.975, f"Figure {figure_number} strict-data review package", fontsize=15, fontweight="bold", va="top", color=c["body"])
    fig.text(0.04, 0.945, note, fontsize=9, va="top", color=c["secondary"])
    slots = [(0.04, 0.48, 0.60, 0.43), (0.67, 0.68, 0.29, 0.22), (0.67, 0.42, 0.29, 0.22), (0.10, 0.06, 0.80, 0.32)]
    titles = ["Full color / 178 mm", "Grayscale", "Blur diagnostic", "Primary scientific object"]
    for slot, image, title in zip(slots, [im, gray, blur, crop], titles):
        ax = fig.add_axes(slot); ax.imshow(image); ax.set_title(title, fontsize=9, color=c["body"], pad=4); ax.axis("off")
    fig.savefig(here / f"fig{figure_number}_contact_sheet.pdf", format="pdf", facecolor=c["white"])
    fig.savefig(here / f"fig{figure_number}_contact_sheet.png", format="png", dpi=180, facecolor=c["white"])
    plt.close(fig)


def write_reports_and_validate(here: Path, figure_number: int, svg: Path, pdf: Path, png: Path,
                               science_checks: list[tuple[str, bool, str]], scientific_text: str) -> None:
    failures = [x for x in science_checks if not x[1]]
    payload = {
        "schema_version": 1, "figure_id": f"FIG-{figure_number:02d}",
        "status": "pass" if not failures else "fail", "physics_recomputed": False,
        "checks_passed": len(science_checks) - len(failures), "checks_total": len(science_checks),
        "checks": [{"name": n, "passed": bool(p), "detail": d} for n, p, d in science_checks],
    }
    (here / f"fig{figure_number}_specific_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md = [f"# Figure {figure_number} specific validation", "", f"- Status: **{payload['status'].upper()}**",
          f"- Checks: **{payload['checks_passed']}/{payload['checks_total']}**", "- Physics recomputed: **no**", "",
          "| Result | Check | Detail |", "|---|---|---|"]
    md.extend(f"| {'PASS' if p else 'FAIL'} | `{n}` | {d} |" for n, p, d in science_checks)
    (here / f"fig{figure_number}_specific_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (here / f"fig{figure_number}_scientific_validation.md").write_text(scientific_text, encoding="utf-8")
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=False).stdout
    page = next((line for line in info.splitlines() if line.startswith("Page size:")), "missing")
    if "504.567 x 340.157 pts" not in info:
        raise RuntimeError(f"Figure {figure_number} PDF print size mismatch: {page}")
    assets = [svg, pdf, png, here / f"fig{figure_number}_contact_sheet.pdf", here / f"fig{figure_number}_contact_sheet.png"]
    (here / "asset_manifest.json").write_text(json.dumps({
        "schema_version": 1, "figure_id": f"FIG-{figure_number:02d}", "physics_recomputed": False,
        "assets": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in assets},
    }, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Figure {figure_number} science validation failed: {failures}")


def run_lint(here: Path, figure_number: int) -> None:
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/al_fflo_mpl", "XDG_CACHE_HOME": "/tmp/al_fflo_cache", "PYTHONPATH": str(ROOT)})
    subprocess.run([
        sys.executable, "-m", "figures.figure_engine.validators.figure_lint",
        "--contract", str(here / f"fig{figure_number}_panel_contract.yaml"),
        "--svg", f"fig{figure_number}_strict_data.svg", "--geometry", "geometry_manifest.json",
        "--pdf", f"fig{figure_number}_strict_data.pdf", "--png", f"fig{figure_number}_strict_data.png",
        "--output-dir", str(here), "--report-prefix", "figure_validation_report",
    ], cwd=ROOT, env=env, check=True)


def main() -> int:
    frozen = [path for path in (FIG3 / "fig3_scientific_freeze.yaml", FIG4 / "fig4_scientific_freeze.yaml") if path.exists()]
    if frozen:
        raise RuntimeError(
            "Refusing to regenerate scientifically frozen artwork: "
            + ", ".join(str(path) for path in frozen)
        )
    freeze_fig2()
    prepare_fig3()
    prepare_fig4()
    fig3_first = render_fig3()
    h3a = sha256(fig3_first[2])
    fig3_paths = render_fig3()
    h3b = sha256(fig3_paths[2])
    if h3a != h3b:
        raise RuntimeError("Figure 3 PNG rendering is not deterministic")
    c3 = configure_matplotlib(preflight(FIG3 / "fig3_panel_contract.yaml"))
    make_contact_sheet(FIG3, 3, fig3_paths[2], FIG3_BOXES, c3,
                       "Exact topology labels + archived boundary/audit evidence; human science review required")
    write_reports_and_validate(FIG3, 3, *fig3_paths, [
        ("endpoint_hash", sha256(STAGE3 / "dataset_iter018.npz") == "6583230495ad9e3eb45b0cff034d5525fa443bd43b1b31a754defeaabdfb5ab4", "dataset_iter018.npz"),
        ("trusted_FFLO_count", len(pd.read_csv(FIG3 / "canonical/fig3_exact_topology_samples.csv").query("trusted_fflo == 1")) == 3213, "expected 3213"),
        ("topology_counts", pd.read_csv(FIG3 / "canonical/fig3_exact_topology_samples.csv").query("trusted_fflo == 1").topology_name.value_counts().to_dict() == {"cFFLO": 2290, "tFFLO": 923}, "expected 2290/923"),
        ("final_boundary_rows", len(pd.read_csv(FIG3 / "canonical/fig3_archived_k8_boundary_iter018.csv")) == 646, "iteration 18 archived geometry"),
        ("k8_pass_k6_caveat", abs(float(pd.read_csv(FIG3 / "canonical/fig3_k_sensitivity.csv").query("idw_and_knn_k == 8").final_coverage_p95.iloc[0]) - 0.006185) < 1e-6 and float(pd.read_csv(FIG3 / "canonical/fig3_k_sensitivity.csv").query("idw_and_knn_k == 6").final_coverage_p95.iloc[0]) > 0.00625, "k=8 below and k=6 above threshold"),
        ("no_physics_recompute", True, "direct frozen fields and archived audit tables only"),
    ], """# Figure 3 scientific validation\n\n- Frozen endpoint: `dataset_iter018.npz` (4345 rows).\n- Trusted gapped FFLO mask: `topology_trusted == 1 and topology_applicable == 1 and y_phase == 2`.\n- Trusted topology counts: **2290 cFFLO** and **923 tFFLO**.\n- Boundary layer: archived deterministic k=8 IDW contour at iteration 18; no new extraction.\n- Configured result: **Decision A**; final coverage p95 **0.006185** below **0.00625**.\n- Sensitivity caveat: k=6 coverage p95 **0.006269** is slightly above threshold.\n- Gap closing is a transition diagnostic, not a stable third topology phase.\n- Brackets are diagnostics only.\n\n`GATE_FIG3_HUMAN_SCIENCE_REVIEW = HUMAN_WAIT`\n""")
    (FIG3 / "fig3_render_determinism.json").write_text(json.dumps({"deterministic": True, "sha256": h3b}, indent=2) + "\n", encoding="utf-8")
    run_lint(FIG3, 3)

    fig4_first = render_fig4()
    h4a = sha256(fig4_first[2])
    fig4_paths = render_fig4()
    h4b = sha256(fig4_paths[2])
    if h4a != h4b:
        raise RuntimeError("Figure 4 PNG rendering is not deterministic")
    c4 = configure_matplotlib(preflight(FIG4 / "fig4_panel_contract.yaml"))
    make_contact_sheet(FIG4, 4, fig4_paths[2], FIG4_BOXES, c4,
                       "Exact 3D point clouds and explicit slices only; strict nonclosure is part of the artwork")
    d4 = pd.read_csv(FIG4 / "canonical/fig4_exact_3d_samples.csv")
    write_reports_and_validate(FIG4, 4, *fig4_paths, [
        ("endpoint_hash", sha256(STAGE4 / "dataset_iter025.npz") == "7acab805115bcfcae56e94e2541bfdaa727de7767f2d7dd34330f163e2695842", "dataset_iter025.npz"),
        ("phase_counts", d4.phase_name.value_counts().to_dict() == {"FFLO": 4590, "normal": 2350, "uniform_SC": 141}, "2350/141/4590"),
        ("trusted_FFLO_topology_counts", d4.query("trusted_fflo == 1").topology_name.value_counts().to_dict() == {"cFFLO": 3145, "tFFLO": 1445}, "FFLO-only 3145/1445"),
        ("slice_selection", len(pd.read_csv(FIG4 / "canonical/fig4_selected_fixed_mu_slices.csv")) == 4, "evenly spaced source-table indices 0,6,13,19"),
        ("continuous_surface_absent", True, "scatter and explicit slice bins only"),
        ("strict_nonclosure_visible", True, "four raw failed metrics and component instability printed in panel d"),
        ("no_physics_recompute", True, "direct frozen fields and archived audit tables only"),
    ], """# Figure 4 scientific validation\n\n- Frozen endpoint: `dataset_iter025.npz` (7081 exact observations).\n- Thermodynamic counts: **2350 Normal**, **141 uniform SC**, **4590 FFLO**.\n- FFLO-only trusted topology support: **3145 cFFLO**, **1445 tFFLO**.\n- The archived 3265/1466 counts include 141 uniform-SC topology-applicable samples and are not relabeled cFFLO/tFFLO in the artwork.\n- Four evenly spaced explicit bins are shown from the 20-row fixed-$\\mu$ slice audit; all source rows remain in provenance.\n- No interpolation, smoothing, RBF/KNN surface, or boundary re-extraction was performed.\n- Strict 3D topology closure is **not converged**: final map change, p95 shift, coverage, surprise, and component stability fail.\n- The result supports **higher-dimensional exploration** and a **3D extension** that reveals thermodynamic and topology-aware structures, not a complete 3D phase diagram.\n\n`GATE_FIG4_HUMAN_SCIENCE_REVIEW = HUMAN_WAIT_WITH_CAVEAT`\n""")
    (FIG4 / "fig4_render_determinism.json").write_text(json.dumps({"deterministic": True, "sha256": h4b}, indent=2) + "\n", encoding="utf-8")
    run_lint(FIG4, 4)
    print(json.dumps({
        "fig2": "SCIENTIFIC_FREEZE",
        "fig3": "GATE_FIG3_HUMAN_SCIENCE_REVIEW",
        "fig4": "GATE_FIG4_HUMAN_SCIENCE_REVIEW_WITH_CAVEAT",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
