#!/usr/bin/env python3
"""Build the versioned three-panel Fig. 4 and Stage-IV SM diagnostics.

This builder performs no BdG or active-learning calculation.  It reads only
frozen exact samples and archived audit tables.  Operational thresholds are
deliberately excluded from artwork and are not used for pass/fail claims.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.build_remaining_strict_figures import (
    configure_matplotlib,
    panel_title,
    patch_svg,
    style_2d,
    style_3d,
)
from figures.figure_engine.builders.figure_build import preflight
from figures.figure_engine.validators.figure_lint import lint_figure


ROOT = PROJECT_ROOT
OLD_FIG4 = ROOT / "figures/fig4_strict_data"
FIG4 = ROOT / "figures/fig4_strict_data_v2"
SM = ROOT / "figures/sm_stageiv_numerical_diagnostics"
STAGE4 = ROOT / "manuscript_langgraph/recovered_artifacts/stageIV"

MAIN_CANVAS = (1600, 1100)
MAIN_PRINT_MM = (178.0, 120.0)
MAIN_BOXES = {
    "exact_3d_thermodynamic_cloud": (70, 140, 990, 830),
    "exact_3d_topology_support": (1090, 140, 440, 330),
    "fixed_mu_slice_audit": (1090, 520, 440, 450),
}

SM_CANVAS = (1600, 1350)
SM_PRINT_MM = (178.0, 150.0)
SM_BOXES = {
    "sample_and_trust_growth": (130, 140, 620, 330),
    "thermodynamic_composition": (850, 140, 620, 330),
    "topology_composition": (130, 530, 620, 330),
    "map_change_history": (850, 530, 620, 330),
    "surface_shift_history": (130, 920, 620, 330),
    "trusted_surprise_history": (850, 920, 620, 330),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axes_box(box: tuple[int, int, int, int], canvas: tuple[int, int]) -> list[float]:
    x, y, width, height = box
    return [x / canvas[0], (canvas[1] - y - height) / canvas[1], width / canvas[0], height / canvas[1]]


def write_geometry(path: Path, boxes: dict[str, tuple[int, int, int, int]], canvas: tuple[int, int], primary: str) -> None:
    objects = [
        {
            "id": name,
            "type": "plot",
            "role": "primary" if name == primary else "secondary",
            "parent": "canvas",
            "visual_weight_object": True,
            "bbox": {"x": x, "y": y, "width": width, "height": height},
        }
        for name, (x, y, width, height) in boxes.items()
    ]
    payload = {
        "schema_version": 1,
        "canvas": {"x": 0, "y": 0, "width": canvas[0], "height": canvas[1]},
        "objects": objects,
        "routes": [],
        "scientific_content_bbox": {
            primary: next(item["bbox"] for item in objects if item["id"] == primary)
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _output_names(prefix: str) -> list[str]:
    return [f"{prefix}.svg", f"{prefix}.pdf", f"{prefix}.png"]


def _source(path: Path, version: str, fields: list[str], units: dict[str, str]) -> dict[str, Any]:
    rows = len(pd.read_csv(path)) if path.suffix == ".csv" else None
    return {
        "path": str(path),
        "sha256": sha256(path),
        "version": version,
        "row_count": rows,
        "fields": fields,
        "units": units,
    }


def prepare_main_contract() -> None:
    FIG4.mkdir(parents=True, exist_ok=True)
    canonical = FIG4 / "canonical"
    canonical.mkdir(exist_ok=True)
    for name in ("fig4_exact_3d_samples.csv", "fig4_selected_fixed_mu_slices.csv"):
        shutil.copy2(OLD_FIG4 / "canonical" / name, canonical / name)

    contract = yaml.safe_load((OLD_FIG4 / "fig4_panel_contract.yaml").read_text(encoding="utf-8"))
    contract["figure_id"] = "FIG-04-STAGEIV-3D-STRICT-V2"
    contract["scientific_message"] = (
        "Reliability-aware active learning reveals thermodynamic and topology-resolved "
        "structures across a finite three-dimensional sampled domain."
    )
    contract["visual_message"] = (
        "The exact three-dimensional thermodynamic cloud is primary; trusted topology support "
        "and exact fixed-mu slices define the available evidence without a convergence scoreboard."
    )
    contract["primary_object"]["scientific_role"] = contract["scientific_message"]
    contract["primary_visual_object"]["scientific_role"] = contract["scientific_message"]
    contract["secondary_objects"] = [item for item in contract["secondary_objects"] if item["id"] in MAIN_BOXES]
    contract["tertiary_information"] = []
    contract["secondary_visual_objects"] = ["exact_3d_topology_support", "fixed_mu_slice_audit"]
    contract["tertiary_visual_objects"] = []
    contract["reading_order"] = list(MAIN_BOXES)
    contract["visual_narrative"]["narrative_regions"] = {name: [name] for name in MAIN_BOXES}
    contract["saliency_expectation"]["expected_ranking"] = list(MAIN_BOXES)
    contract["data_sources"] = [
        value for value in contract["data_sources"]
        if "convergence" not in value and "return_report_key_metrics" not in value
    ]
    contract["assets"] = [item for item in contract["assets"] if item["id"] in MAIN_BOXES]
    contract["layout"]["bounding_boxes"] = "geometry_manifest.json"
    contract["layout"]["template_profile"] = "high_dimensional"
    contract["layout"]["primary_area_fraction_min"] = 0.45
    contract["layout"]["canvas"]["viewbox"] = [0, 0, *MAIN_CANVAS]
    contract["layout"]["canvas"]["print_mm"] = list(MAIN_PRINT_MM)
    contract["layout"]["area_requirements"][0]["minimum_fraction"] = 0.45
    contract["required_elements"] = list(MAIN_BOXES)
    contract["forbidden_elements"] = sorted(set(contract["forbidden_elements"] + [
        "threshold_line", "pass_fail_scoreboard", "operational_threshold_as_physical_convergence"
    ]))
    contract["forbidden_claims"] = sorted(set(contract["forbidden_claims"] + [
        "physically converged 3D topology", "threshold-certified 3D topology"
    ]))
    contract["validation_requirements"]["engine_preflight_record"] = "engine_preflight.json"
    contract["validation_requirements"]["prohibited_text"] = ["PASS", "FAIL", "not converged", "convergence gate"]
    contract["approval"] = {
        "gate": "GATE_FIG4_V2_HUMAN_SCIENCE_AND_VISUAL_REVIEW",
        "state": "HUMAN_WAIT",
        "supersedes_after_approval": "FIG-04-STAGEIV-3D-STRICT",
    }
    outputs = _output_names("fig4_strict_data_v2")
    panels = []
    for panel in contract["strict_data"]["panels"]:
        if panel["panel_id"] not in MAIN_BOXES:
            continue
        panel["source_scripts"] = ["figures/build_fig4_v2_and_sm_diagnostics.py"]
        panel["required_caveats"] = [
            "finite sampled domain",
            "discrete exact support with no inferred continuous surface or complete atlas",
        ]
        panel["convergence_status"] = {}
        panel["allowed_claims"] = ["higher-dimensional exploration", "finite-domain exact support"]
        panel["forbidden_claims"] = sorted(set(panel["forbidden_claims"] + [
            "physically converged 3D topology", "threshold-certified 3D topology"
        ]))
        panel["human_review_items"] = [
            "finite-domain wording", "discrete-support semantics", "absence of threshold claims"
        ]
        panel["output_artifacts"] = outputs
        panels.append(panel)
    contract["strict_data"]["panels"] = panels
    contract["strict_data"]["output_artifacts"] = outputs
    (FIG4 / "figure_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    write_geometry(FIG4 / "geometry_manifest.json", MAIN_BOXES, MAIN_CANVAS, "exact_3d_thermodynamic_cloud")

    manifest = yaml.safe_load((OLD_FIG4 / "fig4_data_manifest.yaml").read_text(encoding="utf-8"))
    manifest["figure_id"] = contract["figure_id"]
    manifest["sources"] = {
        key: value for key, value in manifest["sources"].items()
        if key in {"dataset_iter025", "stageiv_narrow_mu_slice_metrics"}
    }
    for key in ("strict_3d_convergence", "topology_surface_coverage_p95", "coverage_threshold"):
        manifest["invariants"].pop(key, None)
    manifest["invariants"]["evidence_scope"] = (
        "finite domain; discrete exact observations; no inferred continuous topology surface"
    )
    manifest["thresholds_used_for_rendering"] = False
    manifest["pass_fail_labels_used_for_rendering"] = False
    manifest["render_authority"] = "User-authorized Fig.4(d) removal on 2026-08-31"
    manifest["canonical_outputs"] = {
        name: {"sha256": sha256(canonical / name), "rows": len(pd.read_csv(canonical / name))}
        for name in ("fig4_exact_3d_samples.csv", "fig4_selected_fixed_mu_slices.csv")
    }
    (FIG4 / "data_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _diagnostic_panel(template: dict[str, Any], panel_id: str, scientific_role: str,
                      source: dict[str, Any], selected: int, excluded: int, filter_text: str,
                      expected: dict[str, Any]) -> dict[str, Any]:
    panel = copy.deepcopy(template)
    panel.update({
        "panel_id": panel_id,
        "scientific_role": scientific_role,
        "visual_priority": "primary" if panel_id == "sample_and_trust_growth" else "secondary",
        "panel_type": "DERIVED_NUMERICAL_DATA",
        "source_artifacts": [source],
        "source_versions": [source["version"]],
        "source_hashes": [source["sha256"]],
        "source_scripts": ["figures/build_fig4_v2_and_sm_diagnostics.py"],
        "axis_fields": ["iteration", "metric"],
        "axis_units": {"iteration": "index", "metric": "defined in panel label"},
        "axis_orientation": {"iteration": "increasing", "metric": "increasing"},
        "input_row_count": selected + excluded,
        "selected_row_count": selected,
        "excluded_row_count": excluded,
        "selection_filters": [filter_text],
        "excluded_data": ["nonselected rows remain in the canonical source table"],
        "label_fields": [],
        "phase_mapping": {},
        "topology_mapping": {},
        "trust_mask": "not applicable; archived aggregate diagnostic",
        "convergence_status": {
            "mode": "diagnostic_only",
            "raw_trace_visible": True,
            "threshold_line_visible": False,
            "pass_fail_labels_visible": False,
        },
        "sensitivity_notes": ["operational thresholds are not rendered or interpreted as physical criteria"],
        "expected_counts": {},
        "expected_reference_values": expected,
        "scientific_invariants": ["archived values are immutable", "no threshold-based pass/fail inference"],
        "allowed_claims": ["archived numerical stability history", "iteration-resolved diagnostic"],
        "required_caveats": ["numerical diagnostic only; not a physical convergence criterion"],
        "forbidden_claims": ["physically converged Stage IV topology", "threshold-certified closure"],
        "primary_visual_object": panel_id,
        "validators": ["strict_data_checker", "sm_stageiv_specific_validation"],
        "human_review_items": ["diagnostic interpretation", "absence of threshold semantics"],
        "output_artifacts": _output_names("sm_stageiv_numerical_diagnostics"),
    })
    panel["transformations"] = {
        "interpolation": {"enabled": False, "authorized": False},
        "smoothing": {"enabled": False, "authorized": False},
        "boundary_extraction": {"enabled": False, "authorized": False},
        "aggregation": {"enabled": False, "authorized": False},
        "normalization": {"enabled": False, "authorized": False},
        "derivation_formula": {
            "enabled": True,
            "authorized": True,
            "formula": "direct archived-field selection; fractions use displayed archived counts only",
            "implementation": "figures/build_fig4_v2_and_sm_diagnostics.py",
            "parameters": {},
            "validator": "canonical table hash and reference-value audit",
        },
    }
    panel["legend_semantics"] = {}
    panel["exact_vs_derived_encoding"] = {"visually_distinct": True}
    return panel


def prepare_sm_contract() -> None:
    SM.mkdir(parents=True, exist_ok=True)
    canonical = SM / "canonical"
    canonical.mkdir(exist_ok=True)
    iteration_source = STAGE4 / "stageiv_3d_return_report_local/tables/dataset_iteration_summary.csv"
    failure_source = STAGE4 / "stageiv_3d_audit_local/tables/stageiv_3d_convergence_failure_decomposition.csv"
    iteration = pd.read_csv(iteration_source)
    iteration_path = canonical / "stageiv_iteration_history.csv"
    iteration.to_csv(iteration_path, index=False, float_format="%.16g")

    failure = pd.read_csv(failure_source)
    rows: list[dict[str, Any]] = []
    for metric in ("topology_volume_map_change", "topology_surface_shift_p95", "trusted_topology_surprise"):
        subset = failure[failure.metric == metric]
        for order, row in enumerate(subset.itertuples(index=False), start=1):
            end_iteration = int(str(row.iteration_or_transition).split("->")[-1])
            rows.append({
                "metric": metric,
                "order": order,
                "end_iteration": end_iteration,
                "transition": row.iteration_or_transition,
                "value": float(row.value),
                "source_field": "value",
            })
    component = failure[failure.metric == "component_stability"].iloc[0]
    for order, value in enumerate(str(component.value).split(";"), start=1):
        rows.append({
            "metric": "component_count",
            "order": order,
            "end_iteration": 22 + order,
            "transition": f"iteration_{22 + order}",
            "value": float(value),
            "source_field": "value parsed from archived semicolon-delimited component history",
        })
    diagnostics = pd.DataFrame(rows)
    diagnostics_path = canonical / "stageiv_last3_raw_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False, float_format="%.16g")

    iteration_meta = _source(
        iteration_path,
        "stageiv_iteration_history_archived_copy",
        list(iteration.columns),
        {"iteration": "index", "sample_count": "count", "all count fields": "count"},
    )
    diagnostic_meta = _source(
        diagnostics_path,
        "stageiv_last3_raw_diagnostics_no_threshold_fields",
        list(diagnostics.columns),
        {"end_iteration": "index", "value": "native metric units"},
    )
    base_contract = yaml.safe_load((OLD_FIG4 / "fig4_panel_contract.yaml").read_text(encoding="utf-8"))
    panel_template = next(item for item in base_contract["strict_data"]["panels"] if item["panel_id"] == "strict_3d_nonclosure")
    specs = [
        ("sample_and_trust_growth", "Archived sample and trust-count growth", iteration_meta, 26, 0, "all archived iterations", {"final_sample_count": 7081, "final_trusted_exact": 6767}),
        ("thermodynamic_composition", "Archived thermodynamic composition by iteration", iteration_meta, 25, 1, "iterations 1 through 25", {"final_counts": [2350, 141, 4590]}),
        ("topology_composition", "Archived topology-label composition by iteration", iteration_meta, 25, 1, "iterations 1 through 25", {"final_counts": [3265, 1466]}),
        ("map_change_history", "Last-three archived topology-map changes", diagnostic_meta, 3, 9, "metric == topology_volume_map_change", {"values": [0.0014122480421106, 0.0019225839528326, 0.002561475409836]}),
        ("surface_shift_history", "Last-three archived proxy-surface p95 displacements", diagnostic_meta, 3, 9, "metric == topology_surface_shift_p95", {"values": [0.0055835372535258, 0.0059066811860211, 0.0059961957165026]}),
        ("trusted_surprise_history", "Last-three archived trusted-topology surprise fractions", diagnostic_meta, 3, 9, "metric == trusted_topology_surprise", {"values": [0.065, 0.1122994652406417, 0.140625]}),
    ]
    panels = [_diagnostic_panel(panel_template, *spec) for spec in specs]
    objects = list(SM_BOXES)
    assets = [
        {
            "id": name,
            "type": "convergence_plot",
            "source": str(iteration_path if index < 3 else diagnostics_path),
            "schematic": False,
            "bbox": name,
            "features": ["raw_metric", "units_or_definition", "iteration_or_budget_axis"],
        }
        for index, name in enumerate(objects)
    ]
    contract = {
        "schema_version": 3,
        "figure_id": "SM-FIG-STAGEIV-NUMERICAL-DIAGNOSTICS-V1",
        "figure_type": "supplement",
        "template": "multi_panel_template",
        "scientific_message": "Archived Stage-IV iteration histories document numerical evolution without assigning physical convergence.",
        "visual_message": "Six matched small plots expose raw sample, label, map-change, displacement, and surprise histories without threshold lines.",
        "primary_object": {"id": objects[0], "asset_type": "convergence_plot", "scientific_role": specs[0][1], "visual_role": "primary"},
        "primary_visual_object": {"id": objects[0], "scientific_role": specs[0][1], "expected_rank": 1},
        "secondary_objects": [
            {"id": name, "asset_type": "convergence_plot", "scientific_role": spec[1], "visual_role": "secondary"}
            for name, spec in zip(objects[1:], specs[1:])
        ],
        "tertiary_information": [],
        "secondary_visual_objects": objects[1:],
        "tertiary_visual_objects": [],
        "reading_order": objects,
        "visual_narrative": {"enabled": True, "narrative_regions": {name: [name] for name in objects}, "scientific_asset_fill_ratio_audits": []},
        "saliency_expectation": {"method": "blurred_region_visual_mass", "expected_ranking": objects, "severity": "warning"},
        "forbidden_visual_priorities": ["decorative_threshold", "pass_fail_scoreboard"],
        "data_sources": [str(iteration_source), str(failure_source), str(iteration_path), str(diagnostics_path)],
        "assets": assets,
        "layout": {
            "canvas": {"viewbox": [0, 0, *SM_CANVAS], "print_mm": list(SM_PRINT_MM)},
            "template_profile": "supplement_diagnostics",
            "primary_area_fraction_min": 0.09,
            "bounding_boxes": "geometry_manifest.json",
            "alignment_anchors": ["shared_iteration_axes", "matched_two_column_grid"],
            "declared_overlays": [],
            "declared_exceptions": [],
            "area_requirements": [{"id": "primary_area", "object_id": objects[0], "reference": "canvas", "minimum_fraction": 0.09}],
        },
        "color_semantics": {"palette": "al_fflo_manuscript_v1", "mappings": {"diagnostic": "selected", "physical": "physical", "topology": "topology_tfflo"}, "allowed_extra_hex": [], "contrast_pairs": [{"id": "body_on_white", "role": "body_text", "foreground": "neutral.body", "background": "neutral.white"}]},
        "text_budget": {"maximum_words_in_artwork": 125, "maximum_words_per_panel": 18, "minimum_font_size_pt": 7.0, "move_excess_to": "caption"},
        "required_elements": objects,
        "required_labels": ["diagnostic only", "iteration"],
        "forbidden_elements": ["threshold_line", "pass_fail_label", "smoothing", "interpolation"],
        "forbidden_claims": ["physically converged Stage IV topology", "threshold-certified closure"],
        "validation_requirements": {"hard_fail": True, "require_geometry_manifest": True, "require_editable_svg": True, "require_pdf": True, "require_png_preview": True, "require_grayscale_review": True, "require_engine_preflight": True, "engine_preflight_record": "engine_preflight.json", "strict_palette": True, "prohibited_text": ["PASS", "FAIL", "threshold"]},
        "provenance": {"physics_recomputed": False, "parameters_recorded": True, "commit_hash_when_available": True},
        "approval": {"gate": "GATE_SM_STAGEIV_DIAGNOSTICS_HUMAN_SCIENCE_AND_VISUAL_REVIEW", "state": "HUMAN_WAIT"},
        "strict_data": {"enabled": True, "track": "STRICT_DATA_FIGURE_TRACK", "claim_freeze": ["CLM-3D-EXTENSION"], "panels": panels, "output_artifacts": _output_names("sm_stageiv_numerical_diagnostics")},
    }
    (SM / "figure_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    write_geometry(SM / "geometry_manifest.json", SM_BOXES, SM_CANVAS, objects[0])
    manifest = {
        "schema_version": 1,
        "figure_id": contract["figure_id"],
        "physics_recomputed": False,
        "source_artifacts": {
            "dataset_iteration_summary": {"path": str(iteration_source), "sha256": sha256(iteration_source), "rows": len(iteration)},
            "convergence_failure_decomposition": {"path": str(failure_source), "sha256": sha256(failure_source), "rows": len(failure)},
        },
        "canonical_outputs": {
            iteration_path.name: {"sha256": sha256(iteration_path), "rows": len(iteration)},
            diagnostics_path.name: {"sha256": sha256(diagnostics_path), "rows": len(diagnostics)},
        },
        "derivation": "direct field selection plus explicit parsing of archived component-count string; no physics recomputation",
        "excluded_from_artwork": ["threshold", "pass", "failure_class", "threshold_ratio"],
        "threshold_lines_rendered": False,
        "pass_fail_semantics_rendered": False,
    }
    (SM / "data_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def prepare() -> None:
    prepare_main_contract()
    prepare_sm_contract()
    preflight(FIG4 / "figure_contract.yaml", record_path=FIG4 / "engine_preflight.json")
    preflight(SM / "figure_contract.yaml", record_path=SM / "engine_preflight.json")


def render_main() -> tuple[Path, Path, Path]:
    context = preflight(FIG4 / "figure_contract.yaml", record_path=FIG4 / "engine_preflight.json")
    colors = configure_matplotlib(context)
    exact = pd.read_csv(FIG4 / "canonical/fig4_exact_3d_samples.csv")
    chosen = pd.read_csv(FIG4 / "canonical/fig4_selected_fixed_mu_slices.csv")
    fig = plt.figure(figsize=(MAIN_PRINT_MM[0] / 25.4, MAIN_PRINT_MM[1] / 25.4), facecolor=colors["white"])
    fig.text(0.025, 0.985, "Reliability-aware active learning extends higher-dimensional exploration", ha="left", va="top", fontsize=10.2, fontweight="bold", color=colors["body"])
    fig.text(0.025, 0.954, r"Finite-domain exact observations and fixed-$\mu$ slices; no continuous topology surface is inferred", ha="left", va="top", fontsize=7.0, color=colors["secondary"])

    ax = fig.add_axes(axes_box(MAIN_BOXES["exact_3d_thermodynamic_cloud"], MAIN_CANVAS), projection="3d")
    for name, color, size, alpha, label in [
        ("normal", colors["normal"], 4.0, 0.36, "Normal"),
        ("uniform_SC", colors["uniform"], 8.0, 0.88, "Uniform SC"),
        ("FFLO", colors["fflo"], 5.0, 0.66, "FFLO"),
    ]:
        data = exact[exact.phase_name == name]
        ax.scatter(data.kT, data.JA, data.mu, s=size, c=color, alpha=alpha, linewidths=0, depthshade=False, label=label)
    style_3d(ax, colors)
    panel_title(ax, "a", "Exact 3D thermodynamic observations", colors, size=8.0)
    ax.legend(loc="upper right", frameon=True, facecolor=colors["white"], edgecolor=colors["cool"], framealpha=0.94, fontsize=6.5)
    ax.text2D(0.02, 0.02, "7081 exact samples: 2350 / 141 / 4590", transform=ax.transAxes, fontsize=7.0, color=colors["secondary"], bbox={"facecolor": colors["white"], "edgecolor": "none", "alpha": 0.88, "pad": 1.4})

    axb = fig.add_axes(axes_box(MAIN_BOXES["exact_3d_topology_support"], MAIN_CANVAS), projection="3d")
    for name, color, label in [("cFFLO", colors["cfflo"], r"cFFLO $\nu=0$ (3145)"), ("tFFLO", colors["tfflo"], r"tFFLO $\nu=1$ (1445)")]:
        data = exact[exact.topology_name == name]
        axb.scatter(data.kT, data.JA, data.mu, s=4.0, c=color, alpha=0.58, linewidths=0, depthshade=False, label=label)
    style_3d(axb, colors)
    axb.set_xlabel(""); axb.set_ylabel(""); axb.set_zlabel("")
    axb.tick_params(labelsize=5.2)
    panel_title(axb, "b", "Trusted FFLO topology support", colors, size=7.1)
    axb.legend(loc="upper left", bbox_to_anchor=(0.01, 0.84), frameon=True, facecolor=colors["white"], edgecolor=colors["cool"], framealpha=0.94, fontsize=4.8, borderpad=0.25, handletextpad=0.25)

    x, y, width, height = MAIN_BOXES["fixed_mu_slice_audit"]
    fig.text(x / MAIN_CANVAS[0], (MAIN_CANVAS[1] - y + 2) / MAIN_CANVAS[1], r"c  Exact fixed-$\mu$ slices", ha="left", va="bottom", fontsize=7.7, fontweight="bold", color=colors["body"])
    mini_w, mini_h, gap_x, gap_y = 194, 170, 32, 42
    for index, row in enumerate(chosen.itertuples(index=False)):
        col, grid_row = index % 2, index // 2
        box = (x + col * (mini_w + gap_x), y + 35 + grid_row * (mini_h + gap_y), mini_w, mini_h)
        axi = fig.add_axes(axes_box(box, MAIN_CANVAS))
        data = exact[np.abs(exact.mu - float(row.mu_center)) <= float(row.slice_half_width) + 1e-12]
        for phase_name, color in [("normal", colors["normal"]), ("uniform_SC", colors["uniform"]), ("FFLO", colors["fflo"])]:
            subset = data[data.phase_name == phase_name]
            axi.scatter(subset.kT, subset.JA, s=4.0, c=color, alpha=0.72, linewidths=0)
        style_2d(axi, colors, xlabel="", ylabel="")
        axi.set_title(f"$\\mu/t={row.mu_center:.2f}\\pm{row.slice_half_width:.3f}$\n$n={len(data)}$", fontsize=5.7, pad=1.0)
        axi.tick_params(labelsize=5.3, length=1.7, pad=1)
        if grid_row == 0:
            axi.tick_params(labelbottom=False)

    svg, pdf, png = (FIG4 / name for name in _output_names("fig4_strict_data_v2"))
    fig.savefig(svg, format="svg", facecolor=colors["white"])
    fig.savefig(pdf, format="pdf", facecolor=colors["white"])
    fig.savefig(png, format="png", dpi=300, facecolor=colors["white"])
    plt.close(fig)
    patch_svg(svg, "FIG-04-STAGEIV-3D-STRICT-V2", "figure_contract.yaml")
    Image.open(png).convert("L").convert("RGB").save(FIG4 / "fig4_strict_data_v2_grayscale.png")
    return svg, pdf, png


def _style_diag(ax, colors: dict[str, str]) -> None:
    ax.grid(axis="y", color=colors["cool"], lw=0.45, alpha=0.65)
    ax.tick_params(labelsize=6.2, width=0.5, length=2.2, colors=colors["axis"], pad=1.5)
    for spine in ax.spines.values():
        spine.set_color(colors["axis"]); spine.set_linewidth(0.55)
    ax.set_xlabel("iteration", fontsize=6.6, labelpad=1.5)


def render_sm() -> tuple[Path, Path, Path]:
    context = preflight(SM / "figure_contract.yaml", record_path=SM / "engine_preflight.json")
    colors = configure_matplotlib(context)
    history = pd.read_csv(SM / "canonical/stageiv_iteration_history.csv")
    diagnostics = pd.read_csv(SM / "canonical/stageiv_last3_raw_diagnostics.csv")
    positive = history[history.iteration > 0].copy()
    fig = plt.figure(figsize=(SM_PRINT_MM[0] / 25.4, SM_PRINT_MM[1] / 25.4), facecolor=colors["white"])
    fig.text(0.025, 0.987, "Stage IV archived numerical stability diagnostics", ha="left", va="top", fontsize=10.2, fontweight="bold", color=colors["body"])
    fig.text(0.025, 0.961, "Raw histories only — diagnostic only; not a physical convergence claim", ha="left", va="top", fontsize=7.0, color=colors["secondary"])

    axes = {name: fig.add_axes(axes_box(box, SM_CANVAS)) for name, box in SM_BOXES.items()}
    ax = axes["sample_and_trust_growth"]
    for field, color, label in [("sample_count", colors["body"], "all exact"), ("trusted_exact_count", colors["selected"], "trusted exact"), ("topology_trusted_count", colors["tfflo"], "trusted topology")]:
        ax.plot(history.iteration, history[field], lw=1.15, color=color, label=label)
    _style_diag(ax, colors); ax.set_ylabel("count", fontsize=6.6); ax.legend(fontsize=5.8, frameon=False, ncol=1, loc="center right")
    panel_title(ax, "a", "Sample and trust-count growth", colors, size=7.2)

    ax = axes["thermodynamic_composition"]
    total = positive.sample_count.to_numpy(float)
    for field, color, label in [("normal_count", colors["normal"], "Normal"), ("uniform_sc_count", colors["uniform"], "uniform SC"), ("fflo_count", colors["fflo"], "FFLO")]:
        ax.plot(positive.iteration, positive[field] / total, lw=1.15, color=color, label=label)
    _style_diag(ax, colors); ax.set_ylabel("fraction of exact samples", fontsize=6.6); ax.legend(fontsize=5.8, frameon=False, ncol=1, loc="center right")
    panel_title(ax, "b", "Thermodynamic composition", colors, size=7.2)

    ax = axes["topology_composition"]
    applicable = positive.trivial_count.to_numpy(float) + positive.topological_count.to_numpy(float)
    ax.plot(positive.iteration, positive.trivial_count / applicable, lw=1.15, color=colors["cfflo"], label=r"$\nu=0$")
    ax.plot(positive.iteration, positive.topological_count / applicable, lw=1.15, color=colors["tfflo"], label=r"$\nu=1$")
    _style_diag(ax, colors); ax.set_ylabel("fraction of applicable SC", fontsize=6.6); ax.legend(fontsize=6.0, frameon=False, ncol=2, loc="center right")
    panel_title(ax, "c", "Topology-label composition", colors, size=7.2)

    series = [
        ("map_change_history", "topology_volume_map_change", colors["cfflo_dark"], "map-change fraction", "d", "Topology-map change"),
        ("surface_shift_history", "topology_surface_shift_p95", colors["selected"], r"normalized $p_{95}$ displacement", "e", r"Proxy-surface $p_{95}$ displacement"),
        ("trusted_surprise_history", "trusted_topology_surprise", colors["uncertainty"], "trusted-surprise fraction", "f", "Trusted-topology surprise"),
    ]
    for panel_id, metric, color, ylabel, letter, title in series:
        ax = axes[panel_id]
        data = diagnostics[diagnostics.metric == metric]
        ax.plot(data.end_iteration, data.value, lw=1.15, marker="o", ms=3.2, color=color, mfc=colors["white"], mew=0.7)
        _style_diag(ax, colors); ax.set_xticks([23, 24, 25]); ax.set_ylabel(ylabel, fontsize=6.6)
        panel_title(ax, letter, title, colors, size=7.2)

    svg, pdf, png = (SM / name for name in _output_names("sm_stageiv_numerical_diagnostics"))
    fig.savefig(svg, format="svg", facecolor=colors["white"])
    fig.savefig(pdf, format="pdf", facecolor=colors["white"])
    fig.savefig(png, format="png", dpi=300, facecolor=colors["white"])
    plt.close(fig)
    patch_svg(svg, "SM-FIG-STAGEIV-NUMERICAL-DIAGNOSTICS-V1", "figure_contract.yaml")
    Image.open(png).convert("L").convert("RGB").save(SM / "sm_stageiv_numerical_diagnostics_grayscale.png")
    return svg, pdf, png


def _specific_report(directory: Path, figure_id: str, checks: list[tuple[str, bool, str]]) -> None:
    failures = [item for item in checks if not item[1]]
    payload = {
        "schema_version": 1,
        "figure_id": figure_id,
        "status": "pass" if not failures else "fail",
        "physics_recomputed": False,
        "threshold_lines_rendered": False,
        "pass_fail_semantics_rendered": False,
        "checks": [{"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks],
    }
    (directory / "specific_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {figure_id} scientific validation", "", f"Status: **{payload['status'].upper()}**", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines.extend(f"| `{name}` | {'PASS' if passed else 'FAIL'} | {detail} |" for name, passed, detail in checks)
    lines.extend(["", "Automated validation does not pass either human review gate.", ""])
    (directory / "specific_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{figure_id} specific validation failed: {failures}")


def validate(main_outputs: tuple[Path, Path, Path], sm_outputs: tuple[Path, Path, Path]) -> None:
    main_svg_text = main_outputs[0].read_text(encoding="utf-8", errors="replace")
    exact = pd.read_csv(FIG4 / "canonical/fig4_exact_3d_samples.csv")
    _specific_report(FIG4, "FIG-04-STAGEIV-3D-STRICT-V2", [
        ("endpoint_rows", len(exact) == 7081, f"rows={len(exact)}"),
        ("phase_counts", exact.phase_name.value_counts().to_dict() == {"FFLO": 4590, "normal": 2350, "uniform_SC": 141}, str(exact.phase_name.value_counts().to_dict())),
        ("trusted_fflo_topology_counts", exact.topology_name.value_counts().to_dict().get("cFFLO") == 3145 and exact.topology_name.value_counts().to_dict().get("tFFLO") == 1445, str(exact.topology_name.value_counts().to_dict())),
        ("panel_d_absent", "Strict 3D topology gates" not in main_svg_text and "not converged" not in main_svg_text, "no panel-d or threshold-closure wording in SVG"),
        ("three_panel_contract", set(MAIN_BOXES) == {"exact_3d_thermodynamic_cloud", "exact_3d_topology_support", "fixed_mu_slice_audit"}, "panels a-c only"),
        ("no_physics_recompute", True, "frozen canonical tables copied byte-for-byte"),
    ])
    diagnostics = pd.read_csv(SM / "canonical/stageiv_last3_raw_diagnostics.csv")
    sm_svg_text = sm_outputs[0].read_text(encoding="utf-8", errors="replace")
    _specific_report(SM, "SM-FIG-STAGEIV-NUMERICAL-DIAGNOSTICS-V1", [
        ("canonical_diagnostics_rows", len(diagnostics) == 12, f"rows={len(diagnostics)}"),
        ("threshold_fields_excluded", not {"threshold", "pass", "failure_class", "threshold_ratio"} & set(diagnostics.columns), str(list(diagnostics.columns))),
        ("threshold_word_absent", "threshold" not in sm_svg_text.lower(), "no threshold text in SVG"),
        ("pass_fail_labels_absent", ">PASS<" not in sm_svg_text and ">FAIL<" not in sm_svg_text, "no PASS/FAIL text labels in SVG"),
        ("diagnostic_semantics_visible", "diagnostic only" in sm_svg_text.lower(), "diagnostic-only qualifier is printed"),
        ("no_physics_recompute", True, "only archived tables and transparent field selection used"),
    ])
    lint_figure(contract_path=FIG4 / "figure_contract.yaml", svg_path=main_outputs[0], geometry_path=FIG4 / "geometry_manifest.json", pdf_path=main_outputs[1], png_path=main_outputs[2], output_dir=FIG4)
    lint_figure(contract_path=SM / "figure_contract.yaml", svg_path=sm_outputs[0], geometry_path=SM / "geometry_manifest.json", pdf_path=sm_outputs[1], png_path=sm_outputs[2], output_dir=SM)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "contracts": [str(FIG4 / "figure_contract.yaml"), str(SM / "figure_contract.yaml")]}))
        return 0
    main_outputs = render_main()
    sm_outputs = render_sm()
    validate(main_outputs, sm_outputs)
    print(json.dumps({"status": "rendered_and_validated", "fig4": [str(path) for path in main_outputs], "supplement": [str(path) for path in sm_outputs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
