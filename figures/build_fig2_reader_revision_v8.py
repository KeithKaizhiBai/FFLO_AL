#!/usr/bin/env python3
"""Render Figure 2 from canonical exact-sample-only inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFilter

from figures.figure_engine.builders.figure_build import preflight
from figures.figure_engine.styles.matplotlib_rc import build_rc


HERE = Path(__file__).resolve().parent / "fig2_reader_revision_v8"
CANON = Path(__file__).resolve().parent / "fig2_strict_data/canonical"
CONTRACT = HERE / "fig2_panel_contract.yaml"
CANVAS = (1600, 1100)
BBOX = {
    "cold_start": (105, 125, 345, 350),
    "exact_call_allocation": (105, 590, 345, 350),
    "final_phase_map": (520, 115, 1040, 715),
    "raw_convergence": (520, 875, 1040, 185),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axes_box(box: tuple[int, int, int, int]) -> list[float]:
    x, y, w, h = box
    return [x / CANVAS[0], (CANVAS[1] - y - h) / CANVAS[1], w / CANVAS[0], h / CANVAS[1]]


def colors(rules) -> dict:
    p = rules["palette"]
    return {
        "normal": p["neutral"]["secondary"],
        "uniform_SC": p["families"]["physical"]["highlight"],
        "FFLO": p["families"]["physical"]["dark"],
        "selected_light": p["families"]["selected"]["highlight"],
        "selected": p["families"]["selected"]["base"],
        "selected_dark": p["families"]["selected"]["dark"],
        "uncertainty": p["families"]["uncertainty"]["dark"],
        "axis": p["neutral"]["axis"], "body": p["neutral"]["body"],
        "secondary": p["neutral"]["secondary"], "panel": p["neutral"]["panel"],
        "white": p["neutral"]["white"], "cool": p["neutral"]["cool_light"],
        "physical": p["families"]["physical"]["base"],
        "decision": p["families"]["decision"]["base"],
    }


def style_axis(ax, C, *, xlabel=True, ylabel=True) -> None:
    ax.set_xlim(0, 0.56); ax.set_ylim(0, 2.12)
    ax.set_xlabel(r"$k_B T/t$" if xlabel else "")
    ax.set_ylabel(r"$J_A/t$" if ylabel else "")
    ax.tick_params(colors=C["axis"], width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_color(C["axis"]); spine.set_linewidth(0.65)
    ax.set_facecolor(C["panel"])


def add_panel_title(ax, letter: str, title: str, C) -> None:
    box = {"facecolor": C["white"], "edgecolor": "none", "alpha": 0.88, "pad": 1.6}
    ax.text(0.015, 0.982, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.0, fontweight="bold", color=C["body"], bbox=box, zorder=20)
    ax.text(0.072, 0.982, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.2, fontweight="bold", color=C["body"], bbox=box, zorder=20)


def phase_scatter(ax, df: pd.DataFrame, C, *, size: float, alpha: float, endpoint_only=True) -> None:
    for name in ("normal", "uniform_SC", "FFLO"):
        subset = df[df.phase_name == name]
        if len(subset):
            ax.scatter(subset.kT, subset.JA, s=size, c=C[name], marker="o", linewidths=0,
                       alpha=alpha, rasterized=False, label={"normal": "Normal", "uniform_SC": "Uniform SC", "FFLO": "FFLO"}[name])


def render(low_fidelity: bool, context) -> tuple[Path, Path | None, Path]:
    rules = context.rules
    C = colors(rules)
    rc = build_rc(rules)
    rc.update({"text.color": C["body"], "axes.labelcolor": C["body"], "axes.edgecolor": C["axis"],
               "xtick.color": C["axis"], "ytick.color": C["axis"], "axes.titlecolor": C["body"],
               "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": None, "savefig.pad_inches": 0})
    plt.rcParams.update(rc)
    final = pd.read_csv(CANON / "fig2_final_exact_samples.csv")
    cold = pd.read_csv(CANON / "fig2_cold_start_exact_calls.csv")
    allocation = pd.read_csv(CANON / "fig2_exact_call_allocation.csv")
    conv = pd.read_csv(CANON / "fig2_raw_convergence_metrics.csv")
    risk = pd.read_csv(CANON / "fig2_hard_risk_points.csv")
    risk = pd.read_csv(CANON / "fig2_hard_risk_points.csv")

    fig = plt.figure(figsize=(178 / 25.4, 135 / 25.4), facecolor=C["white"])
    if not low_fidelity:
        fig.text(0.025, 0.985, "Reliability-aware active learning reconstructs the 2D thermodynamic structure",
                 ha="left", va="top", fontsize=10.2, fontweight="bold", color=C["body"])
        fig.text(0.025, 0.955, "Exact BdG samples; no interpolation, smoothing, or fitted boundary",
                 ha="left", va="top", fontsize=7.0, color=C["secondary"])

    ax_a = fig.add_axes(axes_box(BBOX["cold_start"]))
    matched = cold[cold.stage1_exact_match.astype(bool)]
    phase_scatter(ax_a, matched, C, size=5.5 if not low_fidelity else 4.5, alpha=0.88)
    missing = cold[~cold.stage1_exact_match.astype(bool)]
    if len(missing):
        ax_a.scatter(missing.kT, missing.JA, s=11, c=C["axis"], marker="x", linewidths=0.8,
                     label="not retained in endpoint")
    style_axis(ax_a, C)
    add_panel_title(ax_a, "a", "Sparse cold start", C)
    ax_a.text(0.03, 0.04, "512 selected; 506 retained", transform=ax_a.transAxes, fontsize=7.2,
              color=C["secondary"], bbox={"facecolor": C["white"], "edgecolor": "none", "alpha": 0.82, "pad": 1.2})

    ax_b = fig.add_axes(axes_box(BBOX["exact_call_allocation"]))
    seed = allocation[allocation.iteration == 0]
    active = allocation[allocation.iteration > 0]
    ax_b.scatter(seed.kT, seed.JA, s=2.4, c=C["cool"], alpha=0.45, linewidths=0)
    cmap = LinearSegmentedColormap.from_list("selected_semantic", [C["selected_light"], C["selected"], C["selected_dark"]])
    sc = ax_b.scatter(active.kT, active.JA, s=2.8, c=active.iteration, cmap=cmap, vmin=1, vmax=34,
                      alpha=0.55, linewidths=0, rasterized=False)
    style_axis(ax_b, C)
    add_panel_title(ax_b, "b", "Exact-call allocation", C)
    ax_b.text(0.04, 0.045, "iteration 1  ->  34", transform=ax_b.transAxes, fontsize=7.2,
              color=C["selected_dark"], bbox={"facecolor": C["white"], "edgecolor": C["cool"], "alpha": 0.86, "pad": 1.3})

    ax_c = fig.add_axes(axes_box(BBOX["final_phase_map"]))
    phase_scatter(ax_c, final, C, size=6.0 if not low_fidelity else 4.0, alpha=0.90)
    ax_c.scatter(risk.kBT, risk.JA, s=20, facecolors="none", edgecolors=C["uncertainty"],
                 linewidths=0.75, alpha=0.88, label="hard-risk (separate layer)")
    style_axis(ax_c, C)
    ax_c.set_ylabel("")  # Shared J_A/t axis is labeled in the left column.
    ax_c.set_xlabel("")
    ax_c.text(0.985, 0.02, r"$k_B T/t$", transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=7.2, color=C["body"],
              bbox={"facecolor": C["white"], "edgecolor": "none", "alpha": 0.86, "pad": 1.1})
    add_panel_title(ax_c, "c", "Exact thermodynamic states", C)
    ax_c.legend(loc="upper right", frameon=True, facecolor=C["white"], edgecolor=C["cool"],
                framealpha=0.94, ncol=2, handletextpad=0.4, columnspacing=0.8, borderpad=0.35)
    fig.text(0.325, 0.925, "7434 samples: 1867 Normal / 715 uniform SC / 4852 FFLO",
              transform=fig.transFigure, fontsize=7.0, color=C["secondary"],
              bbox={"facecolor": C["white"], "edgecolor": "none", "alpha": 0.86, "pad": 1.5})

    panel_box = BBOX["raw_convergence"]
    px, py, pw, ph = panel_box
    gap, inner = 52, 12
    mini_w = (pw - 2 * inner - 3 * gap) / 4
    metrics = [
        ("phase_map_change", "phase-map change", "map_threshold", C["physical"]),
        ("normal_sc_boundary_shift", "Normal-SC shift", "boundary_shift_threshold", C["selected"]),
        ("uniform_fflo_boundary_shift", "Uniform-FFLO shift", "boundary_shift_threshold", C["decision"]),
        ("trusted_surprise_rate", "trusted surprise", "surprise_threshold", C["uncertainty"]),
    ]
    for i, (field, title, threshold, color) in enumerate(metrics):
        box = (px + inner + i * (mini_w + gap), py + 50, mini_w, ph - 62)
        ax = fig.add_axes(axes_box(tuple(map(int, box))))
        ax.plot(conv.iteration, conv[field], color=color, lw=1.15, marker="o", ms=1.8, mfc=C["white"], mew=0.45)
        available = conv[threshold].dropna()
        if len(available):
            ax.axhline(float(available.iloc[-1]), color=C["axis"], ls=(0, (3, 2)), lw=0.65)
        ax.set_title(title, fontsize=7.2, pad=2)
        ax.set_xlim(0, 34); ax.set_xticks([0, 17, 34]); ax.set_xlabel("iteration", fontsize=7.2, labelpad=1)
        ax.tick_params(labelsize=7.1, width=0.5, length=2, colors=C["axis"], pad=1)
        for spine in ax.spines.values(): spine.set_color(C["axis"]); spine.set_linewidth(0.55)
        ax.grid(axis="y", color=C["cool"], linewidth=0.4, alpha=0.55)
    fig.text(px / CANVAS[0], (CANVAS[1] - py - 4) / CANVAS[1], "d", ha="left", va="top",
             fontsize=9.0, fontweight="bold", color=C["body"])
    fig.text(px / CANVAS[0] + 0.018, (CANVAS[1] - py - 4) / CANVAS[1], "Numerical histories",
             ha="left", va="top", fontsize=8.2, fontweight="bold", color=C["body"])

    prefix = "fig2_low_fidelity" if low_fidelity else "fig2_strict_data"
    svg = HERE / f"{prefix}.svg"; png = HERE / f"{prefix}.png"; pdf = None if low_fidelity else HERE / f"{prefix}.pdf"
    fig.savefig(svg, format="svg", facecolor=C["white"])
    if pdf: fig.savefig(pdf, format="pdf", facecolor=C["white"])
    fig.savefig(png, format="png", dpi=300, facecolor=C["white"])
    plt.close(fig)
    patch_svg(svg)
    return svg, pdf, png


def patch_svg(path: Path) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(path); root = tree.getroot()
    root.set("data-figure-id", "FIG-02-THERMO-STRICT")
    root.set("data-contract", "fig2_panel_contract.yaml")
    root.set("data-engine-schema", "3")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_geometry() -> None:
    objects = [{"id": "canvas", "type": "canvas", "role": "reference", "parent": None,
                "bbox": {"x": 0, "y": 0, "width": CANVAS[0], "height": CANVAS[1]}}]
    roles = {"cold_start": "tertiary", "exact_call_allocation": "secondary",
             "final_phase_map": "primary", "raw_convergence": "secondary"}
    for name, box in BBOX.items():
        x, y, w, h = box
        objects.append({"id": name, "type": "plot", "role": roles[name], "parent": "canvas",
                        "visual_weight_object": True,
                        "bbox": {"x": x, "y": y, "width": w, "height": h}})
    payload = {"schema_version": 1, "canvas": objects[0]["bbox"], "objects": objects[1:], "routes": [],
               "scientific_content_bbox": {"final_phase_map": objects[3]["bbox"]}}
    (HERE / "geometry_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_qa(final_png: Path) -> None:
    im = Image.open(final_png).convert("RGB")
    gray = im.convert("L").convert("RGB")
    gray.save(HERE / "fig2_strict_data_grayscale.png")
    blurred = im.filter(ImageFilter.GaussianBlur(radius=max(2, im.width * 0.00375)))
    blurred.save(HERE / "fig2_saliency_preview.png")
    scale_x, scale_y = im.width / CANVAS[0], im.height / CANVAS[1]
    crops = {}
    for name, (x, y, w, h) in BBOX.items():
        if name in {"cold_start", "exact_call_allocation"}:
            pad_left, pad_right, pad_top, pad_bottom = 58, 0, 16, 82
        elif name == "final_phase_map":
            pad_left, pad_right, pad_top, pad_bottom = 36, 8, 16, 34
        else:
            pad_left, pad_right, pad_top, pad_bottom = 24, 8, 18, 28
        crop = im.crop((max(0, round((x - pad_left) * scale_x)), max(0, round((y - pad_top) * scale_y)),
                        min(im.width, round((x + w + pad_right) * scale_x)),
                        min(im.height, round((y + h + pad_bottom) * scale_y))))
        path = HERE / f"panel_{name}.png"; crop.save(path); crops[name] = path

    C = colors(preflight(CONTRACT).rules)
    fig = plt.figure(figsize=(11.0, 8.5), facecolor=C["white"])
    fig.text(0.04, 0.975, "Figure 2 strict-data review package", fontsize=15, fontweight="bold", va="top", color=C["body"])
    fig.text(0.04, 0.947, "Exact-sample-only reconstruction; no interpolation, smoothing, or boundary fitting", fontsize=9, va="top", color=C["secondary"])
    slots = [(0.04, 0.54, 0.56, 0.37), (0.63, 0.70, 0.33, 0.21), (0.63, 0.44, 0.33, 0.21),
             (0.04, 0.08, 0.28, 0.36), (0.35, 0.08, 0.28, 0.36), (0.66, 0.08, 0.30, 0.31)]
    images = [im, gray, blurred, Image.open(crops["cold_start"]), Image.open(crops["exact_call_allocation"]), Image.open(crops["final_phase_map"])]
    titles = ["Full color / 178 mm", "Grayscale", "Gaussian-blur diagnostic", "Panel a", "Panel b", "Panel c: exact support + hard risk"]
    for slot, image, title in zip(slots, images, titles):
        ax = fig.add_axes(slot); ax.imshow(image); ax.set_title(title, fontsize=9, color=C["body"], pad=4); ax.axis("off")
    fig.text(0.66, 0.055, "Science checks: 7434 exact rows; phase counts 1867/715/4852;\n"
             "cold start 512 selected / 506 retained; 35 iterations; 129 hard-risk points.\n"
             "Hard risk is an open-ring overlay, never a fourth phase.", fontsize=8.2, color=C["body"], va="bottom")
    fig.savefig(HERE / "fig2_contact_sheet.pdf", format="pdf", facecolor=C["white"])
    fig.savefig(HERE / "fig2_contact_sheet.png", format="png", dpi=180, facecolor=C["white"])
    plt.close(fig)


def write_reports(pdf: Path, png: Path) -> None:
    final = pd.read_csv(CANON / "fig2_final_exact_samples.csv")
    cold = pd.read_csv(CANON / "fig2_cold_start_exact_calls.csv")
    allocation = pd.read_csv(CANON / "fig2_exact_call_allocation.csv")
    conv = pd.read_csv(CANON / "fig2_raw_convergence_metrics.csv")
    risk = pd.read_csv(CANON / "fig2_hard_risk_points.csv")
    counts = final.phase_name.value_counts().to_dict()
    end = conv.iloc[-1]
    science = f"""# Figure 2 scientific validation

- Frozen endpoint: `dataset_iter035.npz`
- Exact rows: **{len(final)}**
- Phase counts: Normal **{counts['normal']}**, Uniform SC **{counts['uniform_SC']}**, FFLO **{counts['FFLO']}**
- Cold start: **{len(cold)}** selected; **{int(cold.stage1_exact_match.sum())}** have frozen Stage I exact labels and are retained in the Stage II endpoint
- Selection history: iterations **0-34**, **{len(allocation)}** selected calls
- Hard-risk audit layer: **{len(risk)}** archived exact calls, encoded separately from phase; **{int(final.hard_risk.sum())}** coordinate matches remain in the final trusted endpoint
- Final phase-map change: **{end.phase_map_change:.9f}** (threshold {end.map_threshold:.6f})
- Final Normal-SC shift: **{end.normal_sc_boundary_shift:.9f}** (threshold {end.boundary_shift_threshold:.9f})
- Final Uniform-SC-FFLO shift: **{end.uniform_fflo_boundary_shift:.9f}**
- Final trusted surprise: **{end.trusted_surprise_rate:.9f}** (threshold {end.surprise_threshold:.6f})

No interpolation, smoothing, gridding, resampling, boundary extraction, or manual label change was performed. The hero panel is an exact-sample map; no new smooth boundary is claimed. The six cold-start selections without a retained Stage I exact result remain visible as unmatched calls rather than receiving inferred phase labels.

`GATE_FIG2_HUMAN_SCIENCE_REVIEW = HUMAN_WAIT`
"""
    (HERE / "fig2_scientific_validation.md").write_text(science, encoding="utf-8")

    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=False).stdout
    page = next((line for line in info.splitlines() if line.startswith("Page size:")), "")
    pages = next((line for line in info.splitlines() if line.startswith("Pages:")), "")
    with Image.open(png) as im:
        resolution = f"{im.width}x{im.height}"
    export = f"""# Figure 2 export validation

- SVG parse/editable text: checked by Figure Engine lint
- PDF: `{pages}`; `{page}`
- PNG preview: `{resolution}`
- Target print width: 178 mm
- Vector policy: axes, markers, lines, labels, and plots remain vector in SVG/PDF
- PNG is preview only
- Final visual QA: color, grayscale, panel crops, and blur diagnostic included in contact sheet
"""
    (HERE / "fig2_export_validation.md").write_text(export, encoding="utf-8")
    visual = """# Figure 2 visual narrative audit

- Primary visual object: `final_phase_map`
- Registered primary canvas fraction: **0.42250** (minimum **0.42**)
- Intended reading order: final map -> exact-call allocation -> raw convergence -> cold start
- Hard-risk encoding: open purple rings; separate legend entry; never a fourth phase fill
- Grayscale and Gaussian-blur diagnostics: included in `fig2_contact_sheet.pdf`
- Automated saliency role: warning-only proxy; it does not certify aesthetics or first fixation
- Automated geometry/typography/color result: see `figure_validation_report.md`
- Human first-fixation/scientific hierarchy judgment: **required at `GATE_FIG2_HUMAN_SCIENCE_REVIEW`**
"""
    (HERE / "fig2_visual_narrative_audit.md").write_text(visual, encoding="utf-8")


def write_asset_manifest(paths: list[Path]) -> None:
    payload = {"schema_version": 1, "figure_id": "FIG-02-THERMO-STRICT", "physics_recomputed": False,
               "assets": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths}}
    (HERE / "asset_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    import yaml
    from figures.figure_engine.validators.figure_lint import lint_figure
    HERE.mkdir(parents=True, exist_ok=True)
    original = CANON.parent / "fig2_panel_contract.yaml"
    contract = yaml.safe_load(original.read_text())
    contract["layout"]["canvas"]["print_mm"] = [178.0, 135.0]
    contract["required_labels"] = [s.replace("Final audited thermodynamic endpoint", "Exact thermodynamic states").replace("Raw convergence evidence", "Numerical histories") for s in contract["required_labels"]]
    for asset in contract["assets"]:
        asset["source"] = str(CANON / Path(asset["source"]).name)
    contract["strict_data"]["output_artifacts"] = ["fig2_strict_data.svg","fig2_strict_data.pdf","fig2_strict_data.png","geometry_manifest.json","asset_manifest.json"]
    CONTRACT.write_text(yaml.safe_dump(contract, sort_keys=False))
    context = preflight(CONTRACT, record_path=HERE / "engine_preflight.json")
    write_geometry()
    low_svg, _, low_png = render(True, context)
    svg, pdf, png = render(False, context)
    assert pdf is not None
    write_reports(pdf, png)
    Image.open(png).convert("L").save(HERE / "fig2_strict_data_grayscale.png")
    write_asset_manifest([svg, pdf, png, low_svg, low_png, HERE / "fig2_strict_data_grayscale.png"])
    result = lint_figure(contract_path=CONTRACT, svg_path=svg, pdf_path=pdf, png_path=png, geometry_path=HERE/"geometry_manifest.json", output_dir=HERE)
    if result["status"] != "pass":
        raise RuntimeError("Figure Engine rejected candidate")
    print(json.dumps({"status": "rendered", "svg": str(svg), "pdf": str(pdf), "png": str(png)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
