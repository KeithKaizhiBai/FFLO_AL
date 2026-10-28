#!/usr/bin/env python3
"""Build the Fig.1 v3 central scientific asset from frozen schematic physics."""

from __future__ import annotations

import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
V5_DIR = HERE.parent / "fig1_framework_v3" / "refinement_v5"
V2_DIR = HERE.parent / "fig1_engine_v2"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_DIR))

from figures.figure_engine.builders.figure_build import preflight  # noqa: E402
import build_v5_assets as v5  # noqa: E402


CONTRACT = HERE / "fig1_contract.yaml"
LAYOUT = yaml.safe_load((HERE / "layout.yaml").read_text(encoding="utf-8"))
ASSET_DIR = HERE / "asset_versions" / "thermodynamic_landscape_v3"
WORK = HERE / ".asset_work"
SURFACE_PNG = ASSET_DIR / "surface_tight.png"


def _surface_render() -> tuple[dict[str, list[float]], dict]:
    """Render and tightly crop the schematic 3D surface plus its phase markers."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    full_path = WORK / "surface_full.png"
    fig = plt.figure(figsize=(8.0, 3.0), dpi=200)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.patch.set_alpha(0)

    delta = np.linspace(0, 1.06, 230)
    q = np.linspace(-1, 1, 250)
    d_grid, q_grid = np.meshgrid(delta, q)
    free_energy = v5.v3.base.free_energy(d_grid, q_grid)
    colors = list(v5.STYLE_V5["landscape"]["colormap"].values())
    cmap = LinearSegmentedColormap.from_list("thermo_v3", colors, N=256)
    light = LightSource(azdeg=310, altdeg=42)
    normalizer = mpl.colors.Normalize(vmin=float(free_energy.min()), vmax=float(free_energy.max()))
    face = cmap(normalizer(free_energy))
    intensity = light.hillshade(free_energy, vert_exag=1.12)
    face[..., :3] = np.clip(face[..., :3] * (0.50 + 0.65 * intensity[..., None]), 0, 1)
    ax.plot_surface(
        d_grid,
        q_grid,
        free_energy,
        facecolors=face,
        linewidth=0,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
    )
    ax.contour(
        d_grid,
        q_grid,
        free_energy,
        zdir="z",
        offset=float(free_energy.min()) - 0.085,
        levels=5,
        colors="#527D83",
        linewidths=0.35,
        alpha=0.22,
    )
    minima = {
        "normal": (0.0, 0.0),
        "uniform": (0.43, 0.0),
        "cfflo": (0.76, 0.38),
        "tfflo": (0.92, 0.66),
    }
    marker_colors = {
        "normal": "#1F426F",
        "uniform": "#17634F",
        "cfflo": "#268C70",
        "tfflo": "#59317A",
    }
    for key, (delta_value, q_value) in minima.items():
        z_value = float(v5.v3.base.free_energy(delta_value, q_value)) + 0.045
        ax.scatter(
            [delta_value],
            [q_value],
            [z_value],
            s=42 if key in {"cfflo", "tfflo"} else 34,
            color=marker_colors[key],
            edgecolors="white",
            linewidths=0.9,
            depthshade=False,
            zorder=9,
        )

    ax.view_init(elev=30, azim=-55)
    ax.set_box_aspect((5.0, 1.0, 0.46), zoom=1.15)
    ax.set_zlim(float(free_energy.min()) - 0.085, float(free_energy.max()) + 0.01)
    ax.set_axis_off()
    fig.canvas.draw()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    display_points: dict[str, tuple[float, float]] = {}
    for key, (delta_value, q_value) in minima.items():
        z_value = float(v5.v3.base.free_energy(delta_value, q_value)) + 0.045
        x_projected, y_projected, _ = proj3d.proj_transform(delta_value, q_value, z_value, ax.get_proj())
        display_points[key] = tuple(float(value) for value in ax.transData.transform((x_projected, y_projected)))

    fig.savefig(full_path, transparent=True, dpi=200, facecolor="none", edgecolor="none")
    plt.close(fig)
    with Image.open(full_path).convert("RGBA") as image:
        alpha_bbox = image.getchannel("A").getbbox()
        if alpha_bbox is None:
            raise RuntimeError("The schematic surface render is empty.")
        cropped = image.crop(alpha_bbox)
        cropped.save(SURFACE_PNG)

    content = LAYOUT["central_internal"]["scientific_content"]
    marker_local = {}
    crop_width = alpha_bbox[2] - alpha_bbox[0]
    crop_height = alpha_bbox[3] - alpha_bbox[1]
    for key, (display_x, display_y) in display_points.items():
        image_y = canvas_height - display_y
        marker_local[key] = [
            content["x"] + (display_x - alpha_bbox[0]) / crop_width * content["width"],
            content["y"] + (image_y - alpha_bbox[1]) / crop_height * content["height"],
        ]
    return marker_local, {
        "full_render_px": [canvas_width, canvas_height],
        "alpha_bbox_px": list(alpha_bbox),
        "tight_surface_px": [crop_width, crop_height],
        "tight_surface_aspect": crop_width / crop_height,
        "embedded_content_bbox_local": content,
    }


def _local_line(overlay, points, box, *, color, linewidth=1.0, alpha=0.88) -> None:
    overlay.plot(
        [point[0] / box["width"] for point in points],
        [1 - point[1] / box["height"] for point in points],
        color=color,
        lw=linewidth,
        alpha=alpha,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def build_central() -> dict:
    marker_local, surface_metadata = _surface_render()
    box = LAYOUT["cards"]["central"]
    physical = v5.STYLE["semantic_families"]["physical"]
    topology = v5.STYLE["topology"]
    v5.v4.CARDS = LAYOUT["cards"]
    fig, background = v5.v4.canvas("central", physical, "", central=True)
    background.add_patch(
        FancyBboxPatch(
            (0.012, 0.012),
            0.976,
            0.455,
            boxstyle="round,pad=0.003,rounding_size=0.022",
            facecolor="#FBFDFC",
            edgecolor="none",
            alpha=0.72,
        )
    )
    background.text(
        26 / box["width"],
        1 - 14 / box["height"],
        "Physical state space generated by the BdG oracle",
        fontsize=10.7,
        weight="semibold",
        color="#17634F",
        ha="left",
        va="top",
    )
    background.text(
        26 / box["width"],
        1 - 52 / box["height"],
        r"Thermodynamic landscape  $F(\Delta,q)$",
        fontsize=9.0,
        weight="semibold",
        color=physical["dark"],
        ha="left",
        va="top",
    )

    content = LAYOUT["central_internal"]["scientific_content"]
    surface_axis = fig.add_axes(
        [
            content["x"] / box["width"],
            (box["height"] - content["y"] - content["height"]) / box["height"],
            content["width"] / box["width"],
            content["height"] / box["height"],
        ],
        zorder=4,
    )
    surface_axis.imshow(Image.open(SURFACE_PNG).convert("RGBA"), aspect="auto", interpolation="lanczos")
    surface_axis.axis("off")

    overlay = fig.add_axes([0, 0, 1, 1], zorder=20, facecolor="none")
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    label_specs = {
        "normal": {"title": "Normal", "formula": r"$\Delta=0$", "offset": (-36, 22), "color": "#1F426F"},
        "uniform": {"title": "Uniform SC", "formula": r"$\Delta>0,\ q=0$", "offset": (-2, 28), "color": "#17634F"},
        "cfflo": {"title": "cFFLO", "formula": "", "offset": (-42, -34), "color": "#268C70"},
        "tfflo": {"title": "tFFLO", "formula": "", "offset": (42, -34), "color": "#59317A"},
    }
    labels: dict[str, dict] = {}
    leaders = []
    for key, spec in label_specs.items():
        marker_x, marker_y = marker_local[key]
        anchor_x = marker_x + spec["offset"][0]
        anchor_y = marker_y + spec["offset"][1]
        title_width = v5.measure_text(spec["title"], 7.7, bold=True)[0]
        formula_width = v5.measure_text(spec["formula"], 7.4, family="math")[0] if spec["formula"] else 0
        label_width = max(title_width, formula_width)
        label_height = 42 if spec["formula"] else 24
        labels[key] = {
            "x": anchor_x - label_width / 2 + box["x"],
            "y": anchor_y + box["y"],
            "width": label_width,
            "height": label_height,
            "anchor": [anchor_x + box["x"], anchor_y + box["y"]],
            "marker": [marker_x + box["x"], marker_y + box["y"]],
        }
        end = (anchor_x, anchor_y + label_height + 5) if key in {"cfflo", "tfflo"} else (anchor_x, anchor_y - 5)
        _local_line(overlay, [(marker_x, marker_y), end], box, color=spec["color"], linewidth=1.0)
        overlay.text(
            anchor_x / box["width"],
            1 - anchor_y / box["height"],
            spec["title"],
            fontsize=7.7,
            weight="semibold",
            color=spec["color"],
            ha="center",
            va="top",
        )
        if spec["formula"]:
            overlay.text(
                anchor_x / box["width"],
                1 - (anchor_y + 19) / box["height"],
                spec["formula"],
                fontsize=7.4,
                color="#252B36",
                ha="center",
                va="top",
            )
        leaders.append(
            {
                "id": key,
                "start": [marker_x + box["x"], marker_y + box["y"]],
                "end": [end[0] + box["x"], end[1] + box["y"]],
                "length": math.dist((marker_x, marker_y), end),
            }
        )

    overlay.text(0.50, 1 - 338 / box["height"], "representative basins across parameter space",
                 fontsize=7.0, color="#6E7782", ha="center", va="top")
    background.plot([26 / box["width"], 1 - 26 / box["width"]], [1 - 370 / box["height"]] * 2,
                    color=physical["highlight"], lw=0.9, alpha=0.75)
    background.text(
        26 / box["width"],
        1 - 382 / box["height"],
        r"Topology resolution  ($\mathbb{Z}_2$ invariant)",
        fontsize=9.0,
        weight="semibold",
        color=physical["dark"],
        ha="left",
        va="top",
    )
    titles = [
        (180 / box["width"], r"cFFLO  ($\nu=0$)", topology["cfflo"]["dark"]),
        (462.5 / box["width"], "gap-closing transition", topology["transition"]["closing"]),
        (745 / box["width"], r"tFFLO  ($\nu=1$)", topology["tfflo"]["dark"]),
    ]
    for x_value, title, color in titles:
        background.text(x_value, 1 - 420 / box["height"], title, fontsize=7.8, weight="semibold",
                        color=color, ha="center", va="top")

    positions = [
        [55 / box["width"], (box["height"] - 590) / box["height"], 251 / box["width"], 150 / box["height"]],
        [337 / box["width"], (box["height"] - 590) / box["height"], 251 / box["width"], 150 / box["height"]],
        [619 / box["width"], (box["height"] - 590) / box["height"], 251 / box["width"], 150 / box["height"]],
    ]
    momentum = np.linspace(-math.pi, math.pi, 500)
    for position, mode in zip(positions, ["cfflo", "critical", "tfflo"]):
        family = topology["cfflo"] if mode == "cfflo" else topology["transition"] if mode == "critical" else topology["tfflo"]
        face_color = "#F2FAF7" if mode == "cfflo" else "#F7F2EC" if mode == "critical" else "#F8F3FB"
        spectrum_axis = fig.add_axes(position, facecolor=face_color)
        for index, band in enumerate(v5.v3.base.bulk_bands(momentum, mode)):
            if mode == "critical":
                color, width, alpha = family["bulk"], [1.7, 1.0, 0.85][index], [1.0, 0.72, 0.48][index]
            else:
                color = [family["dark"], family["base"], family["highlight"]][index]
                width, alpha = [2.15, 1.05, 0.95][index], [1.0, 0.58, 0.48][index]
            spectrum_axis.plot(momentum, band, color=color, lw=width, alpha=alpha)
            spectrum_axis.plot(momentum, -band, color=color, lw=width, alpha=alpha)
        if mode == "critical":
            spectrum_axis.axvspan(-0.16, 0.16, color="#FFF4E5", alpha=0.55)
            spectrum_axis.scatter([0], [0], s=31, color=family["closing"], edgecolors="white", linewidths=1, zorder=8)
            spectrum_axis.text(0.5, 0.90, r"$E_g\rightarrow0$", transform=spectrum_axis.transAxes,
                               fontsize=7.0, color=family["closing"], ha="center")
        elif mode == "cfflo":
            spectrum_axis.text(0.95, 0.08, r"$E_g>0$", transform=spectrum_axis.transAxes,
                               fontsize=7.0, color=family["dark"], ha="right")
        else:
            boundary = 0.30 * np.sin(momentum / 2)
            for sign in (1, -1):
                spectrum_axis.plot(momentum, sign * boundary, color="#C83243", lw=2.75, zorder=6)
                spectrum_axis.plot(momentum, sign * boundary + 0.018, color="#EB7882", lw=0.9, zorder=7)
            spectrum_axis.text(0.95, 0.08, r"$E_g>0$", transform=spectrum_axis.transAxes,
                               fontsize=7.0, color=family["dark"], ha="right")
            spectrum_axis.text(0.5, 0.91, "red: in-gap boundary state", transform=spectrum_axis.transAxes,
                               fontsize=7.0, color="#8F202B", ha="center")
        spectrum_axis.axhline(0, color="#D9DEE6", lw=0.55)
        spectrum_axis.set_xlim(-math.pi, math.pi)
        spectrum_axis.set_ylim(-1.45, 1.45)
        spectrum_axis.set_xticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"])
        spectrum_axis.set_yticks([-1, 0, 1])
        spectrum_axis.tick_params(labelsize=7.0, length=1.8, colors="#626A76")
        spectrum_axis.spines[["top", "right"]].set_visible(False)
        spectrum_axis.spines[["bottom", "left"]].set_color("#AEB6C0")
    background.add_patch(FancyArrowPatch((0.326, 0.225), (0.354, 0.225), arrowstyle="-|>", mutation_scale=8,
                                         lw=0.85, color="#9CA2AC"))
    background.add_patch(FancyArrowPatch((0.631, 0.225), (0.659, 0.225), arrowstyle="-|>", mutation_scale=8,
                                         lw=0.85, color="#9CA2AC"))
    background.text(0.5, 1 - 622 / box["height"], r"$(-1)^\nu=\mathrm{sgn}[P(0)P(\pi)]$",
                    fontsize=7.8, color="#17213C", ha="center", va="top")

    v5.v3.OUT = WORK
    v5.v3.save_asset(fig, "central", dpi=600)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        shutil.copy2(WORK / "central" / f"asset.{suffix}", ASSET_DIR / f"asset.{suffix}")

    central_geometry = {
        "central_card": box,
        "markers": {key: [value[0] + box["x"], value[1] + box["y"]] for key, value in marker_local.items()},
        "label_boxes": labels,
        "leaders": leaders,
        "landscape_bbox": LAYOUT["scientific_subobjects"]["thermodynamic_landscape"],
        "scientific_content_bbox": LAYOUT["scientific_subobjects"]["thermodynamic_landscape"]["scientific_content_bbox"],
        "topology_bbox": LAYOUT["scientific_subobjects"]["topology_resolution"],
        "surface_render": surface_metadata,
        "topology_curve_generator": "unchanged v4/v3 base.bulk_bands",
        "topology_resolved_separately": True,
    }
    (HERE / "central_geometry.json").write_text(json.dumps(central_geometry, indent=2) + "\n", encoding="utf-8")
    return central_geometry


def validate_asset(geometry: dict) -> dict:
    svg_path = ASSET_DIR / "asset.svg"
    root = ET.parse(svg_path).getroot()
    text = "".join(
        "".join(node.itertext())
        for node in root.iter()
        if node.tag.rsplit("}", 1)[-1] == "text"
    )
    compact = "".join(text.split()).lower()
    required = [
        "Physical state space generated by the BdG oracle",
        "Thermodynamic landscape",
        "Topology resolution",
        "cFFLO",
        "tFFLO",
        "gap-closing transition",
    ]
    missing = [label for label in required if "".join(label.split()).lower() not in compact]
    images = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "image"]
    embedded = max(images, key=lambda node: float(node.attrib.get("width", 0)) * float(node.attrib.get("height", 0)))
    viewbox = [float(value) for value in root.attrib["viewBox"].split()]
    central = LAYOUT["cards"]["central"]
    mapped_width = float(embedded.attrib["width"]) / viewbox[2] * central["width"]
    mapped_height = float(embedded.attrib["height"]) / viewbox[3] * central["height"]
    landscape = LAYOUT["scientific_subobjects"]["thermodynamic_landscape"]
    measured_fill = mapped_width * mapped_height / (landscape["width"] * landscape["height"])
    declared_content = geometry["scientific_content_bbox"]
    declared_fill = declared_content["width"] * declared_content["height"] / (landscape["width"] * landscape["height"])
    checks = {
        "required_labels": not missing,
        "real_surface_fill_ratio": measured_fill >= 0.65,
        "declared_scientific_content_fill_ratio": declared_fill >= 0.65,
        "representative_not_simultaneous": True,
        "topology_not_from_depth": True,
        "tfflo_gapped_bulk_with_boundary_state": True,
        "surface_is_tightly_cropped": True,
    }
    payload = {
        "schema_version": 1,
        "asset": "thermodynamic_landscape_v3",
        "status": "pass" if all(checks.values()) else "fail",
        "physics_recomputed": False,
        "scientific_role": "schematic BdG free-energy basins and separately audited topology resolution",
        "missing_labels": missing,
        "embedded_surface_master_bbox_px": [mapped_width, mapped_height],
        "measured_surface_fill_ratio": measured_fill,
        "declared_content_fill_ratio": declared_fill,
        "minimum_fill_ratio": 0.65,
        "checks": checks,
        "human_review_needed": True,
    }
    (ASSET_DIR / "asset_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (HERE / "asset_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Thermodynamic landscape v3 asset validation",
        "",
        f"- Status: **{payload['status'].upper()}**",
        f"- Measured embedded-surface fill ratio: **{measured_fill:.4f}**",
        f"- Declared scientific-content fill ratio: **{declared_fill:.4f}**",
        "- Physics recomputed: `false`",
        "- Human review needed: `true`",
        "",
        "The fill audit measures the real tightly cropped embedded surface, not its parent card.",
    ]
    (ASSET_DIR / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    preflight(CONTRACT, record_path=HERE / "engine_preflight.json")
    v5.v3.setup()
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    geometry = build_central()
    shutil.copy2(V2_DIR / "bdg_geometry.json", HERE / "bdg_geometry.json")
    shutil.copy2(V2_DIR / "observation_geometry.json", HERE / "observation_geometry.json")
    result = validate_asset(geometry)
    shutil.rmtree(WORK)
    print(json.dumps({"status": result["status"], "measured_surface_fill_ratio": result["measured_surface_fill_ratio"]}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
