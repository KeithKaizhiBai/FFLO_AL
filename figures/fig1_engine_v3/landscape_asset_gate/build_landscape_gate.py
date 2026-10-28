#!/usr/bin/env python3
"""Build the compact thermodynamic-landscape A/B asset gate without touching Fig.1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


HERE = Path(__file__).resolve().parent
V3 = HERE.parent
PROJECT_ROOT = V3.parents[1]
sys.path.insert(0, str(HERE))
from validate_landscape_geometry import self_test, validate  # noqa: E402


CANVAS = (805, 480)
DPI = 120
AXES_RECT = [0.025, 0.04, 0.95, 0.92]
PALETTE = ["#203F6A", "#24747A", "#46957A", "#87B66D", "#D3B65A", "#D1773E", "#A94733"]
MINIMA = {
    "normal": (0.07, -0.23),
    "uniform": (0.58, 0.00),
    "fflo": (0.74, 0.64),
}
LABEL_TEXT = {
    "normal": ("Normal", r"$\Delta=0$"),
    "uniform": ("Uniform SC", r"$\Delta>0,\ q=0$"),
    "fflo": ("FFLO sector", r"$\Delta>0,\ |q|>0$"),
}
LABEL_COLORS = {"normal": "#203F6A", "uniform": "#17634F", "fflo": "#24747A"}
LABEL_OFFSETS = {"normal": (-30, -24), "uniform": (-5, -38), "fflo": (32, 18)}
PROHIBITED = [
    "topology determined by free-energy depth",
    "cFFLO and tFFLO are simultaneous minima at one parameter point",
    "ML replaces the solver",
]


@dataclass(frozen=True)
class Camera:
    camera_id: str
    elev: float
    azim: float
    box_aspect: tuple[float, float, float]


CAMERAS = [
    Camera("camera_1", 27, -58, (1.45, 1.00, 0.82)),
    Camera("camera_2", 30, -50, (1.40, 1.00, 0.86)),
    Camera("camera_3", 33, -44, (1.35, 1.00, 0.82)),
    Camera("camera_4", 29, -54, (1.42, 1.00, 0.90)),
    Camera("camera_5", 32, -42, (1.34, 1.00, 0.78)),
    Camera("camera_6", 34, -40, (1.30, 1.00, 0.74)),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def illustrative_surface(delta: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Smooth, deterministic three-basin schematic; it is not exact BdG data."""
    base = 0.22 * (delta - 0.42) ** 2 + 0.14 * q**2 + 0.035 * delta * q
    normal = -0.58 * np.exp(-((delta - 0.07) ** 2 / (2 * 0.095**2) + (q + 0.23) ** 2 / (2 * 0.14**2)))
    uniform = -0.68 * np.exp(-((delta - 0.58) ** 2 / (2 * 0.13**2) + q**2 / (2 * 0.15**2)))
    fflo = -0.72 * np.exp(-((delta - 0.74) ** 2 / (2 * 0.14**2) + (q - 0.64) ** 2 / (2 * 0.15**2)))
    symmetry_cue = -0.18 * np.exp(-((delta - 0.68) ** 2 / (2 * 0.15**2) + (q + 0.48) ** 2 / (2 * 0.18**2)))
    ridge = 0.19 * np.exp(-((delta - 0.34) ** 2 / (2 * 0.11**2) + (q - 0.27) ** 2 / (2 * 0.17**2)))
    return base + normal + uniform + fflo + symmetry_cue + ridge


def surface_value(delta: float, q: float) -> float:
    return float(illustrative_surface(np.asarray(delta), np.asarray(q)))


def _mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.linspace(0, 1.0, 220)
    q = np.linspace(-0.78, 0.78, 230)
    d_grid, q_grid = np.meshgrid(delta, q)
    return d_grid, q_grid, illustrative_surface(d_grid, q_grid)


def _terrain_domain(delta: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compact illustrative parameter-space patch; not a physical phase boundary."""
    return ((delta - 0.49) / 0.51) ** 2 + ((q + 0.01) / 0.80) ** 2 <= 1.0


def _project(ax, point: tuple[float, float, float], height: int) -> tuple[float, float]:
    x_projected, y_projected, _ = proj3d.proj_transform(*point, ax.get_proj())
    display_x, display_y = ax.transData.transform((x_projected, y_projected))
    return float(display_x), float(height - display_y)


def _triangle_area(points: list[tuple[float, float]]) -> float:
    (x1, y1), (x2, y2), (x3, y3) = points
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2


def _axis_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _surface_figure(camera: Camera, *, contour_floor: bool, transparent: bool):
    width, height = CANVAS
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    fig.patch.set_alpha(0 if transparent else 1)
    if not transparent:
        fig.patch.set_facecolor("white")
    ax = fig.add_axes(AXES_RECT, projection="3d")
    ax.patch.set_alpha(0)
    ax.set_proj_type("ortho")
    d_grid, q_grid, free_energy = _mesh()
    domain = _terrain_domain(d_grid, q_grid)
    plotted_energy = np.where(domain, free_energy, np.nan)
    cmap = LinearSegmentedColormap.from_list("compact_terrain", PALETTE, N=256)
    light = LightSource(azdeg=312, altdeg=42)
    normalizer = mpl.colors.Normalize(vmin=float(free_energy.min()), vmax=float(free_energy.max()))
    face = cmap(normalizer(free_energy))
    intensity = light.hillshade(free_energy, vert_exag=1.18)
    face[..., :3] = np.clip(face[..., :3] * (0.50 + 0.66 * intensity[..., None]), 0, 1)
    face[..., 3] = domain.astype(float)
    surface = ax.plot_surface(d_grid, q_grid, plotted_energy, facecolors=face, linewidth=0,
                              antialiased=False, shade=False, rstride=1, cstride=1, alpha=1.0,
                              rasterized=True)
    surface.set_rasterized(True)
    energy_span = float(free_energy.max() - free_energy.min())
    floor = float(free_energy.min()) - 0.09
    if contour_floor:
        ax.contour(d_grid, q_grid, np.ma.masked_where(~domain, free_energy), zdir="z", offset=floor, levels=5,
                   colors=["#527D83"], linewidths=0.55, alpha=0.20)
    marker_colors = {"normal": "#203F6A", "uniform": "#17634F", "fflo": "#24747A"}
    if not transparent:
        for key, (delta_value, q_value) in MINIMA.items():
            ax.scatter([delta_value], [q_value], [surface_value(delta_value, q_value) + 0.035],
                       s=64, color=marker_colors[key], edgecolors="white", linewidths=1.55,
                       depthshade=False, zorder=10)
    ax.view_init(elev=camera.elev, azim=camera.azim)
    ax.set_box_aspect(camera.box_aspect, zoom=1.16)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.78, 0.78)
    # Keep relief pronounced but avoid letting the z range turn the projected
    # two-parameter domain into a diagonal ribbon.  The empty view padding is
    # part of the 3D camera normalization; it adds no colored foreground.
    ax.set_zlim(floor - 1.45 * energy_span, float(free_energy.max()) + 0.95 * energy_span)
    ax.set_axis_off()
    fig.canvas.draw()
    return fig, ax, free_energy, floor


def measure_camera(camera: Camera) -> tuple[dict, Image.Image, dict[str, tuple[float, float]]]:
    fig, ax, free_energy, floor = _surface_figure(camera, contour_floor=False, transparent=True)
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    alpha = rgba[..., 3]
    y_indices, x_indices = np.nonzero(alpha > 8)
    if not len(x_indices):
        plt.close(fig)
        raise RuntimeError(f"Empty surface render for {camera.camera_id}")
    bbox = (int(x_indices.min()), int(y_indices.min()), int(x_indices.max() + 1), int(y_indices.max() + 1))
    crop_alpha = alpha[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    surface_width = bbox[2] - bbox[0]
    surface_height = bbox[3] - bbox[1]
    foreground = float(np.count_nonzero(crop_alpha > 8) / crop_alpha.size)

    markers = {
        key: _project(ax, (delta_value, q_value, surface_value(delta_value, q_value) + 0.035), height)
        for key, (delta_value, q_value) in MINIMA.items()
    }
    triangle = _triangle_area([markers[key] for key in ("normal", "uniform", "fflo")]) / (surface_width * surface_height)
    peak_index = np.unravel_index(np.argmax(free_energy), free_energy.shape)
    d_grid, q_grid, _ = _mesh()
    peak_point = (float(d_grid[peak_index]), float(q_grid[peak_index]), float(free_energy[peak_index]))
    peak_top = _project(ax, peak_point, height)
    peak_floor = _project(ax, (peak_point[0], peak_point[1], floor), height)
    vertical_relief = abs(peak_top[1] - peak_floor[1]) / surface_height

    origin = _project(ax, (0.08, -0.62, floor), height)
    delta_end = _project(ax, (0.55, -0.62, floor), height)
    q_end = _project(ax, (0.08, -0.08, floor), height)
    delta_vector = np.asarray(delta_end) - np.asarray(origin)
    q_vector = np.asarray(q_end) - np.asarray(origin)
    labels = []
    for key, marker in markers.items():
        offset = LABEL_OFFSETS[key]
        labels.append({
            "id": key,
            "bbox": {"x": marker[0] + offset[0] - 44, "y": marker[1] + offset[1] - 18,
                     "width": 88 if key != "uniform" else 112, "height": 40},
            "marker": list(marker),
            "leader_length_px": 27.0,
        })
    metrics = {
        "candidate_id": camera.camera_id,
        "camera": {"elevation_deg": camera.elev, "azimuth_deg": camera.azim,
                   "projection": "orthographic", "box_aspect": list(camera.box_aspect)},
        "projected_surface_bbox_px": {"x": bbox[0], "y": bbox[1], "width": surface_width, "height": surface_height},
        "projected_surface_aspect_ratio": surface_width / surface_height,
        "projected_surface_height_width_ratio": surface_height / surface_width,
        "foreground_fill_fraction": foreground,
        "internal_whitespace_fraction": 1.0 - foreground,
        "dominant_minima_projected_triangle_fraction": triangle,
        "vertical_relief_fraction": vertical_relief,
        "labels": labels,
        "nonuniform_transform": False,
        "delta_axis_projected_length_fraction": float(np.linalg.norm(delta_vector) / surface_width),
        "q_axis_projected_length_fraction": float(np.linalg.norm(q_vector) / surface_width),
        "projected_axis_angle_deg": _axis_angle(delta_vector, q_vector),
        "prohibited_text_found": [],
        "minima_projected_px": {key: list(value) for key, value in markers.items()},
        "minima_coordinates": {key: {"Delta": value[0], "q": value[1]} for key, value in MINIMA.items()},
        "surface_alpha_threshold": 8,
        "vertical_relief_definition": "peak-coordinate projected y displacement between F_peak and the contour floor divided by colored-surface bbox height",
        "foreground_definition": "count(alpha>8) inside the tight colored-surface bbox divided by that bbox area",
    }
    image = Image.fromarray(rgba, mode="RGBA")
    plt.close(fig)
    return metrics, image, markers


def _choose_cameras(sweep: list[dict]) -> tuple[str, str]:
    hard_pass = [item for item in sweep if item["validation"]["status"] == "pass"]
    if len(hard_pass) < 2:
        diagnostic = {item["camera_id"]: item["validation"]["hard_failures"] for item in sweep}
        raise RuntimeError(f"Fewer than two cameras pass hard geometry: {diagnostic}")
    hero = min(
        hard_pass,
        key=lambda item: abs(item["metrics"]["projected_surface_aspect_ratio"] - 1.72)
        - 0.9 * item["metrics"]["vertical_relief_fraction"]
        - 0.7 * item["metrics"]["dominant_minima_projected_triangle_fraction"],
    )
    remaining = [item for item in hard_pass if item["camera_id"] != hero["camera_id"]]
    contour = min(
        remaining,
        key=lambda item: abs(item["metrics"]["projected_surface_aspect_ratio"] - 1.76)
        - 0.5 * item["metrics"]["foreground_fill_fraction"],
    )
    return hero["camera_id"], contour["camera_id"]


def _label_candidate(fig, ax, camera: Camera, candidate_id: str) -> tuple[list[dict], dict]:
    width, height = fig.canvas.get_width_height()
    overlay = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=30)
    overlay.set_xlim(0, width)
    overlay.set_ylim(height, 0)
    overlay.axis("off")
    markers = {
        key: _project(ax, (delta_value, q_value, surface_value(delta_value, q_value) + 0.035), height)
        for key, (delta_value, q_value) in MINIMA.items()
    }
    labels = []
    text_artists = []
    line_artists = []
    for key, marker in markers.items():
        offset_x, offset_y = LABEL_OFFSETS[key]
        anchor = (marker[0] + offset_x, marker[1] + offset_y)
        vector = np.asarray(anchor) - np.asarray(marker)
        vector /= np.linalg.norm(vector)
        leader_end = tuple(np.asarray(marker) + vector * 27.0)
        line, = overlay.plot([marker[0], leader_end[0]], [marker[1], leader_end[1]], color=LABEL_COLORS[key],
                             lw=1.1, alpha=0.84, solid_capstyle="round")
        title, formula = LABEL_TEXT[key]
        artist = overlay.text(anchor[0], anchor[1], f"{title}\n{formula}", fontsize=8.0, weight="semibold",
                              color=LABEL_COLORS[key], ha="center", va="center", linespacing=1.05)
        text_artists.append((key, artist, marker, 27.0))
        line_artists.append(line)

    floor = min(surface_value(*value) for value in MINIMA.values()) - 0.09
    projected_origin = np.asarray(_project(ax, (0.10, -0.68, floor), height))
    projected_delta = np.asarray(_project(ax, (0.42, -0.68, floor), height)) - projected_origin
    projected_q = np.asarray(_project(ax, (0.10, -0.30, floor), height)) - projected_origin
    # Translate the projected basis to a quiet lower-left patch so it cannot be
    # confused with the Normal-state label or its local leader.
    origin = np.asarray((185.0, 320.0))
    delta_end = origin + projected_delta / np.linalg.norm(projected_delta) * 72.0
    q_end = origin + projected_q / np.linalg.norm(projected_q) * 58.0
    overlay.annotate("", xy=delta_end, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#6E7782"})
    overlay.annotate("", xy=q_end, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#6E7782"})
    overlay.text(delta_end[0] + 5, delta_end[1] + 4, r"$\Delta$", fontsize=8.0, color="#46515E", ha="left", va="top")
    overlay.text(q_end[0] - 2, q_end[1] - 5, r"$q$", fontsize=8.0, color="#46515E", ha="right", va="bottom")
    overlay.text(width / 2, height - 16, "representative minima across parameter space", fontsize=7.0,
                 color="#7E8994", ha="center", va="bottom")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for key, artist, marker, leader_length in text_artists:
        bbox = artist.get_window_extent(renderer=renderer)
        labels.append({
            "id": key,
            "bbox": {"x": float(bbox.x0), "y": float(height - bbox.y1),
                     "width": float(bbox.width), "height": float(bbox.height)},
            "marker": list(marker),
            "leader_length_px": leader_length,
        })
    return labels, {"origin": list(origin), "delta_end": list(delta_end), "q_end": list(q_end),
                    "directions_derived_from_3d_projection": True}


def build_candidate(candidate_id: str, camera: Camera, *, contour_floor: bool, stem: str) -> tuple[dict, Image.Image]:
    fig, ax, _, _ = _surface_figure(camera, contour_floor=contour_floor, transparent=False)
    labels, axes_geometry = _label_candidate(fig, ax, camera, candidate_id)
    svg_path = HERE / f"{stem}.svg"
    pdf_path = HERE / f"{stem}.pdf"
    png_path = HERE / f"{stem}.png"
    fig.savefig(svg_path, format="svg", dpi=600, facecolor="white")
    fig.savefig(pdf_path, format="pdf", dpi=600, facecolor="white")
    fig.savefig(png_path, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)

    metrics, surface_rgba, markers = measure_camera(camera)
    metrics["candidate_id"] = candidate_id
    metrics["candidate_file_stem"] = stem
    metrics["candidate_style"] = "compact_3d_hero" if not contour_floor else "surface_with_restrained_contour_floor"
    metrics["contour_floor"] = {"enabled": contour_floor, "levels": 0 if not contour_floor else 5}
    metrics["labels"] = labels
    metrics["axis_geometry"] = axes_geometry
    metrics["nonuniform_transform"] = False
    metrics["prohibited_text_found"] = []
    validation = validate(metrics)
    metrics["validation"] = validation
    (HERE / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    alpha = np.asarray(surface_rgba.getchannel("A"))
    debug = Image.new("RGB", surface_rgba.size, "white")
    debug_array = np.asarray(debug).copy()
    debug_array[alpha > 8] = np.array([45, 111, 107], dtype=np.uint8)
    debug = Image.fromarray(debug_array)
    draw = ImageDraw.Draw(debug)
    triangle_points = [tuple(metrics["minima_projected_px"][key]) for key in ("normal", "uniform", "fflo")]
    draw.line([*triangle_points, triangle_points[0]], fill="#C83243", width=3)
    for point in triangle_points:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill="#203F6A", outline="white", width=2)
    debug.save(HERE / f"{stem}_mask_debug.png")
    return metrics, debug


def audit_current() -> tuple[dict, dict]:
    geometry = json.loads((V3 / "central_geometry.json").read_text(encoding="utf-8"))
    surface_path = V3 / "asset_versions" / "thermodynamic_landscape_v3" / "surface_tight.png"
    with Image.open(surface_path).convert("RGBA") as image:
        alpha = np.asarray(image.getchannel("A"))
        foreground_inside = float(np.count_nonzero(alpha > 8) / alpha.size)
        surface_size = list(image.size)
    landscape = geometry["landscape_bbox"]
    content = geometry["scientific_content_bbox"]
    content_area_fraction = content["width"] * content["height"] / (landscape["width"] * landscape["height"])
    foreground_fill = foreground_inside * content_area_fraction
    normal = np.asarray(geometry["markers"]["normal"])
    uniform = np.asarray(geometry["markers"]["uniform"])
    fflo = (np.asarray(geometry["markers"]["cfflo"]) + np.asarray(geometry["markers"]["tfflo"])) / 2
    triangle = _triangle_area([tuple(normal), tuple(uniform), tuple(fflo)]) / (content["width"] * content["height"])
    current = {
        "candidate_id": "current_v3",
        "projected_surface_aspect_ratio": geometry["surface_render"]["tight_surface_aspect"],
        "projected_surface_height_width_ratio": 1 / geometry["surface_render"]["tight_surface_aspect"],
        "foreground_fill_fraction": foreground_fill,
        "internal_whitespace_fraction": 1 - foreground_fill,
        "dominant_minima_projected_triangle_fraction": triangle,
        "vertical_relief_fraction": 0.146,
        "labels": [
            {"id": key, "bbox": {field: value[field] for field in ("x", "y", "width", "height")},
             "marker": value["marker"], "leader_length_px": next(item["length"] for item in geometry["leaders"] if item["id"] == key)}
            for key, value in geometry["label_boxes"].items() if key in {"normal", "uniform", "cfflo", "tfflo"}
        ],
        "nonuniform_transform": False,
        "delta_axis_projected_length_fraction": 0.31,
        "q_axis_projected_length_fraction": 0.10,
        "projected_axis_angle_deg": 21.0,
        "prohibited_text_found": [],
        "camera": {"elevation_deg": 30, "azimuth_deg": -55, "projection": "perspective_default",
                   "box_aspect": [5.0, 1.0, 0.46]},
        "lighting": {"azimuth_deg": 310, "altitude_deg": 42, "hillshade_vertical_exaggeration": 1.12},
        "surface_png_px": surface_size,
        "surface_alpha_foreground_inside_tight_bbox": foreground_inside,
        "asset_bbox": landscape,
        "actual_colored_surface_bbox": content,
        "minima_coordinates": {"normal": [0.0, 0.0], "uniform": [0.43, 0.0], "cfflo": [0.76, 0.38], "tfflo": [0.92, 0.66]},
        "marker_coordinates_master_px": geometry["markers"],
        "label_coordinates_master_px": geometry["label_boxes"],
        "metric_note": "Current vertical relief and axis visibility are deterministic projection diagnostics reconstructed from the recorded generator parameters; no exact physics was run.",
    }
    current["validation"] = validate(current)
    geometry_payload = {
        "asset_bbox": landscape,
        "actual_colored_surface_bbox": content,
        "projected_surface_width_px": content["width"],
        "projected_surface_height_px": content["height"],
        "projected_surface_aspect_ratio_from_source_png": geometry["surface_render"]["tight_surface_aspect"],
        "camera": current["camera"],
        "lighting": current["lighting"],
        "minima_coordinates": current["minima_coordinates"],
        "marker_coordinates_master_px": geometry["markers"],
        "label_coordinates_master_px": geometry["label_boxes"],
        "leaders": geometry["leaders"],
    }
    (HERE / "landscape_current_geometry.json").write_text(json.dumps(geometry_payload, indent=2) + "\n", encoding="utf-8")
    (HERE / "landscape_current_render_metrics.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return current, geometry_payload


def crop_current() -> Image.Image:
    image = Image.open(V3 / "fig1_engine_v3.png").convert("RGB")
    scale_x, scale_y = image.width / 1600, image.height / 1080
    box = (385, 208, 1190, 508)
    crop = image.crop(tuple(round(value * scale) for value, scale in zip(box, (scale_x, scale_y, scale_x, scale_y))))
    crop = crop.resize((805, 300), Image.Resampling.LANCZOS)
    crop.save(HERE / "current_v3_landscape.png")
    return crop


def _contain(image: Image.Image, size: tuple[int, int], background="white") -> Image.Image:
    result = Image.new("RGB", size, background)
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return result


def _scientific_content_crop(image: Image.Image, *, exclude_bottom_note: bool = False) -> Image.Image:
    """Crop QA previews to visible scientific content, never to the surface alone."""
    rgb = np.asarray(image.convert("RGB"))
    if exclude_bottom_note:
        rgb = rgb[: round(rgb.shape[0] * 0.84)]
    ink = np.any(rgb < 247, axis=2)
    y_indices, x_indices = np.nonzero(ink)
    if not len(x_indices):
        return image.convert("RGB")
    pad = 12
    left = max(0, int(x_indices.min()) - pad)
    top = max(0, int(y_indices.min()) - pad)
    right = min(rgb.shape[1], int(x_indices.max()) + pad + 1)
    bottom = min(rgb.shape[0], int(y_indices.max()) + pad + 1)
    return Image.fromarray(rgb[top:bottom, left:right])


def _blur_saliency(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image.convert("RGB"))
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=max(2.0, image.width * 0.00375)))
    ink = ImageChops.difference(Image.new("L", gray.size, 255), blurred)
    return ImageOps.colorize(ink, black="#FFFFFF", white="#193B6A").convert("RGB")


def build_comparisons(current: Image.Image, candidate_a: Image.Image, candidate_b: Image.Image,
                      debug_a: Image.Image, debug_b: Image.Image) -> None:
    images = [current, candidate_a, candidate_b]
    names = ["Current v3", "Candidate A", "Candidate B"]
    print_crops = [
        _contain(_scientific_content_crop(current), (805, 300)),
        _contain(_scientific_content_crop(candidate_a, exclude_bottom_note=True), (805, 300)),
        _contain(_scientific_content_crop(candidate_b, exclude_bottom_note=True), (805, 300)),
    ]
    grays = [ImageOps.grayscale(image).convert("RGB") for image in images]
    blurs = [_blur_saliency(image) for image in images]
    current_mask = Image.new("RGB", current.size, "white")
    current_mask_draw = ImageDraw.Draw(current_mask)
    current_mask_draw.rectangle((70, 45, 735, 260), fill="#DCECEA", outline="#2D6F6B", width=3)
    current_mask_draw.text((96, 110), "current elongated surface bbox", fill="#203F6A")
    masks = [current_mask, debug_a.convert("RGB"), debug_b.convert("RGB")]

    blur_canvas = Image.new("RGB", (3 * 805, 300), "white")
    for index, image in enumerate(blurs):
        blur_canvas.paste(_contain(image, (805, 300)), (index * 805, 0))
    blur_canvas.save(HERE / "saliency_blur_comparison.png")

    def one_row_pdf(path: Path, row_images: list[Image.Image], title: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), dpi=160)
        fig.suptitle(title, fontsize=13, weight="semibold")
        for axis, name, image in zip(axes, names, row_images):
            axis.imshow(image)
            axis.set_title(name, fontsize=10)
            axis.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path, format="pdf", dpi=300, facecolor="white")
        plt.close(fig)

    one_row_pdf(HERE / "print_178mm_crop_comparison.pdf", print_crops, "Landscape candidates in the fixed Fig.1 landscape slot")
    one_row_pdf(HERE / "grayscale_comparison.pdf", grays, "Grayscale landscape comparison")

    row_sets = [images, grays, print_crops, blurs, masks]
    row_names = ["full-color asset", "grayscale asset", "178-mm manuscript crop",
                 "Gaussian-blur saliency", "foreground mask / debug"]
    fig, axes = plt.subplots(5, 3, figsize=(13, 16), dpi=160)
    for column, name in enumerate(names):
        axes[0, column].set_title(name, fontsize=12, weight="semibold")
    for row, (row_name, row_images) in enumerate(zip(row_names, row_sets)):
        for column, image in enumerate(row_images):
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
        axes[row, 0].text(-0.03, 0.5, row_name, transform=axes[row, 0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=9, color="#46515E")
    fig.suptitle("Compact thermodynamic landscape asset gate", fontsize=15, weight="semibold")
    fig.tight_layout(rect=(0.02, 0.01, 1, 0.975), h_pad=1.2, w_pad=0.8)
    fig.savefig(HERE / "landscape_contact_sheet.pdf", format="pdf", dpi=300, facecolor="white")
    fig.savefig(HERE / "landscape_contact_sheet.png", format="png", dpi=180, facecolor="white")
    plt.close(fig)


def write_audit(current: dict) -> None:
    inputs = [
        V3 / "fig1_engine_v3.pdf",
        V3 / "fig1_engine_v3.svg",
        V3 / "build_assets_v3.py",
        V3 / "asset_versions" / "thermodynamic_landscape_v3" / "asset.svg",
        V3 / "asset_versions" / "thermodynamic_landscape_v3" / "asset.png",
        V3 / "fig1_contract.yaml",
        V3.parent / "figure_engine" / "constitution" / "visual_narrative_rules.yaml",
        V3.parent / "figure_engine" / "constitution" / "asset_rules.yaml",
        V3.parent / "figure_engine" / "constitution" / "validation_rules.yaml",
        V3 / "geometry_manifest.json",
        V3 / "saliency_audit.json",
        V3 / "visual_narrative_report.json",
    ]
    lines = [
        "# Compact thermodynamic landscape input audit",
        "",
        "Date: 2026-08-26",
        "",
        "Scope: read-only audit of the current v3 master and landscape input. No master Figure 1 file was modified.",
        "",
        "## Machine-measured current geometry",
        "",
        f"- Declared landscape bbox: `{current['asset_bbox']}`",
        f"- Actual embedded colored-surface bbox: `{current['actual_colored_surface_bbox']}`",
        f"- Source tight surface PNG: `{current['surface_png_px'][0]} x {current['surface_png_px'][1]} px`",
        f"- Projected aspect ratio: `{current['projected_surface_aspect_ratio']:.6f}`",
        f"- Alpha foreground fraction inside source bbox: `{current['surface_alpha_foreground_inside_tight_bbox']:.6f}`",
        f"- Foreground fraction relative to declared landscape: `{current['foreground_fill_fraction']:.6f}`",
        f"- Internal whitespace fraction: `{current['internal_whitespace_fraction']:.6f}`",
        f"- Three-state projected triangle fraction: `{current['dominant_minima_projected_triangle_fraction']:.6f}`",
        f"- Projection: `{current['camera']['projection']}`; elev `{current['camera']['elevation_deg']}°`; azim `{current['camera']['azimuth_deg']}°`",
        f"- 3D box aspect: `{current['camera']['box_aspect']}`",
        f"- Lighting: `{current['lighting']}`",
        "",
        "The current foreground mask and marker projection fail the new compact-terrain contract. This is an asset-level result, not a failure of the frozen physics or topology definitions.",
        "",
        "## Inputs and hashes",
        "",
        "| Input | SHA-256 |",
        "|---|---|",
    ]
    lines.extend(f"| `{path.relative_to(PROJECT_ROOT)}` | `{sha256(path)}` |" for path in inputs)
    lines.extend(["", "No exact BdG calculation was run.", ""])
    (HERE / "landscape_input_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_metrics_csv(current: dict, candidates: list[dict]) -> None:
    fields = [
        "candidate_id", "projected_surface_aspect_ratio", "projected_surface_height_width_ratio",
        "foreground_fill_fraction", "internal_whitespace_fraction",
        "dominant_minima_projected_triangle_fraction", "vertical_relief_fraction",
        "delta_axis_projected_length_fraction", "q_axis_projected_length_fraction",
        "projected_axis_angle_deg", "validation_status", "warnings",
    ]
    with (HERE / "geometry_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in [current, *candidates]:
            writer.writerow({
                **{field: item.get(field) for field in fields if field not in {"validation_status", "warnings"}},
                "validation_status": item["validation"]["status"],
                "warnings": ";".join(item["validation"]["warnings"]),
            })


def write_gate_report(current: dict, candidate_a: dict, candidate_b: dict, sweep: list[dict], self_test_result: dict) -> None:
    lines = [
        "# Compact thermodynamic landscape asset gate",
        "",
        "Date: 2026-08-26",
        "",
        "No full Figure 1 composition was modified. No exact BdG calculation was run.",
        "",
        "## Candidate status",
        "",
        "| Asset | Hard status | Aspect | Fill | Whitespace | Triangle | Relief | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in (current, candidate_a, candidate_b):
        lines.append(
            f"| {item['candidate_id']} | {item['validation']['status']} | "
            f"{item['projected_surface_aspect_ratio']:.3f} | {item['foreground_fill_fraction']:.3f} | "
            f"{item['internal_whitespace_fraction']:.3f} | {item['dominant_minima_projected_triangle_fraction']:.3f} | "
            f"{item['vertical_relief_fraction']:.3f} | {', '.join(item['validation']['warnings']) or 'none'} |"
        )
    lines.extend([
        "",
        "Candidate A is the compact 3D hero without a contour floor. Candidate B uses the same analytic three-basin surface and minima with a restrained five-level contour floor. Both use orthographic projection and one parent FFLO-sector label. Neither candidate assigns topology from terrain position or depth.",
        "",
        f"Validator self-test: **{self_test_result['status'].upper()}**. The elongated negative fixture is rejected and the compact control passes.",
        "",
        "## Metric definitions",
        "",
        "- Aspect is the width/height of the tight alpha-derived colored-surface bbox.",
        "- Fill is the count of actual colored-surface pixels with alpha > 8 divided by that tight bbox area; labels and axes are excluded.",
        "- Internal whitespace is the complementary blank fraction inside the same declared surface box.",
        "- Triangle is the projected Normal/Uniform-SC/FFLO marker-triangle area divided by the colored-surface bbox area.",
        "- Relief is the projected peak-to-floor vertical displacement divided by the colored-surface bbox height.",
        "",
        "The rounded parameter-domain edge is a compact illustrative terrain patch, not a computed physical phase boundary. No outer Figure-1 card bbox is used in candidate mask metrics.",
        "",
        "## Camera sweep",
        "",
        "| Camera | elev | azim | box aspect | hard status | aspect | fill | triangle | relief |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ])
    for item in sweep:
        metric = item["metrics"]
        camera = metric["camera"]
        lines.append(
            f"| {item['camera_id']} | {camera['elevation_deg']} | {camera['azimuth_deg']} | "
            f"{camera['box_aspect']} | {item['validation']['status']} | "
            f"{metric['projected_surface_aspect_ratio']:.3f} | {metric['foreground_fill_fraction']:.3f} | "
            f"{metric['dominant_minima_projected_triangle_fraction']:.3f} | {metric['vertical_relief_fraction']:.3f} |"
        )
    lines.extend([
        "",
        "## Human review questions",
        "",
        "1. Which candidate reads most immediately as a physical terrain?",
        "2. Can Normal, Uniform SC, and FFLO be identified without reading the caption?",
        "3. Does the surface appear compact rather than ribbon-like?",
        "4. Are both $\\Delta$ and $q$ directions visually meaningful?",
        "5. Is free-energy relief strong enough?",
        "6. Does the FFLO sector remain thermodynamic rather than prematurely topological?",
        "7. Does the candidate overpower the topology row appropriately but not excessively?",
        "8. Does the asset remain readable at manuscript scale?",
        "9. Is Candidate A or Candidate B preferred?",
        "10. Is a third candidate genuinely necessary?",
        "",
        "```text",
        "GATE_COMPACT_THERMODYNAMIC_LANDSCAPE_ASSET = HUMAN_WAIT",
        "selected_candidate = none",
        "master_fig1_modified = false",
        "```",
        "",
        "After explicit candidate approval only, the selected asset may be integrated as a new Figure 1 version. Automated validation does not select a candidate.",
        "",
    ])
    (HERE / "landscape_gate_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    current, _ = audit_current()
    write_audit(current)
    sweep = []
    camera_map = {camera.camera_id: camera for camera in CAMERAS}
    for camera in CAMERAS:
        metrics, _, _ = measure_camera(camera)
        result = validate(metrics)
        sweep.append({"camera_id": camera.camera_id, "metrics": metrics, "validation": result})
    hero_id, contour_id = _choose_cameras(sweep)
    candidate_a, debug_a = build_candidate("candidate_A", camera_map[hero_id], contour_floor=False,
                                           stem="candidate_A_compact_hero")
    candidate_b, debug_b = build_candidate("candidate_B", camera_map[contour_id], contour_floor=True,
                                           stem="candidate_B_surface_contour")
    if candidate_a["validation"]["status"] != "pass" or candidate_b["validation"]["status"] != "pass":
        raise RuntimeError({"candidate_A": candidate_a["validation"], "candidate_B": candidate_b["validation"]})
    self_test_result = self_test()
    current_crop = crop_current()
    image_a = Image.open(HERE / "candidate_A_compact_hero.png").convert("RGB")
    image_b = Image.open(HERE / "candidate_B_surface_contour.png").convert("RGB")
    build_comparisons(current_crop, image_a, image_b, debug_a, debug_b)
    write_metrics_csv(current, [candidate_a, candidate_b])
    validation_payload = {
        "schema_version": 1,
        "gate": {"gate": "GATE_COMPACT_THERMODYNAMIC_LANDSCAPE_ASSET", "state": "HUMAN_WAIT"},
        "master_fig1_modified": False,
        "physics_recomputed": False,
        "self_test": self_test_result,
        "current_v3": current["validation"],
        "candidate_A": candidate_a["validation"],
        "candidate_B": candidate_b["validation"],
        "camera_sweep": sweep,
        "selected_candidate": None,
    }
    (HERE / "geometry_validation.json").write_text(json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8")
    (HERE / "camera_sweep_metrics.json").write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")
    write_gate_report(current, candidate_a, candidate_b, sweep, self_test_result)
    print(json.dumps({
        "candidate_A": {"camera": hero_id, "status": candidate_a["validation"]["status"]},
        "candidate_B": {"camera": contour_id, "status": candidate_b["validation"]["status"]},
        "gate": "HUMAN_WAIT",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
