#!/usr/bin/env python3
"""Create warning-only visual-mass diagnostics for Fig.1 v3."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


HERE = Path(__file__).resolve().parent


def main() -> int:
    image = Image.open(HERE / "fig1_engine_v3.png").convert("RGB")
    gray = ImageOps.grayscale(image)
    ink = ImageChops.difference(Image.new("L", gray.size, 255), gray.filter(ImageFilter.GaussianBlur(radius=3.0)))
    preview = ImageEnhance.Contrast(ImageOps.colorize(ink, black="#FFFFFF", white="#193B6A")).enhance(1.8)
    preview.save(HERE / "saliency_preview.png")
    layout = yaml.safe_load((HERE / "layout.yaml").read_text(encoding="utf-8"))
    regions = {
        "thermodynamic_landscape": layout["scientific_subobjects"]["thermodynamic_landscape"],
        "topology_resolution": layout["scientific_subobjects"]["topology_resolution"],
        "exact_bdg_oracle_and_reliability": {
            "x": layout["cards"]["bdg"]["x"], "y": layout["cards"]["bdg"]["y"],
            "width": layout["cards"]["bdg"]["width"],
            "height": layout["cards"]["reliability"]["y"] + layout["cards"]["reliability"]["height"] - layout["cards"]["bdg"]["y"],
        },
        "active_learning_cycle": {"x": 0, "y": 0, "width": 1270, "height": 1080},
    }
    scale_x, scale_y = image.width / 1600.0, image.height / 1080.0
    masses = {}
    for name, box in regions.items():
        crop = ink.crop((round(box["x"] * scale_x), round(box["y"] * scale_y),
                         round((box["x"] + box["width"]) * scale_x), round((box["y"] + box["height"]) * scale_y)))
        values = list(crop.getdata())
        total = float(sum(values))
        masses[name] = {"integrated_visual_mass": total, "mean_visual_mass": total / max(len(values), 1)}
    ranking = sorted(masses, key=lambda name: masses[name]["mean_visual_mass"], reverse=True)
    payload = {
        "schema_version": 1,
        "status": "warning_only",
        "method": "Gaussian-blurred grayscale distance from white within declared narrative regions",
        "interpretation": "A diagnostic proxy only; it cannot certify first fixation or aesthetic quality.",
        "ranking_by_mean_visual_mass": ranking,
        "region_visual_mass": masses,
        "human_review_required": True,
    }
    (HERE / "saliency_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "ranking": ranking}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
