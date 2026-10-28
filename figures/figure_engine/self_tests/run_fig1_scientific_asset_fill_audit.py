#!/usr/bin/env python3
"""Read-only Fig.1-v1 audit of rendered surface area inside the landscape bbox."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
ENGINE_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
FIG1 = PROJECT_ROOT / "figures/fig1_engine"
THRESHOLD = 0.65


def _number(value: str | None) -> float:
    if value is None:
        return 0.0
    return float(value.replace("px", "").strip())


def main() -> int:
    contract = yaml.safe_load((FIG1 / "fig1_contract.yaml").read_text(encoding="utf-8"))
    layout = yaml.safe_load((FIG1 / "layout.yaml").read_text(encoding="utf-8"))
    central_asset = next(item for item in contract["assets"] if item["id"] == "central")
    asset_path = (FIG1 / central_asset["source"]).resolve()
    root = ET.parse(asset_path).getroot()
    viewbox = [float(value) for value in root.attrib["viewBox"].split()]
    images = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "image"]
    embedded = max(images, key=lambda node: _number(node.get("width")) * _number(node.get("height")))
    image_area_asset = _number(embedded.get("width")) * _number(embedded.get("height"))
    card = layout["cards"]["central"]
    scale_x = float(card["width"]) / viewbox[2]
    scale_y = float(card["height"]) / viewbox[3]
    image_area_master = image_area_asset * scale_x * scale_y
    landscape = layout["scientific_subobjects"]["thermodynamic_landscape"]
    landscape_area = float(landscape["width"]) * float(landscape["height"])
    ratio = image_area_master / landscape_area
    audit_status = "pass" if ratio >= THRESHOLD else "fail"
    payload = {
        "schema_version": 1,
        "figure_id": "FIG-01-ENGINE-V1",
        "artwork_modified": False,
        "asset": str(asset_path),
        "method": "largest embedded scientific raster area mapped into the master landscape bbox",
        "surface_image_size_asset_units": [_number(embedded.get("width")), _number(embedded.get("height"))],
        "surface_area_master_px2": image_area_master,
        "landscape_bbox_area_master_px2": landscape_area,
        "fill_ratio": ratio,
        "minimum": THRESHOLD,
        "audit_status": audit_status,
        "expected_failure_detected": audit_status == "fail",
        "interpretation": "The declared landscape bbox is large, but the actual rendered free-energy surface fills only a minority of it.",
    }
    output = HERE / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "fig1_scientific_asset_fill_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fig.1 scientific-asset fill-ratio audit", "",
        f"- Artwork modified: **{payload['artwork_modified']}**",
        f"- Audit status: **{audit_status.upper()}**",
        f"- Surface/landscape fill ratio: **{ratio:.4f}**",
        f"- Required proposal minimum: **{THRESHOLD:.2f}**", "",
        "The failure is expected for the frozen engine-v1 benchmark and demonstrates why",
        "bbox area alone is not sufficient. Detecting this known failure is a passing",
        "framework self-test; it does not authorize artwork modification.", "",
    ]
    (output / "fig1_scientific_asset_fill_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload))
    return 0 if payload["expected_failure_detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
