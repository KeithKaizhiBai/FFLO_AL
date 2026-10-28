#!/usr/bin/env python3
"""Inspect SVG editability, raster exceptions, transforms, and metadata."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import Check, check


def _local(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _parse_scale(transform: str) -> tuple[float, float] | None:
    match = re.search(r"scale\(\s*([-+0-9.eE]+)(?:[ ,]+([-+0-9.eE]+))?\s*\)", transform)
    if not match:
        return None
    sx = float(match.group(1)); sy = float(match.group(2) or match.group(1))
    return sx, sy


def inspect(path: str | Path, contract: dict) -> tuple[list[Check], dict]:
    svg_path = Path(path)
    try:
        root = ET.parse(svg_path).getroot()
    except Exception as exc:
        return [check("svg", "parse", False, f"SVG parse failed: {exc}")], {}
    nodes = list(root.iter())
    text_nodes = [node for node in nodes if _local(node) == "text"]
    image_nodes = [node for node in nodes if _local(node) == "image"]
    paths = [node for node in nodes if _local(node) == "path"]
    nonuniform, skewed = [], []
    for node in nodes:
        transform = node.get("transform", "")
        scale = _parse_scale(transform)
        if scale and abs(abs(scale[0]) - abs(scale[1])) > 1e-3:
            nonuniform.append(node.get("id") or _local(node))
        if re.search(r"skew[XY]\(", transform):
            skewed.append(node.get("id") or _local(node))
    required_metadata = {
        "figure_id": root.get("data-figure-id") or root.get("id"),
        "contract": root.get("data-contract"),
        "engine_schema": root.get("data-engine-schema"),
    }
    allow_raster = any(asset.get("allow_embedded_raster") for asset in contract.get("assets", []))
    declared_exceptions = contract.get("layout", {}).get("declared_exceptions", [])
    legacy_metadata_exception = any(
        isinstance(item, dict) and item.get("id") == "legacy_svg_without_engine_metadata"
        for item in declared_exceptions
    )
    raw = svg_path.read_text(encoding="utf-8")
    summary = {
        "text_nodes": len(text_nodes), "image_nodes": len(image_nodes), "path_nodes": len(paths),
        "metadata": required_metadata, "raw_text": " ".join("".join(node.itertext()) for node in text_nodes),
        "hex_colors": sorted({value.upper() for value in re.findall(r"#[0-9a-fA-F]{6}", raw)}),
    }
    checks = [
        check("svg", "parse", True, "SVG parses successfully."),
        check("svg", "editable_text", bool(text_nodes), f"SVG contains {len(text_nodes)} editable text nodes."),
        check("svg", "raster_policy", not image_nodes or allow_raster, f"SVG contains {len(image_nodes)} image nodes; raster exception declared={allow_raster}.", evidence=[node.get("id") for node in image_nodes]),
        check("typography", "svg_nonuniform_scaling", not nonuniform, "No SVG group uses non-uniform scale.", evidence=nonuniform),
        check("typography", "svg_skew", not skewed, "No SVG group uses skew transforms.", evidence=skewed),
        check("svg", "engine_metadata", all(required_metadata.values()),
              "SVG records figure id, contract path, and engine schema.",
              severity="warning" if legacy_metadata_exception else "hard_fail", evidence=required_metadata),
    ]
    return checks, summary
