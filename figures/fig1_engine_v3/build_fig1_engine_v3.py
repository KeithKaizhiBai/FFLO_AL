#!/usr/bin/env python3
"""Assemble the approved Fig.1 v3 visual-narrative review candidate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V2_BUILDER = HERE.parent / "fig1_engine_v2" / "build_fig1_engine_v2.py"


def _load_v2_builder():
    spec = importlib.util.spec_from_file_location("fig1_engine_v2_builder_for_v3", V2_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base assembler: {V2_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = _load_v2_builder()
    base.HERE = HERE
    base.CONTRACT = HERE / "fig1_contract.yaml"
    base.LAYOUT_PATH = HERE / "layout.yaml"
    base.REGISTRY_PATH = HERE / "asset_registry.yaml"
    base.BASE_GEOMETRY = HERE.parent / "fig1_engine" / "geometry_manifest.json"
    base.OUTPUT = HERE / "fig1_engine_v3.svg"
    result = base.main()
    if result:
        return int(result)

    svg = base.OUTPUT.read_text(encoding="utf-8")
    svg = svg.replace("engine-v2-", "engine-v3-")
    base.OUTPUT.write_text(svg, encoding="utf-8")

    geometry_path = HERE / "geometry_manifest.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    geometry["figure_id"] = "FIG-01-ENGINE-V3"
    thermodynamic = next(item for item in geometry["objects"] if item["id"] == "thermodynamic_landscape")
    content_bbox = thermodynamic["bbox"].pop("scientific_content_bbox")
    thermodynamic.update(
        {
            "type": "landscape_3d",
            "role": "primary_scientific_subobject",
            "scientific_role": "representative_bdg_free_energy_state_space",
            "scientific_content": True,
            "scientific_content_bbox": content_bbox,
            "visual_weight_object": False,
        }
    )
    layout = __import__("yaml").safe_load((HERE / "layout.yaml").read_text(encoding="utf-8"))
    geometry["objects"].append(
        {
            "id": "topology_resolution",
            "type": "topology",
            "role": "secondary_scientific_subobject",
            "scientific_role": "gap_and_pfaffian_resolved_fflo_topology",
            "scientific_content": True,
            "parent": "central",
            "bbox": layout["scientific_subobjects"]["topology_resolution"],
            "visual_weight_object": False,
        }
    )
    geometry_path.write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {base.OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
