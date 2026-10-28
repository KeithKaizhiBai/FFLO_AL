#!/usr/bin/env python3
"""Fig.1-v3 scientific, narrative, and frozen-benchmark assertions."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "fig1_engine"
V2 = HERE.parent / "fig1_engine_v2"
ENGINE = HERE.parent / "figure_engine"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def area(box: dict) -> float:
    return float(box["width"]) * float(box["height"])


def check(name: str, passed: bool, detail: str, evidence=None) -> dict:
    return {"check": name, "status": "pass" if passed else "fail", "detail": detail, "evidence": evidence}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    layout = yaml.safe_load((HERE / "layout.yaml").read_text(encoding="utf-8"))
    contract = yaml.safe_load((HERE / "fig1_contract.yaml").read_text(encoding="utf-8"))
    geometry = load_json(HERE / "geometry_manifest.json")
    asset = load_json(HERE / "asset_validation.json")
    rendered = load_json(HERE / "rendered_geometry_audit.json")
    page = load_json(HERE / "pdf_page_audit.json")
    print_preview = load_json(HERE / "print_preview_audit.json")

    root = ET.parse(HERE / "fig1_engine_v3.svg").getroot()
    extracted = " ".join(
        "".join(piece.strip(" \n\r\t") for piece in node.itertext()).replace("\u00a0", " ")
        for node in root.iter()
        if node.tag.rsplit("}", 1)[-1] == "text"
    )
    normalized = "".join(extracted.split()).lower()
    required_missing = [label for label in contract["required_labels"] if "".join(label.split()).lower() not in normalized]
    forbidden_present = [claim for claim in contract["forbidden_claims"] if "".join(claim.split()).lower() in normalized]
    prohibited_present = [claim for claim in contract["validation_requirements"]["prohibited_text"]
                          if "".join(claim.split()).lower() in normalized]

    cards = layout["cards"]
    primary_fraction = area(cards["central"]) / sum(area(box) for box in cards.values())
    landscape = layout["scientific_subobjects"]["thermodynamic_landscape"]
    content = landscape["scientific_content_bbox"]
    landscape_fraction = area(landscape) / area(cards["central"])
    declared_fill = area(content) / area(landscape)
    routes = geometry.get("routes", [])
    main_routes = [route for route in routes if route.get("type") == "main_workflow"]
    secondary_routes = [route for route in routes if route.get("type") == "secondary_workflow"]
    word_count = len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", extracted))

    benchmark = yaml.safe_load((ENGINE / "improvement" / "asset_benchmark_registry.yaml").read_text(encoding="utf-8"))
    expected_v1 = benchmark["benchmarks"]["fig1_engine_v1"]["sha256"]
    v1_hashes = {name: sha256(V1 / name) for name in expected_v1}
    v1_frozen = v1_hashes == expected_v1
    v2_manifest = load_json(V2 / "regeneration_manifest.json")
    expected_v2 = v2_manifest.get("sha256", {})
    v2_hashes = {name: sha256(V2 / name) for name in expected_v2 if (V2 / name).is_file()}
    v2_frozen = len(v2_hashes) == len(expected_v2) and v2_hashes == expected_v2
    topology_checks = asset.get("checks", {})

    checks = [
        check("engine_preflight", (HERE / "engine_preflight.json").is_file(), "Versioned Figure Engine preflight exists."),
        check("proposal_gate_recorded", contract["provenance"]["proposal_gate"]["state"] == "APPROVED",
              "The separate v3 proposal gate is explicitly approved."),
        check("v1_frozen", v1_frozen, "Frozen engine-v1 hashes match the benchmark.", v1_hashes),
        check("v2_frozen", v2_frozen, "Frozen engine-v2 files match its regeneration manifest."),
        check("primary_card_area", primary_fraction >= 0.55, f"Primary card fraction={primary_fraction:.4f}; required >=0.55."),
        check("primary_scientific_area", landscape_fraction >= 0.35, f"Landscape/central fraction={landscape_fraction:.4f}; required >=0.35."),
        check("declared_scientific_fill", declared_fill >= 0.65, f"Scientific content/landscape fraction={declared_fill:.4f}; required >=0.65."),
        check("measured_real_surface_fill", asset.get("measured_surface_fill_ratio", 0) >= 0.65,
              f"Measured embedded surface/landscape fraction={asset.get('measured_surface_fill_ratio', 0):.4f}; required >=0.65."),
        check("primary_is_scientific_subobject", any(item.get("id") == "thermodynamic_landscape" and item.get("type") != "card"
              and item.get("scientific_content") is True for item in geometry.get("objects", [])),
              "Primary visual object resolves to the landscape scientific subobject."),
        check("topology_subobject_registered", any(item.get("id") == "topology_resolution" and item.get("scientific_content") is True
              for item in geometry.get("objects", [])), "Topology is a separate registered scientific subobject."),
        check("workflow_route_count", len(main_routes) == 7 and len(secondary_routes) == 7,
              f"Found {len(main_routes)} main and {len(secondary_routes)} secondary workflow ribbons."),
        check("required_scientific_labels", not required_missing, "All required labels are present.", required_missing),
        check("forbidden_claims_absent", not forbidden_present and not prohibited_present,
              "No forbidden/prohibited claim appears in artwork.", forbidden_present + prohibited_present),
        check("topology_scientific_separation", topology_checks.get("representative_not_simultaneous") is True
              and topology_checks.get("topology_not_from_depth") is True
              and topology_checks.get("tfflo_gapped_bulk_with_boundary_state") is True,
              "Representative minima, gap/Pfaffian topology, and boundary-state semantics remain separate."),
        check("rendered_bbox_audit", rendered.get("status") == "pass", "Browser-measured registered geometry passes."),
        check("exact_print_size", page.get("status") == "pass", "PDF is exactly 178.00 × 120.15 mm."),
        check("print_previews", print_preview.get("status") == "pass" and (HERE / "fig1_engine_v3_grayscale.png").is_file(),
              "300-dpi color and grayscale previews exist."),
        check("artwork_word_budget", word_count <= int(contract["text_budget"]["maximum_words_in_artwork"]),
              f"Artwork word count={word_count}; maximum={contract['text_budget']['maximum_words_in_artwork']}.", word_count),
        check("no_physics_recomputation", asset.get("physics_recomputed") is False, "Asset provenance records physics_recomputed=false."),
        check("human_gate_held", contract["approval"] == {"gate": "GATE_FIG1_ENGINE_V3_FINAL_APPROVAL", "state": "HUMAN_WAIT"},
              "Automated validation leaves final Figure 1 approval to humans."),
    ]
    failures = [item for item in checks if item["status"] == "fail"]
    payload = {
        "schema_version": 1,
        "figure_id": contract["figure_id"],
        "status": "pass" if not failures else "fail",
        "hard_failure_count": len(failures),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "gate": contract["approval"],
    }
    (HERE / "fig1_engine_v3_specific_lint.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Figure 1 engine-v3 specific validation", "", f"- Status: **{payload['status'].upper()}**",
             f"- Hard failures: **{len(failures)}**", "", "| Result | Check | Detail |", "|---|---|---|"]
    lines.extend(f"| {item['status']} | `{item['check']}` | {item['detail']} |" for item in checks)
    lines.extend(["", "Automated validation does not pass `GATE_FIG1_ENGINE_V3_FINAL_APPROVAL`.", ""])
    (HERE / "fig1_engine_v3_specific_lint.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "hard_failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
