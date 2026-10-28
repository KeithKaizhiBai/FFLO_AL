#!/usr/bin/env python3
"""Validate the non-quantitative PRL figure style-gate package."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

SPECS = {
    "fig1_framework": {
        "svg": ["fig1_visual_mockup.svg", "fig1_vector_prototype.svg", "fig1_master.svg"],
        "png": ["fig1_visual_mockup.png", "fig1_vector_prototype.png", "fig1_preview.png"],
        "pdf": ["fig1_vector_prototype.pdf", "fig1_master.pdf"],
        "required_text": ["parameter", "BdG", "physical", "surrogate", "acquisition", "next exact", "reopened gapped phase"],
    },
    "fig2_thermodynamic": {
        "svg": ["fig2_visual_mockup.svg"],
        "png": ["fig2_visual_mockup.png"],
        "pdf": [],
        "required_text": ["initial design", "acquisition evolution", "final thermodynamic", "raw convergence"],
    },
    "fig3_topology": {
        "svg": ["fig3_visual_mockup.svg"],
        "png": ["fig3_visual_mockup.png"],
        "pdf": [],
        "required_text": ["trusted topology", "boundary support only", "neighbor sensitivity", "raw convergence"],
    },
    "fig4_3d_extension": {
        "svg": ["fig4_visual_mockup.svg"],
        "png": ["fig4_visual_mockup.png"],
        "pdf": [],
        "required_text": ["one 3D overview", "slice atlas", "physical evolution", "scaling evidence"],
    },
}

BANNED_ARTWORK_TERMS = [
    "sha256",
    "archive_commit",
    "recovered_artifacts",
    "provenance",
    "not passed",
    "complete 3d phase diagram",
    "fully converged 3d topology atlas",
]
BANNED_WARNING_COLORS = ["#c84545", "#a12626", "#b23a48", "#d62728", "#ff0000"]
REQUIRED_CONTRACT_KEYS = {"scientific_message", "panels", "source_datasets", "claim_ids", "allowed", "forbidden"}


def main() -> int:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for folder_name, spec in SPECS.items():
        folder = FIGURES / folder_name
        contract_path = folder / "figure_contract.yaml"
        manifest_path = folder / "data_manifest.json"
        log_path = folder / "regeneration.log"
        wrapper_path = folder / "build_mockup.py"

        for path in (contract_path, manifest_path, log_path, wrapper_path):
            add(f"{folder_name}:{path.name}:present", path.is_file(), str(path.relative_to(ROOT)))

        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        missing = sorted(REQUIRED_CONTRACT_KEYS - set(contract))
        add(f"{folder_name}:contract_fields", not missing, f"missing={missing}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        add(f"{folder_name}:no_quantitative_data", manifest.get("quantitative_data_used") is False, str(manifest.get("quantitative_data_used")))
        add(f"{folder_name}:no_physics_recompute", manifest.get("physics_recomputed") is False, str(manifest.get("physics_recomputed")))
        add(f"{folder_name}:no_manual_scientific_edits", manifest.get("manual_scientific_edits") is False, str(manifest.get("manual_scientific_edits")))

        svg_text = ""
        for name in spec["svg"]:
            path = folder / name
            add(f"{folder_name}:{name}:present", path.is_file(), str(path.relative_to(ROOT)))
            text = path.read_text(encoding="utf-8")
            svg_text += text.lower()
            add(f"{folder_name}:{name}:editable_text", text.count("<text") >= 5, f"text_nodes={text.count('<text')}")
            banned_terms = [term for term in BANNED_ARTWORK_TERMS if term in text.lower()]
            banned_colors = [color for color in BANNED_WARNING_COLORS if color in text.lower()]
            add(f"{folder_name}:{name}:no_artwork_metadata", not banned_terms, f"found={banned_terms}")
            add(f"{folder_name}:{name}:no_red_warning", not banned_colors, f"found={banned_colors}")

        missing_text = [term for term in spec["required_text"] if term.lower() not in svg_text]
        add(f"{folder_name}:required_hierarchy_text", not missing_text, f"missing={missing_text}")

        for name in spec["png"]:
            path = folder / name
            add(f"{folder_name}:{name}:present", path.is_file(), str(path.relative_to(ROOT)))
            with Image.open(path) as image:
                width, height = image.size
                add(f"{folder_name}:{name}:resolution", width >= 2000 and height >= 1200, f"{width}x{height}")
                corner = image.convert("RGB").getpixel((width - 2, 2))
                add(f"{folder_name}:{name}:white_background", min(corner) >= 245, f"corner_rgb={corner}")

        for name in spec["pdf"]:
            path = folder / name
            reader = PdfReader(path)
            add(f"{folder_name}:{name}:one_page", len(reader.pages) == 1, f"pages={len(reader.pages)}")

    passed = all(item["passed"] for item in checks)
    result = {
        "schema_version": 1,
        "gate": "GATE_FIGURE_STYLE_APPROVAL",
        "gate_status": "pending_human_approval",
        "automated_validation_passed": passed,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "scientific_scope_changed": False,
        "quantitative_figures_rebuilt": False,
    }
    (FIGURES / "style_gate_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    failed = [item for item in checks if not item["passed"]]
    lines = [
        "# Figure Style-Gate Validation",
        "",
        "- Gate: `GATE_FIGURE_STYLE_APPROVAL`",
        "- Gate status: **pending human approval**",
        f"- Automated checks: **{result['checks_passed']}/{result['checks_total']} passed**",
        "- Quantitative figures rebuilt: **no**",
        "- Scientific scope changed: **no**",
        "",
        "Automated validation covers bundle completeness, contract fields, raster",
        "resolution/background, editable SVG text, PDF page count, and prohibited",
        "in-artwork metadata/red-warning elements. Human review remains required for",
        "visual hierarchy and publication suitability.",
    ]
    if failed:
        lines.extend(["", "## Failed checks", ""])
        lines.extend(f"- `{item['name']}`: {item['detail']}" for item in failed)
    (FIGURES / "style_gate_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"passed": passed, "checks": len(checks), "failed": len(failed)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
