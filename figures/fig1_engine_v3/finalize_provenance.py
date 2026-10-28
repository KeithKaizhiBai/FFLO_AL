#!/usr/bin/env python3
"""Record the Fig.1 v3 review-candidate provenance without passing its gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    names = [
        "fig1_contract.yaml", "layout.yaml", "asset_registry.yaml", "asset_manifest.json",
        "asset_validation.json", "engine_preflight.json", "geometry_manifest.json",
        "fig1_engine_v3.svg", "fig1_engine_v3.pdf", "fig1_engine_v3.png",
        "fig1_engine_v3_grayscale.png", "central_panel_preview.png", "saliency_preview.png",
        "rendered_geometry_audit.json", "pdf_page_audit.json", "print_preview_audit.json",
        "saliency_audit.json", "fig1_engine_v3_specific_lint.json", "validation_report.json",
        "visual_narrative_report.json", "refinement_log.yaml", "visual_review.md",
    ]
    missing = [name for name in names if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize provenance; missing: {missing}")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                                text=True, check=False).stdout.strip() or None
    except OSError:
        commit = None
    report = json.loads((HERE / "validation_report.json").read_text(encoding="utf-8"))
    specific = json.loads((HERE / "fig1_engine_v3_specific_lint.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "figure_id": "FIG-01-ENGINE-V3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit_hash": commit,
        "physics_recomputed": False,
        "source_figures_overwritten": False,
        "source_candidates": ["../fig1_engine", "../fig1_engine_v2"],
        "proposal_gate": {"gate": "GATE_FIG1_V3_PROPOSAL", "state": "APPROVED", "date": "2026-08-26"},
        "automated_validation": report["status"],
        "specific_validation": specific["status"],
        "warnings": report["warning_count"],
        "human_review_items": report["human_review_count"],
        "approval": {"gate": "GATE_FIG1_ENGINE_V3_FINAL_APPROVAL", "state": "HUMAN_WAIT"},
        "sha256": {name: sha256(HERE / name) for name in names},
    }
    (HERE / "regeneration_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "artifacts": len(names), "gate": payload["approval"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
