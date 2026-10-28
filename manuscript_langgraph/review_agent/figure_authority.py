from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


# Single source of truth for artwork embedded in the manuscript.  The output
# filenames remain stable for LaTeX, while every source path and hash points to
# the Figure Engine / strict-data scientific freeze.
CANONICAL_MAIN_FIGURES: dict[str, dict[str, str]] = {
    "FIG-01-PIPELINE": {
        "number": "1",
        "source_pdf": "figures/fig1_engine_v4/fig1_engine_v4.pdf",
        "source_png": "figures/fig1_engine_v4/fig1_engine_v4.png",
        "staged_pdf": "Fig1_candidate.pdf",
        "staged_png": "Fig1_candidate.png",
        "pdf_sha256": "5c10a6a9c78e6697f4f8878057e898931344b3eb705d50ccdb435b2b8fccc715",
    },
    "FIG-02-THERMO": {
        "number": "2",
        "source_pdf": "figures/fig2_strict_data/fig2_strict_data.pdf",
        "source_png": "figures/fig2_strict_data/fig2_strict_data.png",
        "staged_pdf": "Fig2_candidate.pdf",
        "staged_png": "Fig2_candidate.png",
        "pdf_sha256": "77eba3a2e61d64f6f7febf16fa0822ab74f5be0fddeee4f1aad3f66126d866ec",
    },
    "FIG-03-STAGEIII-TOPOLOGY": {
        "number": "3",
        "source_pdf": "figures/fig3_strict_data/fig3_strict_data.pdf",
        "source_png": "figures/fig3_strict_data/fig3_strict_data.png",
        "staged_pdf": "Fig3_candidate.pdf",
        "staged_png": "Fig3_candidate.png",
        "pdf_sha256": "752f4645b87c0cf55da822dc949c7e06630e217dfc62d287f510aa05447992d8",
    },
    "FIG-04-STAGEIV-3D-EXTENSION": {
        "number": "4",
        "source_pdf": "figures/fig4_strict_data/fig4_strict_data.pdf",
        "source_png": "figures/fig4_strict_data/fig4_strict_data.png",
        "staged_pdf": "Fig4_candidate.pdf",
        "staged_png": "Fig4_candidate.png",
        "pdf_sha256": "3eb78a0c3c479e5c7837ccb5e175b35349b06ad4b942383d8f569d0e60cb5ccb",
    },
}


# Versioned authority for the Stage-IV scope revision.  Figures 1--3 retain
# their frozen sources; only Figure 4 is replaced by the Figure Engine v2
# candidate that removes the threshold-based panel (d).
CANONICAL_MAIN_FIGURES_STAGEIV_REVISION: dict[str, dict[str, str]] = {
    **{key: dict(value) for key, value in CANONICAL_MAIN_FIGURES.items() if key != "FIG-04-STAGEIV-3D-EXTENSION"},
    "FIG-04-STAGEIV-3D-EXTENSION": {
        "number": "4",
        "source_pdf": "figures/fig4_strict_data_v2/fig4_strict_data_v2.pdf",
        "source_png": "figures/fig4_strict_data_v2/fig4_strict_data_v2.png",
        "staged_pdf": "Fig4_candidate.pdf",
        "staged_png": "Fig4_candidate.png",
        "pdf_sha256": "b1f8f4d812ea913e7005789d60e46dc2529e8ccfc9c2b5eb47c9d8df54cc76f9",
    },
}


CANONICAL_FROZEN_HASHES = {
    "figures/fig1_engine_v4/fig1_engine_v4.svg": "f960829e98e7593aec44e540afe46aabe0da1e767ae28c708bddefd69d86f8d2",
    "figures/fig1_engine_v4/fig1_engine_v4.pdf": "5c10a6a9c78e6697f4f8878057e898931344b3eb705d50ccdb435b2b8fccc715",
    "figures/fig1_engine_v4/fig1_engine_v4.png": "a62af6eff5374917dd28ecd3315e535418564fee24d3cd2fad800ee834c726ed",
    "figures/fig2_strict_data/fig2_strict_data.svg": "f3ca32665cc75f8c1d98851f70e1e7713fd7c22c787caaac9237b633b8c8501f",
    "figures/fig2_strict_data/fig2_strict_data.pdf": "77eba3a2e61d64f6f7febf16fa0822ab74f5be0fddeee4f1aad3f66126d866ec",
    "figures/fig2_strict_data/fig2_strict_data.png": "cfc5a58eb40f1cb3fbbdbd0e4d1467d526e52954a2a48f90a06835fecc471f78",
    "figures/fig3_strict_data/fig3_strict_data.svg": "59fb9962ad4bea0ca3cbe6cf10f12face89ffd4221869bc6dda4580b32f71146",
    "figures/fig3_strict_data/fig3_strict_data.pdf": "752f4645b87c0cf55da822dc949c7e06630e217dfc62d287f510aa05447992d8",
    "figures/fig3_strict_data/fig3_strict_data.png": "6d7136ff1eed94e52d49a7dc052e02f419bec177fd54d1b4a703dbcff014cf51",
    "figures/fig4_strict_data/fig4_strict_data.svg": "33078bb56849fd0954ee097b5aea1602649d493d389d31107cbf526e5cbf310a",
    "figures/fig4_strict_data/fig4_strict_data.pdf": "3eb78a0c3c479e5c7837ccb5e175b35349b06ad4b942383d8f569d0e60cb5ccb",
    "figures/fig4_strict_data/fig4_strict_data.png": "42e5279221afee56b35ebef84dae22868a83430f97bdd50838b3698400b3cbb7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_canonical_frozen_artifacts(project_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative, expected in CANONICAL_FROZEN_HASHES.items():
        path = project_root / relative
        if not path.exists():
            raise RuntimeError(f"canonical frozen figure is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"canonical frozen figure hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual, "status": "PASS"})
    return rows


def stage_canonical_main_figures(
    project_root: Path,
    figures_dir: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    verify_canonical_frozen_artifacts(project_root)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for figure_id, record in CANONICAL_MAIN_FIGURES.items():
        source_pdf = project_root / record["source_pdf"]
        staged_pdf = figures_dir / record["staged_pdf"]
        shutil.copy2(source_pdf, staged_pdf)
        source_hash = sha256(source_pdf)
        staged_hash = sha256(staged_pdf)
        if source_hash != record["pdf_sha256"] or staged_hash != source_hash:
            raise RuntimeError(f"manuscript staging changed the frozen artwork: {figure_id}")
        source_png = project_root / record["source_png"]
        staged_png = figures_dir / record["staged_png"]
        if source_png.exists():
            shutil.copy2(source_png, staged_png)
        rows.append({
            "figure_id": figure_id,
            "source_path": record["source_pdf"],
            "expected_sha256": record["pdf_sha256"],
            "source_sha256": source_hash,
            "staged_path": str(staged_pdf.relative_to(audit_dir)),
            "staged_sha256": staged_hash,
            "status": "PASS",
        })

    tables = audit_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "manuscript_figure_source_hashes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": 1,
        "authority": "Figure Engine / strict-data scientific freeze",
        "status": "PASS",
        "figures": rows,
        "legacy_candidate_bundle_used": False,
    }
    (audit_dir / "manuscript_figure_source_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    markdown_rows = "\n".join(
        f"| {row['figure_id']} | `{row['source_path']}` | `{row['source_sha256']}` | PASS |"
        for row in rows
    )
    (audit_dir / "manuscript_figure_source_audit.md").write_text(
        "# Manuscript figure-source audit\n\n"
        "Status: **PASS**. The manuscript stages exact byte copies of the approved "
        "Figure Engine / strict-data PDFs. Legacy `manuscript_figures/final/` main-figure "
        "candidates are not used.\n\n"
        "| Figure | Authoritative source | SHA-256 | Status |\n"
        "|---|---|---|---|\n"
        f"{markdown_rows}\n",
        encoding="utf-8",
    )
    return payload


def stage_stageiv_revision_main_figures(
    project_root: Path,
    figures_dir: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    """Stage Figures 1--3 from freeze and the versioned Figure-4 revision."""
    verify_canonical_frozen_artifacts(project_root)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for figure_id, record in CANONICAL_MAIN_FIGURES_STAGEIV_REVISION.items():
        source_pdf = project_root / record["source_pdf"]
        source_png = project_root / record["source_png"]
        if not source_pdf.is_file():
            raise RuntimeError(f"versioned manuscript figure is missing: {source_pdf}")
        actual = sha256(source_pdf)
        if actual != record["pdf_sha256"]:
            raise RuntimeError(f"versioned manuscript figure hash mismatch: {figure_id}")
        staged_pdf = figures_dir / record["staged_pdf"]
        staged_png = figures_dir / record["staged_png"]
        shutil.copy2(source_pdf, staged_pdf)
        if source_png.is_file():
            shutil.copy2(source_png, staged_png)
        if sha256(staged_pdf) != actual:
            raise RuntimeError(f"staging changed versioned artwork: {figure_id}")
        rows.append({
            "figure_id": figure_id,
            "source_path": record["source_pdf"],
            "expected_sha256": record["pdf_sha256"],
            "source_sha256": actual,
            "staged_path": str(staged_pdf.relative_to(audit_dir)),
            "staged_sha256": sha256(staged_pdf),
            "status": "PASS",
        })
    tables = audit_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "manuscript_figure_source_hashes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "schema_version": 2,
        "authority": "Figure Engine scientific freezes plus Stage-IV versioned human-review candidate",
        "status": "PASS",
        "figures": rows,
        "stageiv_revision_human_gate": "HUMAN_WAIT",
    }
    (audit_dir / "manuscript_figure_source_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
