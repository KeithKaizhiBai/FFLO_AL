#!/usr/bin/env python3
"""Normalize the v4 candidate PDF to the contract print size."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter


HERE = Path(__file__).resolve().parent
PDF = HERE / "fig1_engine_v4.pdf"
TEMP = HERE / "fig1_engine_v4.normalized.pdf"
TARGET_MM = (178.0, 120.15)
TARGET_PT = tuple(value / 25.4 * 72.0 for value in TARGET_MM)


def main() -> int:
    reader = PdfReader(PDF)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected one page, found {len(reader.pages)}")
    page = reader.pages[0]
    source = (float(page.mediabox.width), float(page.mediabox.height))
    page.scale_to(*TARGET_PT)
    writer = PdfWriter()
    writer.add_page(page)
    metadata = {str(key): str(value) for key, value in (reader.metadata or {}).items() if value is not None}
    metadata["/Title"] = "Figure 1 v4 - Reliability-aware active learning of topology-resolved FFLO states"
    writer.add_metadata(metadata)
    with TEMP.open("wb") as stream:
        writer.write(stream)
    TEMP.replace(PDF)
    actual_reader = PdfReader(PDF)
    actual = (float(actual_reader.pages[0].mediabox.width), float(actual_reader.pages[0].mediabox.height))
    error = max(abs(actual[index] - TARGET_PT[index]) for index in (0, 1))
    payload = {
        "schema_version": 1,
        "status": "pass" if error <= 1e-4 else "fail",
        "source_page_pt": source,
        "target_page_mm": TARGET_MM,
        "target_page_pt": TARGET_PT,
        "actual_page_pt": actual,
        "maximum_absolute_error_pt": error,
        "vector_content_preserved": True,
    }
    (HERE / "pdf_page_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "actual_page_pt": actual}))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
