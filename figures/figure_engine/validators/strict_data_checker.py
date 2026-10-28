#!/usr/bin/env python3
"""Machine checks for strict-data source, semantics, and transformation contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..builders.panel_router import route_panel, validate_panel_contract
from .common import Check, check


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_count(path: Path) -> int | None:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return sum(1 for _ in csv.reader(stream)) - 1
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "x" in data:
                return int(data["x"].shape[0])
            first = data[data.files[0]]
            return int(first.shape[0]) if first.ndim else 1
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return len(value) if isinstance(value, list) else None
    return None


def validate(contract: dict[str, Any], contract_dir: Path) -> list[Check]:
    strict = contract.get("strict_data", {})
    if not isinstance(strict, dict) or strict.get("enabled") is not True:
        return []
    checks: list[Check] = []
    panels = strict.get("panels", [])
    structural: dict[str, list[str]] = {}
    routes = {}
    for panel in panels:
        panel_id = str(panel.get("panel_id", "unnamed"))
        errors = validate_panel_contract(panel)
        structural[panel_id] = errors
        if not errors:
            routes[panel_id] = list(route_panel(panel).gates)
    checks.append(check("strict_data", "panel_contracts", not any(structural.values()),
                        "Every strict-data panel satisfies the versioned structural contract.",
                        evidence={"errors": structural, "routes": routes}))

    source_failures = []
    for panel in panels:
        for source in panel.get("source_artifacts", []):
            path = Path(str(source.get("path", ""))).expanduser()
            path = path if path.is_absolute() else (contract_dir / path).resolve()
            if not path.is_file():
                source_failures.append({"panel": panel.get("panel_id"), "path": str(path), "error": "missing"})
                continue
            actual_hash = _sha256(path)
            expected_hash = source.get("sha256")
            actual_rows = _row_count(path)
            expected_rows = source.get("row_count")
            if actual_hash != expected_hash:
                source_failures.append({"panel": panel.get("panel_id"), "path": str(path), "error": "hash", "actual": actual_hash, "expected": expected_hash})
            if expected_rows is not None and actual_rows is not None and int(expected_rows) != actual_rows:
                source_failures.append({"panel": panel.get("panel_id"), "path": str(path), "error": "rows", "actual": actual_rows, "expected": expected_rows})
            if not source.get("version") or not source.get("fields") or not source.get("units"):
                source_failures.append({"panel": panel.get("panel_id"), "path": str(path), "error": "metadata"})
    checks.append(check("strict_data", "source_provenance", not source_failures,
                        "Every strict-data source exists and matches its registered hash, version, rows, fields, and units.",
                        evidence=source_failures))

    reconciliation = []
    for panel in panels:
        selected = panel.get("selected_row_count")
        excluded = panel.get("excluded_row_count")
        total = panel.get("input_row_count")
        if selected is not None and excluded is not None and int(selected) + int(excluded) != int(total):
            reconciliation.append({"panel": panel.get("panel_id"), "selected": selected, "excluded": excluded, "input": total})
    checks.append(check("strict_data", "row_reconciliation", not reconciliation,
                        "Selected and excluded counts reconcile with every declared input.", evidence=reconciliation))

    expected_outputs = strict.get("output_artifacts", [])
    missing_outputs = []
    for value in expected_outputs:
        path = Path(str(value)); path = path if path.is_absolute() else (contract_dir / path).resolve()
        if not path.is_file():
            missing_outputs.append(str(path))
    checks.append(check("strict_data", "output_manifest_complete", not missing_outputs,
                        "All contract-declared strict-data outputs are present.", evidence=missing_outputs))
    return checks
