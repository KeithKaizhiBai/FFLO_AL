#!/usr/bin/env python3
"""Register scientific assets and write reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    asset_type: str
    visual_role: str
    path: str
    source: str
    schematic: bool
    bbox: dict[str, float] | None = None
    parameters_path: str | None = None
    features: list[str] | None = None
    validation_evidence: str | None = None


class AssetManager:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()
        self.records: list[AssetRecord] = []

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def register(self, record: AssetRecord) -> None:
        if any(existing.asset_id == record.asset_id for existing in self.records):
            raise ValueError(f"Duplicate asset id: {record.asset_id}")
        path = (self.base_dir / record.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.records.append(record)

    def write_manifest(self, output: str | Path, *, figure_id: str, commit_hash: str | None = None) -> Path:
        payload = {
            "schema_version": 1,
            "figure_id": figure_id,
            "commit_hash": commit_hash or "unavailable",
            "assets": [],
        }
        for record in self.records:
            path = (self.base_dir / record.path).resolve()
            payload["assets"].append({**asdict(record), "sha256": self.sha256(path)})
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target
