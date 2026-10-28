"""Shared validation result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Check:
    category: str
    name: str
    severity: str
    passed: bool
    detail: str
    evidence: Any = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check(category: str, name: str, passed: bool, detail: str, *, severity: str = "hard_fail", evidence=None) -> Check:
    return Check(category, name, severity, bool(passed), detail, {} if evidence is None else evidence)

