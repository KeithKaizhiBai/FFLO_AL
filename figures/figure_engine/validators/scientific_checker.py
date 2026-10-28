#!/usr/bin/env python3
"""Check required scientific labels and prohibited claims."""

from __future__ import annotations

import re

from .common import Check, check


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def validate(text: str, contract: dict) -> list[Check]:
    compact = _normalized(text)
    required = [str(value) for value in contract.get("required_labels", [])]
    forbidden = [str(value) for value in contract.get("forbidden_claims", [])]
    forbidden.extend(str(value) for value in contract.get("validation_requirements", {}).get("prohibited_text", []))
    missing = [value for value in required if _normalized(value) not in compact]
    present_forbidden = [value for value in forbidden if _normalized(value) in compact]
    return [
        check("scientific", "required_labels", not missing, "All contract-required scientific labels are present.", evidence=missing),
        check("scientific", "prohibited_claims", not present_forbidden, "No contract-prohibited claim appears in the artwork.", evidence=present_forbidden),
    ]

