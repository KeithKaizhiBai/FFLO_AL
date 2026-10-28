#!/usr/bin/env python3
"""Mandatory preflight/postflight interface for every project figure build."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .contract_loader import FigureContract, load_contract
from .rule_loader import RuleBundle, load_rule_bundle


@dataclass(frozen=True)
class BuildContext:
    contract: FigureContract
    rules: RuleBundle

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "figure_id": self.contract.figure_id,
            "contract": str(self.contract.path),
            "rule_hashes": self.rules.hashes,
            "gate": self.contract["approval"],
        }


def preflight(contract_path: str | Path, *, record_path: str | Path | None = None) -> BuildContext:
    """Load all canonical rules and validate the contract before drawing."""
    rules = load_rule_bundle()
    contract = load_contract(contract_path, rules=rules)
    context = BuildContext(contract=contract, rules=rules)
    if record_path:
        target = Path(record_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(context.metadata(), indent=2) + "\n", encoding="utf-8")
    return context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract")
    parser.add_argument("--record", help="write the loaded rule/contract preflight record")
    args = parser.parse_args()
    context = preflight(args.contract, record_path=args.record)
    print(f"figure-engine preflight passed: {context.contract.figure_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

