from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from figures.figure_engine.builders.figure_build import preflight
from figures.figure_engine.builders.rule_loader import ENGINE_ROOT
from figures.figure_engine.validators.figure_lint import _integration_checks


class PreflightIntegrityTest(unittest.TestCase):
    def test_stale_rule_hash_invalidates_preflight(self) -> None:
        contract_path = ENGINE_ROOT / "examples" / "fig1_example" / "figure_contract.yaml"
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "preflight.json"
            context = preflight(contract_path, record_path=record)
            context.contract.data["validation_requirements"]["engine_preflight_record"] = str(record)
            self.assertTrue(_integration_checks(context.contract)[0].passed)

            payload = json.loads(record.read_text(encoding="utf-8"))
            payload["rule_hashes"]["arrow"] = "stale"
            record.write_text(json.dumps(payload), encoding="utf-8")
            check = _integration_checks(context.contract)[0]
            self.assertFalse(check.passed)
            self.assertFalse(check.evidence["hashes_match_current_rules"])


if __name__ == "__main__":
    unittest.main()
