from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from figures.figure_engine.builders.contract_loader import ContractError, load_contract
from figures.figure_engine.builders.figure_build import preflight
from figures.figure_engine.builders import rule_loader
from figures.figure_engine.builders.rule_loader import ENGINE_ROOT, RuleError, load_rule_bundle


class RulesAndContractsTest(unittest.TestCase):
    def test_canonical_rules_are_complete_and_hashed(self) -> None:
        rules = load_rule_bundle()
        self.assertEqual(set(rules.rules), set(rules.hashes))
        self.assertGreaterEqual(len(rules.rules), 14)
        self.assertIn("visual_narrative", rules.rules)
        self.assertTrue(all(len(value) == 64 for value in rules.hashes.values()))

    def test_all_future_figure_examples_validate(self) -> None:
        examples = sorted((ENGINE_ROOT / "examples").glob("*/figure_contract.yaml"))
        self.assertEqual(len(examples), 4)
        ids = {load_contract(path).figure_id for path in examples}
        self.assertEqual(len(ids), 4)

    def test_current_figure_one_contract_validates(self) -> None:
        contract = ENGINE_ROOT.parent / "fig1_framework_v3" / "refinement_v5" / "fig1_contract.yaml"
        loaded = load_contract(contract)
        self.assertEqual(loaded.figure_id, "FIG-01-PIPELINE-V5-ENGINE")
        self.assertEqual(loaded["approval"]["state"], "HUMAN_WAIT")

    def test_preflight_writes_reproducibility_record(self) -> None:
        contract = ENGINE_ROOT / "examples" / "fig1_example" / "figure_contract.yaml"
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "preflight.json"
            context = preflight(contract, record_path=record)
            payload = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_id"], context.contract.figure_id)
            self.assertEqual(payload["rule_hashes"], context.rules.hashes)
            self.assertEqual(payload["gate"]["state"], "HUMAN_WAIT")

    def test_unapproved_font_floor_is_rejected(self) -> None:
        source = ENGINE_ROOT / "examples" / "phase_map_example" / "figure_contract.yaml"
        text = source.read_text(encoding="utf-8").replace(
            "minimum_font_size_pt: 7.0", "minimum_font_size_pt: 5.0"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_contract.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ContractError):
                load_contract(path)

    def test_conflicting_arrow_rule_is_rejected_when_loaded(self) -> None:
        source = rule_loader.RULE_FILES["arrow"]
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["profiles"]["main_workflow"]["widths_px"] = {"tail": 6, "mid": 6, "near_head": 6}
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "arrow_rules.yaml"
            invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            original = rule_loader.RULE_FILES["arrow"]
            rule_loader.RULE_FILES["arrow"] = invalid
            try:
                with self.assertRaises(RuleError):
                    load_rule_bundle()
            finally:
                rule_loader.RULE_FILES["arrow"] = original


if __name__ == "__main__":
    unittest.main()
