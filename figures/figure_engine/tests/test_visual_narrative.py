from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from figures.figure_engine.builders.contract_loader import ContractError, load_contract
from figures.figure_engine.builders.rule_loader import ENGINE_ROOT, load_rule_bundle
from figures.figure_engine.validators.visual_narrative_checker import validate


class VisualNarrativeTest(unittest.TestCase):
    def test_schema_v1_remains_backward_compatible(self) -> None:
        path = ENGINE_ROOT / "examples" / "fig1_example" / "figure_contract.yaml"
        contract = load_contract(path)
        self.assertEqual(contract["schema_version"], 1)
        self.assertFalse(contract.visual_narrative_enabled)

    def test_schema_v1_opt_in_requires_narrative_fields(self) -> None:
        source = ENGINE_ROOT / "examples" / "fig1_example" / "figure_contract.yaml"
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["visual_narrative"] = {"enabled": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_opt_in.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "visual narrative is enabled"):
                load_contract(path)

    def test_schema_v2_control_contract_validates(self) -> None:
        path = ENGINE_ROOT / "self_tests" / "corrected_figure" / "figure_contract.yaml"
        contract = load_contract(path)
        self.assertEqual(contract["schema_version"], 2)
        self.assertTrue(contract.visual_narrative_enabled)

    def test_parent_card_cannot_replace_primary_scientific_subobject(self) -> None:
        directory = ENGINE_ROOT / "self_tests" / "invalid_figure"
        contract = load_contract(directory / "figure_contract.yaml")
        manifest = yaml.safe_load((directory / "invalid_geometry.json").read_text(encoding="utf-8"))
        checks, _ = validate(manifest, contract.data, contract.rules.rules)
        result = next(item for item in checks if item.name == "narrative_primary_scientific_subobject")
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, "hard_fail")

    def test_control_scientific_subobject_and_fill_ratio_pass(self) -> None:
        directory = ENGINE_ROOT / "self_tests" / "corrected_figure"
        contract = load_contract(directory / "figure_contract.yaml")
        manifest = yaml.safe_load((directory / "corrected_geometry.json").read_text(encoding="utf-8"))
        checks, payload = validate(manifest, contract.data, contract.rules.rules)
        hard = [item for item in checks if item.severity == "hard_fail" and not item.passed]
        self.assertFalse(hard)
        self.assertEqual(payload["status"], "pass")
        saliency = next(item for item in checks if item.name == "narrative_saliency_evidence")
        self.assertEqual(saliency.severity, "warning")
        first_fixation = next(
            item for item in checks if item.name == "narrative_first_fixation_human_review"
        )
        self.assertEqual(first_fixation.severity, "human_review")

    def test_visual_narrative_rule_is_loaded_and_hashed(self) -> None:
        rules = load_rule_bundle()
        self.assertIn("visual_narrative", rules.rules)
        self.assertEqual(len(rules.hashes["visual_narrative"]), 64)


if __name__ == "__main__":
    unittest.main()
