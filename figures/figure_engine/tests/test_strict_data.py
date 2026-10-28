from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from figures.figure_engine.builders.panel_router import route_panel, validate_panel_contract
from figures.figure_engine.builders.rule_loader import ENGINE_ROOT, load_rule_bundle


class StrictDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.control = yaml.safe_load((ENGINE_ROOT / "self_tests/strict_data_fixtures/control_panel.yaml").read_text(encoding="utf-8"))

    def test_strict_data_rule_loaded(self) -> None:
        rules = load_rule_bundle()
        self.assertIn("strict_data", rules.rules)
        self.assertEqual(rules["strict_data"]["track_id"], "STRICT_DATA_FIGURE_TRACK")

    def test_control_panel_passes(self) -> None:
        self.assertEqual(validate_panel_contract(self.control), [])

    def test_router_adds_transformation_gate(self) -> None:
        panel = copy.deepcopy(self.control)
        panel["panel_type"] = "DERIVED_NUMERICAL_DATA"
        self.assertIn("GATE_TRANSFORMATION_AUTHORIZATION", route_panel(panel).gates)

    def test_unauthorized_interpolation_fails(self) -> None:
        panel = copy.deepcopy(self.control)
        panel["transformations"]["interpolation"]["enabled"] = True
        self.assertIn("unauthorized_interpolation", validate_panel_contract(panel))


if __name__ == "__main__":
    unittest.main()
