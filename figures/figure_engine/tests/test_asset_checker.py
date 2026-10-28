from __future__ import annotations

import unittest

from figures.figure_engine.builders.rule_loader import load_rule_bundle
from figures.figure_engine.validators.asset_checker import validate


class AssetCheckerTest(unittest.TestCase):
    def test_topology_schematic_requires_reopened_gapped_tfflo_features(self) -> None:
        rules = load_rule_bundle().rules
        contract = {
            "primary_object": {"id": "topology"},
            "secondary_objects": [],
            "tertiary_information": [],
            "assets": [{
                "id": "topology", "type": "topology_schematic", "source": "synthetic",
                "schematic": True, "bbox": "topology", "features": ["cfflo_gapped_nu0"],
            }],
        }
        checks = validate(contract, rules)
        feature_check = next(item for item in checks if item.name == "required_scientific_features")
        self.assertFalse(feature_check.passed)
        self.assertIn("tfflo_reopened_gapped_nu1", feature_check.evidence[0][2])

        contract["assets"][0]["features"] = rules["asset"]["topology_schematic"]["required"]
        checks = validate(contract, rules)
        feature_check = next(item for item in checks if item.name == "required_scientific_features")
        self.assertTrue(feature_check.passed)


if __name__ == "__main__":
    unittest.main()
