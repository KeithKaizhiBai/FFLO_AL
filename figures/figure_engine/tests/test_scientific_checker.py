from __future__ import annotations

import unittest

from figures.figure_engine.validators.scientific_checker import validate


class ScientificCheckerTest(unittest.TestCase):
    def test_required_and_forbidden_claims(self) -> None:
        contract = {
            "required_labels": ["higher-dimensional exploration", "3D extension"],
            "forbidden_claims": ["complete 3D phase diagram"],
            "forbidden_elements": [],
        }
        checks = validate(
            "3D extension for higher-dimensional exploration", contract
        )
        self.assertTrue(all(item.passed for item in checks))

        checks = validate(
            "3D extension: complete 3D phase diagram", contract
        )
        self.assertFalse(any(
            item.passed for item in checks if item.name == "prohibited_claims"
        ))


if __name__ == "__main__":
    unittest.main()
