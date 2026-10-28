from __future__ import annotations

import unittest

from figures.figure_engine.builders.rule_loader import load_rule_bundle
from figures.figure_engine.validators.color_checker import validate


class ColorCheckerTest(unittest.TestCase):
    def test_registered_contrast_passes_and_fails(self) -> None:
        rules = load_rule_bundle().rules
        base = {
            "color_semantics": {
                "mappings": {"body": "physical"},
                "allowed_extra_hex": [],
                "contrast_pairs": [
                    {"id": "body", "role": "body_text", "foreground": "neutral.body", "background": "neutral.white"}
                ],
            },
            "validation_requirements": {"strict_palette": False},
        }
        checks = validate({"hex_colors": []}, base, rules)
        contrast = next(item for item in checks if item.name == "minimum_contrast")
        self.assertTrue(contrast.passed)

        base["color_semantics"]["contrast_pairs"] = [
            {"id": "bad", "role": "body_text", "foreground": "neutral.cool_light", "background": "neutral.white"}
        ]
        checks = validate({"hex_colors": []}, base, rules)
        contrast = next(item for item in checks if item.name == "minimum_contrast")
        self.assertFalse(contrast.passed)


if __name__ == "__main__":
    unittest.main()
