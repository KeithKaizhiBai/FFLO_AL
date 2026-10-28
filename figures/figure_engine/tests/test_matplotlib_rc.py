from __future__ import annotations

import unittest

from figures.figure_engine.builders.rule_loader import load_rule_bundle
from figures.figure_engine.styles.matplotlib_rc import build_rc


class MatplotlibRuleInheritanceTest(unittest.TestCase):
    def test_rc_is_derived_from_canonical_tokens(self) -> None:
        rules = load_rule_bundle()
        rc = build_rc(rules)
        fonts = rules["fonts"]
        self.assertEqual(rc["font.sans-serif"][0], fonts["sans"]["preferred"])
        self.assertEqual(rc["mathtext.fontset"], fonts["matplotlib"]["mathtext_fontset"])
        self.assertEqual(rc["axes.linewidth"], rules["svg_style"]["stroke"]["axis_pt"])
        self.assertEqual(rc["figure.facecolor"], rules["palette"]["neutral"]["white"])


if __name__ == "__main__":
    unittest.main()
