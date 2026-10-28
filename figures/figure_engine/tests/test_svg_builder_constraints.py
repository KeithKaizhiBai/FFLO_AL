from __future__ import annotations

import unittest

from figures.figure_engine.builders.figure_build import preflight
from figures.figure_engine.builders.rule_loader import ENGINE_ROOT
from figures.figure_engine.builders.svg_builder import SVGBuilder


class SVGBuilderConstraintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        contract = ENGINE_ROOT / "examples" / "fig1_example" / "figure_contract.yaml"
        cls.context = preflight(contract)

    def make_builder(self) -> tuple[SVGBuilder, object]:
        builder = SVGBuilder.from_context(400, 240, self.context)
        return builder, builder.group("primary", role="primary")

    def test_rejects_font_below_contract_floor(self) -> None:
        builder, group = self.make_builder()
        with self.assertRaisesRegex(ValueError, "below contract minimum"):
            builder.text(group, 20, 30, "too small", role="body", size_pt=5.0,
                         bbox={"x": 20, "y": 20, "width": 40, "height": 12})

    def test_rejects_unapproved_color(self) -> None:
        builder, group = self.make_builder()
        with self.assertRaisesRegex(ValueError, "unapproved color"):
            builder.text(group, 20, 30, "bad color", role="body", size_pt=7.0,
                         bbox={"x": 20, "y": 20, "width": 50, "height": 12}, fill="#00FF00")

    def test_requires_measured_text_bbox(self) -> None:
        builder, group = self.make_builder()
        with self.assertRaisesRegex(ValueError, "measured positive bbox"):
            builder.text(group, 20, 30, "no width", role="body", size_pt=7.0,
                         bbox={"x": 20, "y": 20, "width": 0, "height": 12})

    def test_compliant_text_uses_canonical_font_and_is_registered(self) -> None:
        builder, group = self.make_builder()
        node = builder.text(group, 20, 30, "valid", role="body", size_pt=7.0,
                            bbox={"x": 20, "y": 20, "width": 30, "height": 12})
        self.assertEqual(node.get("font-family"), self.context.rules["fonts"]["sans"]["preferred"])
        self.assertEqual(len(builder.geometry), 1)

    def test_canvas_background_is_created_from_neutral_white_token(self) -> None:
        builder, _ = self.make_builder()
        background = next(
            node for node in builder.root
            if node.tag.rsplit("}", 1)[-1] == "rect" and node.get("data-role") == "canvas-background"
        )
        self.assertEqual(background.get("fill"), self.context.rules["palette"]["neutral"]["white"])


if __name__ == "__main__":
    unittest.main()
