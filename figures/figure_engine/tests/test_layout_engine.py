from __future__ import annotations

import unittest

from figures.figure_engine.builders.layout_engine import BBox, snap, visual_area_fraction


class LayoutEngineTest(unittest.TestCase):
    def test_bbox_geometry(self) -> None:
        outer = BBox(0, 0, 100, 100)
        inner = BBox(10, 10, 40, 30)
        separate = BBox(120, 10, 20, 20)
        self.assertTrue(outer.contains(inner))
        self.assertTrue(outer.overlaps(inner))
        self.assertFalse(inner.overlaps(separate))
        self.assertEqual(inner.gap(separate), 70)

    def test_grid_snap_and_visual_hierarchy(self) -> None:
        self.assertEqual(snap(27, 8), 24)
        primary = BBox(0, 0, 80, 80)
        secondary = BBox(90, 0, 40, 40)
        self.assertAlmostEqual(
            visual_area_fraction([primary], [primary, secondary]), 0.8
        )


if __name__ == "__main__":
    unittest.main()
