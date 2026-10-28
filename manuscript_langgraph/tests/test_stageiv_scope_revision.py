from __future__ import annotations

import unittest

from review_agent.figure_authority import CANONICAL_MAIN_FIGURES_STAGEIV_REVISION
from review_agent.graphs.stageiv_scope_revision_graph import NODE_SEQUENCE
from review_agent.stageiv_scope_revision import TITLE


class StageIVScopeRevisionTest(unittest.TestCase):
    def test_title_is_venue_neutral(self) -> None:
        for token in ("PRL", "Physical Review", "Nature", "Science"):
            self.assertNotIn(token.lower(), TITLE.lower())

    def test_versioned_figure4_authority_is_used(self) -> None:
        record = CANONICAL_MAIN_FIGURES_STAGEIV_REVISION["FIG-04-STAGEIV-3D-EXTENSION"]
        self.assertEqual(record["source_pdf"], "figures/fig4_strict_data_v2/fig4_strict_data_v2.pdf")

    def test_graph_stops_at_human_gate(self) -> None:
        self.assertEqual(NODE_SEQUENCE[-1][0], "revision_human_gate")
        self.assertNotIn("automatic_approval", [name for name, _ in NODE_SEQUENCE])


if __name__ == "__main__":
    unittest.main()
