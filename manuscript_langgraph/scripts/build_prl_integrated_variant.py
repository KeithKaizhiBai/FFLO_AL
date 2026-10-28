from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.figure_authority import CANONICAL_MAIN_FIGURES, sha256  # noqa: E402
from review_agent.graphs.prl_integrated_manuscript_graph import graph  # noqa: E402


def main() -> None:
    old_pdf = PROJECT_ROOT / "output/pdf/manuscript_methods_sm_discussion_review_v1/manuscript_intermediate_review.pdf"
    old_hash = sha256(old_pdf)
    figure_hashes = {
        figure_id: sha256(PROJECT_ROOT / record["source_pdf"])
        for figure_id, record in CANONICAL_MAIN_FIGURES.items()
    }
    result = graph.invoke({
        "project_root": str(PROJECT_ROOT),
        "manuscript_root": str(PACKAGE_ROOT),
        "figure_approval_package_path": str(PACKAGE_ROOT / "GATE_FIGURE_APPROVAL_PACKAGE"),
        "scope_approved": True,
        "claim_approved": True,
        "figure_approved": True,
        "results_human_approved": True,
        "external_retrieval_approved": False,
        "final_manuscript_approved": False,
        "gate_results": [],
        "metadata": {
            "publication_profile": "PRL_INTEGRATED_MAIN_TEXT",
            "new_physics_permission": False,
            "external_retrieval_approved": False,
        },
    })
    if result["workflow_status"] != "awaiting_prl_integrated_manuscript_human_review":
        raise SystemExit("integrated workflow did not stop at the required human gate")
    if sha256(old_pdf) != old_hash:
        raise SystemExit("parent intermediate review was modified")
    for figure_id, record in CANONICAL_MAIN_FIGURES.items():
        if sha256(PROJECT_ROOT / record["source_pdf"]) != figure_hashes[figure_id]:
            raise SystemExit(f"frozen figure changed: {figure_id}")
    gate = json.loads((PACKAGE_ROOT / "runs/active_learning_prl_integrated_v1/gate_records/prl_integrated_manuscript_human_review_gate.json").read_text(encoding="utf-8"))
    if gate["status"] != "HUMAN_WAIT":
        raise SystemExit("human gate was passed automatically")
    print("PRL_INTEGRATED_VARIANT_BUILD_PASS")
    print(f"candidate_pdf={PROJECT_ROOT / gate['candidate_pdf']}")
    print("current_gate=GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW")
    print("gate_status=HUMAN_WAIT")


if __name__ == "__main__":
    main()
