from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.graphs.prl_manuscript_graph import NODE_SEQUENCE, graph  # noqa: E402


def main() -> None:
    for path in [
        PACKAGE_ROOT / "registries/claim_registry.yaml",
        PACKAGE_ROOT / "registries/evidence_registry.yaml",
        PACKAGE_ROOT / "registries/figure_registry.yaml",
        PACKAGE_ROOT / "vertical_slice_stageIII/claim_contract.yaml",
        PACKAGE_ROOT / "vertical_slice_stageIII/figure_contract.yaml",
    ]:
        json.loads(path.read_text(encoding="utf-8"))

    def invoke(**approvals):
        state = {
            "project_root": str(PROJECT_ROOT),
            "manuscript_root": str(PACKAGE_ROOT),
            "scope_approved": False,
            "claim_approved": False,
            "figure_approved": False,
            "external_retrieval_approved": False,
            "final_manuscript_approved": False,
            "gate_results": [],
        }
        state.update(approvals)
        return graph.invoke(state)

    scope_wait = invoke()
    claim_wait = invoke(scope_approved=True)
    figure_wait = invoke(scope_approved=True, claim_approved=True)
    with tempfile.TemporaryDirectory() as temporary:
        results_path = str(Path(temporary) / "results")
        results_wait = invoke(
            scope_approved=True,
            claim_approved=True,
            figure_approved=True,
            results_package_path=results_path,
        )
        retrieval_wait = invoke(
            scope_approved=True,
            claim_approved=True,
            figure_approved=True,
            results_human_approved=True,
            results_package_path=results_path,
        )
        full = invoke(
            scope_approved=True,
            claim_approved=True,
            figure_approved=True,
            results_human_approved=True,
            results_package_path=results_path,
            external_retrieval_approved=True,
        )

    staged_expectations = [
        (scope_wait, "GATE_SCOPE_APPROVAL", "awaiting_scope_approval"),
        (claim_wait, "GATE_CLAIM_APPROVAL", "awaiting_claim_approval"),
        (figure_wait, "GATE_FIGURE_APPROVAL", "awaiting_figure_approval"),
        (results_wait, "GATE_RESULTS_HUMAN_REVIEW", "awaiting_results_human_review"),
        (retrieval_wait, "INTRO_CITATION_LOOP_GATE", "awaiting_introduction_retrieval_permission"),
        (full, "GATE_FINAL_MANUSCRIPT_APPROVAL", "awaiting_human_approval"),
    ]
    for result, expected_gate, expected_status in staged_expectations:
        gate_ids = {gate["gate_id"] for gate in result["gate_results"]}
        if expected_gate not in gate_ids:
            raise SystemExit(f"staged traversal did not reach {expected_gate}")
        if result["workflow_status"] != expected_status:
            raise SystemExit(
                f"unexpected status at {expected_gate}: {result['workflow_status']}"
            )

    gate_ids = {gate["gate_id"] for gate in full["gate_results"]}
    required = {
        "ARTIFACT_PROVENANCE_AUDIT",
        "PUBLICATION_READINESS_GATE",
        "GATE_SCOPE_APPROVAL",
        "GATE_CLAIM_APPROVAL",
        "FIGURE_EVIDENCE_CONTRACT",
        "GATE_FIGURE_APPROVAL",
        "GATE_RESULTS_HUMAN_REVIEW",
        "METHODS_WRITER_PREREQUISITES",
        "SUPPLEMENT_BUILDER_PREREQUISITES",
        "DISCUSSION_SCOPE_GUARD",
        "GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR",
        "INTRO_CITATION_LOOP_GATE",
        "ABSTRACT_TITLE_BOUNDARY",
        "CROSS_SECTION_CONSISTENCY_AUDIT",
        "CITATION_AUDIT",
        "PRL_LENGTH_CLAIM_GATE",
        "LATEX_INTEGRATION_BOUNDARY",
        "GATE_FINAL_MANUSCRIPT_APPROVAL",
    }
    missing = required - gate_ids
    if missing:
        raise SystemExit(f"missing gates: {sorted(missing)}")
    if full["workflow_status"] != "awaiting_human_approval":
        raise SystemExit("workflow bypassed final human approval")
    if len(NODE_SEQUENCE) != 17:
        raise SystemExit("unexpected node count")
    print("WORKFLOW_VALIDATION_PASS")
    print(f"nodes={len(NODE_SEQUENCE)} full_traversal_gates={len(full['gate_results'])}")
    print(f"unapproved_status={scope_wait['workflow_status']}")
    print(f"next_action={scope_wait['next_action']}")


if __name__ == "__main__":
    main()
