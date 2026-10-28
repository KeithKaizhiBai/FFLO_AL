from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
OUTPUT = PACKAGE_ROOT / "audits/current_workflow_state.json"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.graphs.prl_manuscript_graph import graph  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def persisted_approvals() -> dict[str, bool]:
    scope = load_json(PACKAGE_ROOT / "manuscript_scope.yaml")
    claims = load_json(PACKAGE_ROOT / "registries/claim_registry.yaml")
    figure = load_json(PACKAGE_ROOT / "runs/active_learning_prl_v1/gate_records/figure_approval_record.json")
    results_approval_path = PACKAGE_ROOT / "runs/active_learning_prl_v1/gate_records/results_human_approval_record.json"
    results_approval = load_json(results_approval_path) if results_approval_path.exists() else {}
    gates = load_json(PACKAGE_ROOT / "audits/gate_status.yaml")["gates"]
    return {
        "scope_approved": scope.get("GATE_SCOPE_APPROVAL") == "approved",
        "claim_approved": claims.get("approval_status") in {"approved", "approved_with_modifications"},
        "figure_approved": figure.get("decision") in {"approve", "approve_with_edits"},
        "results_human_approved": results_approval.get("decision") == "approved",
        "results_correction_approved": gates.get("GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR") == "approved_and_applied",
        "external_retrieval_approved": gates.get("INTRO_CITATION_LOOP_GATE") == "approved",
        "final_manuscript_approved": gates.get("GATE_FINAL_MANUSCRIPT_APPROVAL") == "approved",
    }


def run() -> dict[str, Any]:
    approvals = persisted_approvals()
    result = graph.invoke(
        {
            "project_root": str(PROJECT_ROOT),
            "manuscript_root": str(PACKAGE_ROOT),
            "figure_approval_package_path": str(PACKAGE_ROOT / "GATE_FIGURE_APPROVAL_PACKAGE"),
            "results_package_path": str(PROJECT_ROOT / "output/pdf/manuscript_results_review_v1"),
            "downstream_package_path": str(PROJECT_ROOT / "output/pdf/manuscript_methods_sm_discussion_review_v1"),
            **approvals,
            "gate_results": [],
        }
    )
    gate_by_id = {item["gate_id"]: item["status"] for item in result["gate_results"]}

    if gate_by_id.get("GATE_SCOPE_APPROVAL") != "PASS":
        raise SystemExit("persisted scope approval did not pass GATE_SCOPE_APPROVAL")
    if approvals["claim_approved"] and gate_by_id.get("GATE_CLAIM_APPROVAL") != "PASS":
        raise SystemExit("persisted explicit claim approval did not pass GATE_CLAIM_APPROVAL")
    if not approvals["claim_approved"] and gate_by_id.get("GATE_CLAIM_APPROVAL") != "HUMAN_WAIT":
        raise SystemExit("unapproved claim registry did not stop at GATE_CLAIM_APPROVAL")
    if approvals["claim_approved"] and not approvals["figure_approved"]:
        if gate_by_id.get("GATE_FIGURE_APPROVAL") != "HUMAN_WAIT":
            raise SystemExit("current workflow did not stop for human figure approval")
        current_stop = "GATE_FIGURE_APPROVAL"
    elif not approvals["claim_approved"]:
        if "GATE_FIGURE_APPROVAL" in gate_by_id:
            raise SystemExit("workflow bypassed the unapproved claim gate")
        current_stop = "GATE_CLAIM_APPROVAL"
    elif not approvals["results_human_approved"]:
        if gate_by_id.get("GATE_RESULTS_HUMAN_REVIEW") != "HUMAN_WAIT":
            raise SystemExit("workflow bypassed Results human review")
        if "METHODS_WRITER_PREREQUISITES" in gate_by_id:
            raise SystemExit("workflow entered Methods before Results approval")
        current_stop = "GATE_RESULTS_HUMAN_REVIEW"
    elif gate_by_id.get("GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR") != "PASS":
        raise SystemExit("workflow did not pass the manuscript Figure Engine source audit")
    elif not approvals["external_retrieval_approved"]:
        if gate_by_id.get("INTRO_CITATION_LOOP_GATE") != "HUMAN_WAIT":
            raise SystemExit("workflow bypassed external retrieval permission")
        current_stop = "INTRO_CITATION_LOOP_GATE"
    else:
        current_stop = "GATE_FINAL_MANUSCRIPT_APPROVAL"
    if approvals["external_retrieval_approved"]:
        raise SystemExit("external Introduction retrieval was unexpectedly approved")

    payload = {
        "schema_version": 1,
        "evaluated_at": "2026-08-27",
        "approval_source": "persisted manuscript workflow records",
        "derived_approvals": approvals,
        "workflow_status": result["workflow_status"],
        "current_stop": current_stop,
        "next_action": result["next_action"],
        "gate_results": result["gate_results"],
        "safety_checks": {
            "claim_gate_explicitly_approved": approvals["claim_approved"],
            "figure_gate_reached": "GATE_FIGURE_APPROVAL" in gate_by_id,
            "figure_gate_explicitly_approved": approvals["figure_approved"],
            "results_gate_auto_passed": False,
            "manuscript_figure_source_repair_verified": gate_by_id.get("GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR") == "PASS",
            "external_introduction_retrieval_authorized": False,
            "physics_recomputed": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    current = run()
    print("CURRENT_WORKFLOW_STATE_WRITTEN")
    print(f"workflow_status={current['workflow_status']}")
    print(f"current_stop={current['current_stop']}")
