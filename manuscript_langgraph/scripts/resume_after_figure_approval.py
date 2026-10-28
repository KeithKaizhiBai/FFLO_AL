from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
RUN_ROOT = PACKAGE_ROOT / "runs/active_learning_prl_v1"
RESULTS_ROOT = PROJECT_ROOT / "output/pdf/manuscript_results_review_v1"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.graphs.prl_manuscript_graph import build_graph  # noqa: E402


def main() -> None:
    approval_path = RUN_ROOT / "gate_records/figure_approval_record.json"
    if not approval_path.exists():
        raise SystemExit("GATE_FIGURE_APPROVAL is still pending; no approval record exists")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("decision") not in {"approve", "approve_with_edits"}:
        raise SystemExit("figure approval record does not authorize resume")

    connection = sqlite3.connect(RUN_ROOT / "checkpoints/manuscript.sqlite", check_same_thread=False)
    saver = SqliteSaver(connection)
    workflow = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "active_learning_prl_v1"}}
    workflow.update_state(
        config,
        {
            "figure_approved": True,
            "results_human_approved": False,
            "results_claims_frozen": False,
            "results_package_path": str(RESULTS_ROOT),
            "metadata": {
                "compile_results_review_pdf": True,
                "new_physics_permission": False,
                "external_retrieval_approved": False,
            },
        },
        as_node="figure_evidence_contract",
    )
    result = workflow.invoke(None, config=config)
    snapshot = workflow.get_state(config)
    gate_map = {item["gate_id"]: item["status"] for item in result.get("gate_results", [])}
    if gate_map.get("GATE_FIGURE_APPROVAL") != "PASS":
        raise SystemExit("figure approval did not pass during checkpoint resume")
    if gate_map.get("GATE_RESULTS_HUMAN_REVIEW") != "HUMAN_WAIT":
        raise SystemExit("workflow did not stop at GATE_RESULTS_HUMAN_REVIEW")
    forbidden_downstream = {
        "METHODS_WRITER_PREREQUISITES",
        "SUPPLEMENT_BUILDER_PREREQUISITES",
        "DISCUSSION_SCOPE_GUARD",
        "INTRO_CITATION_LOOP_GATE",
    }
    if forbidden_downstream & set(gate_map):
        raise SystemExit("workflow entered a downstream node before Results human approval")
    if result.get("results_claims_frozen"):
        raise SystemExit("Results were frozen without human approval")

    state_path = RUN_ROOT / "results_review_state.json"
    state_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_export = {
        "schema_version": 1,
        "thread_id": "active_learning_prl_v1",
        "checkpoint_backend": "SQLiteSaver",
        "checkpoint_database": str((RUN_ROOT / "checkpoints/manuscript.sqlite").relative_to(PROJECT_ROOT)),
        "snapshot_config": snapshot.config,
        "snapshot_next": list(snapshot.next),
        "values": dict(snapshot.values),
        "current_gate": "GATE_RESULTS_HUMAN_REVIEW",
        "resume_semantics": "After explicit Results/caption approval, update this same thread as results_human_review_gate with results_human_approved=true and invoke with input=None.",
    }
    checkpoint_path = RUN_ROOT / "checkpoints/checkpoint_results_review.json"
    checkpoint_path.write_text(json.dumps(checkpoint_export, indent=2, default=str) + "\n", encoding="utf-8")
    results_gate = next(item for item in result["gate_results"] if item["gate_id"] == "GATE_RESULTS_HUMAN_REVIEW")
    gate_record = {
        "schema_version": 1,
        **results_gate,
        "decision": "pending_human_review",
        "results_package": str(RESULTS_ROOT.relative_to(PROJECT_ROOT)),
        "results_scientific_freeze": False,
        "external_retrieval_permission": False,
        "new_physics_permission": False,
    }
    gate_path = RUN_ROOT / "gate_records/results_human_review_gate.json"
    gate_path.write_text(json.dumps(gate_record, indent=2) + "\n", encoding="utf-8")
    node_output = {
        "active_figure_id": result.get("active_figure_id"),
        "results_package_path": result.get("results_package_path"),
        "paragraph_metadata": result.get("paragraph_metadata", []),
        "results_claims_frozen": result.get("results_claims_frozen"),
        "workflow_status": result.get("workflow_status"),
        "next_action": result.get("next_action"),
    }
    (RUN_ROOT / "node_outputs/results_human_review_gate.json").write_text(
        json.dumps(node_output, indent=2) + "\n", encoding="utf-8"
    )
    log = (
        "LANGGRAPH_RESULTS_RESUME_PASS\n"
        "graph_id=evidence_first_prl_workflow\n"
        "runtime=langgraph\n"
        "checkpoint_backend=SQLiteSaver\n"
        "current_stop=GATE_RESULTS_HUMAN_REVIEW\n"
        "results_auto_approved=false\n"
        "methods_executed=false\n"
        "external_retrieval=false\n"
        "new_physics=false\n"
    )
    (RUN_ROOT / "logs/results_resume.log").write_text(log, encoding="utf-8")
    connection.close()
    print(log, end="")
    print(f"checkpoint={checkpoint_path}")
    print(f"results_package={RESULTS_ROOT}")


if __name__ == "__main__":
    main()
