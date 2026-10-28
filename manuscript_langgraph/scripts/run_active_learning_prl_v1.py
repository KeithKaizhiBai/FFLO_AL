from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
RUN_ROOT = PACKAGE_ROOT / "runs/active_learning_prl_v1"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.graphs.prl_manuscript_graph import build_graph  # noqa: E402


def main() -> None:
    for directory in ["checkpoints", "gate_records", "node_outputs", "logs"]:
        (RUN_ROOT / directory).mkdir(parents=True, exist_ok=True)
    initial = json.loads((RUN_ROOT / "initial_state.json").read_text(encoding="utf-8"))
    checkpoint_path = RUN_ROOT / "checkpoints/manuscript.sqlite"
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    saver = SqliteSaver(connection)
    workflow = build_graph(checkpointer=saver, interrupt_after=["results_by_figure_writer"])
    config = {"configurable": {"thread_id": "active_learning_prl_v1"}}
    result = workflow.invoke(initial, config=config)
    snapshot = workflow.get_state(config)
    gates = result.get("gate_results", [])
    gate_map = {item["gate_id"]: item["status"] for item in gates}
    if gate_map.get("GATE_SCOPE_APPROVAL") != "PASS":
        raise SystemExit("scope approval did not pass")
    if gate_map.get("GATE_CLAIM_APPROVAL") != "PASS":
        raise SystemExit("claim approval with modifications did not pass")
    if gate_map.get("GATE_FIGURE_APPROVAL") != "HUMAN_WAIT":
        raise SystemExit("workflow did not stop at GATE_FIGURE_APPROVAL")
    if result.get("figure_contracts_approved"):
        raise SystemExit("figure approval was passed automatically")
    if result.get("external_retrieval_approved"):
        raise SystemExit("external retrieval permission changed unexpectedly")

    checkpoint_export = {
        "schema_version": 1,
        "thread_id": "active_learning_prl_v1",
        "checkpoint_backend": "SQLiteSaver",
        "checkpoint_database": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "snapshot_config": snapshot.config,
        "snapshot_next": list(snapshot.next),
        "values": dict(snapshot.values),
        "resume_semantics": "After explicit figure approval, update this thread as results_by_figure_writer and invoke with input=None.",
    }
    (RUN_ROOT / "checkpoints/checkpoint_figure_gate.json").write_text(
        json.dumps(checkpoint_export, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (RUN_ROOT / "gate_records/gate_records.json").write_text(
        json.dumps(gates, indent=2) + "\n", encoding="utf-8"
    )
    (RUN_ROOT / "node_outputs/results_by_figure_writer.json").write_text(
        json.dumps(
            {
                "active_figure_id": result.get("active_figure_id"),
                "paragraph_metadata": result.get("paragraph_metadata", []),
                "results_claims_frozen": result.get("results_claims_frozen"),
                "figure_contracts_approved": result.get("figure_contracts_approved"),
                "workflow_status": result.get("workflow_status"),
                "next_action": result.get("next_action"),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "final_state.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log = (
        "LANGGRAPH_INVOCATION_PASS\n"
        "graph_id=evidence_first_prl_workflow\n"
        "runtime=langgraph\n"
        "checkpoint_backend=SQLiteSaver\n"
        "current_stop=GATE_FIGURE_APPROVAL\n"
        "figure_auto_approved=false\n"
        "external_retrieval=false\n"
        "new_physics=false\n"
    )
    (RUN_ROOT / "logs/invocation.log").write_text(log, encoding="utf-8")
    print(log, end="")
    connection.close()


if __name__ == "__main__":
    main()
