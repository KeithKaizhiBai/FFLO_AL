from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
RUN_ROOT = PACKAGE_ROOT / "runs/active_learning_prl_v1"
RESULTS_ROOT = PROJECT_ROOT / "output/pdf/manuscript_results_review_v1"
DOWNSTREAM_ROOT = PROJECT_ROOT / "output/pdf/manuscript_methods_sm_discussion_review_v1"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.graphs.prl_manuscript_graph import build_graph  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [
        "results_candidate.tex",
        "results_candidate.md",
        "figure_captions_candidate.tex",
        "results_sentence_evidence_trace.yaml",
        "figure_caption_evidence_trace.yaml",
        "results_review_packet.pdf",
    ]
    missing = [name for name in required if not (RESULTS_ROOT / name).exists()]
    if missing:
        raise SystemExit(f"approved Results package is incomplete: {missing}")

    approval = {
        "schema_version": 1,
        "gate_id": "GATE_RESULTS_HUMAN_REVIEW",
        "decision": "approved",
        "approved_at": str(date.today()),
        "approval_authority": "explicit_user_instruction_in_active_task",
        "scope": "Results text and Figure 1-4 captions",
        "results_scientific_freeze": True,
        "external_retrieval_permission": False,
        "new_physics_permission": False,
        "frozen_artifacts": {name: sha256(RESULTS_ROOT / name) for name in required},
    }
    approval_path = RUN_ROOT / "gate_records/results_human_approval_record.json"
    approval_path.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")

    connection = sqlite3.connect(RUN_ROOT / "checkpoints/manuscript.sqlite", check_same_thread=False)
    saver = SqliteSaver(connection)
    workflow = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "active_learning_prl_v1"}}
    prior = workflow.get_state(config)
    prior_gates = [
        gate for gate in prior.values.get("gate_results", [])
        if gate.get("gate_id") not in {"INTRO_CITATION_LOOP_GATE", "GATE_RESULTS_FIG4_CAPTION_CORRECTION"}
    ]
    workflow.update_state(
        config,
        {
            "results_human_approved": True,
            "results_claims_frozen": False,
            "downstream_package_path": str(DOWNSTREAM_ROOT),
            "external_retrieval_approved": False,
            "gate_results": prior_gates,
            "metadata": {
                "compile_results_review_pdf": True,
                "compile_downstream_review_pdfs": True,
                "new_physics_permission": False,
                "external_retrieval_approved": False,
            },
        },
        as_node="results_by_figure_writer",
    )
    result = workflow.invoke(None, config=config)
    snapshot = workflow.get_state(config)
    gate_map = {item["gate_id"]: item["status"] for item in result.get("gate_results", [])}
    if gate_map.get("GATE_RESULTS_HUMAN_REVIEW") != "PASS":
        raise SystemExit("Results approval did not pass")
    for gate in ["METHODS_WRITER_PREREQUISITES", "SUPPLEMENT_BUILDER_PREREQUISITES", "DISCUSSION_SCOPE_GUARD"]:
        if gate_map.get(gate) != "PASS":
            raise SystemExit(f"downstream gate failed: {gate}")
    if gate_map.get("GATE_RESULTS_FIG4_CAPTION_CORRECTION") != "HUMAN_WAIT":
        raise SystemExit("workflow did not stop at the newly detected Figure 4 caption correction gate")
    if any(gate in gate_map for gate in ["ABSTRACT_TITLE_BOUNDARY", "LATEX_INTEGRATION_BOUNDARY", "GATE_FINAL_MANUSCRIPT_APPROVAL"]):
        raise SystemExit("workflow crossed the Introduction retrieval permission gate")

    (RUN_ROOT / "introduction_gate_state.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    checkpoint_export = {
        "schema_version": 1,
        "thread_id": "active_learning_prl_v1",
        "checkpoint_backend": "SQLiteSaver",
        "checkpoint_database": str((RUN_ROOT / "checkpoints/manuscript.sqlite").relative_to(PROJECT_ROOT)),
        "snapshot_config": snapshot.config,
        "snapshot_next": list(snapshot.next),
        "values": dict(snapshot.values),
        "current_gate": "GATE_RESULTS_FIG4_CAPTION_CORRECTION",
        "resume_semantics": "After explicit approval of the two-fragment Figure 4 text correction, amend and rehash the Results package, then resume this same thread toward INTRO_CITATION_LOOP_GATE.",
    }
    checkpoint_path = RUN_ROOT / "checkpoints/checkpoint_fig4_caption_correction_gate.json"
    checkpoint_path.write_text(json.dumps(checkpoint_export, indent=2, default=str) + "\n", encoding="utf-8")
    gate_record = next(item for item in result["gate_results"] if item["gate_id"] == "GATE_RESULTS_FIG4_CAPTION_CORRECTION")
    (RUN_ROOT / "gate_records/results_fig4_caption_correction_gate.json").write_text(
        json.dumps({"schema_version": 1, **gate_record, "decision": "pending_human_review", "external_retrieval_executed": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "node_outputs/methods_writer.json").write_text(json.dumps({"complete": result.get("methods_draft_complete"), "path": str(DOWNSTREAM_ROOT / "methods_candidate.tex")}, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "node_outputs/supplement_builder.json").write_text(json.dumps({"complete": result.get("supplement_draft_complete"), "path": str(DOWNSTREAM_ROOT / "supplement_candidate.tex")}, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "node_outputs/discussion_scope_guard.json").write_text(json.dumps({"discussion_complete": result.get("discussion_draft_complete"), "outlook_complete": result.get("outlook_draft_complete")}, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "logs/introduction_gate_resume.log").write_text(
        "LANGGRAPH_RESULTS_APPROVAL_RESUME_PASS\ncurrent_stop=GATE_RESULTS_FIG4_CAPTION_CORRECTION\nmethods_executed=true\nsupplement_executed=true\ndiscussion_executed=true\noutlook_executed=true\nexternal_retrieval=false\nnew_physics=false\n",
        encoding="utf-8",
    )
    connection.close()
    print("LANGGRAPH_RESULTS_APPROVAL_RESUME_PASS")
    print(f"checkpoint={checkpoint_path}")
    print(f"downstream_package={DOWNSTREAM_ROOT}")


if __name__ == "__main__":
    main()
