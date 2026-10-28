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

from review_agent.downstream_package import audit_fig4_caption_consistency  # noqa: E402
from review_agent.graphs.prl_manuscript_graph import build_graph  # noqa: E402
from review_agent.results_package import build_results_review_package  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    build_results_review_package(
        PROJECT_ROOT,
        RESULTS_ROOT,
        compile_pdf=True,
        scientifically_frozen=True,
    )
    audit = audit_fig4_caption_consistency(PROJECT_ROOT, DOWNSTREAM_ROOT, RESULTS_ROOT)
    if not audit["results_caption_consistent"]:
        raise SystemExit("the approved Fig. 4 text correction did not pass the artwork/text audit")

    amendment_files = [
        "results_candidate.tex",
        "results_candidate.md",
        "figure_captions_candidate.tex",
        "results_sentence_evidence_trace.yaml",
        "figure_caption_evidence_trace.yaml",
        "paragraph_metadata.json",
        "results_review_packet.tex",
        "results_review_packet.pdf",
    ]
    amendment = {
        "schema_version": 1,
        "gate_id": "GATE_RESULTS_FIG4_CAPTION_CORRECTION",
        "decision": "approved_and_applied",
        "approved_at": str(date.today()),
        "approval_authority": "explicit_user_instruction_in_active_task",
        "scope": "Fig. 4 panel-C slice count/centers and Results-review approved-artwork binding",
        "changes": [
            "Results RES-FIG4-001: four explicit bins -> three explicit bins",
            "Figure 4 caption panel c: centers [-0.45,0.15,0.85,1.45] -> [-0.35,0.55,1.25]",
            "Results review packet: earlier four-slice strict-data prototype -> approved final three-slice Fig4_candidate.pdf",
        ],
        "unchanged": ["Figure 4 artwork", "frozen data", "physics", "counts", "convergence metrics", "scientific claim"],
        "corrected_artifacts": {name: sha256(RESULTS_ROOT / name) for name in amendment_files},
        "external_retrieval_permission": False,
        "new_physics_permission": False,
    }
    amendment_path = RUN_ROOT / "gate_records/results_fig4_caption_correction_approval.json"
    amendment_path.write_text(json.dumps(amendment, indent=2) + "\n", encoding="utf-8")

    approval_path = RUN_ROOT / "gate_records/results_human_approval_record.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["results_scientific_freeze"] = True
    approval["amendment"] = str(amendment_path.relative_to(PROJECT_ROOT))
    approval["frozen_artifacts"] = {
        name: sha256(RESULTS_ROOT / name)
        for name in [
            "results_candidate.tex",
            "results_candidate.md",
            "figure_captions_candidate.tex",
            "results_sentence_evidence_trace.yaml",
            "figure_caption_evidence_trace.yaml",
            "results_review_packet.tex",
            "results_review_packet.pdf",
        ]
    }
    approval_path.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")

    connection = sqlite3.connect(RUN_ROOT / "checkpoints/manuscript.sqlite", check_same_thread=False)
    saver = SqliteSaver(connection)
    workflow = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "active_learning_prl_v1"}}
    prior = workflow.get_state(config)
    prior_gates = [
        gate for gate in prior.values.get("gate_results", [])
        if gate.get("gate_id") not in {"GATE_RESULTS_FIG4_CAPTION_CORRECTION", "INTRO_CITATION_LOOP_GATE"}
    ]
    workflow.update_state(
        config,
        {
            "results_human_approved": True,
            "results_claims_frozen": True,
            "results_correction_approved": True,
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
        as_node="discussion_scope_guard",
    )
    result = workflow.invoke(None, config=config)
    snapshot = workflow.get_state(config)
    gate_map = {item["gate_id"]: item["status"] for item in result.get("gate_results", [])}
    if gate_map.get("GATE_RESULTS_FIG4_CAPTION_CORRECTION") != "PASS":
        raise SystemExit("Fig. 4 caption correction gate did not pass")
    if gate_map.get("INTRO_CITATION_LOOP_GATE") != "HUMAN_WAIT":
        raise SystemExit("workflow did not stop at the external Introduction retrieval permission gate")
    if any(gate in gate_map for gate in ["ABSTRACT_TITLE_BOUNDARY", "LATEX_INTEGRATION_BOUNDARY", "GATE_FINAL_MANUSCRIPT_APPROVAL"]):
        raise SystemExit("workflow crossed the external Introduction retrieval permission gate")

    (RUN_ROOT / "introduction_gate_state.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    checkpoint_export = {
        "schema_version": 1,
        "thread_id": "active_learning_prl_v1",
        "checkpoint_backend": "SQLiteSaver",
        "checkpoint_database": str((RUN_ROOT / "checkpoints/manuscript.sqlite").relative_to(PROJECT_ROOT)),
        "snapshot_config": snapshot.config,
        "snapshot_next": list(snapshot.next),
        "values": dict(snapshot.values),
        "current_gate": "INTRO_CITATION_LOOP_GATE",
        "resume_semantics": "After explicit external Introduction-retrieval approval, update this same thread as results_figure_consistency_gate with external_retrieval_approved=true and invoke with input=None.",
    }
    checkpoint_path = RUN_ROOT / "checkpoints/checkpoint_introduction_gate.json"
    checkpoint_path.write_text(json.dumps(checkpoint_export, indent=2, default=str) + "\n", encoding="utf-8")
    intro_gate = next(item for item in result["gate_results"] if item["gate_id"] == "INTRO_CITATION_LOOP_GATE")
    (RUN_ROOT / "gate_records/introduction_citation_loop_gate.json").write_text(
        json.dumps({"schema_version": 1, **intro_gate, "decision": "permission_not_granted", "external_retrieval_executed": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "gate_records/results_fig4_caption_correction_gate.json").write_text(
        json.dumps({"schema_version": 1, "gate_id": "GATE_RESULTS_FIG4_CAPTION_CORRECTION", "status": "PASS", "decision": "approved_and_applied", "approval_record": str(amendment_path.relative_to(PROJECT_ROOT))}, indent=2) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "logs/introduction_gate_resume.log").write_text(
        "LANGGRAPH_FIG4_CAPTION_CORRECTION_PASS\ncurrent_stop=INTRO_CITATION_LOOP_GATE\nresults_refrozen=true\nexternal_retrieval=false\nnew_physics=false\n",
        encoding="utf-8",
    )
    connection.close()
    print("LANGGRAPH_FIG4_CAPTION_CORRECTION_PASS")
    print(f"checkpoint={checkpoint_path}")
    print(f"main_pdf={DOWNSTREAM_ROOT / 'manuscript_intermediate_review.pdf'}")
    print(f"supplement_pdf={DOWNSTREAM_ROOT / 'supplement_review.pdf'}")


if __name__ == "__main__":
    main()
