from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.figure_authority import CANONICAL_MAIN_FIGURES, sha256  # noqa: E402
from review_agent.graphs.prl_integrated_cited_graph import graph  # noqa: E402
from review_agent.prl_cited_variant import PROFILE_ID, RUN_ID  # noqa: E402


def main() -> None:
    base_pdf = PROJECT_ROOT / "output/pdf/manuscript_prl_integrated_v1/manuscript_prl_integrated_candidate.pdf"
    base_hash = sha256(base_pdf)
    base_intro = PROJECT_ROOT / "manuscript_variants/prl_integrated/sections/introduction.tex"
    base_intro_hash = sha256(base_intro)
    cited_v1_pdf = PROJECT_ROOT / "output/pdf/manuscript_prl_integrated_cited_v1/manuscript_prl_integrated_cited_candidate.pdf"
    cited_v1_intro = PROJECT_ROOT / "manuscript_variants/prl_integrated_cited/sections/introduction.tex"
    cited_v1_hashes = {
        path: sha256(path)
        for path in (cited_v1_pdf, cited_v1_intro)
        if path.exists()
    }
    figure_hashes = {key: sha256(PROJECT_ROOT / record["source_pdf"]) for key, record in CANONICAL_MAIN_FIGURES.items()}
    result = graph.invoke({
        "project_root": str(PROJECT_ROOT),
        "manuscript_root": str(PACKAGE_ROOT),
        "figure_approval_package_path": str(PACKAGE_ROOT / "GATE_FIGURE_APPROVAL_PACKAGE"),
        "scope_approved": True,
        "claim_approved": True,
        "figure_approved": True,
        "results_human_approved": True,
        "external_retrieval_approved": True,
        "final_manuscript_approved": False,
        "gate_results": [],
        "metadata": {
            "publication_profile": PROFILE_ID,
            "external_retrieval_approved": True,
            "new_physics_permission": False,
        },
    })
    if result["workflow_status"] != "awaiting_cited_introduction_manuscript_human_review":
        raise SystemExit("cited workflow did not stop at its human review gate")
    if sha256(base_pdf) != base_hash or sha256(base_intro) != base_intro_hash:
        raise SystemExit("base PRL-integrated candidate was modified")
    for path, expected in cited_v1_hashes.items():
        if sha256(path) != expected:
            raise SystemExit(f"cited v1 artifact was modified: {path}")
    for key, record in CANONICAL_MAIN_FIGURES.items():
        if sha256(PROJECT_ROOT / record["source_pdf"]) != figure_hashes[key]:
            raise SystemExit(f"frozen figure changed: {key}")
    gate_path = PACKAGE_ROOT / f"runs/{RUN_ID}/gate_records/cited_introduction_manuscript_human_review_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["status"] != "HUMAN_WAIT" or not gate["retrieval_executed"]:
        raise SystemExit("cited human gate/provenance is incorrect")
    print("PRL_INTEGRATED_CITED_V2_VARIANT_BUILD_PASS")
    print(f"candidate_pdf={PROJECT_ROOT / gate['candidate_pdf']}")
    print(f"current_gate={gate['gate_id']}")
    print("gate_status=HUMAN_WAIT")


if __name__ == "__main__":
    main()
