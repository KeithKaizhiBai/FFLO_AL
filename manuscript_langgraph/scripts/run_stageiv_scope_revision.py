#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "manuscript_langgraph"))

from review_agent.graphs.stageiv_scope_revision_graph import graph  # noqa: E402


def main() -> int:
    state = graph.invoke({"project_root": str(ROOT)})
    print(json.dumps({
        "workflow_status": state.get("workflow_status"),
        "manuscript_pdf": state.get("revised_manuscript_pdf"),
        "supplement_pdf": state.get("revised_supplement_pdf"),
        "next_action": state.get("next_action"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
