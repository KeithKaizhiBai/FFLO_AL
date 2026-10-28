#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "manuscript_langgraph"))

from review_agent.graphs.scientific_depth_revision_graph import graph


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-after-human-review", action="store_true")
    args = parser.parse_args()
    if args.resume_after_human_review:
        print("Human approval must be recorded explicitly before any scientific or publication freeze.")
        return 2
    state = graph.invoke({"project_root": str(ROOT)})
    keys = [
        "workflow_status",
        "current_gate",
        "gate_status",
        "selected_layout",
        "expanded_manuscript_pdf",
        "fallback_manuscript_pdf",
        "revised_supplement_pdf",
        "compiled_review_package_pdf",
        "resume_command",
    ]
    print(json.dumps({key: state.get(key) for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
