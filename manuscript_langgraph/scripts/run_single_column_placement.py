#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "manuscript_langgraph"))

from review_agent.graphs.single_column_placement_graph import build_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["legacy_v6", "synchronized_v7", "reader_revision_v8", "authorship_v9"], default="synchronized_v7")
    args = parser.parse_args()
    state = build_graph(args.profile).invoke({"project_root": str(ROOT)})
    keys = ["workflow_status", "current_gate", "gate_status", "selected_single_column_layout", "single_column_manuscript_pdf", "compiled_review_package_pdf"]
    print(json.dumps({key: state.get(key) for key in keys}, indent=2))
