from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.dataset_builder import load_flat_dataset
from ml_phase.stagev_v3 import StageV3Config, sparse_arc_replay_from_dataset


DEFAULT_TAIL_ROOT = Path("stagev2_topology_surface_coverage_tail_v1_hpc/ML_Phase_StageV_V2_TopologyCoverageTail")
DEFAULT_TAIL_RUN = "stagev2_topology_surface_coverage_tail_v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage V-v3 offline sparse-arc replay on Stage V-v2-tail data.")
    p.add_argument("--tail-root", type=Path, default=DEFAULT_TAIL_ROOT)
    p.add_argument("--tail-run-id", default=DEFAULT_TAIL_RUN)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("reports/stagev_v3_sparse_arc_replay"))
    p.add_argument("--config", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StageV3Config() if args.config is None else StageV3Config.from_json(args.config)
    dataset_path = args.dataset or args.tail_root / "active_runs" / args.tail_run_id / "dataset_iter101.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    dataset = load_flat_dataset(dataset_path)
    contributors, patch_table, summary = sparse_arc_replay_from_dataset(dataset, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contributors.to_csv(args.output_dir / "p95_coverage_contributors.csv", index=False)
    patch_table.to_csv(args.output_dir / "sparse_topology_patch_table.csv", index=False)
    decision = {
        "run_id": "stagev_v3_sparse_arc_replay_on_v2_tail",
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": sha256_file(dataset_path),
        "production_data_leakage": False,
        "replay_use_only": True,
        "recommended_generic_proposal_mechanisms": [
            "p0_zero_surface_sparse_patch",
            "ppi_zero_surface_sparse_patch",
            "opposite_Z2_bracket_gap",
            "topology_surface_normal_jitter",
            "topology_surface_tangent_jitter",
            "component_boundary_guard",
            "sparse_patch_fill_distance",
            "gap_topology_relevant",
        ],
        **summary,
    }
    write_json(args.output_dir / "sparse_arc_localization_summary.json", decision)
    md = [
        "# Stage V-v3 Sparse-Arc Localization Replay",
        "",
        "This is a diagnostic-only replay on Stage V-v2 topology-surface coverage-tail data.",
        "It is not used to seed Stage V-v3 production training or selected coordinates.",
        "",
        "## Input",
        "",
        f"- dataset: `{dataset_path}`",
        f"- dataset sha256: `{decision['source_dataset_sha256']}`",
        "",
        "## Summary",
        "",
    ]
    for key in ["status", "surface_samples", "sparse_patch_samples", "component_count", "coverage_p50", "coverage_p90", "coverage_p95", "p0_cross_fraction", "ppi_cross_fraction"]:
        md.append(f"- {key}: `{decision.get(key)}`")
    md += [
        "",
        "## Generic v3 Proposal Mechanisms",
        "",
    ]
    for item in decision["recommended_generic_proposal_mechanisms"]:
        md.append(f"- `{item}`")
    md += [
        "",
        "## Output Tables",
        "",
        "- `p95_coverage_contributors.csv`",
        "- `sparse_topology_patch_table.csv`",
        "- `sparse_arc_localization_summary.json`",
    ]
    (args.output_dir / "sparse_arc_localization_report.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
