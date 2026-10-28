from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.labels import PHASE_NAMES


TOPOLOGY_LABEL_NAMES = {
    -1: "not_applicable",
    0: "trivial",
    1: "topological",
    2: "gapless_SC",
    3: "unresolved",
}


def _latest_dataset(run_dir: Path) -> Path:
    datasets = sorted(run_dir.glob("dataset_iter*.npz"))
    if not datasets:
        raise FileNotFoundError(f"No dataset_iter*.npz files found under {run_dir}")
    return datasets[-1]


def _iter_from_name(path: Path) -> int:
    stem = path.stem
    return int(stem.replace("dataset_iter", ""))


def _counts(values: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code in sorted(set(np.asarray(values, dtype=np.int64).tolist())):
        rows.append({"code": int(code), "label": names.get(int(code), str(int(code))), "count": int(np.sum(values == code))})
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    rows = [[str(col) for col in df.columns]]
    for _, row in df.iterrows():
        rows.append([str(row[col]) for col in df.columns])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def fmt(parts: list[str]) -> str:
        return "| " + " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts))) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    return "\n".join([fmt(rows[0]), sep, *[fmt(r) for r in rows[1:]]])


def build_report(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    dataset_path = _latest_dataset(run_dir)
    iteration = _iter_from_name(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as z:
        x = z["x"]
        y_phase = z["y_phase"]
        topo = z["topology_label_code"] if "topology_label_code" in z.files else np.full(x.shape[0], -1, dtype=np.int64)
        trusted = z["trusted_exact"] if "trusted_exact" in z.files else np.ones(x.shape[0], dtype=np.int8)
        topo_trusted = z["topology_trusted"] if "topology_trusted" in z.files else np.zeros(x.shape[0], dtype=np.int8)
        mu = x[:, 2] if x.shape[1] >= 3 else z["mu"] if "mu" in z.files else np.full(x.shape[0], 0.55)

    phase_counts = _counts(y_phase, PHASE_NAMES)
    topo_counts = _counts(topo, TOPOLOGY_LABEL_NAMES)
    phase_counts.to_csv(tables / "phase_counts.csv", index=False)
    topo_counts.to_csv(tables / "topology_counts.csv", index=False)
    mu_summary = pd.DataFrame(
        [
            {
                "sample_count": int(x.shape[0]),
                "mu_min": float(np.nanmin(mu)) if x.shape[0] else float("nan"),
                "mu_max": float(np.nanmax(mu)) if x.shape[0] else float("nan"),
                "mu_mean": float(np.nanmean(mu)) if x.shape[0] else float("nan"),
                "trusted_exact_count": int(np.sum(np.asarray(trusted).astype(bool))),
                "topology_trusted_count": int(np.sum(np.asarray(topo_trusted).astype(bool))),
            }
        ]
    )
    mu_summary.to_csv(tables / "run_summary.csv", index=False)
    decision = {
        "run_dir": str(run_dir),
        "latest_dataset": str(dataset_path),
        "latest_dataset_iteration": int(iteration),
        "sample_count": int(x.shape[0]),
        "phase_counts": phase_counts.to_dict("records"),
        "topology_counts": topo_counts.to_dict("records"),
        "postrun_status": "summary_only",
        "notes": [
            "This is a lightweight post-run summary, not a publication-grade 3D convergence audit.",
            "Run the dedicated Stage IV 3D convergence audit before making scientific closure claims.",
        ],
    }
    (output_dir / "stageiv_3d_postrun_summary.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    md = [
        "# Stage IV 3D Post-Run Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- latest_dataset: `{dataset_path}`",
        f"- latest_dataset_iteration: `{iteration}`",
        f"- sample_count: `{x.shape[0]}`",
        "",
        "## Phase Counts",
        "",
        df_to_markdown(phase_counts),
        "",
        "## Topology Counts",
        "",
        df_to_markdown(topo_counts),
        "",
        "This file is a summary-only handoff.  It does not replace the Stage IV 3D convergence audit.",
    ]
    (output_dir / "stageiv_3d_postrun_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a lightweight Stage IV 3D post-run summary.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    decision = build_report(args.run_dir, args.output_dir)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
