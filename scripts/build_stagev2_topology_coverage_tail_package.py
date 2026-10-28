from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RUN_ID = "stagev2_topology_surface_coverage_tail_v1"
OUTPUT_ROOT = "ML_Phase_StageV_V2_TopologyCoverageTail"
SOURCE_RUN_ID = "stagev_v2_multihead_boundary_learning_3d_v1"
SOURCE_OUTPUT_ROOT = "ML_Phase_StageV_V2_Multihead"
SOURCE_PACKAGE_REMOTE = (
    "/public_hw/home/sci_bfu/bkz/Fu_FFLO/"
    "stagev_v2_multihead_boundary_learning_3d_v1_hpc"
)
WORLD_SIZE = 8
START_ITER = 97
TAIL_BATCHES = 4
POINTS_PER_BATCH = 64
TOTAL_POINTS = TAIL_BATCHES * POINTS_PER_BATCH


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_coords(points, mins, spans):
    return [
        (
            (float(p[0]) - mins[0]) / spans[0],
            (float(p[1]) - mins[1]) / spans[1],
            (float(p[2]) - mins[2]) / spans[2],
        )
        for p in points
    ]


def greedy_select(candidates: list[dict], n: int, mins, spans) -> list[dict]:
    if len(candidates) <= n:
        return candidates

    coords = norm_coords([(c["kT"], c["JA"], c["mu"]) for c in candidates], mins, spans)
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    first = max(remaining, key=lambda i: (candidates[i]["priority"], candidates[i]["edge_length_norm"]))
    selected.append(first)
    remaining.remove(first)

    min_d2 = [float("inf")] * len(candidates)
    while len(selected) < n and remaining:
        last = selected[-1]
        lx, ly, lz = coords[last]
        for i in list(remaining):
            x, y, z = coords[i]
            d2 = (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2
            if d2 < min_d2[i]:
                min_d2[i] = d2
        chosen = max(
            remaining,
            key=lambda i: (
                math.sqrt(min_d2[i]) * 0.70
                + float(candidates[i]["edge_length_norm"]) * 0.30
                + float(candidates[i]["priority"]) * 1.0e-6,
                candidates[i]["edge_length_norm"],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)

    return [candidates[i] for i in selected]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def build_candidates(final_df: pd.DataFrame, brackets: pd.DataFrame) -> tuple[list[dict], dict]:
    coord_cols = ["kT", "JA", "mu"]
    mins = tuple(float(final_df[c].min()) for c in coord_cols)
    maxs = tuple(float(final_df[c].max()) for c in coord_cols)
    spans = tuple(max(1.0e-12, maxs[i] - mins[i]) for i in range(3))

    existing = set(
        (
            round(float(r.kT), 12),
            round(float(r.JA), 12),
            round(float(r.mu), 12),
        )
        for r in final_df[coord_cols].itertuples(index=False)
    )

    rows: list[dict] = []
    uid = 0

    def add_candidate(row, kT, JA, mu, proposal_source, interp_t, endpoint_role):
        nonlocal uid
        kT = clamp(kT, mins[0], maxs[0])
        JA = clamp(JA, mins[1], maxs[1])
        mu = clamp(mu, mins[2], maxs[2])
        key = (round(kT, 12), round(JA, 12), round(mu, 12))
        if key in existing:
            return
        source_bonus = {
            "ppi_zero_surface": 2.0,
            "p0_zero_surface": 1.5,
            "opposite_z2_bracket_support": 1.2,
            "topology_component_guard": 1.0,
            "gap_surface_guard": 0.8,
        }.get(proposal_source, 0.5)
        edge = float(row.edge_length_norm)
        rows.append(
            {
                "candidate_id": f"tail_{uid:05d}",
                "kT": kT,
                "JA": JA,
                "mu": mu,
                "proposal_source": proposal_source,
                "source_bracket_kind": str(row.kind),
                "source_i": int(row.i),
                "source_j": int(row.j),
                "edge_length_norm": edge,
                "p0_cross": bool(row.p0_cross),
                "ppi_cross": bool(row.ppi_cross),
                "interp_t": float(interp_t),
                "endpoint_role": endpoint_role,
                "priority": edge + source_bonus,
            }
        )
        existing.add(key)
        uid += 1

    valid = brackets[
        (brackets["i"] >= 0)
        & (brackets["j"] >= 0)
        & (brackets["i"] < len(final_df))
        & (brackets["j"] < len(final_df))
    ].copy()
    valid = valid.sort_values(["edge_length_norm", "ppi_cross"], ascending=[False, False])

    for row in valid.itertuples(index=False):
        a = final_df.iloc[int(row.i)]
        b = final_df.iloc[int(row.j)]
        source = "ppi_zero_surface" if bool(row.ppi_cross) else "p0_zero_surface"
        add_candidate(row, row.kT, row.JA, row.mu, source, 0.5, "midpoint")
        add_candidate(
            row,
            0.35 * float(a.kT) + 0.65 * float(b.kT),
            0.35 * float(a.JA) + 0.65 * float(b.JA),
            0.35 * float(a.mu) + 0.65 * float(b.mu),
            "opposite_z2_bracket_support",
            0.65,
            "interior",
        )
        add_candidate(
            row,
            0.65 * float(a.kT) + 0.35 * float(b.kT),
            0.65 * float(a.JA) + 0.35 * float(b.JA),
            0.65 * float(a.mu) + 0.35 * float(b.mu),
            "opposite_z2_bracket_support",
            0.35,
            "interior",
        )

    # Add a few long-edge component guards after the main bracket proposals.
    for row in valid.head(64).itertuples(index=False):
        add_candidate(row, row.kT, row.JA, row.mu, "topology_component_guard", 0.5, "guard")

    meta = {
        "domain_min": dict(zip(coord_cols, mins)),
        "domain_max": dict(zip(coord_cols, maxs)),
        "candidate_pool_size": len(rows),
        "valid_bracket_count": int(len(valid)),
        "p0_bracket_count": int(valid["p0_cross"].sum()),
        "ppi_bracket_count": int(valid["ppi_cross"].sum()),
    }
    return rows, meta


def assign_batches(selected: list[dict]) -> list[dict]:
    ppi = [r for r in selected if r["ppi_cross"]]
    p0 = [r for r in selected if not r["ppi_cross"]]
    ordered: list[dict] = []
    while ppi or p0:
        if ppi:
            ordered.append(ppi.pop(0))
        if p0:
            ordered.append(p0.pop(0))
        if p0:
            ordered.append(p0.pop(0))
        if p0:
            ordered.append(p0.pop(0))
    ordered = ordered[:TOTAL_POINTS]
    for i, row in enumerate(ordered):
        row["tail_iteration"] = START_ITER + i // POINTS_PER_BATCH
        row["tail_batch_index"] = i // POINTS_PER_BATCH
        row["tail_batch_row"] = i % POINTS_PER_BATCH
    return ordered


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def make_package(root: Path) -> dict:
    source_root = root / "stagev_v2_multihead_boundary_learning_3d_v1_hpc"
    source_run = source_root / SOURCE_OUTPUT_ROOT / "active_runs" / SOURCE_RUN_ID
    report_root = source_root / SOURCE_OUTPUT_ROOT / "reports" / "stagev_v2_formal_audit_local"
    final_csv = source_run / "dataset_iter097.csv"
    final_npz = source_run / "dataset_iter097.npz"
    brackets_csv = report_root / "tables" / "stagev_v2_final_topology_brackets.csv"
    decision_json = report_root / "stagev_v2_decision.json"

    missing = [p for p in [final_csv, final_npz, brackets_csv, decision_json] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(str(p) for p in missing))

    final_df = pd.read_csv(final_csv)
    brackets = pd.read_csv(brackets_csv)
    candidates, candidate_meta = build_candidates(final_df, brackets)
    mins = tuple(float(final_df[c].min()) for c in ["kT", "JA", "mu"])
    maxs = tuple(float(final_df[c].max()) for c in ["kT", "JA", "mu"])
    spans = tuple(maxs[i] - mins[i] for i in range(3))
    selected = assign_batches(greedy_select(candidates, TOTAL_POINTS, mins, spans))
    if len(selected) != TOTAL_POINTS:
        raise RuntimeError(f"Expected {TOTAL_POINTS} selected points, got {len(selected)}")

    package_dir = root / "hpc_packages" / f"{RUN_ID}_hpc"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    for sub in ["configs", "data", "scripts", "docs"]:
        (package_dir / sub).mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "source_run_id": SOURCE_RUN_ID,
        "source_output_root": SOURCE_OUTPUT_ROOT,
        "source_package_remote": SOURCE_PACKAGE_REMOTE,
        "source_final_dataset_iteration": 97,
        "tail_start_iteration": START_ITER,
        "tail_batches": TAIL_BATCHES,
        "tail_final_exact_iteration": START_ITER + TAIL_BATCHES - 1,
        "tail_expected_final_dataset_iteration": START_ITER + TAIL_BATCHES,
        "world_size": WORLD_SIZE,
        "points_per_batch": POINTS_PER_BATCH,
        "max_new_exact_points": TOTAL_POINTS,
        "hard_cap_new_exact_points": 512,
        "selection_mode": "targeted_topology_surface_coverage_tail",
        "selection_inputs": {
            "final_dataset": "dataset_iter097",
            "formal_audit_bracket_table": "stagev2_final_topology_brackets",
        },
        "slurm": {
            "partition": "NV_H100",
            "exclude_nodes": "gpuh01,gpuh14",
            "time": "05:00:00",
            "cpus_per_task": 8,
            "gpus": 1,
        },
        "oracle": {
            "oracle_mode": "robust_incremental",
            "enable_topology_classification": True,
            "topology_gap_nk": 2048,
            "topology_gap_backend": "gpu",
            "topology_gap_tol_rel": 1e-8,
            "topology_gap_tol_abs": 0.0,
            "topology_gap_k_chunk": 512,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            "dataset_iter097_csv_sha256": sha256_file(final_csv),
            "dataset_iter097_npz_sha256": sha256_file(final_npz),
            "stagev2_final_topology_brackets_csv_sha256": sha256_file(brackets_csv),
            "stagev2_decision_json_sha256": sha256_file(decision_json),
        },
        "candidate_meta": candidate_meta,
    }

    write_text(package_dir / "configs" / "stagev2_topology_coverage_tail_config.json", json.dumps(config, indent=2))
    write_text(package_dir / "data" / "source_hashes.json", json.dumps(config["source_hashes"], indent=2))
    shutil.copy2(decision_json, package_dir / "data" / "source_stagev2_decision.json")

    meta_fields = [
        "candidate_id",
        "tail_iteration",
        "tail_batch_index",
        "tail_batch_row",
        "kT",
        "JA",
        "mu",
        "proposal_source",
        "source_bracket_kind",
        "source_i",
        "source_j",
        "edge_length_norm",
        "p0_cross",
        "ppi_cross",
        "interp_t",
        "endpoint_role",
        "priority",
    ]
    write_csv(package_dir / "data" / "tail_selected_points_metadata.csv", selected, meta_fields)

    point_fields = ["kT", "JA", "mu"]
    for batch in range(TAIL_BATCHES):
        iteration = START_ITER + batch
        rows = [r for r in selected if r["tail_iteration"] == iteration]
        write_csv(package_dir / "data" / f"selected_points_iter{iteration:03d}.csv", rows, point_fields)
        for rank in range(WORLD_SIZE):
            shard_rows = [r for idx, r in enumerate(rows) if idx % WORLD_SIZE == rank]
            write_csv(
                package_dir / "data" / f"selected_points_iter{iteration:03d}_rank{rank:03d}_of{WORLD_SIZE:03d}.csv",
                shard_rows,
                point_fields,
            )

    scripts = script_payloads()
    for rel, text in scripts.items():
        write_text(package_dir / rel, text)

    readme = f"""# Stage V-v2 Topology Surface Coverage Tail

This package performs a targeted coverage tail for the completed Stage V-v2 run.

- Source run: `{SOURCE_RUN_ID}`
- Source final dataset: `dataset_iter097`
- Tail run: `{RUN_ID}`
- Tail iterations: `{START_ITER:03d}` to `{START_ITER + TAIL_BATCHES - 1:03d}`
- New exact points: `{TOTAL_POINTS}` in `{TAIL_BATCHES}` batches of `{POINTS_PER_BATCH}`
- Default excluded nodes: `gpuh01,gpuh14`

The package does not modify the original Stage V-v2 run. It seeds a new tail
run directory from the source `dataset_iter097` and appends trusted exact tail
points there.

Submit on GBU H100:

```bash
cd /public_hw/home/sci_bfu/bkz/Fu_FFLO/{RUN_ID}_hpc
CONFIRM_STAGEV2_TAIL=1 nohup bash scripts/submit_stagev2_topology_coverage_tail.sh \\
  > {RUN_ID}.nohup.log 2>&1 &
```

Monitor:

```bash
bash scripts/status_stagev2_topology_coverage_tail.sh
tail -f {RUN_ID}.nohup.log
squeue -u "$USER"
```
"""
    write_text(package_dir / "README.md", readme)

    manifest = {
        "package_dir": str(package_dir),
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "selected_point_count": len(selected),
        "tail_iterations": list(range(START_ITER, START_ITER + TAIL_BATCHES)),
        "proposal_source_counts": pd.Series([r["proposal_source"] for r in selected]).value_counts().to_dict(),
        "p0_selected": int(sum(1 for r in selected if r["p0_cross"])),
        "ppi_selected": int(sum(1 for r in selected if r["ppi_cross"])),
        "max_edge_length_norm_selected": float(max(r["edge_length_norm"] for r in selected)),
        "min_edge_length_norm_selected": float(min(r["edge_length_norm"] for r in selected)),
        "created_at_utc": config["created_at_utc"],
    }
    write_text(package_dir / "data" / "tail_selection_manifest.json", json.dumps(manifest, indent=2))

    archive_path = root / "hpc_packages" / f"{RUN_ID}_hpc.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(package_dir, arcname=package_dir.name)
    manifest["archive_path"] = str(archive_path)
    manifest["archive_sha256"] = sha256_file(archive_path)
    write_text(package_dir / "data" / "tail_package_manifest.json", json.dumps(manifest, indent=2))
    return manifest


def script_payloads() -> dict[str, str]:
    return {
        "scripts/stagev2_tail_preflight.py": r'''#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def count_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stagev2_topology_coverage_tail_config.json")
    ap.add_argument("--source-project-dir", default=None)
    ap.add_argument("--check-source", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    cfg_path = root / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    errors = []

    meta = root / "data" / "tail_selected_points_metadata.csv"
    if not meta.exists():
        errors.append(f"missing {meta}")
    else:
        with meta.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != int(cfg["max_new_exact_points"]):
            errors.append(f"selected metadata row count mismatch: {len(rows)}")
        if any(not r.get("proposal_source") for r in rows):
            errors.append("empty proposal_source exists")

    for iteration in range(int(cfg["tail_start_iteration"]), int(cfg["tail_final_exact_iteration"]) + 1):
        p = root / "data" / f"selected_points_iter{iteration:03d}.csv"
        if not p.exists():
            errors.append(f"missing {p}")
        elif count_rows(p) != int(cfg["points_per_batch"]):
            errors.append(f"{p} row count is not {cfg['points_per_batch']}")

    slurm = (root / "scripts" / "slurm_stagev2_tail_exact_array.sh").read_text(encoding="utf-8")
    if "gpuh01" not in slurm:
        errors.append("slurm exact script does not exclude/refuse gpuh01")

    if args.check_source:
        source = Path(args.source_project_dir or cfg["source_package_remote"])
        if not source.exists():
            errors.append(f"source package missing: {source}")
        source_run = (
            source
            / cfg["source_output_root"]
            / "active_runs"
            / cfg["source_run_id"]
        )
        for suffix in ("csv", "npz"):
            p = source_run / f"dataset_iter{cfg['source_final_dataset_iteration']:03d}.{suffix}"
            if not p.exists():
                errors.append(f"source final dataset missing: {p}")
        if not (source / "ml_phase" / "exact_oracle.py").exists():
            errors.append("source ml_phase exact_oracle.py missing")

    status = "pass" if not errors else "fail"
    print(json.dumps({"status": status, "errors": errors, "config": cfg}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
''',
        "scripts/prepare_stagev2_tail_run.py": r'''#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stagev2_topology_coverage_tail_config.json")
    ap.add_argument("--source-project-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    package = Path.cwd()
    cfg = json.loads((package / args.config).read_text(encoding="utf-8"))
    source_project = Path(args.source_project_dir)
    source_run = source_project / cfg["source_output_root"] / "active_runs" / cfg["source_run_id"]
    run_dir = package / cfg["output_root"] / "active_runs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    start = int(cfg["tail_start_iteration"])
    source_iter = int(cfg["source_final_dataset_iteration"])
    if start != source_iter:
        raise SystemExit(f"tail start iteration {start} must match source dataset iteration {source_iter}")

    for suffix in ("npz", "csv"):
        src = source_run / f"dataset_iter{source_iter:03d}.{suffix}"
        dst = run_dir / f"dataset_iter{start:03d}.{suffix}"
        if dst.exists() and not args.force:
            continue
        shutil.copy2(src, dst)

    world = int(cfg["world_size"])
    for iteration in range(start, int(cfg["tail_final_exact_iteration"]) + 1):
        iter_dir = run_dir / f"iter{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        rows = load_rows(package / "data" / f"selected_points_iter{iteration:03d}.csv")
        write_csv(iter_dir / "selected_points.csv", rows, ["kT", "JA", "mu"])
        for rank in range(world):
            shard = [row for i, row in enumerate(rows) if i % world == rank]
            write_csv(
                iter_dir / f"selected_points_rank{rank:03d}_of{world:03d}.csv",
                shard,
                ["kT", "JA", "mu"],
            )

    print(json.dumps({"status": "prepared", "run_dir": str(run_dir), "tail_start_iteration": start}, indent=2))


if __name__ == "__main__":
    main()
''',
        "scripts/slurm_stagev2_tail_exact_array.sh": r'''#!/bin/bash
#SBATCH --job-name=stagev2_tail
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --exclude=gpuh01,gpuh14
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

host_name="$(hostname 2>/dev/null || true)"
short_host="${host_name%%.*}"
if [ "${short_host}" = "gpuh01" ] || [ "${short_host}" = "gpuh14" ]; then
  echo "[error] refusing to run on excluded node ${short_host}" >&2
  exit 42
fi

PACKAGE_DIR="${PACKAGE_DIR:?PACKAGE_DIR must be exported}"
SOURCE_PROJECT_DIR="${SOURCE_PROJECT_DIR:?SOURCE_PROJECT_DIR must be exported}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageV_V2_TopologyCoverageTail}"
RUN_ID="${RUN_ID:-stagev2_topology_surface_coverage_tail_v1}"
ITER="${ITER:?ITER must be set}"
WORLD_SIZE="${WORLD_SIZE:-8}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
ACTIVE_ROOT="${PACKAGE_DIR}/${OUTPUT_ROOT}/active_runs"
export PYTHONPATH="${SOURCE_PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PACKAGE_DIR}"

echo "run_id=${RUN_ID}"
echo "iteration=${ITER}"
echo "rank=${RANK}"
echo "world_size=${WORLD_SIZE}"
echo "hostname=${host_name}"
echo "source_project_dir=${SOURCE_PROJECT_DIR}"
echo "tail_active_root=${ACTIVE_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch.__version__=", torch.__version__)
print("torch.version.cuda=", torch.version.cuda)
print("torch.cuda.is_available()=", torch.cuda.is_available())
PY

"${PYTHON_BIN}" -m ml_phase.exact_oracle \
  --active-root "${ACTIVE_ROOT}" \
  --run-id "${RUN_ID}" \
  --iteration "${ITER}" \
  --rank "${RANK}" \
  --world-size "${WORLD_SIZE}" \
  --device cuda:0 \
  --save-every 1 \
  --oracle-mode robust_incremental \
  --enable-q-expansion \
  --enable-incremental-q-expansion \
  --enable-delta-refinement \
  --allow-ambiguous-output \
  --enable-local-box-instrumentation \
  --enable-basin-clustering \
  --enable-selective-refinement \
  --max-refined-minima 3 \
  --max-optional-refined-basins 3 \
  --no-mandatory-basins-can-exceed-cap \
  --high-risk-overflow-policy rank_and_cap \
  --max-edge-risk-basins 1 \
  --max-delta-near-eps-basins 2 \
  --max-near-degenerate-basins 2 \
  --enable-topology-classification \
  --topology-gap-nk "${TOPOLOGY_GAP_NK:-2048}" \
  --topology-gap-backend "${TOPOLOGY_GAP_BACKEND:-gpu}" \
  --topology-gap-tol-rel "${TOPOLOGY_GAP_TOL_REL:-1e-8}" \
  --topology-gap-tol-abs "${TOPOLOGY_GAP_TOL_ABS:-0.0}" \
  --topology-gap-k-chunk "${TOPOLOGY_GAP_K_CHUNK:-512}"
''',
        "scripts/submit_stagev2_topology_coverage_tail.sh": r'''#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
SOURCE_PROJECT_DIR="${SOURCE_PROJECT_DIR:-/public_hw/home/sci_bfu/bkz/Fu_FFLO/stagev_v2_multihead_boundary_learning_3d_v1_hpc}"
CONFIG_JSON="${CONFIG_JSON:-configs/stagev2_topology_coverage_tail_config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageV_V2_TopologyCoverageTail}"
RUN_ID="${RUN_ID:-stagev2_topology_surface_coverage_tail_v1}"
WORLD_SIZE="${WORLD_SIZE:-8}"
START_ITER="${START_ITER:-97}"
FINAL_EXACT_ITER="${FINAL_EXACT_ITER:-100}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01,gpuh14}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"
RUN_DIR="${PACKAGE_DIR}/${OUTPUT_ROOT}/active_runs/${RUN_ID}"
SQUEUE_BIN="${SQUEUE_BIN:-/opt/gridview/slurm-2.3.3/bin/squeue}"
SACCT_BIN="${SACCT_BIN:-/opt/gridview/slurm-2.3.3/bin/sacct}"
SBATCH_BIN="${SBATCH_BIN:-/opt/gridview/slurm-2.3.3/bin/sbatch}"

if [ "${CONFIRM_STAGEV2_TAIL:-0}" != "1" ]; then
  echo "[error] set CONFIRM_STAGEV2_TAIL=1 to submit Stage V-v2 topology coverage tail." >&2
  exit 2
fi

cd "${PACKAGE_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"

"${PYTHON_BIN}" scripts/stagev2_tail_preflight.py \
  --config "${CONFIG_JSON}" \
  --source-project-dir "${SOURCE_PROJECT_DIR}" \
  --check-source
"${PYTHON_BIN}" scripts/prepare_stagev2_tail_run.py \
  --config "${CONFIG_JSON}" \
  --source-project-dir "${SOURCE_PROJECT_DIR}"

wait_for_job() {
  local job_id="$1"
  local label="$2"
  while true; do
    active="$("${SQUEUE_BIN}" -h -j "${job_id}" -o "%i|%T" 2>/dev/null | paste -sd, - || true)"
    if [ -z "${active}" ]; then
      break
    fi
    echo "[wait] Stage V-v2 tail ${label}: squeue active: ${active}"
    sleep "${SLEEP_SECONDS}"
  done

  for attempt in $(seq 1 10); do
    rows="$("${SACCT_BIN}" -j "${job_id}" --format=JobID,State,ExitCode --parsable2 2>/dev/null || true)"
    failures="$(printf "%s\n" "${rows}" | awk -F'|' 'NR>1 && $1 !~ /\.batch|\.extern/ && $2 ~ /FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE/ {print $1":"$2":"$3}' | paste -sd, -)"
    running="$(printf "%s\n" "${rows}" | awk -F'|' 'NR>1 && $1 !~ /\.batch|\.extern/ && $2 ~ /RUNNING|PENDING|CONFIGURING|COMPLETING/ {print $1":"$2":"$3}' | paste -sd, -)"
    completed_count="$(printf "%s\n" "${rows}" | awk -F'|' 'NR>1 && $1 !~ /\.batch|\.extern/ && $2 ~ /COMPLETED/ {n++} END{print n+0}')"
    if [ -n "${failures}" ]; then
      echo "[error] Stage V-v2 tail ${label}: job ${job_id} failed: ${failures}" >&2
      exit 5
    fi
    if [ -z "${running}" ] && [ "${completed_count}" -gt 0 ]; then
      return 0
    fi
    echo "[wait] Stage V-v2 tail ${label}: sacct not settled yet; attempt ${attempt}; rows=${rows//$'\n'/;}"
    sleep 20
  done
  echo "[error] Stage V-v2 tail ${label}: sacct did not settle for job ${job_id}" >&2
  exit 6
}

export PACKAGE_DIR SOURCE_PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID WORLD_SIZE
export TOPOLOGY_GAP_NK="${TOPOLOGY_GAP_NK:-2048}"
export TOPOLOGY_GAP_BACKEND="${TOPOLOGY_GAP_BACKEND:-gpu}"
export TOPOLOGY_GAP_TOL_REL="${TOPOLOGY_GAP_TOL_REL:-1e-8}"
export TOPOLOGY_GAP_TOL_ABS="${TOPOLOGY_GAP_TOL_ABS:-0.0}"
export TOPOLOGY_GAP_K_CHUNK="${TOPOLOGY_GAP_K_CHUNK:-512}"
export PYTHONPATH="${SOURCE_PROJECT_DIR}:${PYTHONPATH:-}"

for ITER in $(seq "${START_ITER}" "${FINAL_EXACT_ITER}"); do
  TAG="$(printf "%03d" "${ITER}")"
  NEXT="$(printf "%03d" "$((ITER + 1))")"
  if [ -f "${RUN_DIR}/dataset_iter${NEXT}.npz" ]; then
    echo "[skip] Stage V-v2 tail iter ${TAG}; dataset_iter${NEXT} already exists"
    continue
  fi

  export ITER
  JOB_ID="$("${SBATCH_BIN}" --parsable --exclude="${EXCLUDE_NODES}" --array=0-$((WORLD_SIZE - 1)) \
    --export=ALL,PACKAGE_DIR="${PACKAGE_DIR}",SOURCE_PROJECT_DIR="${SOURCE_PROJECT_DIR}",PYTHON_BIN="${PYTHON_BIN}",OUTPUT_ROOT="${OUTPUT_ROOT}",RUN_ID="${RUN_ID}",ITER="${ITER}",WORLD_SIZE="${WORLD_SIZE}" \
    scripts/slurm_stagev2_tail_exact_array.sh)"
  echo "[submit] Stage V-v2 tail exact iter ${TAG}: job ${JOB_ID}"
  wait_for_job "${JOB_ID}" "exact iter ${TAG}"

  "${PYTHON_BIN}" -m ml_phase.hpc \
    --merge \
    --run-dir "${RUN_DIR}" \
    --iteration "${ITER}" \
    --world-size "${WORLD_SIZE}" \
    --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"

  "${PYTHON_BIN}" -m ml_phase.append_trusted \
    --dataset "${RUN_DIR}/dataset_iter${TAG}.npz" \
    --trusted-exact "${RUN_DIR}/iter${TAG}/exact_trusted_iter${TAG}.npz" \
    --output-npz "${RUN_DIR}/dataset_iter${NEXT}.npz" \
    --output-csv "${RUN_DIR}/dataset_iter${NEXT}.csv" \
    --output-root "${PACKAGE_DIR}/${OUTPUT_ROOT}"

  echo "[done] Stage V-v2 tail iter ${TAG}; wrote dataset_iter${NEXT}"
done

echo "[done] Stage V-v2 topology coverage tail completed through exact iter $(printf "%03d" "${FINAL_EXACT_ITER}")"
''',
        "scripts/status_stagev2_topology_coverage_tail.sh": r'''#!/bin/bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageV_V2_TopologyCoverageTail}"
RUN_ID="${RUN_ID:-stagev2_topology_surface_coverage_tail_v1}"
RUN_DIR="${PACKAGE_DIR}/${OUTPUT_ROOT}/active_runs/${RUN_ID}"
SQUEUE_BIN="${SQUEUE_BIN:-/opt/gridview/slurm-2.3.3/bin/squeue}"

latest_dataset="$(find "${RUN_DIR}" -maxdepth 1 -name 'dataset_iter*.npz' 2>/dev/null | sed -E 's/.*dataset_iter([0-9]+)\.npz/\1/' | sort -n | tail -1 || true)"
latest_iter="$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'iter[0-9][0-9][0-9]' 2>/dev/null | sed -E 's/.*iter([0-9]+)/\1/' | sort -n | tail -1 || true)"
echo "{"
echo "  \"run_id\": \"${RUN_ID}\","
echo "  \"run_dir\": \"${RUN_DIR}\","
echo "  \"latest_dataset_iteration\": \"${latest_dataset}\","
echo "  \"latest_iteration_dir\": \"${latest_iter}\","
echo "  \"squeue\": \"$("${SQUEUE_BIN}" -h -u "$USER" -o '%i %j %T %M %R' 2>/dev/null | tr '\n' ';' | sed 's/"/\\"/g')\""
echo "}"
''',
        "scripts/collect_stagev2_topology_coverage_tail_results.sh": r'''#!/bin/bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageV_V2_TopologyCoverageTail}"
RUN_ID="${RUN_ID:-stagev2_topology_surface_coverage_tail_v1}"
ARCHIVE="${OUTPUT_ROOT}/stagev2_topology_surface_coverage_tail_results.tar.gz"
cd "${PACKAGE_DIR}"
mkdir -p "${OUTPUT_ROOT}"
tar -czf "${ARCHIVE}" \
  configs data README.md \
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}"/dataset_iter*.csv \
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}"/dataset_iter*.npz \
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}"/iter*/exact_merged_iter*.npz \
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}"/iter*/exact_trusted_iter*.npz \
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}"/iter*/exact_shard_rank*_of*.json \
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}"/iter*/selected_points*.csv \
  slurm-*.out \
  *.nohup.log 2>/dev/null || true
echo "[done] archive=${ARCHIVE}"
''',
    }


if __name__ == "__main__":
    manifest = make_package(Path.cwd())
    print(json.dumps(manifest, indent=2))
