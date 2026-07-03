from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.stageiv_3d import StageIV3DConfig, normal_band_count_scan


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path | None) -> StageIV3DConfig:
    return StageIV3DConfig.from_json(path) if path is not None else StageIV3DConfig()


def audit_mu_propagation() -> list[dict[str, Any]]:
    checks = [
        {
            "component": "stageiv_selector",
            "file": "ml_phase/stageiv_3d.py",
            "required_patterns": ["sobol_points_3d", "\"mu\"", "candidate_points[:, 2]", "write_empty_stageiv_dataset"],
        },
        {
            "component": "hpc_partition",
            "file": "ml_phase/hpc.py",
            "required_patterns": ["optional mu", "(n, 3) [kT, JA, mu]", "[\"mu\"] if points.shape[1] == 3"],
        },
        {
            "component": "exact_oracle_cli",
            "file": "ml_phase/exact_oracle.py",
            "required_patterns": ["point_columns = [\"kT\", \"JA\"]", "mu = float(row[2])", "replace(base_cfg, mu=mu)"],
        },
        {
            "component": "exact_output",
            "file": "ml_phase/exact_oracle.py",
            "required_patterns": ["mu: float = 0.55", "mu: np.ndarray", "return asdict(self)", "point.mu = mu"],
        },
        {
            "component": "dataset_append",
            "file": "ml_phase/active_refine.py",
            "required_patterns": ["result.get(\"mu\"", "old_x.shape[1]", "\"mu\": x_all[:, 2]"],
        },
        {
            "component": "dataset_load",
            "file": "ml_phase/dataset_builder.py",
            "required_patterns": ["\"mu\": 0.55", "x[:, 2] if x.shape[1] >= 3"],
        },
        {
            "component": "topology_pfaffian",
            "file": "ml_phase/topology_oracle.py",
            "required_patterns": ["def analytic_pfaffians", "mu: np.ndarray | float | None", "mu_a = np.asarray"],
        },
        {
            "component": "topology_bulk_gap",
            "file": "ml_phase/topology_oracle.py",
            "required_patterns": ["def compute", "mu: np.ndarray | None", "mu=mu_t"],
        },
    ]
    rows: list[dict[str, Any]] = []
    for check in checks:
        path = ROOT / str(check["file"])
        text = read_text(path) if path.exists() else ""
        missing = [pat for pat in check["required_patterns"] if pat not in text]
        rows.append(
            {
                "component": check["component"],
                "file": check["file"],
                "status": "pass" if not missing else "fail",
                "missing_patterns": ";".join(missing),
                "sha256": sha256_file(path) if path.exists() else "missing",
            }
        )
    return rows


def config_rows(cfg: StageIV3DConfig) -> list[dict[str, Any]]:
    mu_reference_inside = float(cfg.mu_min) <= float(cfg.mu_reference) <= float(cfg.mu_max)
    production_mu_exact = float(cfg.mu_min) == -0.5 and float(cfg.mu_max) == 1.5
    guard_contains = float(cfg.guard_mu_min) <= float(cfg.mu_min) and float(cfg.guard_mu_max) >= float(cfg.mu_max)
    return [
        {"item": "run_id", "value": cfg.run_id, "status": "pass"},
        {"item": "output_root", "value": cfg.output_root, "status": "pass"},
        {"item": "kBT_range", "value": f"[{cfg.kt_min}, {cfg.kt_max}]", "status": "pass"},
        {"item": "JA_range", "value": f"[{cfg.ja_min}, {cfg.ja_max}]", "status": "pass"},
        {"item": "mu_range", "value": f"[{cfg.mu_min}, {cfg.mu_max}]", "status": "pass" if production_mu_exact else "fail"},
        {"item": "mu_reference", "value": cfg.mu_reference, "status": "pass" if mu_reference_inside else "fail"},
        {"item": "guard_mu_range", "value": f"[{cfg.guard_mu_min}, {cfg.guard_mu_max}]", "status": "pass" if guard_contains else "fail"},
        {"item": "initial_seed_size", "value": cfg.initial_seed_size, "status": "pass" if int(cfg.initial_seed_size) == 1024 else "fail"},
        {"item": "batch_size", "value": cfg.batch_size, "status": "pass" if int(cfg.batch_size) == 256 else "fail"},
        {"item": "max_acquisition_batches", "value": cfg.max_acquisition_batches, "status": "pass" if int(cfg.max_acquisition_batches) == 24 else "fail"},
        {"item": "t", "value": cfg.t, "status": "pass" if float(cfg.t) == 1.0 else "fail"},
        {"item": "U", "value": cfg.u, "status": "pass"},
    ]


def write_report(
    path: Path,
    cfg: StageIV3DConfig,
    mu_rows: list[dict[str, Any]],
    config_audit: list[dict[str, Any]],
    scan_summary: dict[str, Any],
) -> None:
    failed = [row for row in mu_rows + config_audit if row.get("status") != "pass"]
    lines = [
        "# Stage IV 3D Preflight Check",
        "",
        f"- run_id: `{cfg.run_id}`",
        f"- output_root: `{cfg.output_root}`",
        f"- production mu/t range: `[{cfg.mu_min}, {cfg.mu_max}]`",
        f"- guard mu/t range: `[{cfg.guard_mu_min}, {cfg.guard_mu_max}]`",
        f"- preflight_status: `{'pass' if not failed else 'fail'}`",
        "",
        "## Mu Propagation Audit",
        "",
        "The 3D parameter `mu` is propagated through point selection, shard partitioning, exact-oracle input, exact-oracle output, dataset append, and topology Pfaffian/bulk-gap diagnostics.",
        "",
        "## Normal-State Band-Count Guard Scan",
        "",
        f"- scanned rows: `{scan_summary['rows']}`",
        f"- no-FS fraction: `{scan_summary['no_fs_fraction']:.6g}`",
        f"- single-pair fraction: `{scan_summary['single_pair_fraction']:.6g}`",
        f"- multi-pair fraction: `{scan_summary['multi_pair_fraction']:.6g}`",
        "",
        "This scan is a guard diagnostic over the wider mu range; it does not change the production physical definitions.",
    ]
    if failed:
        lines += ["", "## Failed Checks", ""]
        for row in failed:
            lines.append(f"- {row.get('component', row.get('item'))}: {row.get('missing_patterns', row.get('value'))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight audit for Stage IV 3D topology-aware cold-start package.")
    p.add_argument("--config", type=Path, default=None, help="Stage IV JSON config.")
    p.add_argument("--output-dir", type=Path, default=Path("reports/stageiv_3d_preflight"), help="Output directory.")
    p.add_argument("--n-ja", type=int, default=48)
    p.add_argument("--n-mu", type=int, default=64)
    p.add_argument("--n-k", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = args.output_dir
    tables = out / "tables"
    mu_rows = audit_mu_propagation()
    cfg_rows = config_rows(cfg)
    write_csv(tables / "mu_propagation_audit.csv", mu_rows)
    write_csv(tables / "stageiv_config_audit.csv", cfg_rows)
    scan = normal_band_count_scan(cfg, n_ja=int(args.n_ja), n_mu=int(args.n_mu), n_k=int(args.n_k))
    scan.to_csv(tables / "normal_band_count_scan.csv", index=False)
    scan_summary = {
        "rows": int(scan.shape[0]),
        "no_fs_fraction": float(scan["no_fs_diagnostic"].mean()),
        "single_pair_fraction": float(scan["single_pair_diagnostic"].mean()),
        "multi_pair_fraction": float(scan["multi_pair_diagnostic"].mean()),
        "min_abs_band_energy_min": float(scan["min_abs_band_energy"].min()),
    }
    write_json(out / "stageiv_preflight_summary.json", scan_summary)
    write_json(out / "stageiv_config_snapshot.json", cfg.to_dict())
    write_report(out / "stageiv_preflight_check.md", cfg, mu_rows, cfg_rows, scan_summary)
    print(f"Wrote Stage IV preflight outputs under {out}")


if __name__ == "__main__":
    main()
