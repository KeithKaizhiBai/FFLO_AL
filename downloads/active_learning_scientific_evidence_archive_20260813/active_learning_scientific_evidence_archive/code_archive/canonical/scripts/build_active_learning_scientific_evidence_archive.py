from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


ARCHIVE_NAME = "active_learning_scientific_evidence_archive"

BUILD_JUNK_SUFFIXES = {
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pyc",
    ".synctex.gz",
    ".toc",
}

REPORT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

CODE_SUFFIXES = {".m", ".py", ".ps1", ".sh"}

FINAL_DATASETS = [
    {
        "stage": "stage_i",
        "run_id": "active_boundary_discovery_rankcap_k3_full_loop_v1",
        "root": "rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/active_runs/active_boundary_discovery_rankcap_k3_full_loop_v1",
        "dataset": "dataset_iter031",
        "purpose": "Final state of the 2D thermodynamic baseline full loop before the trusted-surprise tail.",
    },
    {
        "stage": "stage_ii",
        "run_id": "active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1",
        "root": "rankcap_k3_tail_surprise_continuation_results/ML_Phase_512_RankCapK3_TailContinuation/active_runs/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1",
        "dataset": "dataset_iter035",
        "purpose": "Frozen publication dataset for the final 2D thermodynamic phase map.",
    },
    {
        "stage": "stage_iii",
        "run_id": "active_phase_topology_from_scratch_full_loop_v2",
        "root": "active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/active_runs/active_phase_topology_from_scratch_full_loop_v2",
        "dataset": "dataset_iter018",
        "purpose": "Final cumulative dataset of the converged 2D topology-aware cold-start loop.",
    },
    {
        "stage": "stage_iv",
        "run_id": "active_phase_topology_3d_t_ja_mu_from_scratch_v1",
        "root": "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/active_runs/active_phase_topology_3d_t_ja_mu_from_scratch_v1",
        "dataset": "dataset_iter025",
        "purpose": "Final local 3D thermodynamic/topology dataset used by the Stage IV reports.",
    },
    {
        "stage": "stage_v_v1",
        "run_id": "stagev_acqv2_boundary_support_learned_residual_3d_v1",
        "root": "stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/ML_Phase_StageV_AcqV2/active_runs/stagev_acqv2_boundary_support_learned_residual_3d_v1",
        "dataset": "dataset_iter093",
        "purpose": "Latest locally returned Stage V-v1 learned-residual prototype dataset.",
    },
    {
        "stage": "stage_v_v2",
        "run_id": "stagev_v2_multihead_boundary_learning_3d_v1",
        "root": "stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/active_runs/stagev_v2_multihead_boundary_learning_3d_v1",
        "dataset": "dataset_iter097",
        "purpose": "Final locally available Stage V-v2 multi-head dataset.",
    },
    {
        "stage": "stage_v_v2_tail",
        "run_id": "stagev2_topology_surface_coverage_tail_v1",
        "root": "stagev2_topology_surface_coverage_tail_v1_hpc/ML_Phase_StageV_V2_TopologyCoverageTail/active_runs/stagev2_topology_surface_coverage_tail_v1",
        "dataset": "dataset_iter101",
        "purpose": "Final Stage V-v2 topology-surface coverage-tail dataset.",
    },
    {
        "stage": "stage_v_v3",
        "run_id": "stagev_v3_sparse_arc_boundary_support_3d_v1",
        "root": "stagev_v3_sparse_arc_boundary_support_3d_v1_hpc/ML_Phase_StageV_V3_SparseArc/active_runs/stagev_v3_sparse_arc_boundary_support_3d_v1",
        "dataset": "dataset_iter097",
        "purpose": "Final Stage V-v3 sparse-arc acquisition dataset.",
    },
]

REPORT_UNITS = [
    (
        "stage_i",
        "rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/full_loop_enhanced_report",
        "rankcap_k3_full_loop_enhanced",
    ),
    (
        "stage_i",
        "reports/_phase2_supporting_reports/trusted_surprise_counterfactual",
        "trusted_surprise_counterfactual",
    ),
    (
        "stage_i",
        "reports/_phase2_supporting_reports/rankcap_k3_tail_surprise_continuation_return",
        "rankcap_k3_tail_surprise_continuation_return",
    ),
    ("stage_i_v", "reports/stagei_to_stagev_comprehensive_report", "stagei_to_stagev_comprehensive_report"),
    ("stage_ii", "project_history/reports/report_phase2_robust_al_final_202606", "phase2_robust_al_final_202606"),
    ("stage_ii", "reports/_phase2_supporting_reports/hard_risk_boundary_impact_audit_v2", "hard_risk_boundary_impact_audit_v2"),
    (
        "stage_ii",
        "reports/_phase2_supporting_reports/hard_risk_boundary_impact_audit",
        "hard_risk_boundary_impact_audit_v1",
    ),
    ("stage_iii", "reports/topology_pass_dataset_iter035_v1", "topology_pass_dataset_iter035_v1"),
    (
        "stage_iii",
        "active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topology_convergence_audit_full_loop_v2",
        "topology_convergence_audit_full_loop_v2",
    ),
    (
        "stage_iv",
        "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local",
        "stageiv_3d_return_report_local",
    ),
    (
        "stage_iv",
        "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_audit_local",
        "stageiv_3d_audit_local",
    ),
    (
        "stage_iv",
        "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_mu_slice_boundary_audit_local",
        "stageiv_mu_slice_boundary_audit_local",
    ),
    (
        "stage_iv",
        "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_curve_extraction_fix_local",
        "stageiv_curve_extraction_fix_local",
    ),
    ("stage_iv", "reports/stageiv_3d_traceability_audit", "stageiv_3d_traceability_audit"),
    ("stage_iv", "reports/stageiv_3d_external_dependency_audit", "stageiv_3d_external_dependency_audit"),
    (
        "stage_v_v1",
        "stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/ML_Phase_StageV_AcqV2/reports/stagev_acqv2_return_report_local",
        "stagev_acqv2_return_report_local",
    ),
    (
        "stage_v_v2",
        "stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_return_report_local",
        "stagev_v2_return_report_local",
    ),
    (
        "stage_v_v2",
        "stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_formal_audit_local",
        "stagev_v2_formal_audit_local",
    ),
    (
        "stage_v_v2_tail",
        "stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_tail_latex_report_local",
        "stagev_v2_tail_latex_report_local",
    ),
    (
        "stage_v_v3",
        "stagev_v3_sparse_arc_boundary_support_3d_v1_hpc/ML_Phase_StageV_V3_SparseArc/reports/stagev_v3_formal_audit_local",
        "stagev_v3_formal_audit_local",
    ),
    (
        "stage_v_v3",
        "stagev_v3_sparse_arc_boundary_support_3d_v1_hpc/ML_Phase_StageV_V3_SparseArc/reports/stagev_v3_return_report_local",
        "stagev_v3_return_report_local",
    ),
    ("stage_v_v4", "reports/stagev_v4_graph_acq_smoke", "stagev_v4_graph_acq_smoke"),
    ("stage_v_v4", "reports/stagev_v4_hpc_package", "stagev_v4_hpc_package"),
]

CODE_UNITS = [
    ("canonical", "ml_phase", "ml_phase"),
    ("stage_iii", "active_phase_topology_from_scratch_full_loop_v1_hpc/ml_phase", "ml_phase"),
    (
        "stage_iv",
        "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ml_phase",
        "ml_phase",
    ),
    ("stage_v_v3", "hpc_packages/stagev_v3_sparse_arc_boundary_support_3d_v1_hpc/ml_phase", "ml_phase"),
    ("stage_v_v4", "hpc_packages/stagev_v4_phase_structure_graph_acquisition_3d_v1_hpc/ml_phase", "ml_phase"),
]

PACKAGE_SUPPORT_UNITS = [
    (
        "stage_iii",
        "hpc_packages/active_phase_topology_from_scratch_full_loop_v1_hpc",
        "stage_iii_hpc_package_support",
    ),
    (
        "stage_iv",
        "hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624",
        "stage_iv_hpc_package_support",
    ),
    (
        "stage_v_v3",
        "hpc_packages/stagev_v3_sparse_arc_boundary_support_3d_v1_hpc",
        "stage_v_v3_hpc_package_support",
    ),
    (
        "stage_v_v4",
        "hpc_packages/stagev_v4_phase_structure_graph_acquisition_3d_v1_hpc",
        "stage_v_v4_hpc_package_support",
    ),
]

KEY_FIGURES = [
    {
        "id": "stage_ii_final_phase_map",
        "purpose": "Frozen thermodynamic normal/uniform-SC/FFLO map with hard-risk uncertainty overlay.",
        "source": "project_history/reports/report_phase2_robust_al_final_202606/figures/final_phase_map_with_uncertainty.png",
        "dataset": "data_archive/stage_ii/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/dataset_iter035.npz",
        "script": "figures_archive/plotting_scripts/build_phase2_final_audit_and_report.py",
    },
    {
        "id": "stage_ii_hard_risk_band",
        "purpose": "Hard-risk uncertainty band used by the publication boundary audit.",
        "source": "project_history/reports/report_phase2_robust_al_final_202606/figures/hard_risk_uncertainty_band.png",
        "dataset": "figures_archive/figure_data/stage_ii/hard_risk_boundary_impact_audit_v2",
        "script": "figures_archive/plotting_scripts/build_hard_risk_boundary_impact_audit.py",
    },
    {
        "id": "stage_iii_final_topology_map",
        "purpose": "Final 2D cFFLO/tFFLO contour with trusted exact support.",
        "source": "active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topology_convergence_audit_full_loop_v2/figures/fig01_final_map_contour.png",
        "dataset": "data_archive/stage_iii/topology_pass_dataset_iter035_v1/dataset_iter035_topology_ground_v1.npz",
        "script": "figures_archive/plotting_scripts/build_topology_convergence_audit.py",
    },
    {
        "id": "stage_iii_boundary_convergence",
        "purpose": "Overlay of final-five topology contours for the formal convergence audit.",
        "source": "active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topology_convergence_audit_full_loop_v2/figures/fig02_final5_contour_overlay.png",
        "dataset": "figures_archive/figure_data/stage_iii/topology_convergence_audit_full_loop_v2",
        "script": "figures_archive/plotting_scripts/build_topology_convergence_audit.py",
    },
    {
        "id": "stage_iv_3d_phase_map",
        "purpose": "Transparent 3D thermodynamic phase map in (kBT, JA, mu).",
        "source": "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/figures/phase_3d_transparent.png",
        "dataset": "data_archive/stage_iv/active_phase_topology_3d_t_ja_mu_from_scratch_v1/dataset_iter025.npz",
        "script": "figures_archive/plotting_scripts/build_stageiv_3d_return_report.py",
    },
    {
        "id": "stage_iv_mu_slice_atlas",
        "purpose": "Fixed-mu thermodynamic slice atlas for 2D/3D comparison.",
        "source": "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/figures/phase_mu_slice_atlas.png",
        "dataset": "data_archive/stage_iv/active_phase_topology_3d_t_ja_mu_from_scratch_v1/dataset_iter025.npz",
        "script": "figures_archive/plotting_scripts/build_stageiv_3d_return_report.py",
    },
    {
        "id": "stage_v_v2_topology_audit",
        "purpose": "Stage V-v2 final 3D topology map used by the multi-head audit.",
        "source": "stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_formal_audit_local/figures/fig02_final_3d_fflo_topology_map.png",
        "dataset": "data_archive/stage_v_v2/stagev_v2_multihead_boundary_learning_3d_v1/dataset_iter097.npz",
        "script": "figures_archive/plotting_scripts/build_stagev_v2_formal_audit.py",
    },
    {
        "id": "stage_v_v3_sparse_arc_result",
        "purpose": "Stage V-v3 final topology point cloud after sparse-arc-aware acquisition.",
        "source": "stagev_v3_sparse_arc_boundary_support_3d_v1_hpc/ML_Phase_StageV_V3_SparseArc/reports/stagev_v3_return_report_local/figures/stagev3_final_topology_point_cloud.png",
        "dataset": "data_archive/stage_v_v3/stagev_v3_sparse_arc_boundary_support_3d_v1/dataset_iter097.npz",
        "script": "figures_archive/plotting_scripts/build_stagev_v3_return_report.py",
    },
    {
        "id": "stage_v_v3_reward_learning",
        "purpose": "Reward/value-model learning diagnostics for the sparse-arc acquisition function.",
        "source": "stagev_v3_sparse_arc_boundary_support_3d_v1_hpc/ML_Phase_StageV_V3_SparseArc/reports/stagev_v3_return_report_local/figures/stagev3_reward_learning.png",
        "dataset": "figures_archive/figure_data/stage_v_v3/stagev_v3_return_report_local",
        "script": "figures_archive/plotting_scripts/build_stagev_v3_return_report.py",
    },
]

PLOTTING_SCRIPTS = [
    "scripts/build_phase2_final_audit_and_report.py",
    "scripts/build_hard_risk_boundary_impact_audit.py",
    "scripts/run_topology_pass_dataset_iter035.py",
    "scripts/build_topology_convergence_audit.py",
    "scripts/build_stageiv_3d_return_report.py",
    "scripts/stageiv_3d_convergence_audit.py",
    "scripts/stageiv_3d_hidden_slice_audit.py",
    "scripts/build_stagev_v2_formal_audit.py",
    "scripts/build_stagev_v2_return_report.py",
    "scripts/build_stagev_v3_return_report.py",
    "plot_eta_phase_diagram.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the filtered Active Learning scientific evidence archive."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Fu_FFLO project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Archive directory (default: <project-root>/{ARCHIVE_NAME}).",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Build and verify the directory but do not create a ZIP package.",
    )
    return parser.parse_args()


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_build_junk(path: Path) -> bool:
    lower_name = path.name.lower()
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts.intersection({"__pycache__", ".pytest_cache", "pdf_render_check"}):
        return True
    if lower_name.endswith(".synctex.gz"):
        return True
    return path.suffix.lower() in BUILD_JUNK_SUFFIXES


def report_filter(path: Path) -> bool:
    if is_build_junk(path):
        return False
    if path.name.lower() in {"build_log.txt", "figure_generation_log.txt"}:
        return False
    return path.suffix.lower() in REPORT_SUFFIXES


def code_filter(path: Path) -> bool:
    if is_build_junk(path):
        return False
    return path.suffix.lower() in CODE_SUFFIXES


def package_support_filter(path: Path) -> bool:
    if is_build_junk(path):
        return False
    lower = path.as_posix().lower()
    if any(token in lower for token in ("candidate_scores", "active_runs/", "nohup", "slurm-")):
        return False
    if path.suffix.lower() in {".npz", ".npy", ".h5", ".pt", ".pkl", ".zip", ".gz"}:
        return False
    return path.suffix.lower() in CODE_SUFFIXES.union(REPORT_SUFFIXES) or path.name in {
        "AGENTS.md",
        "MODEL_SPEC.md",
        "README.md",
        "run_manifest.json",
        "package_validation.json",
        "package_validation.md",
        "SHA256SUMS.txt",
    }


def classify_unselected(relative_path: str, size: int) -> tuple[str, str, str]:
    lower = relative_path.lower()
    name = Path(relative_path).name.lower()
    parts = set(Path(relative_path).parts)
    lower_parts = {part.lower() for part in parts}

    secret_tokens = ("id_rsa", "id_ed25519", ".pem", ".env", "credential", "private_key", "token")
    if any(token in lower for token in secret_tokens):
        return "D", "Sensitive or private configuration is outside the scientific archive scope.", "EXCLUDE"
    if lower.startswith(f"{ARCHIVE_NAME}/") or lower.startswith(".git/"):
        return "D", "Archive output or Git implementation metadata.", "EXCLUDE"
    if any(token in lower for token in ("manuscript", "langgraph", "qahe3d", "writing_workflow")):
        return "D", "Migrated manuscript/workflow scope is explicitly excluded.", "EXCLUDE"
    if lower.startswith("shared_prompts/"):
        return "D", "Cross-session prompt workflow is not scientific evidence.", "EXCLUDE"
    if lower_parts.intersection({"__pycache__", ".pytest_cache", ".mypy_cache", "cache", "tmp", "temp", "build"}):
        return "D", "Cache, temporary, or build artifact.", "EXCLUDE"
    if is_build_junk(Path(relative_path)):
        return "D", "Generated build/debug artifact.", "EXCLUDE"
    if any(token in lower for token in ("candidate_scores", "candidate_pool", "candidate_cloud")):
        return "D", "Historical candidate pool is redundant after final selected/exact states are retained.", "EXCLUDE"
    if name.startswith("slurm-") or "nohup" in name or name.endswith(".trace"):
        return "D", "Intermediate scheduler/debug log without unique scientific content.", "EXCLUDE"
    if path_is_archive_bundle(name):
        return "D", "Redundant ZIP/TAR package; curated source artifacts are archived separately.", "EXCLUDE"
    if "smoke" in lower and "stagev_v4_graph_acq_smoke" not in lower:
        return "D", "Smoke-test output is superseded by validated production/report artifacts.", "EXCLUDE"
    if "debug" in lower or "hotfix" in lower or "migration" in lower:
        return "D", "Temporary debug, hotfix, or one-time migration artifact.", "EXCLUDE"
    if "dataset_iter" in name:
        return "C", "Intermediate cumulative checkpoint; final frozen state is retained separately.", "OPTIONAL_ARCHIVE"
    if path_has_checkpoint_semantics(lower, name):
        return "C", "Model/checkpoint state is optional unless it is a small final state.", "OPTIONAL_ARCHIVE"
    if lower.startswith("reports/") or "/reports/" in lower:
        if path_is_formal_report(path=Path(relative_path)):
            return "B", "Scientific report material that may support later interpretation.", "KEEP_IF_SMALL"
        return "C", "Intermediate report material or derived visualization.", "OPTIONAL_ARCHIVE"
    if Path(relative_path).suffix.lower() in CODE_SUFFIXES:
        return "B", "Scientific or operational source code; canonical copies are retained by whitelist.", "KEEP_IF_SMALL"
    if Path(relative_path).suffix.lower() in {".csv", ".json", ".npz", ".parquet", ".yaml", ".yml"}:
        return "C", "Potentially useful derived/intermediate scientific data.", "OPTIONAL_ARCHIVE"
    if size > 100 * 1024 * 1024:
        return "D", "Large non-whitelisted artifact without final-scientific-data status.", "EXCLUDE"
    return "C", "Not required by the curated scientific evidence package.", "OPTIONAL_ARCHIVE"


def path_is_archive_bundle(name: str) -> bool:
    return name.endswith((".tar", ".tar.gz", ".tgz", ".zip", ".7z"))


def path_has_checkpoint_semantics(lower: str, name: str) -> bool:
    return "checkpoint" in lower or name.endswith((".pt", ".pth", ".pkl", ".joblib"))


def path_is_formal_report(path: Path) -> bool:
    lower = path.as_posix().lower()
    return any(token in lower for token in ("audit", "final", "return_report", "convergence", "decision"))


def enumerate_source_files(project_root: Path, output_dir: Path) -> list[Path]:
    output_resolved = output_dir.resolve()
    files: list[Path] = []
    for base, dirs, names in os.walk(project_root):
        base_path = Path(base)
        dirs[:] = [
            name
            for name in dirs
            if (base_path / name).resolve() != output_resolved
            and name not in {".git"}
        ]
        for name in names:
            path = base_path / name
            if path.resolve() in {
                (project_root / f"{ARCHIVE_NAME}.zip").resolve(),
                (project_root / f"{ARCHIVE_NAME}.zip.sha256").resolve(),
            }:
                continue
            if path.is_file():
                files.append(path)
    files.sort(key=lambda item: item.as_posix().lower())
    return files


def write_inventory(
    project_root: Path,
    output_path: Path,
    source_files: Iterable[Path],
    selected: dict[str, dict[str, str]],
) -> Counter:
    counts: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "size_bytes",
                "modified_time",
                "category",
                "scientific_relevance",
                "recommended_action",
                "archive_path",
            ],
        )
        writer.writeheader()
        for path in source_files:
            relative = relative_posix(path, project_root)
            stat = path.stat()
            if relative in selected:
                record = selected[relative]
                category = record["tier"]
                relevance = record["purpose"]
                action = "MUST_KEEP" if category == "A" else "KEEP_IF_SMALL"
                archive_path = record["archive_path"]
            else:
                category, relevance, action = classify_unselected(relative, stat.st_size)
                archive_path = ""
            counts[(category, "files")] += 1
            counts[(category, "bytes")] += stat.st_size
            writer.writerow(
                {
                    "path": relative,
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "category": category,
                    "scientific_relevance": relevance,
                    "recommended_action": action,
                    "archive_path": archive_path,
                }
            )
    return counts


def register_copy(
    project_root: Path,
    archive_root: Path,
    source: Path,
    destination_relative: str,
    selected: dict[str, dict[str, str]],
    copy_records: list[dict[str, str | int]],
    stage: str,
    tier: str,
    purpose: str,
) -> None:
    if not source.is_file():
        return
    source_relative = relative_posix(source, project_root)
    destination = archive_root / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"Hash mismatch while copying {source_relative}")
    selected[source_relative] = {
        "tier": tier,
        "purpose": purpose,
        "archive_path": destination_relative,
        "stage": stage,
    }
    copy_records.append(
        {
            "source_path": source_relative,
            "archive_path": destination_relative,
            "size_bytes": source.stat().st_size,
            "sha256": destination_hash,
            "hash_check": "pass",
            "category": tier,
            "stage": stage,
            "purpose": purpose,
        }
    )


def register_tree(
    project_root: Path,
    archive_root: Path,
    source_relative: str,
    destination_relative: str,
    selected: dict[str, dict[str, str]],
    copy_records: list[dict[str, str | int]],
    stage: str,
    tier: str,
    purpose: str,
    predicate: Callable[[Path], bool],
    missing: list[str],
) -> None:
    source_root = project_root / source_relative
    if not source_root.is_dir():
        missing.append(source_relative)
        return
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or not predicate(source):
            continue
        child = source.relative_to(source_root).as_posix()
        register_copy(
            project_root,
            archive_root,
            source,
            f"{destination_relative}/{child}",
            selected,
            copy_records,
            stage,
            tier,
            purpose,
        )


def copy_dataset_and_run_evidence(
    project_root: Path,
    archive_root: Path,
    spec: dict[str, str],
    selected: dict[str, dict[str, str]],
    copy_records: list[dict[str, str | int]],
    missing: list[str],
) -> None:
    run_root = project_root / spec["root"]
    destination_root = f"data_archive/{spec['stage']}/{spec['run_id']}"
    if not run_root.is_dir():
        missing.append(spec["root"])
        return

    final_names = {
        f"{spec['dataset']}.npz",
        f"{spec['dataset']}.csv",
        f"{spec['dataset']}.append.json",
        "metrics_history.json",
        "run_config.json",
        "stop_metrics_history.json",
        "stop_state.json",
        "tail_restart_manifest.json",
        "stagev_acqv2_reward_history.csv",
        "stagev_v2_reward_history.csv",
        "stagev_v3_reward_history.csv",
        "stagev_v4_reward_history.csv",
        "graph_history.csv",
        "graph_state.json",
    }
    found_dataset = False
    for name in sorted(final_names):
        source = run_root / name
        if source.is_file():
            found_dataset = found_dataset or name == f"{spec['dataset']}.npz"
            register_copy(
                project_root,
                archive_root,
                source,
                f"{destination_root}/{name}",
                selected,
                copy_records,
                spec["stage"],
                "A",
                spec["purpose"],
            )
    if not found_dataset:
        missing.append(f"{spec['root']}/{spec['dataset']}.npz")

    selected_names = {
        "selected_points.csv",
        "selected_points_metadata.csv",
        "selection_diagnostics.json",
        "stagev_v2_selection_summary.json",
        "stagev_v3_selection_summary.json",
        "stagev_v4_selection_summary.json",
        "graph_snapshot.json",
        "graph_update_summary.json",
    }
    for iteration_dir in sorted(run_root.glob("iter[0-9][0-9][0-9]")):
        for source in sorted(iteration_dir.iterdir()):
            if not source.is_file():
                continue
            lower_name = source.name.lower()
            keep = source.name in selected_names or (
                lower_name.endswith(".json")
                and any(token in lower_name for token in ("reward", "graph", "selection_summary"))
            )
            if not keep:
                continue
            register_copy(
                project_root,
                archive_root,
                source,
                f"{destination_root}/selection_history/{iteration_dir.name}/{source.name}",
                selected,
                copy_records,
                spec["stage"],
                "B",
                "Selected-point, proposal-source, reward, or graph-update provenance.",
            )


def copy_topology_ground_dataset(
    project_root: Path,
    archive_root: Path,
    selected: dict[str, dict[str, str]],
    copy_records: list[dict[str, str | int]],
    missing: list[str],
) -> None:
    source_root = project_root / "reports/topology_pass_dataset_iter035_v1/tables"
    names = {
        "dataset_iter035_topology_ground_v1.csv",
        "dataset_iter035_topology_ground_v1.npz",
        "pfaffian_validation.csv",
        "topology_boundary_candidate_edges.csv",
        "topology_coverage_holes.csv",
        "topology_count_summary.csv",
    }
    for name in sorted(names):
        source = source_root / name
        if source.is_file():
            register_copy(
                project_root,
                archive_root,
                source,
                f"data_archive/stage_iii/topology_pass_dataset_iter035_v1/{name}",
                selected,
                copy_records,
                "stage_iii",
                "A",
                "Final topology labels, Pfaffian validation, and surface-extraction support data.",
            )
        else:
            missing.append(relative_posix(source, project_root))


def copy_canonical_docs_and_code(
    project_root: Path,
    archive_root: Path,
    selected: dict[str, dict[str, str]],
    copy_records: list[dict[str, str | int]],
    missing: list[str],
) -> None:
    root_files = [
        "AGENTS.md",
        "MODEL_SPEC.md",
        "eta_phase_diagram_cuda.py",
        "tfflo_1d_cuda.py",
        "plot_eta_phase_diagram.py",
        "finite_T_phase_diagram.m",
    ]
    for relative in root_files:
        source = project_root / relative
        if not source.is_file():
            missing.append(relative)
            continue
        destination_root = "code_archive/canonical" if source.suffix.lower() in CODE_SUFFIXES else "code_archive/documentation"
        register_copy(
            project_root,
            archive_root,
            source,
            f"{destination_root}/{source.name}",
            selected,
            copy_records,
            "cross_stage",
            "A" if source.suffix.lower() in CODE_SUFFIXES else "B",
            "Canonical physical model, solver, plotting, or agent provenance needed for scientific reproduction.",
        )

    canonical_docs = [
        "docs/PROJECT_SUMMARY.md",
        "docs/MODEL_SPEC.md",
        "docs/NUMERICS_SPEC.md",
        "docs/DECISIONS.md",
    ]
    for relative in canonical_docs:
        source = project_root / relative
        if source.is_file():
            register_copy(
                project_root,
                archive_root,
                source,
                f"code_archive/documentation/{source.name}",
                selected,
                copy_records,
                "cross_stage",
                "A",
                "Canonical scientific/numerical project memory.",
            )
        else:
            missing.append(relative)

    report_qa = project_root / "docs/report_qa"
    if report_qa.is_dir():
        for source in sorted(report_qa.glob("*.md")):
            register_copy(
                project_root,
                archive_root,
                source,
                f"code_archive/documentation/report_qa/{source.name}",
                selected,
                copy_records,
                "cross_stage",
                "B",
                "Scientific interpretation note retained for later evidence review.",
            )

    for stage, source_relative, destination_name in CODE_UNITS:
        register_tree(
            project_root,
            archive_root,
            source_relative,
            f"code_archive/{stage}/{destination_name}",
            selected,
            copy_records,
            stage,
            "A",
            "Scientific implementation of exact-oracle, labeling, topology, acquisition, or audit logic.",
            code_filter,
            missing,
        )

    script_root = project_root / "scripts"
    if script_root.is_dir():
        excluded_tokens = ("debug", "hotfix", "migrate", "migration", "recover", "smoke")
        for source in sorted(script_root.iterdir()):
            if not source.is_file() or source.suffix.lower() not in CODE_SUFFIXES:
                continue
            lower = source.name.lower()
            if any(token in lower for token in excluded_tokens):
                continue
            register_copy(
                project_root,
                archive_root,
                source,
                f"code_archive/canonical/scripts/{source.name}",
                selected,
                copy_records,
                "cross_stage",
                "B",
                "Small scientific data-processing, plotting, validation, or HPC script.",
            )

    for stage, source_relative, destination_name in PACKAGE_SUPPORT_UNITS:
        source_root = project_root / source_relative
        if not source_root.is_dir():
            missing.append(source_relative)
            continue
        for subdir in ("configs", "scripts", "tests", "docs"):
            child = source_root / subdir
            if not child.is_dir():
                continue
            register_tree(
                project_root,
                archive_root,
                relative_posix(child, project_root),
                f"code_archive/{stage}/{destination_name}/{subdir}",
                selected,
                copy_records,
                stage,
                "B",
                "Small package configuration, validation, documentation, or execution support.",
                package_support_filter,
                missing,
            )
        for name in ("AGENTS.md", "MODEL_SPEC.md", "README.md", "run_manifest.json", "package_validation.json", "package_validation.md"):
            source = source_root / name
            if source.is_file():
                register_copy(
                    project_root,
                    archive_root,
                    source,
                    f"code_archive/{stage}/{destination_name}/{name}",
                    selected,
                    copy_records,
                    stage,
                    "B",
                    "HPC package provenance and reproducibility metadata.",
                )


def copy_reports(
    project_root: Path,
    archive_root: Path,
    selected: dict[str, dict[str, str]],
    copy_records: list[dict[str, str | int]],
    missing: list[str],
) -> None:
    for stage, source_relative, destination_name in REPORT_UNITS:
        register_tree(
            project_root,
            archive_root,
            source_relative,
            f"reports_archive/{stage}/{destination_name}",
            selected,
            copy_records,
            stage,
            "A",
            "Formal scientific report, audit, decision log, table, or associated final figure.",
            report_filter,
            missing,
        )


def copy_figure_reproducibility_material(
    project_root: Path,
    archive_root: Path,
    selected: dict[str, dict[str, str]],
    copy_records: list[dict[str, str | int]],
    missing: list[str],
) -> list[dict[str, str]]:
    manifest_entries: list[dict[str, str]] = []
    for item in KEY_FIGURES:
        source = project_root / item["source"]
        generated_paths: list[str] = []
        if source.is_file():
            stem_destination = f"figures_archive/generated_figures/{item['id']}"
            for candidate in (source, source.with_suffix(".pdf"), source.with_suffix(".svg")):
                if not candidate.is_file():
                    continue
                destination = f"{stem_destination}{candidate.suffix.lower()}"
                register_copy(
                    project_root,
                    archive_root,
                    candidate,
                    destination,
                    selected,
                    copy_records,
                    item["id"].split("_")[0],
                    "A",
                    item["purpose"],
                )
                generated_paths.append(destination)
        else:
            missing.append(item["source"])
        manifest_entries.append({**item, "generated": generated_paths})

    for relative in PLOTTING_SCRIPTS:
        source = project_root / relative
        if not source.is_file():
            missing.append(relative)
            continue
        register_copy(
            project_root,
            archive_root,
            source,
            f"figures_archive/plotting_scripts/{source.name}",
            selected,
            copy_records,
            "cross_stage",
            "A",
            "Plot/report source needed to regenerate archived scientific figures.",
        )

    table_units = [
        (
            "stage_ii",
            "reports/_phase2_supporting_reports/hard_risk_boundary_impact_audit_v2/tables",
            "hard_risk_boundary_impact_audit_v2",
        ),
        (
            "stage_iii",
            "active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topology_convergence_audit_full_loop_v2/tables",
            "topology_convergence_audit_full_loop_v2",
        ),
        (
            "stage_iv",
            "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_mu_slice_boundary_audit_local/tables",
            "stageiv_mu_slice_boundary_audit_local",
        ),
        (
            "stage_v_v2",
            "stagev_v2_multihead_boundary_learning_3d_v1_hpc/ML_Phase_StageV_V2_Multihead/reports/stagev_v2_formal_audit_local/tables",
            "stagev_v2_formal_audit_local",
        ),
        (
            "stage_v_v3",
            "stagev_v3_sparse_arc_boundary_support_3d_v1_hpc/ML_Phase_StageV_V3_SparseArc/reports/stagev_v3_return_report_local/tables",
            "stagev_v3_return_report_local",
        ),
    ]
    for stage, source_relative, destination_name in table_units:
        register_tree(
            project_root,
            archive_root,
            source_relative,
            f"figures_archive/figure_data/{stage}/{destination_name}",
            selected,
            copy_records,
            stage,
            "A",
            "Numerical table used to reconstruct a report figure or boundary diagnostic.",
            lambda path: path.suffix.lower() in {".csv", ".json", ".npz", ".yaml", ".yml"},
            missing,
        )
    return manifest_entries


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_figure_manifest(path: Path, entries: list[dict[str, str]]) -> None:
    lines = ["figures:"]
    for item in entries:
        lines.extend(
            [
                f"  {item['id']}:",
                f"    purpose: {yaml_quote(item['purpose'])}",
                f"    source_dataset: {yaml_quote(item['dataset'])}",
                f"    plotting_script: {yaml_quote(item['script'])}",
                "    required_inputs:",
                f"      - {yaml_quote(item['dataset'])}",
                f"      - {yaml_quote(item['script'])}",
                "    generated_files:",
            ]
        )
        generated = item.get("generated", [])
        if generated:
            lines.extend(f"      - {yaml_quote(value)}" for value in generated)
        else:
            lines.append("      - missing")
        lines.append(f"    original_source: {yaml_quote(item['source'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage_status(path: Path, project_root: Path) -> None:
    stage_records = [
        {
            "stage": "Stage I",
            "run_ids": ["active_boundary_discovery_rankcap_k3_full_loop_v1"],
            "final_dataset": "data_archive/stage_i/active_boundary_discovery_rankcap_k3_full_loop_v1/dataset_iter031.npz",
            "reports": ["reports_archive/stage_i/rankcap_k3_full_loop_enhanced"],
            "figure_sources": ["reports_archive/stage_i_v/stagei_to_stagev_comprehensive_report"],
            "scripts": ["code_archive/canonical", "code_archive/canonical/scripts"],
            "audit_status": "completed_baseline; final main-phase convergence is established by the Stage II tail audit",
            "missing_artifacts": ["standalone Stage-I-only final report"],
        },
        {
            "stage": "Stage II",
            "run_ids": ["active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1"],
            "final_dataset": "data_archive/stage_ii/active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/dataset_iter035.npz",
            "reports": [
                "reports_archive/stage_ii/phase2_robust_al_final_202606",
                "reports_archive/stage_ii/hard_risk_boundary_impact_audit_v2",
            ],
            "figure_sources": ["figures_archive/figure_data/stage_ii"],
            "scripts": ["code_archive/canonical/ml_phase"],
            "audit_status": "main phase map publication audit passed; hard-risk frontier documented",
            "missing_artifacts": [],
        },
        {
            "stage": "Stage III",
            "run_ids": ["topology_pass_dataset_iter035_v1", "active_phase_topology_from_scratch_full_loop_v2"],
            "final_dataset": "data_archive/stage_iii/topology_pass_dataset_iter035_v1/dataset_iter035_topology_ground_v1.npz",
            "reports": [
                "reports_archive/stage_iii/topology_pass_dataset_iter035_v1",
                "reports_archive/stage_iii/topology_convergence_audit_full_loop_v2",
            ],
            "figure_sources": ["figures_archive/figure_data/stage_iii"],
            "scripts": ["code_archive/stage_iii"],
            "audit_status": "2D cFFLO/tFFLO topology classification and convergence audit passed",
            "missing_artifacts": [],
        },
        {
            "stage": "Stage IV",
            "run_ids": ["active_phase_topology_3d_t_ja_mu_from_scratch_v1"],
            "final_dataset": "data_archive/stage_iv/active_phase_topology_3d_t_ja_mu_from_scratch_v1/dataset_iter025.npz",
            "reports": [
                "reports_archive/stage_iv/stageiv_3d_return_report_local",
                "reports_archive/stage_iv/stageiv_mu_slice_boundary_audit_local",
            ],
            "figure_sources": ["figures_archive/figure_data/stage_iv"],
            "scripts": ["code_archive/stage_iv"],
            "audit_status": "3D structures recovered; formal 3D convergence not passed",
            "missing_artifacts": [],
        },
        {
            "stage": "Stage V-v1",
            "run_ids": ["stagev_acqv2_boundary_support_learned_residual_3d_v1"],
            "final_dataset": "data_archive/stage_v_v1/stagev_acqv2_boundary_support_learned_residual_3d_v1/dataset_iter093.npz",
            "reports": ["reports_archive/stage_v_v1/stagev_acqv2_return_report_local"],
            "figure_sources": [],
            "scripts": ["code_archive/canonical"],
            "audit_status": "prototype; scalar reward learned but acquisition remained imbalanced",
            "missing_artifacts": ["dataset_iter100 final merged local copy"],
        },
        {
            "stage": "Stage V-v2 and coverage tail",
            "run_ids": ["stagev_v2_multihead_boundary_learning_3d_v1", "stagev2_topology_surface_coverage_tail_v1"],
            "final_dataset": "data_archive/stage_v_v2_tail/stagev2_topology_surface_coverage_tail_v1/dataset_iter101.npz",
            "reports": [
                "reports_archive/stage_v_v2/stagev_v2_formal_audit_local",
                "reports_archive/stage_v_v2_tail/stagev_v2_tail_latex_report_local",
            ],
            "figure_sources": ["figures_archive/figure_data/stage_v_v2"],
            "scripts": ["code_archive/canonical"],
            "audit_status": "Decision C; topology map change, coverage, and surprise/support remained limiting",
            "missing_artifacts": [],
        },
        {
            "stage": "Stage V-v3",
            "run_ids": ["stagev_v3_sparse_arc_boundary_support_3d_v1"],
            "final_dataset": "data_archive/stage_v_v3/stagev_v3_sparse_arc_boundary_support_3d_v1/dataset_iter097.npz",
            "reports": ["reports_archive/stage_v_v3/stagev_v3_return_report_local"],
            "figure_sources": ["figures_archive/figure_data/stage_v_v3"],
            "scripts": ["code_archive/stage_v_v3"],
            "audit_status": "Decision C; coverage improved by about 47.7% versus v2-tail but remained above gate",
            "missing_artifacts": [],
        },
        {
            "stage": "Stage V-v4",
            "run_ids": ["stagev_v4_phase_structure_graph_acquisition_3d_v1"],
            "final_dataset": "missing locally",
            "reports": ["reports_archive/stage_v_v4/stagev_v4_graph_acq_smoke"],
            "figure_sources": [],
            "scripts": ["code_archive/stage_v_v4"],
            "audit_status": "implementation/package validated; local final production result not present",
            "missing_artifacts": ["final cumulative dataset", "formal production convergence audit", "final return report"],
        },
        {
            "stage": "Stage VI",
            "run_ids": [],
            "final_dataset": "not present",
            "reports": [],
            "figure_sources": [],
            "scripts": [],
            "audit_status": "not_present",
            "missing_artifacts": ["no independent Stage VI world-model experiment was found"],
        },
    ]

    lines = ["stages:"]
    for record in stage_records:
        lines.append(f"  - stage_name: {yaml_quote(record['stage'])}")
        for key in ("run_ids", "reports", "figure_sources", "scripts", "missing_artifacts"):
            lines.append(f"    {key}:")
            values = record[key]
            if values:
                lines.extend(f"      - {yaml_quote(value)}" for value in values)
            else:
                lines.append("      []")
        lines.append(f"    final_dataset: {yaml_quote(record['final_dataset'])}")
        lines.append(f"    audit_status: {yaml_quote(record['audit_status'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_code_readmes(archive_root: Path) -> None:
    readmes = {
        "code_archive/README.md": """# Scientific Code Archive

This directory contains source code required to reproduce the retained active-learning scientific evidence. It excludes caches, debug scripts, one-time migrations, model caches, and full historical run outputs.

- `canonical/`: current exact-oracle, phase-label, acquisition, plotting, validation, and report code.
- `stage_iii/`: topology-aware 2D implementation snapshot and package support.
- `stage_iv/`: 3D topology-aware implementation snapshot and package support.
- `stage_v_v3/`: sparse-arc acquisition implementation and production configuration.
- `stage_v_v4/`: phase-structure graph acquisition implementation and validated package configuration.

Inputs are the frozen datasets under `../data_archive/`. Outputs are reports, tables, and figures under `../reports_archive/` and `../figures_archive/`. Python/CUDA/Slurm dependencies are documented in the retained package READMEs and manifests.
""",
        "data_archive/README.md": """# Frozen and Final Data Archive

This directory stores selected final cumulative datasets and compact run provenance. Intermediate candidate pools and redundant cumulative checkpoints are intentionally excluded. Dataset bytes are copied unchanged and verified by SHA256.
""",
        "reports_archive/README.md": """# Scientific Reports Archive

This directory stores formal convergence reports, audit reports, final run reports, and scientifically relevant failure analyses. LaTeX build junk, page-render checks, and routine scheduler logs are excluded.
""",
        "figures_archive/README.md": """# Figure Reproducibility Archive

- `generated_figures/`: selected final PNG/PDF/SVG assets.
- `figure_data/`: source tables used by the selected figures.
- `plotting_scripts/`: scripts that regenerate or assemble the figures.
- `FIGURE_MANIFEST.yaml`: source dataset, script, inputs, and generated files for each logical figure.

The smooth surfaces and contours retained here remain subject to the scientific caveats in their source audits; they are not silently promoted to definitive boundaries.
""",
    }
    for relative, content in readmes.items():
        path = archive_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def write_copy_manifest(path: Path, records: list[dict[str, str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "archive_path",
                "size_bytes",
                "sha256",
                "hash_check",
                "category",
                "stage",
                "purpose",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: str(row["archive_path"]).lower()))


def write_checksums(path: Path, archive_root: Path) -> None:
    lines: list[str] = []
    for file_path in sorted(archive_root.rglob("*")):
        if not file_path.is_file() or file_path == path:
            continue
        relative = file_path.relative_to(archive_root).as_posix()
        lines.append(f"{sha256_file(file_path)}  {relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_archive_manifest(
    path: Path,
    project_root: Path,
    counts: Counter,
    copy_records: list[dict[str, str | int]],
    missing: list[str],
) -> None:
    git_commit = "unknown"
    git_head = project_root / ".git/HEAD"
    if git_head.is_file():
        head_text = git_head.read_text(encoding="utf-8", errors="replace").strip()
        if head_text.startswith("ref: "):
            ref_path = project_root / ".git" / head_text[5:]
            if ref_path.is_file():
                git_commit = ref_path.read_text(encoding="utf-8", errors="replace").strip()
        elif head_text:
            git_commit = head_text

    copied_bytes = sum(int(record["size_bytes"]) for record in copy_records)
    lines = [
        "project:",
        "  name: \"BdG FFLO topology-aware active learning\"",
        "archive:",
        "  purpose: \"scientific evidence for independent manuscript writing, figure regeneration, data audit, and result verification\"",
        f"  created_at_utc: {yaml_quote(datetime.now(timezone.utc).isoformat())}",
        f"  source_root: {yaml_quote(str(project_root.resolve()))}",
        f"  git_commit: {yaml_quote(git_commit)}",
        "  scope:",
        "    - exact-oracle and active-learning computational history",
        "    - frozen/final datasets",
        "    - convergence and reliability audits",
        "    - figure source data and plotting scripts",
        "  excluded_scope:",
        "    - manuscript LangGraph and claim registry",
        "    - AI writing workflow and shared prompt handoff",
        "    - QAHE3D-related material",
        "    - caches, debug logs, candidate pools, and redundant checkpoints",
        "summary:",
        f"  source_file_count: {sum(counts[(tier, 'files')] for tier in 'ABCD')}",
        f"  archived_source_file_count: {len(copy_records)}",
        f"  archived_source_bytes: {copied_bytes}",
        "  category_counts:",
    ]
    for tier in "ABCD":
        lines.append(f"    {tier}: {{files: {counts[(tier, 'files')]}, bytes: {counts[(tier, 'bytes')]}}}")
    lines.extend(
        [
            "integrity:",
            "  copied_file_hash_check: pass",
            "  algorithm: sha256",
            "  checksum_file: checksums_sha256.txt",
            "missing_or_unavailable_artifacts:",
        ]
    )
    if missing:
        lines.extend(f"  - {yaml_quote(value)}" for value in sorted(set(missing)))
    else:
        lines.append("  []")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit_report(
    path: Path,
    counts: Counter,
    copy_records: list[dict[str, str | int]],
    missing: list[str],
    zip_path: Path | None,
) -> None:
    copied_bytes = sum(int(record["size_bytes"]) for record in copy_records)
    lines = [
        "# Active Learning Scientific Evidence Archive Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope Decision",
        "",
        "This archive covers Active Learning scientific computation, exact-oracle development, active-learning loops, HPC execution provenance, audits, reports, and figure-generation source data. Manuscript LangGraph, claim registry, AI writing workflow, QAHE3D files, caches, and temporary development artifacts are excluded.",
        "",
        "No original project file was deleted or modified by the archive builder.",
        "",
        "## Inventory Summary",
        "",
        "| Category | Meaning | Files | Bytes |",
        "| --- | --- | ---: | ---: |",
    ]
    meanings = {
        "A": "MUST_KEEP",
        "B": "KEEP_IF_SMALL",
        "C": "OPTIONAL_ARCHIVE",
        "D": "EXCLUDE",
    }
    for tier in "ABCD":
        lines.append(
            f"| {tier} | {meanings[tier]} | {counts[(tier, 'files')]} | {counts[(tier, 'bytes')]} |"
        )
    lines.extend(
        [
            "",
            "The per-file classification, relevance statement, action, and archive destination are recorded in `archive_inventory.csv`.",
            "",
            "## Stage Status",
            "",
            "- Stage I: 2D thermodynamic baseline retained; final convergence semantics are superseded by Stage II.",
            "- Stage II: frozen `dataset_iter035` and publication-grade thermodynamic/hard-risk audits retained.",
            "- Stage III: final topology dataset, Pfaffian/bulk-gap evidence, and the formally converged 2D topology-boundary audit retained.",
            "- Stage IV: final 3D dataset, fixed-mu slices, phase/topology maps, selected-point history, and audit reports retained; formal 3D convergence remains unpassed.",
            "- Stage V-v1/v2/v2-tail/v3: final locally available datasets, reward/proposal histories, formal audits, and matched comparisons retained.",
            "- Stage V-v4: source, config, smoke validation, and package provenance retained; a local final production dataset/report was not found.",
            "- Stage VI: no independent formal world-model experiment was found.",
            "",
            "Full run IDs, artifact paths, status, and missing items are in `stage_archive_status.yaml`.",
            "",
            "## Archive Construction",
            "",
            f"- Copied source files: {len(copy_records)}",
            f"- Copied source bytes: {copied_bytes}",
            "- Source-to-destination SHA256 checks: pass",
            "- Final data formats were preserved byte-for-byte; already-compressed NPZ files were not recomputed.",
            "- CSV-to-Parquet conversion was not applied because it would create a second representation and is unnecessary at the retained data sizes.",
            "- Redundant ZIP/TAR bundles, candidate-score clouds, historical cumulative checkpoints, Slurm/nohup logs, LaTeX build junk, and caches were not copied.",
        ]
    )
    if zip_path is not None:
        lines.append(f"- Transfer package: `{zip_path.name}`")
    lines.extend(
        [
            "",
            "## Figure Reproducibility",
            "",
            "`figures_archive/FIGURE_MANIFEST.yaml` maps each selected logical figure to its purpose, source dataset, plotting script, required inputs, original source, and retained generated files. Source data and scripts are retained separately from final PNG/PDF/SVG assets.",
            "",
            "## Reports and Code",
            "",
            "Formal scientific reports are under `reports_archive/`. Reproduction-relevant exact-oracle, topology, acquisition, plotting, data-processing, validation, and HPC code is under `code_archive/`. Each top-level retained area has a README describing purpose, inputs, outputs, and dependencies.",
            "",
            "## Missing or Limited Evidence",
            "",
        ]
    )
    if missing:
        lines.extend(f"- `{value}`" for value in sorted(set(missing)))
    else:
        lines.append("- None detected among the explicit whitelist artifacts.")
    lines.extend(
        [
            "",
            "## Scientific Caveats",
            "",
            "- Provisional or hard-risk labels remain provisional; the archive does not promote them to definitive phases.",
            "- Smooth 3D surfaces and report-only contours retain the caveats of their source audits.",
            "- Stage IV and Stage V-v1/v2/v3 are not relabeled as converged by archival inclusion.",
            "- Stage V-v4 production completion cannot be inferred from package/smoke artifacts alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_readme(path: Path) -> None:
    path.write_text(
        """# Active Learning Scientific Evidence Archive

This is a filtered, hash-verified archive of the Windows-side BdG/FFLO active-learning scientific evidence. It is intended for independent manuscript writing on another machine, figure regeneration, audit, and numerical result verification.

Start with:

1. `archive_audit_report.md`
2. `stage_archive_status.yaml`
3. `ACTIVE_LEARNING_ARCHIVE_MANIFEST.yaml`
4. `figures_archive/FIGURE_MANIFEST.yaml`
5. `copied_file_manifest.csv` and `checksums_sha256.txt`

The archive intentionally excludes manuscript LangGraph/claim-registry work, shared prompts, QAHE3D material, caches, temporary files, candidate pools, redundant checkpoints, and routine scheduler logs.
""",
        encoding="utf-8",
    )


def make_zip(archive_root: Path) -> Path:
    zip_path = archive_root.parent / f"{archive_root.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in sorted(archive_root.rglob("*")):
            if path.is_file():
                package.write(path, arcname=f"{archive_root.name}/{path.relative_to(archive_root).as_posix()}")
    return zip_path


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    archive_root = (args.output_dir or (project_root / ARCHIVE_NAME)).resolve()
    if project_root == archive_root:
        raise SystemExit("Archive output must not be the project root.")

    archive_root.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict[str, str]] = {}
    copy_records: list[dict[str, str | int]] = []
    missing: list[str] = []

    # Phase 1: inventory first. A final inventory is rewritten after the curated
    # copy so archive destinations are explicit for every retained source file.
    source_files = enumerate_source_files(project_root, archive_root)
    preliminary_counts = write_inventory(
        project_root,
        archive_root / "archive_inventory.csv",
        source_files,
        selected,
    )
    write_audit_report(
        archive_root / "archive_audit_report.md",
        preliminary_counts,
        copy_records,
        missing,
        None,
    )

    # Phase 2: curated evidence copy.
    for spec in FINAL_DATASETS:
        copy_dataset_and_run_evidence(
            project_root, archive_root, spec, selected, copy_records, missing
        )
    copy_topology_ground_dataset(
        project_root, archive_root, selected, copy_records, missing
    )
    copy_reports(project_root, archive_root, selected, copy_records, missing)
    copy_canonical_docs_and_code(
        project_root, archive_root, selected, copy_records, missing
    )
    figure_entries = copy_figure_reproducibility_material(
        project_root, archive_root, selected, copy_records, missing
    )

    write_code_readmes(archive_root)
    write_root_readme(archive_root / "README.md")
    write_figure_manifest(
        archive_root / "figures_archive/FIGURE_MANIFEST.yaml", figure_entries
    )
    write_stage_status(archive_root / "stage_archive_status.yaml", project_root)
    write_copy_manifest(archive_root / "copied_file_manifest.csv", copy_records)

    # Phase 3: final inventory, manifests, and integrity package.
    final_counts = write_inventory(
        project_root,
        archive_root / "archive_inventory.csv",
        source_files,
        selected,
    )
    write_archive_manifest(
        archive_root / "ACTIVE_LEARNING_ARCHIVE_MANIFEST.yaml",
        project_root,
        final_counts,
        copy_records,
        missing,
    )
    planned_zip_path = (
        archive_root.parent / f"{archive_root.name}.zip"
        if not args.no_zip
        else None
    )
    write_audit_report(
        archive_root / "archive_audit_report.md",
        final_counts,
        copy_records,
        missing,
        planned_zip_path,
    )

    archive_summary = {
        "status": "pass",
        "archive_root": str(archive_root),
        "source_file_count": len(source_files),
        "archived_source_file_count": len(copy_records),
        "archived_source_bytes": sum(int(row["size_bytes"]) for row in copy_records),
        "missing_explicit_artifacts": sorted(set(missing)),
        "zip_path": str(planned_zip_path) if planned_zip_path else None,
        "zip_sha256": "recorded in the adjacent .zip.sha256 sidecar"
        if planned_zip_path
        else None,
    }
    (archive_root / "archive_build_summary.json").write_text(
        json.dumps(archive_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_checksums(archive_root / "checksums_sha256.txt", archive_root)

    zip_path = None
    zip_hash = None
    if planned_zip_path is not None:
        zip_path = make_zip(archive_root)
        zip_hash = sha256_file(zip_path)
        zip_hash_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
        zip_hash_path.write_text(
            f"{zip_hash}  {zip_path.name}\n", encoding="utf-8"
        )

    runtime_summary = dict(archive_summary)
    runtime_summary["zip_sha256"] = zip_hash
    print(json.dumps(runtime_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
