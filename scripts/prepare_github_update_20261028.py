#!/usr/bin/env python3
"""Prepare an allowlisted local checkout update; never perform Git/network actions.

The default is a read-only dry run. Use --apply only after reviewing its summary.
Copies preserve source bytes and relative paths, skip identical destination files,
and never delete destination files. Scientific calculations are never imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKOUT = Path("/tmp/fflo_github_upload_20261028")
MAX_BYTES = 50 * 1024 * 1024
MAX_SMALL_RUN_BYTES = 2 * 1024 * 1024
META = Path("release_metadata/20261028")
ARCHIVE = Path("downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive")
STAGE_V_ROOTS = (
    Path("downloads/stagev_v3_v4_matched_audit_raw_20260712/v3/ML_Phase_StageV_V3_SparseArc/active_runs/stagev_v3_sparse_arc_boundary_support_3d_v1"),
    Path("downloads/stagev_v4_min_report_audit_20260711/extracted/ML_Phase_StageV_V4_GraphAcq/active_runs/stagev_v4_phase_structure_graph_acquisition_3d_v1"),
)
PARENT_OUTPUTS = (
    Path("output/pdf/manuscript_synchronized_review_v7"),
    Path("output/pdf/manuscript_reader_revision_v8"),
)
CURRENT_OUTPUT = Path("output/pdf/manuscript_author_revision_v9")
COMMON_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".history", ".cache", ".codex", ".agents",
    ".aws", ".ssh", "tmp", "temp", "build", "dist", "built", "built_logs",
}
ALWAYS_BLOCKED_DIRS = {".git", ".aws", ".ssh", "private", "credentials", "secrets"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".7z", ".rar")
TEXT_SUFFIXES = {
    ".py", ".sh", ".ps1", ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".tex", ".bib", ".sty",
    ".cls", ".html", ".css", ".js", ".mjs", ".ts", ".svg", ".xml",
    ".log", ".aux", ".bbl", ".blg", ".out", ".diff", ".patch", ".sha256",
}
CODE_CONFIG_SUFFIXES = TEXT_SUFFIXES - {".log", ".aux", ".bbl", ".blg", ".out"}
SMALL_RUN_SUFFIXES = {".json", ".yaml", ".yml", ".md", ".csv", ".txt", ".tex", ".bib"}
RECOVERED_SUFFIXES = CODE_CONFIG_SUFFIXES | {".npz", ".npy"}
EXCLUDED_DOCS = {"docs/hpc_access_gbu_h100.md"}
FIGURE_FOLDERS = (
    "figure_engine", "strict_data_precontracts", "fig1_engine_v3", "fig1_engine_v4",
    "fig2_strict_data", "fig2_reader_revision_v8", "fig3_strict_data",
    "fig4_strict_data_v2", "sm_stageiv_numerical_diagnostics",
    "supplementary_publication_methods", "supplementary_publication_methods_v7",
    "supplementary_publication_methods_v8",
)
LIMITATIONS = [
    "Stage V supplemental downloads include endpoints and compact histories only; intermediate cumulative CSVs are not included by this allowlist. Stage VI sequential replay therefore is not fully reproducible from this update alone.",
    "Stage V learned acquisition and Stage VI world-model code are exploratory, not evidence of learned-acquisition superiority or validated EIG.",
    "Historical archive status manifests may predate the separate V4/VI exports. Current endpoint inclusions do not rewrite those historical records.",
    "Some Figure Engine contracts retain absolute historical paths. Upload does not establish cross-machine engine regeneration or per-run executable binding.",
    "Existing checkout files outside this allowlist are not deleted or scanned by this script. The final staged Git diff needs a separate review.",
    "v7/v8 parent output trees deliberately retain all non-sensitive files, including small source ZIPs and generated files, to preserve exact parent hash guards.",
]
SECRET_RULES = {
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "github_classic_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "github_fine_grained_token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "credential_in_url": re.compile(r"https?://[^\s/:@]+:[^\s/@]{8,}@"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sensitive_name(relative: Path) -> bool:
    parts = {part.lower() for part in relative.parts}
    name = relative.name.lower()
    return bool(parts & ALWAYS_BLOCKED_DIRS) or (
        name == ".env" or name.startswith(".env.")
        or name.startswith(("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"))
        or name.endswith((".pem", ".key", ".p12", ".pfx", ".keystore"))
        or any(word in name for word in ("credentials", "private_key", "access_token", "api_key", "password"))
    )


def excluded(relative: Path, protected_parent: bool = False) -> str | None:
    if sensitive_name(relative):
        return "sensitive_filename"
    if relative.as_posix() in EXCLUDED_DOCS or relative.name.lower() == "hpc_access_gbu_h100.md":
        return "cluster_access_document"
    if relative.name == ".DS_Store":
        return "macos_metadata"
    if protected_parent:
        return None
    if any(part.lower() in COMMON_DIRS for part in relative.parts):
        return "cache_or_build_directory"
    lower = relative.name.lower()
    if lower.endswith(ARCHIVE_SUFFIXES):
        return "duplicate_archive"
    if lower.endswith((".pyc", ".pyo", ".tmp", ".bak", ".sqlite", ".sqlite3", ".db", ".db-wal", ".db-shm")):
        return "cache_or_checkpoint"
    return None


def candidate_files(root: Path) -> tuple[dict[Path, str], dict[str, int], list[str]]:
    selected: dict[Path, str] = {}
    skipped: dict[str, int] = {}
    missing: list[str] = []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    def add(path: Path, group: str, *, protected: bool = False, suffixes: set[str] | None = None) -> None:
        relative = path.relative_to(root)
        reason = excluded(relative, protected)
        if reason:
            skip(reason)
            return
        if path.is_symlink():
            raise ValueError(f"Symlink is not allowed: {relative.as_posix()}")
        if not path.is_file():
            return
        if suffixes is not None and path.suffix.lower() not in suffixes:
            skip("outside_suffix_allowlist")
            return
        if group == "manuscript_runs" and path.stat().st_size > MAX_SMALL_RUN_BYTES:
            skip("large_historical_run_artifact")
            return
        selected[relative] = group

    def tree(relative: Path | str, group: str, *, protected: bool = False, suffixes: set[str] | None = None, no_previews: bool = False) -> None:
        folder = root / relative
        if not folder.is_dir():
            missing.append(str(relative))
            return
        for base, directories, files in os.walk(folder, followlinks=False):
            base_path = Path(base)
            for name in list(directories):
                sub = base_path / name
                rel = sub.relative_to(root)
                reason = excluded(rel, protected)
                if no_previews and name.lower() == "previews":
                    reason = "current_candidate_previews"
                if reason:
                    skip(reason)
                    directories.remove(name)
                elif sub.is_symlink():
                    raise ValueError(f"Symlink directory is not allowed: {rel.as_posix()}")
            for name in sorted(files):
                add(base_path / name, group, protected=protected, suffixes=suffixes)

    for path in root.iterdir():
        if path.suffix.lower() in {".py", ".sh"} or path.name in {"AGENTS.md", "MODEL_SPEC.md", "PROJECT_SUMMARY.md", "README.md"}:
            add(path, "root_source")
    for folder in ("ml_phase", "scripts", "tests", "hpc_hotfix_scripts", "hpc_packages"):
        tree(folder, "program", suffixes=CODE_CONFIG_SUFFIXES)
    tree("docs", "documentation")
    tree("manuscript_variants", "manuscript_variants")
    language = root / "manuscript_langgraph"
    for path in language.iterdir():
        if path.is_file():
            add(path, "manuscript_workflow", suffixes=CODE_CONFIG_SUFFIXES)
        elif path.name == "runs":
            tree(path.relative_to(root), "manuscript_runs", suffixes=SMALL_RUN_SUFFIXES)
        elif path.name == "recovered_artifacts":
            tree(path.relative_to(root), "recovered_evidence", suffixes=RECOVERED_SUFFIXES)
        else:
            tree(path.relative_to(root), "manuscript_workflow", suffixes=CODE_CONFIG_SUFFIXES)
    for path in (root / "figures").iterdir():
        if path.is_file():
            add(path, "figure_builders", suffixes=CODE_CONFIG_SUFFIXES)
    for folder in FIGURE_FOLDERS:
        tree(Path("figures") / folder, "figure_engine_and_assets")
    for relative in PARENT_OUTPUTS:
        tree(relative, "immutable_parent_output", protected=True)
    tree(CURRENT_OUTPUT, "current_manuscript", no_previews=True)
    tree(ARCHIVE / "code_archive", "historical_scientific_code", suffixes=CODE_CONFIG_SUFFIXES)
    tree(ARCHIVE / "data_archive", "scientific_data")
    for path in (root / ARCHIVE).iterdir():
        if path.is_file():
            add(path, "historical_archive_manifest", suffixes=CODE_CONFIG_SUFFIXES)
    for relative in STAGE_V_ROOTS:
        folder = root / relative
        if not folder.is_dir():
            missing.append(relative.as_posix())
            continue
        for path in folder.iterdir():
            if path.name in {"dataset_iter097.csv", "dataset_iter097.npz", "dataset_iter097.append.json", "run_config.json"}:
                add(path, "stage_v_endpoint")
            elif path.is_file() and path.suffix.lower() in {".json", ".csv", ".yaml", ".yml"} and any(
                token in path.stem.lower() for token in ("history", "diagnostic", "stop", "summary")
            ):
                if path.stat().st_size <= MAX_SMALL_RUN_BYTES:
                    add(path, "stage_v_compact_history")
                else:
                    skip("large_stage_v_history")
    for folder in ("world_model_descriptor_dev", "stage_vi0_hierarchical_world_model_validation", "stage_vi0_2_hierarchical_world_model_v2"):
        tree(folder, "exploratory_world_model")
    return selected, skipped, missing


def scan_text(data: bytes, display_path: str) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return []
    return [{"path": display_path, "rule": name} for name, pattern in SECRET_RULES.items() if pattern.search(text)]


def scan_file(path: Path, relative: Path) -> list[dict[str, str]]:
    if path.suffix.lower() != ".zip":
        return scan_text(path.read_bytes(), relative.as_posix())
    findings: list[dict[str, str]] = []
    with zipfile.ZipFile(path) as archive:
        total = 0
        for member in archive.infolist():
            if member.is_dir():
                continue
            nested = Path(member.filename)
            display = relative.as_posix() + "!" + member.filename
            total += member.file_size
            if member.file_size > MAX_BYTES or total > 250 * 1024 * 1024:
                raise ValueError(f"Archive scan size limit exceeded: {relative.as_posix()}")
            if nested.is_absolute() or ".." in nested.parts or sensitive_name(nested):
                findings.append({"path": display, "rule": "unsafe_archive_member_name"})
            if nested.name.lower().endswith(ARCHIVE_SUFFIXES):
                findings.append({"path": display, "rule": "nested_archive_requires_manual_review"})
            else:
                findings.extend(scan_text(archive.read(member), display))
    return findings


def checked_destination(checkout: Path, relative: Path) -> Path:
    destination = checkout / relative
    for part in (destination, *destination.parents):
        if part == checkout:
            break
        if part.is_symlink():
            raise ValueError(f"Destination contains a symlink: {relative.as_posix()}")
    if not destination.resolve().is_relative_to(checkout):
        raise ValueError(f"Destination escapes checkout: {relative.as_posix()}")
    if destination.exists() and not destination.is_file():
        raise ValueError(f"Destination is not a regular file: {relative.as_posix()}")
    return destination


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".release-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--apply", action="store_true", help="Copy selected changes locally; no Git/network action")
    args = parser.parse_args(argv)
    root, checkout = args.root.resolve(), args.checkout.resolve()
    if root == checkout or checkout.is_relative_to(root) or root.is_relative_to(checkout):
        raise ValueError("Source and destination must be separate non-nested directories")
    if not checkout.is_dir() or not (checkout / ".git").exists():
        raise ValueError("Destination must be an existing Git checkout")
    selected, skipped, missing = candidate_files(root)
    rows: list[dict[str, object]] = []
    findings: list[dict[str, str]] = []
    groups: dict[str, dict[str, int]] = {}
    for relative, group in sorted(selected.items()):
        source = root / relative
        size = source.stat().st_size
        if size > MAX_BYTES:
            raise ValueError(f"Single file exceeds 50 MiB: {relative.as_posix()} ({size} bytes)")
        findings.extend(scan_file(source, relative))
        sha = digest(source)
        destination = checked_destination(checkout, relative)
        status = "new" if not destination.exists() else "unchanged" if digest(destination) == sha else "modified"
        rows.append({"path": relative.as_posix(), "bytes": size, "sha256": sha, "group": group, "copy_status": status})
        totals = groups.setdefault(group, {"files": 0, "bytes": 0})
        totals["files"] += 1
        totals["bytes"] += size
    if findings:
        print(json.dumps({"status": "blocked", "reason": "possible_secrets", "findings": findings}, indent=2))
        return 2
    summary = {
        "date": "2026-10-28", "mode": "apply" if args.apply else "dry_run", "status": "checked",
        "source_root": str(root), "checkout": str(checkout),
        "selected_files": len(rows), "selected_bytes": sum(int(row["bytes"]) for row in rows),
        "copy_files": sum(row["copy_status"] != "unchanged" for row in rows),
        "copy_bytes": sum(int(row["bytes"]) for row in rows if row["copy_status"] != "unchanged"),
        "groups": groups, "skipped_counts": skipped, "missing_allowlist_roots": missing,
        "secret_scan": "no_obvious_pattern_hits_not_a_security_guarantee",
        "deletions": 0, "git_actions": False, "network_actions": False, "physics_runs": False,
        "limitations": LIMITATIONS,
    }
    if missing:
        summary["status"] = "blocked_missing_allowlist_roots"
        print(json.dumps(summary, indent=2))
        return 2
    if args.apply:
        # All scanning and destination validation complete before the first copy.
        # Fail if an input changed during review; never silently copy a new version.
        for row in rows:
            source = root / str(row["path"])
            if digest(source) != row["sha256"]:
                raise ValueError(f"Input changed during staging: {row['path']}; rerun")
        for row in rows:
            if row["copy_status"] == "unchanged":
                continue
            source = root / str(row["path"])
            destination = checked_destination(checkout, Path(str(row["path"])))
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".release-", dir=destination.parent)
            os.close(fd)
            try:
                shutil.copy2(source, temporary)
                if digest(Path(temporary)) != row["sha256"]:
                    raise ValueError(f"Input changed while copying: {row['path']}; rerun")
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        manifest = io.StringIO(newline="")
        writer = csv.DictWriter(manifest, fieldnames=["path", "bytes", "sha256", "group", "copy_status"])
        writer.writeheader()
        writer.writerows(rows)
        summary["status"] = "local_copy_complete_not_committed_or_uploaded"
        for name in ("file_manifest.csv", "summary.json"):
            checked_destination(checkout, META / name)
        atomic_text(checkout / META / "file_manifest.csv", manifest.getvalue())
        atomic_text(checkout / META / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}), file=sys.stderr)
        sys.exit(2)
