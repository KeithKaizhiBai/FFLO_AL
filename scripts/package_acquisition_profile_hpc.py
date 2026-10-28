from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.config import ActiveLearningConfig


OUT_ROOT = ROOT / "hpc_packages"
SELF_CONTAINED_NAME = "hpc_upload_robust_oracle_label_closure_acq_compare_20260601"
FULL_RUN_ID = "active_boundary_discovery_robust_oracle_full_acq_label_closed_v1"
SIMPLE_RUN_ID = "active_boundary_discovery_robust_oracle_simple_acq_label_closed_v1"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, ignore: shutil.IgnorePattern | None = None) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def _normalize_shell_scripts(root: Path) -> None:
    for path in root.rglob("*.sh"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def _manifest(
    run_id: str,
    profile: str,
    cfg: ActiveLearningConfig,
    package_name: str,
) -> dict[str, Any]:
    return {
        "package_name": package_name,
        "run_id": run_id,
        "acquisition_profile": profile,
        "oracle_mode": "robust_al",
        "run_mode": "discovery",
        "candidate_domain_mode": "full",
        "initialization": "random_grid",
        "finite_t_band_width": None,
        "selection_mode": "stochastic",
        "initial_seed_size": int(cfg.initial_seed_size),
        "batch_size_max": int(cfg.batch_size_max),
        "points_per_iter": int(cfg.points_per_iter),
        "iterations": int(cfg.iterations),
        "random_seed": int(cfg.random_seed),
        "notes": [
            "Do not overwrite existing run directories.",
            "Use standalone run_id output paths.",
            "Generate md + pdf + csv report artifacts.",
        ],
    }


def _readme(manifest: dict[str, Any]) -> str:
    return f"""# {manifest['package_name']}

Run ID: `{manifest['run_id']}`
Acquisition profile: `{manifest['acquisition_profile']}`
Oracle mode: `{manifest['oracle_mode']}`

## Submission

```bash
bash scripts/submit_active_loop.sh
```

## Resume

```bash
START_ITER=17 N_ITERS=83 bash scripts/submit_active_loop.sh
```

## Collect report

```bash
bash scripts/collect_and_report.sh
```
"""


def _submit_script(manifest: dict[str, Any]) -> str:
    return f"""#!/bin/bash
set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-$PWD}}"
LOG_DIR="${{PROJECT_DIR}}/logs"
mkdir -p "${{LOG_DIR}}"
ENV_LOG="${{LOG_DIR}}/env_snapshot.txt"

echo "timestamp=$(date -Iseconds)" > "${{ENV_LOG}}"
echo "hostname=$(hostname)" >> "${{ENV_LOG}}"
echo "run_id={manifest['run_id']}" >> "${{ENV_LOG}}"
echo "acquisition_profile={manifest['acquisition_profile']}" >> "${{ENV_LOG}}"
echo "oracle_mode={manifest['oracle_mode']}" >> "${{ENV_LOG}}"
echo "cuda_visible_devices=${{CUDA_VISIBLE_DEVICES:-}}" >> "${{ENV_LOG}}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >> "${{ENV_LOG}}" || true
python - <<'PY' >> "${{ENV_LOG}}" 2>&1
import torch,sys
print("python", sys.version)
print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
PY
git rev-parse --short HEAD >> "${{ENV_LOG}}" 2>/dev/null || echo "git_commit=N/A" >> "${{ENV_LOG}}"

START_ITER="${{START_ITER:-0}}"
N_ITERS="${{N_ITERS:-{manifest['iterations']}}}"

nohup env \\
RUN_ID="{manifest['run_id']}" \\
RUN_MODE="discovery" \\
CANDIDATE_DOMAIN_MODE="full" \\
INITIALIZATION="random_grid" \\
SELECTION_MODE="stochastic" \\
INITIAL_SEED_SIZE="{manifest['initial_seed_size']}" \\
BATCH_SIZE_MAX="{manifest['batch_size_max']}" \\
POINTS_PER_ITER="{manifest['points_per_iter']}" \\
START_ITER="${{START_ITER}}" \\
N_ITERS="${{N_ITERS}}" \\
RANDOM_SEED="{manifest['random_seed']}" \\
ORACLE_MODE="robust_al" \\
ACQUISITION_PROFILE="{manifest['acquisition_profile']}" \\
bash hpc_active_loop.sh > "${{LOG_DIR}}/active_loop.nohup.log" 2>&1 &
echo $! > "${{LOG_DIR}}/active_loop.pid"
echo "started pid=$(cat "${{LOG_DIR}}/active_loop.pid")"
"""


def _collect_script(run_id: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
python -m ml_phase.report_builder --run-id "{run_id}" --run-root "ML_Phase/active_runs" --output "ML_Phase/reports/{run_id}.tex"
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory ML_Phase/reports "ML_Phase/reports/{run_id}.tex"
fi
python scripts/export_active_run_report_md.py --run-id "{run_id}" --run-root "ML_Phase/active_runs" --out-dir "ML_Phase/reports"
echo "report ready: ML_Phase/reports/{run_id}.tex"
"""


def _decision_log(profile: str) -> str:
    return f"""# Decision Log

Profile: `{profile}`

- Purpose: compare discovery active-learning behavior under robust exact oracle.
- Constraint: do not alter thermodynamic phase criterion.
- Output: standalone run directory and report artifacts.
"""


def _selfcontained_launch_script(run_id: str, profile: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PYTHON_BIN="${{PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}}"
WORLD_SIZE="${{WORLD_SIZE:-8}}"
START_ITER="${{START_ITER:-0}}"
N_ITERS="${{N_ITERS:-100}}"
EXCLUDE_NODES="${{EXCLUDE_NODES:-gpuh01}}"
LOG_DIR="${{PROJECT_DIR}}/logs/{profile}"
mkdir -p "${{LOG_DIR}}"

echo "timestamp=$(date -Iseconds)" > "${{LOG_DIR}}/env_snapshot.txt"
echo "hostname=$(hostname)" >> "${{LOG_DIR}}/env_snapshot.txt"
echo "project_dir=${{PROJECT_DIR}}" >> "${{LOG_DIR}}/env_snapshot.txt"
echo "run_id={run_id}" >> "${{LOG_DIR}}/env_snapshot.txt"
echo "acquisition_profile={profile}" >> "${{LOG_DIR}}/env_snapshot.txt"
echo "oracle_mode=robust_al" >> "${{LOG_DIR}}/env_snapshot.txt"
echo "cuda_visible_devices=${{CUDA_VISIBLE_DEVICES:-}}" >> "${{LOG_DIR}}/env_snapshot.txt"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >> "${{LOG_DIR}}/env_snapshot.txt" || true
"${{PYTHON_BIN}}" - <<'PY' >> "${{LOG_DIR}}/env_snapshot.txt" 2>&1
import sys
try:
    import torch
    print("python", sys.version)
    print("torch", torch.__version__)
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("torch.version.cuda", torch.version.cuda)
except Exception as exc:
    print("torch_check_failed", repr(exc))
PY

nohup env \\
PROJECT_DIR="${{PROJECT_DIR}}" \\
PYTHON_BIN="${{PYTHON_BIN}}" \\
RUN_ID="{run_id}" \\
RUN_MODE="discovery" \\
CANDIDATE_DOMAIN_MODE="full" \\
INITIALIZATION="random_grid" \\
INITIAL_SEED_SIZE=512 \\
BATCH_SIZE_MAX=256 \\
POINTS_PER_ITER=256 \\
START_ITER="${{START_ITER}}" \\
N_ITERS="${{N_ITERS}}" \\
RANDOM_SEED=42 \\
ORACLE_MODE="robust_al" \\
ACQUISITION_PROFILE="{profile}" \\
WORLD_SIZE="${{WORLD_SIZE}}" \\
EXCLUDE_NODES="${{EXCLUDE_NODES}}" \\
bash "${{PROJECT_DIR}}/hpc_active_loop.sh" > "${{LOG_DIR}}/active_loop.nohup.log" 2>&1 &
echo $! > "${{LOG_DIR}}/active_loop.pid"
echo "started {profile}: pid=$(cat "${{LOG_DIR}}/active_loop.pid")"
echo "log: ${{LOG_DIR}}/active_loop.nohup.log"
"""


def _selfcontained_readme() -> str:
    return """# Robust Oracle Acquisition Comparison HPC Package

This is a self-contained HPC run directory. After extracting the tarball, run
commands from this directory. Do not copy files out of code_snapshot.

This package includes the 2026-06-01 robust-oracle label-closure patch. Stable
normal points are expected to enter the training dataset with
q_status=not_applicable and q_unresolved=false.

## Recommended first run: mini AL

Run a short validation before a full 100-iteration comparison:

```bash
N_ITERS=5 bash run_full_acquisition.sh
N_ITERS=5 bash run_simple_phase_acquisition.sh
```

## Start baseline full acquisition

```bash
bash run_full_acquisition.sh
```

## Start simplified phase acquisition

```bash
bash run_simple_phase_acquisition.sh
```

## Check status

```bash
tail -n 80 logs/full/active_loop.nohup.log
tail -n 80 logs/simple_phase/active_loop.nohup.log
squeue -u $USER
```

## Resume examples

```bash
START_ITER=17 N_ITERS=83 bash run_full_acquisition.sh
START_ITER=17 N_ITERS=83 bash run_simple_phase_acquisition.sh
```

## Output directories

```text
ML_Phase/active_runs/active_boundary_discovery_robust_oracle_full_acq_label_closed_v1
ML_Phase/active_runs/active_boundary_discovery_robust_oracle_simple_acq_label_closed_v1
```
"""


def _build_selfcontained_package() -> Path:
    root = OUT_ROOT / SELF_CONTAINED_NAME
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    code_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache")
    ml_phase_ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".mypy_cache",
        "active_runs",
        "datasets",
        "figures",
        "reports",
        "models",
    )
    _copy_tree(ROOT / "ml_phase", root / "ml_phase", ignore=ml_phase_ignore)
    _copy_tree(ROOT / "scripts", root / "scripts", ignore=code_ignore)
    _copy_tree(ROOT / "report", root / "report", ignore=shutil.ignore_patterns("figures"))
    _copy_tree(ROOT / "docs", root / "docs", ignore=code_ignore)
    validation_report = ROOT / "reports" / "robust_oracle_label_closure_validation"
    if validation_report.exists():
        _copy_tree(validation_report, root / "reports" / "robust_oracle_label_closure_validation", ignore=code_ignore)

    for p in [
        "hpc_active_loop.sh",
        "eta_phase_diagram_cuda.py",
        "tfflo_1d_cuda.py",
        "MODEL_SPEC.md",
        "AGENTS.md",
    ]:
        if (ROOT / p).exists():
            _copy(ROOT / p, root / p)

    manifests = [
        _manifest(
            run_id=FULL_RUN_ID,
            profile="full",
            cfg=ActiveLearningConfig(acquisition_profile="full", oracle_mode="robust_al", iterations=100, points_per_iter=256),
            package_name=SELF_CONTAINED_NAME,
        ),
        _manifest(
            run_id=SIMPLE_RUN_ID,
            profile="simple_phase",
            cfg=ActiveLearningConfig(acquisition_profile="simple_phase", oracle_mode="robust_al", iterations=100, points_per_iter=256),
            package_name=SELF_CONTAINED_NAME,
        ),
    ]
    _write_text(root / "RUN_MANIFEST.json", json.dumps({"runs": manifests}, indent=2))
    _write_text(root / "README.md", _selfcontained_readme())
    _write_text(
        root / "run_full_acquisition.sh",
        _selfcontained_launch_script(FULL_RUN_ID, "full"),
    )
    _write_text(
        root / "run_simple_phase_acquisition.sh",
        _selfcontained_launch_script(SIMPLE_RUN_ID, "simple_phase"),
    )
    _write_text(
        root / "collect_compare.sh",
        """#!/bin/bash
set -euo pipefail
python scripts/compare_acquisition_profiles.py \\
  --full-run ML_Phase/active_runs/active_boundary_discovery_robust_oracle_full_acq_label_closed_v1 \\
  --simple-run ML_Phase/active_runs/active_boundary_discovery_robust_oracle_simple_acq_label_closed_v1
""",
    )
    _normalize_shell_scripts(root)
    for p in [
        "run_full_acquisition.sh",
        "run_simple_phase_acquisition.sh",
        "collect_compare.sh",
        "hpc_active_loop.sh",
        "scripts/slurm_active_refine.sh",
        "scripts/slurm_exact_oracle_array.sh",
    ]:
        target = root / p
        if target.exists():
            target.chmod(0o755)
    return root


def _build_package(package_name: str, run_id: str, profile: str) -> Path:
    cfg = ActiveLearningConfig(
        acquisition_profile=profile,
        run_mode="discovery",
        candidate_domain_mode="full",
        initialization="random_grid",
        initial_seed_size=512,
        batch_size_max=256,
        points_per_iter=256,
        iterations=100,
        random_seed=42,
        oracle_mode="robust_al",
        finite_t_band_width=None,
    )
    root = OUT_ROOT / package_name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    manifest = _manifest(run_id=run_id, profile=profile, cfg=cfg, package_name=package_name)
    _write_text(root / "RUN_MANIFEST.json", json.dumps(manifest, indent=2))
    _write_text(root / "README.md", _readme(manifest))
    _write_text(root / "config" / "active_learning_config.json", json.dumps(asdict(cfg), indent=2))
    _write_text(root / "scripts" / "submit_active_loop.sh", _submit_script(manifest))
    _write_text(root / "scripts" / "submit_exact_array.sh", "#!/bin/bash\nset -euo pipefail\nsbatch scripts/slurm_exact_oracle_array.sh\n")
    _write_text(root / "scripts" / "collect_and_report.sh", _collect_script(run_id))
    _write_text(root / "reports" / "decision_log.md", _decision_log(profile))

    for p in [
        "ml_phase/acquisition.py",
        "ml_phase/config.py",
        "ml_phase/exact_oracle.py",
        "ml_phase/active_refine.py",
        "hpc_active_loop.sh",
        "scripts/slurm_active_refine.sh",
        "scripts/slurm_exact_oracle_array.sh",
        "scripts/export_active_run_report_md.py",
        "scripts/check_acquisition_profiles.py",
        "scripts/compare_acquisition_profiles.py",
        "scripts/compare_incremental_qexpansion_regression.py",
        "scripts/run_qwindow_incremental_benchmark.py",
        "scripts/build_qwindow_incremental_performance_report.py",
    ]:
        _copy(ROOT / p, root / "code_snapshot" / p)

    for p in ["scripts/submit_active_loop.sh", "scripts/submit_exact_array.sh", "scripts/collect_and_report.sh"]:
        (root / p).chmod(0o755)
    return root


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selfcontained = _build_selfcontained_package()
    p1 = _build_package(
        package_name="robust_oracle_full_acquisition",
        run_id=FULL_RUN_ID,
        profile="full",
    )
    p2 = _build_package(
        package_name="robust_oracle_simple_phase_acquisition",
        run_id=SIMPLE_RUN_ID,
        profile="simple_phase",
    )
    print(str(selfcontained))
    print(str(p1))
    print(str(p2))


if __name__ == "__main__":
    main()
