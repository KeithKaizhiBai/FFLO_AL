# Local-Refinement Stage 1 Import-Path Fix Q&A

Date: 2026-06-03

## Question

Why did the Stage 1 HPC fixed-point regression jobs fail with:

```text
ModuleNotFoundError: No module named 'eta_phase_diagram_cuda'
```

## Short Answer

The package did contain `eta_phase_diagram_cuda.py` at the package root, but
the fixed-point regression job was launched as:

```bash
python scripts/run_local_refinement_fixed_point_regression.py
```

In that execution mode Python puts `scripts/` on `sys.path`, not the package
root.  The runner then tried to import `eta_phase_diagram_cuda` from the root
module namespace and failed.

## Fix

The fixed-point runner now inserts its package/repository root into `sys.path`
before importing root-level modules:

```text
scripts/run_local_refinement_fixed_point_regression.py
```

It also resolves relative input and output paths against the package root, not
the caller's current working directory.  This means the runner can be launched
from a different HPC working directory as long as the script path points into
the extracted package.

The Stage 1 preflight now also performs isolated import checks for:

```text
eta_phase_diagram_cuda
ml_phase.exact_oracle
```

This makes package-root import errors visible before Slurm exact jobs are
submitted.  The import checks run with bytecode writing disabled, so the
preflight does not create `__pycache__` files inside the extracted package.

The compare, gate-verification, collection, and preflight entry points now use
the same package-root convention for their default relative paths.  The
generated Slurm scripts also derive `PROJECT_DIR` from their own location under
`scripts/` instead of assuming `$PWD` is the package root.

After the HPC permission failure on `mkdir logs` and `mkdir reports`, the
package was further split into:

```text
PACKAGE_ROOT = extracted package directory, used for code and fixed_points/
RUN_ROOT     = writable run/output directory, used for logs/, reports/, return archive
```

Set `RUN_ROOT` explicitly on HPC when the extracted package directory is
read-only or otherwise not writable.

## What Did Not Change

This fix does not change the Hamiltonian, phase criterion, q-window policy,
Delta-refinement policy, fixed-point CSV, local-box instrumentation semantics,
or Stage 1 pass criteria.

## Required Action

Upload the rebuilt Stage 1 package and sidecars, rerun:

```bash
sha256sum -c local_refinement_refactor_stage01_instrumentation.tar.gz.sha256
tar -xzf local_refinement_refactor_stage01_instrumentation.tar.gz
cd local_refinement_refactor_stage01_instrumentation
export RUN_ROOT="$HOME/local_refinement_refactor_stage1_run"
mkdir -p "$RUN_ROOT"
python scripts/preflight_local_refinement_stage1_hpc.py --package-root . --run-root "$RUN_ROOT"
bash scripts/submit_stage1_regression_workflow.sh
```

Do not reuse the failed extracted package for the fixed-point jobs.  Stage 2
remains pending until the returned Stage 1 gate status reports `status=pass`.
