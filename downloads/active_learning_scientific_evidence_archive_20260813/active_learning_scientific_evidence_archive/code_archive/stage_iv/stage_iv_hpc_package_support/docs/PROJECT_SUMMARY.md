# Project Summary

## 1. Project Goal

This self-contained HPC package runs the Stage IV cold-start active-learning
calculation for the three-dimensional parameter space `(kBT/t, J_A/t, mu/t)`.
It is intended to generate a fresh 3D thermodynamic and topology-aware dataset,
not to continue any previous 2D active-learning run.

## 2. Current Architecture

- `ml_phase/` contains the copied production code needed by the exact oracle,
  active-learning selection, topology diagnostics, merge, append, and report
  utilities.
- `scripts/stageiv_3d_select.py` creates the seed and acquisition batches.
- `scripts/slurm_stageiv_exact_array.sh` evaluates exact oracle shards on the
  GBU `NV_H100` partition.
- `scripts/submit_stageiv_3d_full_loop.sh` orchestrates selection, Slurm exact
  shards, merge, and trusted append for the full Stage IV run.
- `configs/stageiv_3d_production.json` is the frozen production configuration.

## 3. Important Files

- `RUN_MANIFEST.json`: package provenance, validation results, and hashes.
- `HPC_COMMANDS_STAGEIV_3D.md`: upload and Slurm command checklist.
- `ENVIRONMENT_STAGEIV_3D_HPC.md`: cluster runtime and encoding notes.
- `configs/stageiv_3d_production.json`: expected `run_id`, `output_root`, and
  Stage IV acquisition settings.

## 4. Major Decisions

- Date: 2026-06-24
- Decision: Freeze the package run identity to `active_phase_topology_3d_t_ja_mu_from_scratch_v1` under
  `ML_Phase_StageIV_Topology3D`.
- Reason: A previous HPC shell environment allowed stale 2D `OUTPUT_ROOT` or
  `RUN_ID` variables to redirect a Stage IV submission into an older run.
- Consequences: Generated shell scripts now reject conflicting `OUTPUT_ROOT`,
  `RUN_ID`, or `CONFIG_JSON` values instead of silently using them.

## 5. Completed Milestones

- Date: 2026-06-24
- Milestone: Built a self-contained Stage IV 3D HPC package with identity
  guards, LF-normalized shell scripts, `gpuh01` exclusion, preflight smoke
  checks, status checks, return checks, failed-rank recovery, and post-run
  report-only audit scripts.
- Evidence: `RUN_MANIFEST.json` and package validation fields.

## 6. Recent Major Changes

- Date: 2026-06-24
- Files changed: generated package shell scripts and package-local docs.
- Summary: The package refuses stale 2D run identity variables and avoids
  copying local historical project-memory text into the upload bundle.
- Why it matters: The uploaded package can be run from a contaminated login
  shell without accidentally writing to a prior 2D output namespace.

## 7. Validation and Tests

- Python package compile check: `python -m py_compile`.
- Package preflight smoke check: `scripts/stageiv_3d_preflight.py` with a small
  grid.
- Submit-ready smoke check: `scripts/stageiv_3d_submit_check.py`.
- Status and return-bundle smoke checks are report-only and do not launch exact
  calculations.
- Shell scripts are LF-normalized and ASCII-safe.

## 8. Open Problems

- Scientific convergence cannot be assessed until the external Stage IV HPC
  outputs return.
- Hidden fixed-mu validation requires an explicit `REFERENCE_DATASET` path at
  post-run report time.

## 9. Next Steps

- Upload this package to the GBU cluster.
- Extract into a fresh directory.
- Do not export stale `OUTPUT_ROOT`, `RUN_ID`, or `CONFIG_JSON`; the scripts
  will reject conflicting values.
- Run `bash scripts/check_stageiv_3d_submit_ready.sh`.
- Submit with `CONFIRM_STAGEIV_FULL_LOOP=1`.
