# Local-Refinement Stage 1 Downloaded Package Check

Date: 2026-06-03

## Question

The Slurm queue is empty, but the returned result archive was not found in the
downloaded `local_refinement_refactor_stage01_instrumentation` directory.  Did
the Stage 1 regression finish?

## Finding

The downloaded directory is the package root, not the writable `RUN_ROOT`.
Its top level contains Slurm output files:

```text
slurm-70684.out
slurm-70685.out
slurm-70686.out
```

The exact jobs finished and the postprocess job reported Stage 1 gate pass:

```text
n_common_points = 32
n_missing_in_candidate = 0
n_extra_in_candidate = 0
flag_mismatch_count = 0
max_deltaf_abs_diff = 0.0
max_q_opt_abs_diff = 0.0
max_delta_opt_abs_diff = 0.0
local_box_rows = 192
gate_status = pass
```

The return archive was written on HPC under `RUN_ROOT`:

```text
/public_hw/home/sci_bfu/local_refinement_refactor_stage1_run/local_refinement_refactor_stage1_regression_results.tar.gz
```

The downloaded package root does not contain that archive or the pointwise CSV
outputs because those were written to `RUN_ROOT`, not `PACKAGE_ROOT`.

## Follow-up Local Import

The returned RUN_ROOT archive was later downloaded into the local downloaded
package directory:

```text
local_refinement_refactor_stage01_instrumentation/local_refinement_refactor_stage1_run/local_refinement_refactor_stage1_regression_results.tar.gz
```

It was imported locally under the downloaded package tree, not under the
repository root:

```text
local_refinement_refactor_stage01_instrumentation/imported_results/stage1_regression_results
```

The local gate report confirms:

```text
status = pass
expected_points = 32
baseline_rows = 32
candidate_rows = 32
comparison_rows = 32
local_box_rows = 192
missing_files = []
failures = []
flag_mismatch_count = 0
max_deltaf_abs_diff = 0.0
max_q_opt_abs_diff = 0.0
max_delta_opt_abs_diff = 0.0
```

The direct import script was also hardened so that, when `--import-root` is not
given, it writes `imported_results/` next to the returned archive instead of
defaulting to the repository-root `reports/` tree.  This keeps future imported
outputs inside the downloaded or extracted package area unless an explicit
destination is supplied.

Stage 1 is now locally confirmed as passed.  Stage 2 may be planned next, but
the root-level goal-run report was not rewritten during this check because the
current output-location rule is to keep generated/imported outputs inside the
downloaded package tree.
