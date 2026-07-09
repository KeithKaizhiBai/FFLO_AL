# Local Refinement Refactor Decision Log

Date: 2026-06-03

## D1. Freeze before optimizing local refinement

Decision:

```text
Start with Stage 0 baseline freeze before enabling clustering, pruning, branch
reuse, adaptive boxes, or GPU batching.
```

Reason:

```text
Current reports show that local refinement dominates point-level compute time,
but current metadata is point-level, not local-box-level. Tightening the top-k
cap or pruning boxes before box-level evidence would risk removing mandatory
branches.
```

Consequences:

```text
Stage 0 adds documentation, fixed-point regression scaffolds, and report-path
guardrails only. It does not change the exact-oracle physics path.
```

## D2. Treat report-generation failure separately from numerical completion

Decision:

```text
The active-loop final report step should not turn a completed numerical run
into a failed active-learning loop solely because a LaTeX template is missing.
```

Reason:

```text
The robust-incremental five-iteration mini-run completed all numerical
iterations, but the final local report failed because
report/active_learning_phase_boundary_report.tex was missing in the HPC package
path. That is a report packaging failure, not an exact-oracle failure.
```

Consequences:

```text
The report builder now resolves the default template relative to the package
root and falls back to a built-in minimal template for the default report. The
active loop logs report-generation failure as a warning and records a status
file instead of hiding completed numerical outputs.
```

## D3. Add Basin Clustering as a Disabled-by-Default Stage 2 Layer

Decision:

```text
Implement basin clustering as an explicit feature-flagged layer between coarse
local-minimum detection and local-box refinement.
```

Reason:

```text
Stage 1 GPU regression established equivalence for the unclustered local-box
path.  Stage 2 should reduce duplicate coarse minima only when explicitly
enabled, so baseline behavior remains available for comparison.
```

Consequences:

```text
The new clustering helper groups candidates by coarse q, Delta, and DeltaF
proximity.  Representatives carry cluster_size, merged_branch_ids,
cluster_reason, and mandatory_basin_reasons.  Mandatory-risk reasons are
aggregated across merged members so global-best, edge-risk,
Delta-near-epsilon, and near-degenerate basins are not silently discarded.

No production active-learning run, pruning, branch reuse, adaptive boxes, GPU
batching, Hamiltonian caching, physics criterion, q/Delta tolerance, or
acquisition behavior changes in this stage.
```

## D4. Add Selective Refinement as an Opt-In Stage 3 Policy

Decision:

```text
Implement local-refinement target selection as a helper with two modes:
legacy behavior by default, and explicit selective refinement when
enable_selective_refinement=true.
```

Reason:

```text
The old code selected mandatory-like branches and then applied a hard total
cap.  This can drop mandatory-risk basins if too many risks occur in a point.
Stage 3 needs a policy that preserves mandatory basins while capping ordinary
basins.
```

Consequences:

```text
The default path preserves the previous selection and cap behavior.  The
selective path keeps global-best, edge-risk, Delta-near-epsilon, and
near-degenerate basins first, then adds ordinary basins up to
max_optional_refined_basins.  The setting mandatory_basins_can_exceed_cap
controls whether mandatory basins may exceed the total cap.

No energy-window pruning, branch reuse, adaptive boxes, GPU batching,
Hamiltonian caching, physics criterion, q/Delta tolerance, or acquisition
behavior changes in this stage.
```

## D5. Limit Energy-Window Pruning to Ordinary Basins

Decision:

```text
Implement energy-window pruning as an opt-in Stage 4 policy that marks and
skips only ordinary non-mandatory basins above the pruning window.
```

Reason:

```text
The optimization should reduce local-box work on high-energy ordinary basins
without removing global-best, edge-risk, Delta-near-epsilon, or
near-degenerate basins.
```

Consequences:

```text
Rows pruned by this policy carry pruned_reason=ordinary_above_energy_window.
Mandatory basins remain eligible for refinement regardless of the energy
window.  The policy is disabled unless energy_window_pruning_enabled is
explicitly set.

No branch reuse, adaptive boxes, GPU batching, Hamiltonian caching, physics
criterion, q/Delta tolerance, acquisition behavior, or StopController behavior
changes in this stage.
```

## D6. Prototype Branch-Reuse Decisions Before Loop Integration

Decision:

```text
Add branch reuse signature and decision helpers, but do not connect branch
reuse to the production local-refinement loop in this stage.
```

Reason:

```text
Branch reuse can create silent scientific errors if a newly lower-energy
competing branch appears after q-window expansion.  The reuse decision must
first be explicit, rejectable, and testable.
```

Consequences:

```text
Reuse requires matching solver and local-box signatures, close q/Delta/energy
agreement, and no lower-energy competing branch.  Every rejection has a named
reason.  The exact-oracle loop still computes local boxes normally until a
future integration stage records box-level reuse/rejection diagnostics and
passes GPU fixed-point regression.
```

## D7. Keep Adaptive Local Boxes as Diagnostics First

Decision:

```text
Stage 6 records basin geometry proxies and bounded adaptive half-width
suggestions, but does not use adaptive boxes in the production local scan.
```

Reason:

```text
Changing local box bounds can change refined minima and phase labels.  The
project needs diagnostic evidence and GPU fixed-point regression before
adaptive boxes can replace the fixed-box baseline.
```

Consequences:

```text
Branch candidate diagnostics now include basin_q_width, basin_Delta_width,
basin_energy_span, and basin_curvature_proxy.  The adaptive half-width helper
returns fixed defaults unless explicitly enabled by a future integration path.
```

## D8. Defer GPU Batching and Hamiltonian Cache Implementation

Decision:

```text
Stage 7 records GPU batching, Hamiltonian cache, profiler, validation, and HPC
package-set requirements, but does not implement batching or caching.
```

Reason:

```text
GPU batching and caching can alter memory layout, execution order, and failure
surfaces.  They need explicit variant-level GPU regression gates after the
local Stage 2-6 optimization logic is ready.
```

Consequences:

```text
No production exact calculation changes in Stage 7.  The next concrete step is
to wire variant-runner/package support for explicit Stage 2/3/4 GPU fixed-point
validation and to design integration diagnostics for Stage 5/6 before creating
the combined HPC package set.
```

## D9. Package Only Runnable Local-Refinement Variants

Decision:

```text
Generate the current variant-suite package with baseline, cluster_only,
cluster_optional_k3, cluster_optional_k2, and cluster_energy_window.  Do not
submit cluster_energy_reuse until branch reuse is integrated into the
production exact-oracle loop with explicit reuse/rejection diagnostics.
```

Reason:

```text
The Stage 2/3/4 feature flags are implemented in the production fixed-point
runner and can be validated by GPU regression.  Stage 5 branch reuse is only a
tested decision prototype; running it as a production variant would imply a
level of integration that does not exist yet.
```

Consequences:

```text
The generated package writes logs, reports, comparisons, and the return archive
under RUN_ROOT, defaulting to
$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run when the extracted
package is writable.  This avoids the earlier Permission denied failure from
trying to create logs or reports in a non-writable current/root directory.

The package is structurally generated and locally preflighted, but Stage 2/3/4
GPU fixed-point validation remains pending until it is run on the target
cluster.
```

## D10. Include Runtime Diagnostics in Returned Variant Archives

Decision:

```text
The variant-suite postprocess calls
scripts/collect_local_refinement_performance_report.sh to build a performance
report before collecting the return archive.
```

Reason:

```text
Equivalence gates alone say whether a variant preserves the fixed-point
physics outputs.  Runtime totals, local-refinement fractions, refined-box
counts, pruning counts, and local-box timing diagnostics are also needed to
judge whether the optimization is useful.
```

Consequences:

```text
The returned archive includes
reports/local_refinement_refactor/variant_regression/performance_report/
with runtime_summary.csv, local_box_summary.csv, performance_summary.json, and
performance_report.md.  This is report-only postprocessing and does not change
physical definitions, numerical tolerances, or exact-oracle behavior.

The importer also builds or refreshes this report after extraction when the
variant-suite gate passes.  It reports `gate_status` and `import_status`
separately so a physics-equivalent return bundle cannot silently miss required
performance companion files.

The goal-run audit treats Stage 2-4 GPU validation as passed only when
`gate_status`, `import_status`, and `performance_report_status` all pass.
```

## D11. Define Stage 5/6 Diagnostic Contracts Before Production Integration

Decision:

```text
Stage 5 branch reuse now has explicit cache-record and diagnostic-record
builders.  Stage 6 adaptive boxes now have an explicit suggestion diagnostic
record.  Neither path is wired into the production exact-oracle local scan.
```

Reason:

```text
Branch reuse and adaptive boxes can silently change local minima if integrated
without complete evidence rows.  The project needs stable records of cache
validity, reuse rejection reasons, signature mismatches, competing branches,
default box widths, suggested box widths, basin geometry, and bounding reasons
before any future GPU variant can be called runnable.
```

Consequences:

```text
Future Stage 5/6 production integration must persist these diagnostic records
and pass fixed-point GPU regression before branch reuse or adaptive local boxes
can replace the fixed local-box baseline.
```

## D12. Keep Stage 7 Batching and Cache as Auditable Interfaces First

Decision:

```text
Stage 7 now defines pure local-box batch-plan, Hamiltonian cache-signature,
cache diagnostic, and profiler-event helpers.  These helpers are not wired into
the production exact-oracle local scan and do not enable GPU batching or cache
reuse.
```

Reason:

```text
Batching and caching can alter tensor shapes, execution order, and memory/cache
validity assumptions.  Before any GPU-level optimization is runnable, the
project needs explicit records for batch dimensions, q/Delta grid shape,
tensor construction location, cache signature inputs, and cache rejection
reasons.
```

Consequences:

```text
The new tests validate only the interface contract.  Physics equivalence,
GPU cache correctness, and runtime speedup remain unproven until a future
fixed-point GPU variant is implemented and passes the variant-suite gate.
```

## D13. Use Package-Local Run Directories for Stage 1 Reference Outputs

Decision:

```text
The Stage 1 reference/instrumentation package now defaults runtime output to
$PACKAGE_ROOT/local_refinement_refactor_stage1_run when RUN_ROOT is unset and
the extracted package is writable.
```

Reason:

```text
The package source root should remain separate from generated logs, reports,
preflight JSON, and return-bundle metadata.  This also matches the Stage 2-4
variant-suite rule, where runtime outputs live under a package-local run
directory by default.
```

Consequences:

```text
Explicit RUN_ROOT still wins.  Non-writable package roots still fall back to
SCRATCH, TMPDIR, or HOME.  This changes only output placement and does not
modify exact-oracle physics, numerical tolerances, or local-refinement feature
flags.
```

## D14. Make Clustered Risk Annotation Explicitly Basin-Level

Decision:

```text
For clustered local-refinement candidates, attach explicit basin-level risk
flags and per-risk member counts to the representative, and make mandatory
selection/pruning consume those basin flags when present.
```

Reason:

```text
The returned variant-array audit showed target construction was the blocker.
The code already clustered before selection, but risk interpretation could
still depend on the representative row.  A basin whose representative is
ordinary but whose merged member is edge-risk or Delta-near-epsilon should
remain auditable and mandatory at basin level.
```

Consequences:

```text
Unclustered baseline behavior still falls back to candidate-level risk logic.
Clustered variants now emit basin_risk_flags and per-risk member counts in
branch-candidate and local-box diagnostics.  This stage does not introduce
rank-and-cap, hard total cap enforcement, per-risk caps, energy-window
semantic changes, physics criterion changes, q/Delta tolerance changes,
acquisition changes, or Slurm reruns.
```

## D15. Add Rank-and-Cap as an Opt-In Mandatory Overflow Policy

Decision:

```text
Add `high_risk_overflow_policy = rank_and_cap` and new `rank_and_cap_*`
variants instead of changing historical `cluster_optional_*` variants in
place.
```

Reason:

```text
The variant-array return report showed that `cluster_optional_k3`,
`cluster_optional_k2`, and `cluster_energy_window` timed out because mandatory
targets could exceed the intended cap.  Those failed variants are now evidence
and should remain reproducible.  The corrected behavior needs explicit new
variant names and a hard local selector gate before any expensive rerun.
```

Consequences:

```text
The new policy ranks mandatory basins by risk type, applies per-risk caps,
then enforces `max_total_refined_basins` before adding ordinary optional
targets.  It records selected and dropped target diagnostics in branch
candidate CSVs.  This does not change thermodynamic phase criteria, Delta
tolerances, final ambiguity tolerance, acquisition, or Slurm submission logic.
```

## D16. Keep Energy-Window Pruning Ordinary-Only

Decision:

```text
Energy-window pruning remains restricted to ordinary non-mandatory basins under
the new rank-and-cap variants.
```

Reason:

```text
Global-best, edge-risk, Delta-near-epsilon, and near-degenerate basins are
physics-safety guardrails.  Pruning those branches by the ordinary energy
window would silently weaken target construction.
```

Consequences:

```text
Energy-window pruning may have no selected-target-count effect if mandatory
rank-and-cap already fills the total cap.  Its effectiveness must be reported
as ordinary-pruned count, and runtime reduction must not be assumed before the
32 fixed-point dry-run.
```
