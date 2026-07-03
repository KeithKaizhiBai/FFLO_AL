# Decisions

Last updated: 2026-06-04

## D1. Use Existing Exact Data as Warm Start

Decision:

```text
Use the existing 21528-point exact BdG dataset as the initial supervised
training set.
```

Reason:

```text
Starting active learning from no data would spend expensive GPU time
rediscovering the coarse phase structure. The existing grid already gives a
usable first surrogate and uncertainty map.
```

## D2. Treat ML as Scheduler, Not Solver

Decision:

```text
The ML model proposes exact BdG points. It does not replace exact labels.
```

Reason:

```text
This preserves physics correctness and makes active learning acceptable for
phase-boundary refinement.
```

## D3. Separate CPU-Side Acquisition from H100 Exact Evaluation

Decision:

```text
Train models and select points in a CPU-side job. Run only exact BdG point
evaluation on H100 array jobs.
```

Reason:

```text
Avoid wasting allocated H100 time on CPU-heavy model training, plotting, or
report generation.
```

## D4. Use GBU Partition NV_H100

Decision:

```text
Use partition NV_H100 for H100 GPU jobs.
```

Reason:

```text
The cluster module/sinfo output showed NV_H100 as the available H100 partition.
Earlier Intel partition usage failed on the remote workflow.
```

## D5. Do Not Rely on Conda Activation in SLURM

Decision:

```text
Prefer PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python in
batch scripts.
```

Reason:

```text
The cluster reported "CondaError: Run 'conda init' before 'conda activate'".
Calling the environment Python directly is more robust.
```

## D6. q-Window Must Be Adaptive

Decision:

```text
Future exact oracle outputs must detect q_edge_hit and support q-window
expansion or rerun lists.
```

Reason:

```text
For large J_A, q_opt can grow rapidly. A fixed q range can truncate the true
minimum and create false phase labels.
```

## D7. Delta Must Be Refined Near the Normal/SC Boundary

Decision:

```text
Future exact oracle outputs must refine Delta near Delta_opt ~= DELTA_EPS or
when the free-energy minimum is shallow.
```

Reason:

```text
The normal/SC boundary is sensitive to coarse Delta sampling. Boundary labels
must be based on refined Delta values.
```

## D8. Normal-State q Is Not a Physical q-Edge Failure

Decision:

```text
When Delta_opt < DELTA_EPS and the Delta status is stable, the active-learning
exact oracle sets q_status=not_applicable and q_edge_hit=false.
```

Reason:

```text
In the normal state the free energy is independent of the FFLO order parameter
in the same sense as the superconducting state, so a reported q_opt at q_min is
usually a grid tie artifact rather than a physical q-window truncation.
```

## D9. Merge Only Trusted Exact Points Into the Training Pool

Decision:

```text
HPC shard merge writes both exact_merged_iterXXX.npz and
exact_trusted_iterXXX.npz. Future append logic should use the trusted file by
default.
```

Reason:

```text
Unresolved q-edge and Delta-boundary points should drive reruns or further
refinement, not silently train the surrogate as final exact labels.
```

## D10. Delta=0 Normal Points Require a Positive-Delta Gap Check

Decision:

```text
When Delta_opt = 0, the oracle does not use omega_global = 0 by itself as a
Delta-boundary ambiguity signal. It computes positive_delta_gap from a
strictly positive-Delta scan and marks the point ambiguous only if this gap is
near zero.
```

Reason:

```text
For a true normal-state optimum, the free-energy gain relative to the
Delta=0 baseline is exactly zero by construction. Treating this identity as a
boundary signal misclassifies stable normal points as unresolved.
```

## D11. Exclude gpuh01 from Automatic H100 Active-Learning Jobs

Decision:

```text
Set EXCLUDE_NODES=gpuh01 by default in the automatic H100 submission helpers.
```

Reason:

```text
The completed cluster test showed gpuh01 failing with a PyTorch/CUDA driver
mismatch while other H100 nodes such as gpuh04 and gpuh11 completed the exact
oracle shards.
```

Consequence:

```text
The automatic loop submits both candidate-generation jobs and H100 exact-oracle
arrays with sbatch --exclude=gpuh01 unless the user explicitly overrides
EXCLUDE_NODES. This improves reliability but may reduce available scheduling
capacity by one node.
```

## D12. Treat Sub-Tolerance Delta Boundary Points as Boundary-Band Normal

Decision:

```text
If Delta_opt = 0 and the best strictly positive-Delta state is not lower than
the normal state by more than the adopted energy tolerance, interpret the point
as normal-side or normal/SC boundary-band data rather than as a failed exact
calculation.
```

Reason:

```text
The 25-iteration 128-point active-learning run found 147 Delta-unresolved
points, all with delta_boundary_unresolved;max_delta_refinement_reached. Most
had Delta_opt = 0 and positive_delta_gap in the 1e-9 to 1e-8 range. This is
below or comparable to the chosen physical/numerical resolution for the present
phase diagram, and the positive-Delta state does not show a resolvable
condensation-energy gain.
```

Consequence:

```text
These points should be carried with explicit boundary-band metadata and used as
normal-side constraints for phase-boundary construction. Future active-learning
iterations should avoid repeatedly spending the same budget on these
sub-tolerance points unless the study goal changes to resolving a much finer
normal/SC critical line.
```

## D26. Separate Phase Label, Confidence, and Training Eligibility

Decision:

```text
The robust production oracle must store thermodynamic phase label,
confidence_state, training_eligible_exact, and rerun_required as separate
metadata. Stable normal points with Delta_opt=0 and
positive_delta_gap > positive_delta_gap_tol are trusted normal labels.
normal_q_not_applicable is not a q-window failure.
```

Reason:

```text
The first robust-oracle acquisition comparison rejected many stable normal
points because the adaptive-box path used free_energy_ambiguity_tol=1e-6 as a
positive-Delta ambiguity threshold and then made boundary_ambiguous imply
trusted_exact=false. This removed normal labels from the dataset and broke the
active-learning closed loop.
```

Consequence:

```text
Delta_opt=0 points are classified using positive_delta_gap_tol for training
eligibility. Boundary-band normal points remain explicitly marked as lower
confidence but can still enter the dataset as normal-side constraints. Coverage
unresolved, solver failed, and response reliability remain separate from the
basic thermodynamic phase label.
```

## D13. Hard-Exclude Existing Exact Coordinates During Acquisition

Decision:

```text
During candidate selection, hard-exclude dense-grid candidates whose rounded
(kT, JA) coordinate already appears in the current exact training dataset.
Use 4 decimal places by default.
```

Reason:

```text
The 10-iteration continuation after the boundary-band semantic fix showed that
most exact records became training-eligible, but many selected candidates were

## D25. Use Robust Adaptive-Box Oracle for New Discovery Reruns

Decision:

```text
Keep the thermodynamic phase criterion unchanged, but switch production
discovery reruns from the legacy exact oracle to oracle_mode=robust_al.
```

Reason:

```text
Audits showed that high-JA/low-T phase-side errors were mainly caused by
insufficient q-window coverage and coarse local minima resolution, not by
acquisition logic. The robust adaptive-box oracle adds deterministic q-window
expansion plus local q-Delta refinement while preserving per-point
independence and reproducibility.
```

Consequence:

```text
New discovery runs should use robust_al by default and persist detailed
q-window/refinement metadata. Legacy mode remains available for backward
compatibility and controlled A/B comparisons.
```

## D26. Add Configurable Acquisition Profiles for Controlled A/B Runs

Decision:

```text
Introduce acquisition_profile in active-learning config:
  full
  simple_phase
```

Reason:

```text
We need a fair discovery-mode comparison where the exact-oracle side is fixed
to robust_al while acquisition complexity is varied. This requires profile
switching by configuration, not by maintaining two divergent code copies.
```

Consequence:

```text
Profile `full` preserves the existing production formula.
Profile `simple_phase` keeps phase-boundary terms as the main score and
removes numerical-risk/response terms from A0_main while preserving diagnostics.

Two standalone HPC package snapshots are generated for reproducible runs:
  robust_oracle_full_acquisition
  robust_oracle_simple_phase_acquisition
```
already present in the dataset. Repeating exact calls at existing coordinates
does not improve phase-boundary resolution.
```

Consequence:

```text
Existing exact-data exclusion is not relaxed. Recent selected points and
previous boundary-band points use the same 4-decimal cooldown, but those two
cooldowns may be relaxed if fewer than points_per_iter finite candidates remain.
This keeps the active-learning loop productive while avoiding an overly small
rounding threshold such as 8 decimals.
```

## D14. Use the Diversity Radius as the Existing-Data Exclusion Radius

Decision:

```text
In addition to 4-decimal rounded-coordinate exclusion, hard-exclude any
candidate whose normalized distance to the existing exact dataset is smaller
than existing_min_dist = 0.015. This matches the selected-selected diversity
radius.
```

Reason:

```text
The 3-iteration validation after D13 reduced repeated existing coordinates
from 82-97 per 128-point iteration to 1-4, but a few high-JA existing points
still leaked through the rounded-key rule. The selected-selected diversity rule
was already enforcing the same 0.015 normalized radius within each new batch,
so applying that radius to the existing dataset makes the "one point per
radius" rule consistent.
```

Consequence:

```text
The existing-data exclusion is now stronger and will remove much of the
already dense warm-start region from the candidate pool. A check against
dataset_iter038 still leaves more than 1000 finite candidates, so a short
128-point continuation remains feasible. The next validation must confirm that
selected points still follow physically meaningful phase-boundary regions.

This 0.015 radius was later superseded for ordinary dense-grid acquisition by
D17, which halves the radius to 0.0075 for the next 512x50 production upload.
```

## D15. Do Not Blindly Apply Existing-Min-Distance to Boundary-Bracket Midpoints

Decision:

```text
Dense-grid existing-distance rules are valid for ordinary active-learning
candidate selection, but they should not be blindly used to reject explicit
boundary-bracket midpoint refinement targets.
```

Reason:

```text
The boundary extraction recheck on 2026-05-12 found that all prioritized
midpoint targets from the extracted boundary brackets were within the then-used
0.015 existing_min_dist radius of an already computed exact point. This is
expected: a bracket midpoint is geometrically close to the two exact points
that define the bracket. Applying a dense-grid acquisition radius to these
midpoint targets can reject valid normal_sc, uniform_fflo, strong_diode, and
eta_zero candidates.
```

Consequence:

```text
Future boundary refinement must use a boundary-specific target policy. The
policy should quota-limit eta_zero targets, prioritize normal_sc and
uniform_fflo brackets, and either relax the existing-distance rule for bracket
midpoints or replace it with a smaller boundary-specific exclusion radius.
The archived recheck writes both strict radius-checked outputs and
basic-filtered candidate outputs so this decision remains auditable.
```

## D16. Stop Active-Learning Loops When Unique Data Growth Stalls

Decision:

```text
The active-learning loop now stops early when an iteration selects no candidate
points, appends zero new unique training samples, or appends fewer than
MIN_NEW_POINTS_PER_ITER unique samples for MAX_LOW_APPEND_ITERS consecutive
iterations.
```

Reason:

```text
The boundary-extraction recheck showed that dataset_iter039 through
dataset_iter042 can be content-identical when later iterations are
empty-append or ineffective postprocessing aliases. Continuing the loop in
that state wastes exact-oracle budget and creates misleading dataset indices.
```

Consequence:

```text
hpc_active_loop.sh reads the append summary written by ml_phase.append_trusted
and stops by default at zero unique additions or after two consecutive
iterations with fewer than eight new unique samples. The thresholds are
configurable through ENABLE_EARLY_STOP, MIN_NEW_POINTS_PER_ITER, and
MAX_LOW_APPEND_ITERS. The local active_refine loop uses matching configuration
fields and writes local_append_summary.json for traceability.
```

## D17. Halve Dense-Grid Acquisition Radius for the 512x50 Production Upload

Decision:

```text
For the next warm-start production active-learning upload, use
diversity_min_dist = existing_min_dist = 0.0075 instead of 0.015 for ordinary
dense-grid acquisition.
```

Reason:

```text
The 0.015 normalized radius was useful for eliminating repeated or near-repeat
exact calls, but it is coarse relative to the boundary detail now being
targeted. Halving the radius keeps an explicit diversity and existing-data
exclusion rule while allowing denser sampling near normal/SC and
uniform_SC/FFLO boundaries.
```

Consequence:

```text
The production upload defaults to 512 selected points per loop for 50 loops
starting from the warm-start dataset. More candidates can survive the
existing-data exclusion than with the previous 0.015 radius, so the loop should
be less likely to stall from over-aggressive exclusion. D15 still applies:
explicit boundary-bracket midpoint refinement should use a boundary-specific
policy rather than blindly reusing the dense-grid radius.
```

## D18. Add Boundary-Local Hybrid Refinement Mode (Superseded by D19)

Decision:

```text
Add an optional boundary_refinement_mode with values off, hybrid, and local.
The default Python active_refine behavior remains off for backward
compatibility. The current HPC loop wrappers default to hybrid for the next
boundary-refinement production runs.
```

Reason:

```text
Fixed-radius global acquisition can saturate after it has covered the broad
candidate pool, even though exact phase boundaries still need local refinement.
Boundary-bracket midpoints are valid refinement targets precisely because they
lie close to existing exact bracket endpoints, so they require a separate
boundary-local radius instead of the ordinary dense-grid existing-distance
radius.
```

Consequence:

```text
Hybrid mode extracts normal_sc and uniform_fflo boundary brackets from the
current accepted exact dataset, selects midpoint targets with a smaller
boundary_local_min_dist=0.00375, and fills the remaining point budget with
high-JA q-risk and ordinary global acquisition points. Eta-zero and
strong-diode brackets are still extracted for diagnostics, but their default
selection quotas are zero because they are response boundaries rather than the
primary thermodynamic phase boundaries.

This decision is superseded by D19 for the next active-learning stage.
```

## D19. Disable Midpoint-Based Selection for the Next Active-Learning Stage

Decision:

```text
The production active-learning selector no longer uses geometric boundary
midpoints as selected exact-call candidates. Boundary extraction remains
available in diagnostic mode, but selected_points.csv is produced only by
ML-guided dense-grid acquisition.
```

Reason:

```text
Midpoint selection is a deterministic bisection rule on already known exact
brackets. It can narrow a bracket, but it does not let the trained surrogate
decide where a new exact calculation would most reduce uncertainty, boundary
ambiguity, or numerical risk. A large fixed midpoint quota therefore conflicts
with the active-learning objective.
```

Consequence:

```text
boundary_refinement_mode=diagnostic extracts and reports exact boundary
segments without generating midpoint targets. The legacy hybrid/local modes
now raise an error. Candidate selection uses the acquisition score on the
dense grid, a hard exact-coordinate duplicate mask, soft observation repulsion
near existing exact data, and soft batch repulsion among newly selected points.
```

## D20. Use StopController Metrics Instead of Candidate Exhaustion

Decision:

```text
Add an independent StopController for the acquisition-only active-learning
loop. The controller evaluates convergence after each exact-oracle merge and
trusted append, using phase-map stability, main thermodynamic boundary shifts,
label surprise, selected acquisition score saturation, q-edge/rerun rates, and
main-boundary exact-data coverage.
```

Reason:

```text
With soft observation and batch repulsion, the selector can usually continue
assigning nonzero scores to uncomputed dense-grid candidates. Therefore the
existence of more candidates is not a useful convergence criterion. The stop
decision must instead ask whether new exact BdG calls are still changing the
main phase-boundary information or exposing numerical-risk failures.
```

Consequence:

```text
No-available-candidates remains an exceptional stop reason with
stop_reason=no_available_candidates. It is not the main convergence logic.
Eta-zero response boundaries and topology are excluded from the current main
stop rule.

The original 2026-05-14 implementation used q-edge/rerun and boundary coverage
as mandatory gates in a seven-condition rule. This was superseded by D21:
selected A0 ratio and q-edge/rerun rates are now diagnostics/cleanup warnings,
while the main stop rule uses five phase-map and thermodynamic-boundary
conditions.
```

## D21. Add Discovery-Mode Active Learning

Decision:

```text
Add run_mode=discovery as a first-class active-learning workflow distinct from
run_mode=refinement.
```

Reason:

```text
Warm-start refinement measures how efficiently ML improves a known exact phase
diagram. It does not test whether active learning can discover the phase
structure from sparse exact information. Discovery mode is needed for that
benchmark.
```

Consequences:

```text
Discovery mode starts from random exact seed points on the full rectangular
candidate grid, does not load the large warm-start dataset as the initial
training set, and forbids the finite-T prior candidate-band mask.

Refinement mode remains available for warm-start boundary refinement and may
explicitly use candidate_domain_mode=prior_band with finite_t_band_width.

Discovery-mode acquisition uses stochastic score-weighted sampling with dynamic
batch repulsion instead of fixed regional quotas. The main stop controller now
uses the five main phase-map and thermodynamic-boundary conditions; selected
A0 ratio, q-edge rate, and rerun-required rate are diagnostics and cleanup
warnings, not mandatory gates for stopping the main phase-boundary loop.
```

## D22. Gate Discovery Sampling by A0 Main Before Stochastic Draws

Decision:

```text
Discovery-mode active-learning selection now separates physical/model
information value from stochastic sampling weights.  A0_main defines a
high-information active pool on the full candidate grid.  The actual stochastic
draws are then made inside that pool with weights proportional to
(A0_main * R_obs * R_batch)^gamma.
```

Reason:

```text
The first full discovery run converged in the StopController sense, but late
selected points still looked close to random full-domain coverage.  The old
sampler applied stochastic sampling over the broad corrected score field and
used a strong historical R_obs, so selection behaved partly like a coverage
sampler rather than a phase-boundary information sampler.
```

Consequences:

```text
The high-information pool is selected by A0_main quantile and relative-to-p95
thresholds.  R_obs is now a mild historical downweight with default floor 0.5
and length 0.02; it no longer decides whether a point is worth considering.
R_batch remains stronger and only discourages within-batch clustering.

Discovery batches are adaptive: before the minimum-iteration stage the active
pool threshold can relax to maintain a useful exploratory batch, while later
iterations no longer force the batch to fill batch_size_max when only fewer
high-information candidates remain.

Selection diagnostics now track active-pool size, threshold relaxation,
effective sample size, selected A0 concentration, R_obs statistics, and
selected-to-predicted-boundary distances against a random active-pool baseline.
```

## D23. Sharpen Discovery Active Pool with Gated Delta-Boundary Scores

Decision:

```text
Discovery-mode acquisition now uses a normal/SC competition gate for the
Delta-boundary score, a stricter max-threshold active-pool rule, a soft
high-confidence interior penalty, piecewise sampling-power annealing, and
piecewise exploration-weight annealing.
```

Reason:

```text
The 2026-05-20 selection-region diagnostics showed that the late discovery
batch was dominated by predicted normal-interior points.  Component attribution
identified the broad Delta-boundary score as the main source: deep predicted
normal points can have Delta_pred close to zero, making
exp(-abs(Delta_pred - Delta_eps) / delta_scale) large even when the classifier
is highly confident the point is not near the normal/SC boundary.
```

Consequences:

```text
B_delta_raw is still saved for diagnostics, but A_phase uses
B_delta_gated = B_delta_raw * U_NS with U_NS = 4 * P_normal * P_SC.

The high-information pool is built from A0_for_pool rather than raw A0_main.
A0_for_pool includes the gated acquisition components and a soft penalty for
high-confidence, low-information phase interiors.  The active-pool threshold is
max(quantile threshold, relative-p95 threshold) rather than the previous loose
OR rule, and a scheduled active-pool fraction cap can tighten the threshold.

R_obs remains mild and does not decide active-pool membership.  Stochastic
selection still operates inside the score-defined pool and does not introduce
JA/kT region quotas or midpoint targets.

Future discovery reports should inspect B_delta_raw, U_NS, B_delta_gated,
A0_for_pool, active_pool_fraction, selected boundary-band fraction, and
N_eff/active_pool_size to verify that late batches become more boundary
focused.
```

## D24. Separate Phase Robustness Audits from Eta Response Audits

Decision:

```text
Add an audit-only production-oriented phase robustness workflow for high-JA,
low-T boundary-sensitive points.  The workflow expands the free-energy q window,
saves F_min(q), records low-energy local q minima, and applies stricter
near-zero Delta refinement.
```

Reason:

```text
The numerical reliability audit showed that high-JA positive eta anomalies are
response-extraction pathologies unless eta_response_valid=true.  The remaining
important uncertainty is phase-side robustness: whether the scanned q window
and near-zero Delta resolution are sufficient for high-JA normal/SC boundary
points.
```

Consequences:

```text
The basic phase criterion is unchanged.  Finding one positive-Delta state with
F_SC < F_N is sufficient for superconducting classification within the scanned
window.  Expanded q-window scans are used to test q_opt stability, branch
identity, boundary robustness, and future topology readiness; they are not a
new requirement for proving superconductivity.

Audit outputs must remain separate from active-learning datasets unless the
user explicitly asks to append them.  Reports must distinguish basic SC
classification, branch-resolved completeness, topology readiness, and eta
response validity.
```

## D27. Keep Local-Refinement Variant Outputs Under Package RUN_ROOT

Decision:

```text
The local-refinement variant-suite HPC package writes runtime logs, reports,
comparisons, and the return archive under RUN_ROOT.  If RUN_ROOT is unset and
the extracted package directory is writable, RUN_ROOT defaults to
$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run.
```

Reason:

```text
The cluster-side current directory may be non-writable, which caused earlier
mkdir logs and mkdir reports failures.  Runtime outputs should therefore stay
inside the extracted upload package or an explicitly chosen writable run root.
```

Consequences:

```text
The generated package no longer relies on writing outputs to the repository
root or login directory.  The current runnable fixed-point variants are
baseline, cluster_only, cluster_optional_k3, cluster_optional_k2, and
cluster_energy_window.  Branch reuse remains a prototype until production-loop
integration and reuse/rejection diagnostics are implemented.
```

## D28. Define GPU Batching and Cache Interfaces Before Production Use

Decision:

```text
The local-refinement refactor defines pure Stage 7 helper records for
local-box batch plans, Hamiltonian cache signatures, cache hit/miss
diagnostics, and profiler events before implementing GPU batching or
Hamiltonian cache reuse.
```

Reason:

```text
GPU batching and Hamiltonian caching can change tensor shapes, execution
order, memory pressure, and cache validity assumptions.  A future optimization
must first expose auditable batch dimensions, q/Delta grid shapes, tensor
construction locations, cache signature inputs, and explicit rejection reasons.
```

Consequences:

```text
The Stage 7 helpers are interface contracts only.  They do not change the
thermodynamic phase criterion, q/Delta refinement policy, local-refinement
selection, exact-oracle production path, or active-learning workflow.  GPU
batching and Hamiltonian cache remain disabled until a future fixed-point GPU
variant passes regression.
```

## D29. Keep Stage 1 Reference Outputs in a Package-Local Run Directory

Decision:

```text
The Stage 1 local-refinement reference/instrumentation package defaults
runtime outputs to $PACKAGE_ROOT/local_refinement_refactor_stage1_run when
RUN_ROOT is unset and the extracted package root is writable.
```

Reason:

```text
Earlier Stage 1 scripts could write logs, reports, preflight JSON, and return
metadata directly into the extracted package root.  A package-local run
subdirectory keeps source/package files separate from generated outputs and
matches the Stage 2-4 variant-suite output policy.
```

Consequences:

```text
Users can still set RUN_ROOT explicitly.  If the package root is not writable,
the scripts still fall back to SCRATCH, TMPDIR, or HOME.  This is only an HPC
output-location policy change and does not change physical definitions,
numerical safeguards, or exact-oracle behavior.
```

## D30. Provide Runbook-Compatible HPC Submit Aliases

Decision:

```text
The generated Stage 1 and variant-suite HPC packages include runbook-named
submit aliases:

scripts/submit_local_refinement_fixed_point_regression.sh
scripts/submit_local_refinement_instrumented_benchmark.sh
```

Reason:

```text
The long-form refactor runbook names these generic entry points, while the
package-specific implementation scripts use stage-specific workflow names.
Providing thin aliases makes the handoff easier to run and audit without
duplicating workflow logic or changing the validated Slurm chain.
```

Consequences:

```text
The aliases only exec the existing package workflow scripts.  They do not
introduce one-iteration or intermediate active-learning validation wrappers,
do not enable branch reuse, adaptive local boxes, GPU batching, or Hamiltonian
cache production paths, and do not change any physical definition or numerical
gate.  Runtime outputs still go under package-local RUN_ROOT directories by
default.
```

## D31. Inspect Nested HPC Packages in the Upload-Set Verifier

Decision:

```text
The generated upload-set verifier now opens each nested package archive and
checks its RUN_MANIFEST, README, runbook-compatible submit aliases, workflow
targets, and package-local RUN_ROOT suffix in addition to archive hashes,
sidecars, and metadata.
```

Reason:

```text
Earlier HPC handoff failures came from import-path assumptions, non-writable
logs/reports directories, and confusion between package roots and runtime
output roots.  A checksum-only upload-set verifier can prove archive integrity
but cannot prove that the uploaded package still exposes the expected runbook
entry points or package-local runtime-output policy.
```

Consequences:

```text
The handoff check now fails before upload/extraction if a nested package lacks
scripts/submit_local_refinement_fixed_point_regression.sh, maps an alias to
the wrong workflow script, omits RUN_ROOT from its manifest, or no longer
documents the expected package-local run directory.  This remains a packaging
and audit check only; it does not change exact-oracle physics, feature flags,
q/Delta safeguards, production branch reuse, adaptive boxes, GPU batching, or
Hamiltonian cache behavior.
```

## D32. Enforce RUN_ROOT for Nested Package Shell Outputs

Decision:

```text
The variant-suite collector writes return-bundle metadata under
$RUN_ROOT/reports/local_refinement_refactor/variant_regression/return_bundle_metadata,
and the upload-set verifier now scans nested package shell scripts to ensure
any logs/reports output path is expressed through RUN_ROOT.
```

Reason:

```text
The earlier HPC failures included permission errors from attempts to create
logs or reports in a non-writable current directory.  Checking only the
RUN_MANIFEST contract is not enough if a lower-level collector or Slurm helper
still contains a bare reports/ or logs/ write.
```

Consequences:

```text
The handoff verifier now fails before upload if any nested package shell
script writes logs or reports without RUN_ROOT.  This is a runtime-output
policy and packaging-gate change only.  It does not change exact-oracle
physics, numerical thresholds, feature flags, branch reuse, adaptive boxes,
GPU batching, or Hamiltonian cache behavior.
```

## D33. Enforce GPU Slurm Node/Runtime Safety in the Upload Set

Decision:

```text
The upload-set verifier now scans GPU Slurm scripts inside each nested package
archive and requires every GPU script to exclude gpuh01 and run a real CUDA
tensor-allocation probe that prints cuda_runtime_probe=pass.
```

Reason:

```text
gpuh01 has an older NVIDIA driver than the PyTorch CUDA runtime used by the
environment, which previously caused CUDA initialization to fail at runtime.
Checking only Slurm submission structure is not enough; the handoff verifier
must prove that GPU jobs exclude the known bad node and fail early if CUDA
cannot initialize on the allocated node.
```

Consequences:

```text
The current upload set verifies 7 GPU Slurm scripts across the Stage 1
reference package and the Stage 2-4 variant-suite package, with zero policy
violations.  This is an HPC scheduling/runtime safety gate only.  It does not
change exact-oracle physics, numerical thresholds, feature flags, branch
reuse, adaptive boxes, GPU batching, or Hamiltonian cache behavior.
```

## D34. Include Variant-Suite HPC Status Diagnostics in the Package

Decision:

```text
The variant-suite HPC package includes scripts/check_variant_suite_hpc_status.py
and documents it in README.md and RUN_MANIFEST.json.
```

Reason:

```text
After Slurm jobs disappear from squeue, users still need a package-local way to
distinguish a successful postprocess return archive from missing postprocess
output, CUDA runtime failures, old-driver failures, and failed Slurm states.
Relying only on squeue is insufficient because completed or failed jobs may no
longer appear there.
```

Consequences:

```text
The status checker reads RUN_ROOT logs, jobid files, and the expected return
archive.  It can optionally query squeue/sacct when available.  This is an HPC
diagnostic and handoff change only; it does not alter Slurm dependencies,
exact-oracle calculations, feature flags, numerical thresholds, or the formal
variant-suite import gate.
```

## D35. Add a Top-Level Upload-Set Run Helper

Decision:

```text
The upload-set bundle includes run_required_variant_suite.sh at its top level.
The helper verifies the upload set, extracts the required variant-suite archive
from archives/ if needed, and then calls the existing package-local
scripts/submit_local_refinement_fixed_point_regression.sh alias.
```

Reason:

```text
The previous handoff required several manual cluster-side steps: verify the
upload set, locate the nested required archive, extract it, cd into the package,
and submit the runbook-named workflow.  Automating only those existing steps
reduces directory mistakes and stale-package submissions without changing the
validated nested package workflow.
```

Consequences:

```text
The upload-set verifier now checks that the helper is present, documented,
performs upload-set verification first, extracts the required archive from
archives/, calls the existing submit alias, and contains no destructive delete
command.  This is a cluster handoff convenience and audit check only; it does
not change exact-oracle physics, numerical thresholds, Slurm job logic inside
the nested package, feature flags, branch reuse, adaptive boxes, GPU batching,
or Hamiltonian cache behavior.
```

## D36. Run the Stage 2-4 Variant Gate as a Point-Wise Slurm Array

Decision:

```text
The Stage 2-4 local-refinement variant-suite package now submits one Slurm
array over variant x fixed_point_id instead of one long serial job per
variant.  The package recomputes baseline and all candidate variants from
scratch, writes one result per point task, aggregates them into the existing
variant-level CSV/manifest layout, and submits postprocess with afterany so a
diagnostic return archive is produced even when some point tasks fail.
```

Reason:

```text
The previous serial variant-suite design let baseline and cluster_only finish
in about two hours, but cluster_optional_k3, cluster_optional_k2, and
cluster_energy_window each timed out after 36 hours without useful pointwise
checkpoint output.  This made the postprocess job DependencyNeverSatisfied and
left no return archive.  The failure was a scheduling/checkpointing design
problem, not a reason to weaken the physics-equivalence gate.
```

Consequences:

```text
The complete gate still compares exact-oracle phase labels, q_opt, Delta_opt,
DeltaF, trusted/training/rerun flags, q/delta unresolved flags, and timing
metadata against a freshly computed baseline.  The change is limited to HPC
task granularity, restartability, diagnostics, and return packaging.  It does
not change physical definitions, numerical thresholds, local-refinement
feature flags, or the accepted equivalence tolerances.
```

## D37. Make Mandatory Overflow Rank-and-Cap an Explicit Optimized Variant Policy

Decision:

```text
Preserve historical `cluster_optional_*` variants and add explicit
`rank_and_cap_*` variants with `high_risk_overflow_policy = rank_and_cap`.
Energy-window pruning remains restricted to ordinary non-mandatory basins.
```

Reason:

```text
The returned variant-array evidence showed that the historical optimized
variants over-selected mandatory basins and timed out.  Those configurations
should remain reproducible as failed evidence.  The corrected policy should
therefore be opt-in, named, and locally gated before any expensive GPU rerun.
Energy-window pruning should not remove selected global-best, edge-risk,
Delta-near-epsilon, or near-degenerate basins because those branches are
physics-safety guardrails.
```

Consequences:

```text
The new policy ranks mandatory basins by risk type, applies per-risk caps, and
enforces the total target cap before ordinary optional targets are added.
Ordinary energy-window pruning can reduce only ordinary optional candidates and
must be reported as ordinary-pruned count, not as a guaranteed runtime
reduction.  No thermodynamic phase criterion, Delta tolerance, final ambiguity
tolerance, acquisition logic, or Slurm submission behavior is changed.
```

## D38. Run Target-Construction Dry-Run as One Slurm Array Task per Fixed Point

Decision:

```text
The 32 fixed-point target-construction-only gate is packaged as a Slurm array
over fixed_point_id.  Each task computes the shared coarse/q-expansion
candidate set once, then applies baseline, cluster_only, rank_and_cap_k3,
rank_and_cap_k2, and rank_and_cap_energy_window selection policies to that
same candidate set.
```

Reason:

```text
The goal of this gate is to test target construction, not local-box runtime.
Repeating the coarse scan separately for every variant would waste GPU time
and obscure whether the new rank-and-cap policy fixes target explosion.
Running one task per point also keeps the package restartable and makes
per-point failures easy to inspect.
```

Consequences:

```text
The package does not run local refinement boxes and does not run active
learning.  Slurm scripts and the workflow submitter exclude gpuh01 by default.
The return archive contains target-construction tables and a gate status JSON;
full local-box GPU regression remains blocked until this gate passes.
```

## D39. Prepare Separate Rank-and-Cap K3 Five-Iteration and Full-Loop Packages

Decision:

```text
Generate two independent active-learning upload packages for rank_and_cap_k3:
one package for seed plus five acquisition-batch closed-loop validation, and a
separate package for the full active-learning loop.  The full-loop package is
uploadable now but its submit script requires CONFIRM_FULL_LOOP=1 so it cannot
be started accidentally before the five-iteration validation is reviewed.
```

Reason:

```text
The one-iteration validation passed, but it cannot establish multi-round
stability or full-loop convergence.  A five-iteration package checks repeated
train/select/exact/merge/append behavior under the accepted rank_and_cap_k3
oracle.  The full-loop package should not depend on the five-iteration output,
because that would make the full package unusable if the validation archive is
delayed or if the user wants to upload both packages at the same time.
```

Consequences:

```text
Both packages carry complete code snapshots, evidence reports, submit wrappers,
collection/report scripts, and package manifests.  Both write outputs under
their own package-local output roots and exclude gpuh01 by default through
EXCLUDE_NODES.  The five-iteration package disables early stopping to force
the planned validation length.  The full-loop package leaves the StopController
enabled but requires explicit confirmation before submission.  No phase
criterion, Delta tolerance, final ambiguity tolerance, acquisition formula,
candidate-domain strategy, local-refinement physics path, k2, energy-window,
branch reuse, Powell, adaptive box, GPU batching, or Hamiltonian cache behavior
is changed.
```

## D40. Split StopController Surprise into All-Selected, Trusted, and Hard-Risk Layers

Decision:

```text
Keep the historical all-selected label-surprise metric as the default
StopController behavior and add an explicit opt-in `stop_surprise_mode=trusted`
gate.  Record hard-risk surprise as a numerical-frontier diagnostic rather
than using it as a direct main-phase convergence veto.
```

Reason:

```text
The rankcap_k3 full-loop replay shows that the final all-selected surprise is
47/256, while the final trusted surprise is 0/137.  The last-five strict
surprise points are all rerun-required, so the historical selected-batch
surprise mixes acquisition difficulty and numerical-frontier provisional
labels.  A trusted gate better tests clean exact-label consistency while
preserving the all-selected metric for backward-compatible diagnostics.
```

Consequences:

```text
`label_surprise_rate` remains an all-selected alias.  New metadata records
all-selected, trusted, hard-risk, and selected-for-gate surprise rates and
denominators.  Trusted surprise requires trusted_exact, training_eligible_exact,
non-rerun, q-resolved, and delta-resolved labels; q_expanded alone does not
exclude a point.  A trusted denominator floor and denominator fraction guard
prevent zero-denominator or tiny-denominator false convergence.  The replay
script exactly reproduces the historical all-selected StopController and gives
trusted-mode counterfactual stop iteration 17.  No thermodynamic phase
criterion, Delta tolerance, final ambiguity tolerance, rankcap_k3 local
refinement, q-window expansion, acquisition behavior, candidate domain,
surprise tolerance, required_pass_count, patience, or boundary thresholds were
changed.
```

## D41. Package a Self-Contained Tail Continuation from the Rankcap K3 Full Loop

Decision:

```text
Create an independent `rankcap_k3_tail_surprise_continuation_v1` HPC package
that starts from the downloaded full-loop endpoint `dataset_iter031.npz` and
continues a small number of acquisition batches with `STOP_SURPRISE_MODE=trusted`.
The package carries the restart dataset, tail datasets, previous monitor
predictions, stop/metrics history, run config, selected tail artifacts, and a
complete runnable code snapshot.
```

Reason:

```text
The trusted-surprise short validation from scratch verified the engineering
path but did not test the late-stage question that motivated the change.  The
right empirical test is a package-local continuation from the full-loop tail:
it asks whether additional late active-learning batches reduce all-selected
surprise, preserve low trusted surprise, or expose boundary coverage as the
remaining blocker.
```

Consequences:

```text
The continuation package writes only under
`ML_Phase_512_RankCapK3_TailContinuation/`, starts at `START_ITER=31`, defaults
to five continuation batches, and excludes `gpuh01` by default.  It preserves
rankcap_k3 settings and trusted-surprise denominator guards.  It does not
change the thermodynamic phase criterion, Delta refinement trigger tolerance,
final ambiguity tolerance, acquisition formula, candidate-domain strategy,
rankcap_k3 local-refinement policy, StopController thresholds, k2,
energy-window pruning, branch reuse, Powell, adaptive box, GPU batching, or
Hamiltonian cache behavior.
```

## D42. Stage V Uses Boundary-Support Acquisition with a Shadow Learned Residual

Decision:

```text
Stage V cold-start active learning uses
stagev_acqv2_boundary_support_learned_residual_3d_v1.  The acquisition score is
split into a transparent physical boundary-support base score A0 and an
optional learned residual g_theta:

A(x) = A0(x) * exp(lambda_t * g_theta(phi(x))).

lambda_t starts at zero.  The learned residual remains in shadow mode until
enough logged reward samples exist and validation shows improvement over A0.
```

Reason:

```text
Stage IV-A showed that the 3D topology run can find tFFLO, but visual boundary
roughness and support gaps remain.  Stage V therefore targets boundary support
directly rather than adding another opaque global acquisition heuristic.  The
base score explicitly combines normal/SC, uniform/FFLO, P0 topology, Ppi
topology, and gap/nodal boundary channels, each using margin likelihood,
uncertainty, and sparse-support factors.  The learned component can improve
ranking only after it proves useful from online rewards.
```

Consequences:

```text
Stage V is a cold-start package: it does not train from Stage III or Stage IV
datasets/checkpoints.  Stage IV artifacts may be used only for offline
comparison and reporting.  The exact oracle remains robust_incremental with
rankcap_k3 and active-loop topology diagnostics; thermodynamic phase criteria,
topology definitions, StopController thresholds, Delta/q tolerances, and
Hamiltonian definitions are unchanged.  Candidate logs preserve selected
metadata, top-K unselected controls, A0 components, propensities, and reward
model state.  Slurm scripts exclude gpuh01 and gpuh14.
```

## D43. Stage V-v2 Splits Learned Acquisition into Per-boundary Heads

Decision:

```text
Create a new cold-start Stage V-v2 package:

run_id = stagev_v2_multihead_boundary_learning_3d_v1
output_root = ML_Phase_StageV_V2_Multihead

The Stage V-v2 acquisition keeps the transparent boundary-support A0 channels
but replaces the Stage V-v1 scalar learned residual with independent heads for
normal/SC, uniform-SC/FFLO, P0 topology, Ppi topology, and gap/nodal support.
Each head has its own reward normalization, validation metric, lambda_s, and
fallback to A0_s.
```

Reason:

```text
The Stage V-v1 return report showed that the scalar residual learned useful
rank structure, but the online learning signal was dominated by normal/SC and
global Sobol proposals while P0/Ppi topology support remained weak.  A single
scalar reward mixes rare topology rewards with frequent thermodynamic boundary
rewards.  Per-boundary reward normalization and independent lambdas make that
failure mode explicit and testable.
```

Consequences:

```text
Stage V-v2 remains strict cold-start.  It does not use Stage III, Stage IV, or
Stage V-v1 data/checkpoints for training initialization or seed coordinates.
Those artifacts may be used only for offline comparison after the run.

No physical definition changes were made.  The exact oracle, thermodynamic
phase rule, Hamiltonian, Pfaffian convention, topology labels, q/Delta search,
rankcap_k3 local refinement, and numerical tolerances are unchanged.

Selection is submitted as a Slurm job, not run as CPU-heavy login-node work.
All generated Slurm scripts exclude gpuh01 and gpuh14.  Production submission
requires CONFIRM_STAGEV2_PRODUCTION=1 and refuses to overwrite an existing run
directory when START_ITER=0.
```
