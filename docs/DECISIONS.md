# Decisions

## D86 — Synchronized author-review candidate (2026-09-11)

At explicit user direction, replace v6's per-result float barriers and reduced
complex figures with normal wide floats, restore the approved engine-v4 Fig.1,
and build both main layouts and Supplement as one source-fingerprinted set.
This supersedes D85's figure sizing for the new candidate only. Prior outputs
remain untouched. Reuse the existing LangGraph topology and Figure Engine;
native-width SM method derivatives correct stage-specific presentation, not
physical definitions. Threshold-free Stage-IV diagnostics remain in SM.

No Hamiltonian, objective, phase labels, boundaries, or physics code changes.
Mechanism-sensitive wording remains a separate author decision. Automated
checks are not figure/claim approval; final state is HUMAN_WAIT.

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

## D46 — Matched V3/V4 audit gives Decision D (2026-07-12)

Use the common offline audit rather than either independent report for V3/V4
method comparison. V3 has better final topology-surface support p95 and lower
unified offline graph uncertainty; V4 has lower sequential trusted topology
surprise. The phase-map ordering changes across the required sensitivity grid.
None of V4's four stored candidate nodes is validated as a provisional phase,
and online/offline graph-uncertainty calibration is incomplete.

Therefore neither acquisition is declared superior. Do not request new exact
calculations yet. First repair/calibrate graph and candidate diagnostics; any
later exact request must name a specific node, edge, or surprise region.

## D47 — Frozen descriptor/world model is not ready for EIG (2026-07-12)

The frozen V3/V4 development audit yields Decision C. Raw field semantics are
consistent enough to build a prototype, but the proposed fixed descriptor and
mask system is scientifically incomplete for strict cold-start information
gain: strict spatial topology holdouts are too sparse, important continuous
fields extrapolate poorly, phase recovery is weak, and ensemble epistemic
uncertainty is neither uniformly error-aligned nor monotonic with budget.

Only descriptor/mask/transform/architecture/loss source and unit tests may be
carried forward. Data rows, scalers, checkpoints, weights, optimizer state,
latent/cluster state, coordinates, rewards, and replay buffers are forbidden.
Revise schema and spatial coverage diagnostics before calibration or EIG.

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

## D44. Stage V-v3 Makes Sparse Topology-Surface Coverage an Acquisition Target

Decision:

```text
Create a strict cold-start Stage V-v3 package:

run_id = stagev_v3_sparse_arc_boundary_support_3d_v1
output_root = ML_Phase_StageV_V3_SparseArc

The Stage V-v3 acquisition keeps the Stage V-v2 multi-head structure but adds
an explicit topology-surface support-distance reduction score C_topo for the
P0/Ppi topology heads.  Sparse topology-surface patches are localized online
from local opposite-Z2 and P0/Ppi sign-change brackets; candidate points near
these sparse patches receive higher topology acquisition score only through
the unified scoring pipeline.
```

Reason:

```text
The Stage V-v2 topology-surface coverage tail ended at dataset_iter101 and
remained Decision C.  The surface proxy was geometrically stable, but coverage
p95 stayed around 0.036 rather than approaching the 0.00625 gate, and one of
the last-three topology-map changes still exceeded the 0.002 gate.  Extending
Stage V-v2 unchanged is therefore inefficient; the acquisition must directly
ask whether a candidate reduces sparse topology-surface support gaps.
```

Consequences:

```text
Stage V-v3 does not use Stage IV, Stage V-v1, Stage V-v2, Stage V-v2-tail
datasets, checkpoints, or selected coordinates for production initialization.
The completed v2-tail data may be used only for offline sparse-arc replay and
regression diagnostics.

No exact-oracle, Hamiltonian, thermodynamic phase criterion, topology
criterion, q/Delta search rule, or numerical tolerance is changed.  The gap
head is guarded by topology relevance, proposal-source attribution is a hard
validation gate, and topology-head learned lambdas are reduced if reward
correlation improves without topology-coverage improvement.

The generated H100 package excludes gpuh01 and gpuh14, uses absolute Slurm
paths under /opt/gridview/slurm-2.3.3/bin for non-interactive SSH submission,
and requires CONFIRM_STAGEV3_PRODUCTION=1 before production submission.
```

## D45. Freeze Stage V-v4 as Decision C Pending Matched Diagnostics

Decision:

```text
Classify the completed Stage V-v4 phase-structure-graph production run as
Decision C (not converged).  Do not submit an unchanged continuation or claim
publication readiness.  Before any new exact calls, run a matched V3/V4 audit,
validate the persistence and descriptor separation of the four final candidate
nodes, and localize the remaining trusted-topology-surprise points.
```

Reason:

```text
Stage V-v4 completed through exact iter096 and produced dataset_iter097 with
7080 training-eligible points and no q-unresolved, Delta-unresolved, or
rerun-required records.  Fixed-cloud phase and topology map-change gates pass,
the topology-surface shift proxy passes, and the last-three topology component
counts are stable at [5, 5, 5].  However, topology-surface coverage p95 is
0.020982 against the 0.00625 gate, and trusted topology surprise remains
[0.129032, 0.030303, 0.029412] over the last three transitions.

The graph-value model is operational and improves its reward rank correlation
relative to the graph prior, but this does not compensate for the failed
scientific coverage and surprise gates.  The previous V3 coverage value was
not recomputed in the same matched audit, so V4 superiority over V3 is not
established.
```

Consequences:

```text
Production completion and numerical label closure are accepted, but scientific
convergence is not.  No Hamiltonian, thermodynamic phase rule, Pfaffian
convention, topology-gap rule, q/Delta refinement rule, or tolerance is
changed.

The inherited source validator also leaves stale source_not_in_allowlist text
on 4362 reward rows even though proposal_source and action_family are nonblank
and valid under the V4 mapping.  Treat this as a logging-semantics defect, not
an exact-label failure, and repair it before another production run.
```

## D48 — Stage VI-0 hierarchy helps but schema remains Decision C (2026-07-12)

Adopt the four-level physical masking design as the next descriptor-development
direction, because it removes normal/not-applicable points from SC-branch and
topology losses and makes P0/Ppi regression meaningful. Do not declare the
world model validated: deltaF_UF is absent, spectral Level 3 is weak, Pfaffian
margin and uncertainty calibration remain inconsistent, and replay probe
uncertainty is non-monotonic.

The formal Stage VI-0 result is Decision C. No EIG acquisition is authorized.
Any future cold-start study must recreate data preparation, preprocessing, and
weights from scratch; only schema, masks, architecture/loss source, validation
source, and tests may be reused.

## D49 — Stage VI-0.2 blocked by missing spectral artifacts (2026-07-12)

Classify Stage VI-0.2 as Decision D. The frozen V3/V4 packages do not contain
the spectral information required to test a fingerprint, PCA/autoencoder, or
low-energy-moment descriptor. Removing k-gap and retaining log-gap is only an
ablation and does not constitute the requested spectral redesign.

Do not run the five-member full-local configuration or EIG. First locate and
export any already-computed fixed-grid spectra or physical spectral moments. If
none exist, update future exact-output logging in a separately authorized task.
No new exact calculation is authorized here, and no learned development state
may transfer to production.

## D50 — Use an evidence-first PRL graph and stop at human scope/slice gates (2026-08-07)

Decision:

```text
Create an isolated manuscript_langgraph/ package whose generation order is
evidence/claims -> Results -> Methods -> Discussion -> Introduction ->
Abstract/Title -> audits -> LaTeX. Preserve claim status in registries and
require explicit human approval for scope, the Stage III vertical slice,
production LaTeX merge, and final manuscript approval.
```

Reason:

```text
The QAHE3D audit found only an Introduction/citation graph, governance concepts,
and editorial prompts, not a complete PRL workflow. The Active Learning project
also has uneven evidence maturity: Stage II/III are the strongest results, Stage
IV is not formally converged, and Stage V/VI do not support central positive
claims. Results must therefore be written from frozen evidence before the
Introduction positions the work.
```

Consequences:

```text
The recommended paper scope is Stages I--III, with Stage III as the central
result. Stage IV is conditional/caveated; Stage V--VI are Discussion/Outlook.
The first Stage III Results-by-figure vertical slice is complete to a human gate,
but final figure reproduction remains blocked because the raw Stage III return
bundle is absent locally. No full PRL, new exact calculation, live citation
retrieval, or production manuscript merge is authorized by this decision.
```

## D51 — Revalidate manuscript contracts against the recorded prompt (2026-08-07)

Decision:

```text
Use the prompt's exact claim vocabulary (`scientific_status`, `evidence_ids`)
and exact human-gate names (`GATE_SCOPE_APPROVAL`, `GATE_CLAIM_APPROVAL`,
`GATE_FIGURE_APPROVAL`, `GATE_FINAL_MANUSCRIPT_APPROVAL`). Test each staged
stop independently instead of claiming that an unapproved invocation traverses
downstream manuscript nodes.
```

Reason:

```text
Revalidation found drift between the executable graph, registries, tests, and
earlier implementation report. The old tests referenced superseded node names
and fields, while the current Results metadata lacked the prompt-required
observation/interpretation/source/figure keys.
```

Consequences:

```text
The graph now has 15 registered nodes, stops at scope by default, and reaches
claim, figure, retrieval, and final-human gates only after explicit preceding
approvals. The curated manifest contains 37 records (34 present, 3 missing),
and 14 tests plus staged workflow validation pass. No physical definition,
numerical rule, frozen data, QAHE3D file, exact calculation, or manuscript
claim was changed.
```

## D52 — Approve the physics-first scope and retain downstream human gates (2026-08-08)

Decision:

```text
Record GATE_SCOPE_APPROVAL = approved for a main narrative consisting of
Stage I 2D thermodynamic active-learning discovery, Stage II audited exact
oracle and thermodynamic closure, and Stage III audited cFFLO/tFFLO topology.
Keep Stage IV conditional and explicitly non-converged, Stage V in Discussion,
and Stage VI in Outlook. Do not auto-pass claim or figure approval.
```

Reason:

```text
The user explicitly approved the recommended physics-first manuscript scope.
Stage II/III remain the strongest evidence, while Stage IV is not formally 3D
topology-converged, Stage V does not establish acquisition superiority, and
Stage VI does not authorize EIG.
```

Consequences:

```text
The persisted current-state workflow passes GATE_SCOPE_APPROVAL and stops at
GATE_CLAIM_APPROVAL. A read-only search of the workspace, all 10 local
archives, the parent GBU_SC tree, and Codex attachments did not recover the
Stage II dataset_iter035 endpoint, Stage III dataset_iter018 endpoint, or the
Stage III convergence/figure bundle. Stage V CSV files with matching iteration
numbers were rejected as wrong-run false positives. The rebuilt artifact
manifest has 41 rows: 38 present and 3 missing.

Figures 1-3 remain unexpanded until claim and figure approvals. Figure 4
prefers a separately authorized matched multi-seed benchmark, otherwise an
explicitly non-converged Stage IV demonstration, otherwise omission. External
Introduction retrieval remains unauthorized until Results claims are frozen
and permission is explicit. No Hamiltonian, phase definition, Pfaffian rule,
numerical tolerance, frozen dataset, exact calculation, or QAHE3D file changed.
```

## D53 — Approve reliability-aware Stage I--IV claims and stop at the figure gate (2026-08-12)

Decision:

```text
Supersede the Stage I--III-only manuscript positioning. Use the central line
that reliability-aware active learning enables thermodynamic,
topology-resolved, and higher-dimensional FFLO exploration. Put Stages I--IV
in the main narrative, Stage V in Discussion, and Stage VI in Outlook. Approve
all six bounded claim positions, including Stage IV as supported_with_caveat
and candidate Figure 4. Do not pass GATE_FIGURE_APPROVAL automatically.
```

Reason:

```text
Stage IV has signed-off report-level evidence for a completed
higher-dimensional exploration and discovered thermodynamic/topology-aware
structure, although its strict topology-convergence gates did not pass. That
supports a visibly caveated scalability/extension result, but not a complete or
fully converged 3D atlas. Stage V still does not establish learned-acquisition
superiority, and Stage VI still does not validate EIG.
```

Consequences:

```text
GATE_SCOPE_APPROVAL and GATE_CLAIM_APPROVAL now pass; the current workflow stop
is GATE_FIGURE_APPROVAL. Figure 1 is the reliability-aware framework, Figure 2
the 2D thermodynamic reconstruction, Figure 3 the topology-aware cFFLO/tFFLO
result, and Figure 4 the caveated 3D extension.

The transfer audit confirms that claim-level records and source code are
present, while the frozen Stage II endpoint, Stage III endpoint/convergence
bundle, and Stage IV endpoint/return/figure bundle are absent. The 51-row
manifest therefore has 45 present and six missing entries. Final Figures 2--4
remain blocked; no Stage V data may substitute for those runs. External
Introduction retrieval remains unauthorized. No physical definition, numerical
rule, frozen data, exact calculation, or QAHE3D file changed.
```

## D54 — Accept recovered Stage II--IV evidence and retain the figure gate (2026-08-13)

Decision:

```text
Accept the supplied scientific-evidence archive as the recovered frozen source
for the requested Stage II endpoint, Stage III endpoint/convergence bundle, and
Stage IV endpoint/return/figure bundle. Close the three raw-artifact gaps after
checksum and NPZ/CSV consistency checks. Do not treat recovery as figure
approval, do not recompute physics, and do not authorize Introduction retrieval.
```

Reason:

```text
The ZIP is path-safe, decompression-valid, and internally checksum-complete.
The Stage II and Stage IV endpoint hashes match their archived reproduction
manifests, and every Stage II--IV endpoint NPZ array matches its paired CSV.
The Stage III convergence report, decision record, tables, boundary data,
figures, and source are present. The sole provenance exception is a frozen
Stage III JSON field that labels the source as NPZ while storing the paired CSV
hash; this is a metadata-label issue because the provenance table is correct
and the numerical arrays agree.
```

Consequences:

```text
The curated artifact manifest now has 51 present rows and zero missing rows.
ARTIFACT_PROVENANCE_AUDIT, PUBLICATION_READINESS_GATE, GATE_SCOPE_APPROVAL,
GATE_CLAIM_APPROVAL, and FIGURE_EVIDENCE_CONTRACT pass. The workflow remains at
GATE_FIGURE_APPROVAL = HUMAN_WAIT. Figures 2--4 are data-ready for redraw but
not approved or manuscript-final. Stage III keeps the k=6 sensitivity and
diagnostic-bracket caveats; Stage IV keeps the strict 3D topology non-closure
caveat. The archive's intentional omission of redundant cumulative NPZ
checkpoints prevents a ground-up rerun of every historical convergence step,
but does not remove the evidence required for manuscript redraw and bounded
claim audit. No frozen record was edited and no physics was recomputed.
```

## D55 — Freeze the regenerated candidate bundle and stop at figure approval (2026-08-13)

Decision:

```text
Use the regenerated Figures 1--4 as the evidence-traced human-review
candidates. Keep Figure 4 in the main-text candidate set with explicit strict
3D non-convergence. Treat Supplementary Figure S1 as an audit candidate only.
Invoke the registered LangGraph with persistent SQLite checkpointing and stop
at GATE_FIGURE_APPROVAL without merging production LaTeX.
```

Reason:

```text
All authoritative endpoint hashes and key counts pass automated validation;
all candidate PDFs render correctly; and the Results paragraph metadata covers
all four figures without unsupported-sentence flags. Human scientific approval
is nevertheless distinct from technical validation.
```

Consequences:

```text
The figure approval package is the next scientific review object. A real
LangGraph thread is checkpointed under active_learning_prl_v1 and can resume
downstream only after an explicit approve/approve_with_edits record. External
Introduction retrieval remains unauthorized. No new physics, frozen-label
change, or final manuscript integration occurred.
```

## D56 — Freeze the PRL visual language before quantitative redraw (2026-08-14)

Decision:

```text
Insert a separate human GATE_FIGURE_STYLE_APPROVAL before rebuilding the
quantitative manuscript panels. Use a physics-first, white-background visual
language and generate four scripted layout mockups plus a high-fidelity Figure
1 vector prototype. Do not overwrite the evidence-traced candidate figures and
do not pass either figure gate automatically.
```

Reason:

```text
The existing candidates are scientifically auditable but read as dashboards:
Figure 1 is a set of disconnected rectangular boxes, Figures 2--3 normalize
convergence by thresholds, and Figure 4 duplicates 3D point clouds and uses a
red failure banner. Publication hierarchy should be settled before registered
frozen data are replotted.
```

Consequences:

```text
The approved scientific status is unchanged. Figure 2 will use raw convergence
metrics; Figure 3 will separate raw convergence from k=6/8/12 sensitivity and
show support only in its boundary diagnostic; Figure 4 will use one 3D overview
and move strict non-closure to caption/Supplement text. Figure 2--4 mockups are
synthetic layout aids, not numerical evidence. Quantitative reconstruction and
manuscript-LaTeX integration remain blocked on explicit style approval.
```

## D57 — Supersede the Figure 1 prototype with a direct-SVG physical-manifold master (2026-08-14)

Decision:

```text
Make fig1_master.svg/pdf/png the current Figure 1 review target. Author the
whole composition directly as editable SVG rather than Matplotlib. Visually
prioritize one audited BdG physical-state manifold inside a six-stage clockwise
active-learning cycle. Retain the earlier prototype only as design history.
```

Reason:

```text
The prototype established the loop but used three unrelated one-dimensional
free-energy curves and made the final tFFLO spectral cartoon look too close to
a gap-closing state. The publication figure must show a unified F(Delta,q)
landscape and distinguish the isolated critical closure from the final reopened
and gapped tFFLO state.
```

Consequences:

```text
The master contains no raster SVG element, embeds all PDF fonts, and passes
26/26 master checks plus 80/80 package checks. The physical conventions are
unchanged: exact BdG creates all labels, FFLO has finite q, topology is assigned
only after the gap audit, and (-1)^nu = sgn[P(0)P(pi)]. Human style approval is
still required; quantitative Figures 2--4 and manuscript LaTeX remain blocked.
```

## D58 — Refine Figure 1 v2 around the physical state space (2026-08-14)

Decision:

```text
Retain the existing direct-SVG Figure 1 structure and redraw only its visual
hierarchy. Make the central physical-state manifold dominant, replace the flat
free-energy contour by a vector three-dimensional F(Delta,q) surface, enlarge
the exact BdG oracle, separate the critical topology transition from the final
tFFLO phase, and reduce the opacity and typography of the outer scheduling
cycle. Keep the surrogate and acquisition explicitly subordinate to physics.
```

Reason:

```text
The previous v2 was scientifically correct but read first as a six-stage
process infographic. The manuscript figure must answer which thermodynamic and
topological state space is explored before explaining how exact calculations
are scheduled.
```

Consequences:

```text
The master now shows representative normal, uniform-SC, and finite-q FFLO
minima on one three-dimensional vector surface; H_BdG(k), eigensolve, and
free-energy generation in the oracle; and separate cFFLO (gapped, nu=0),
critical gap-closing, and tFFLO (reopened, gapped, nu=1) zones. No quantitative
data were read and no physics was recomputed. Figure 1 passes 29/29 master
checks, the style package passes 80/80 checks, and the 16 manuscript workflow
tests pass. GATE_FIGURE_STYLE_APPROVAL remains HUMAN_WAIT; Figures 2--4,
manuscript LaTeX, and retrieval permission are unchanged.
```

## D59 — Freeze Figure 1 physics while enforcing metric-driven independent-card layout (2026-08-22)

Decision:

```text
Use the approved v2 scientific assets as frozen inputs and create a v3
manuscript master through typography, spacing, card, and arrow reflow only.
Independent cards may neither touch nor overlap. Register true-font text and
plot geometry, require at least 14 px internal padding and 20 px declared
gutters, and hard-fail the build when a geometry check fails. Stop at
GATE_FIG1_LAYOUT_APPROVAL.
```

Reason:

```text
The scientific hierarchy was mature, but the prior compact composition used
deliberate docking and contained several print-scale near-collisions that were
not detectable from card rectangles alone. A reproducible metric-driven pass
was required before Figure 1 could be frozen.
```

Consequences:

```text
The v3 layout passes with 0 hard failures, a 20 px minimum gutter, 14 px
minimum registered padding, and 8 px route endpoint clearance. The central
physical card remains dominant at 60.99% of total card area. The acquisition
footer retains the required boundary-distance term as d_boundary in a math
subscript. No physical/numerical convention, frozen dataset, manuscript LaTeX,
or Figures 2--4 changed. Human approval is still required.
```

## D60 — Accept metric-bounded Figure 1 v4 as the targeted-final freeze candidate (2026-08-22)

Decision:

```text
Preserve the v3 architecture and scientific semantics. Enlarge only the
central thermodynamic surface and its local label system, strengthen the outer
cycle within fixed opacity/width bounds, normalize the right column, and keep
reliability and learned scheduling visually tertiary. Require a 12-category
hard lint, exact 178 x 120.15 mm output, and stop at GATE_FIG1_FINAL_FREEZE.
```

Reason:

```text
The v3 figure was structurally mature, but the physical landscape remained too
small, several outer routes were partially hidden by cards, and the first v4
assembly contained one acquisition-label collision. A targeted correction and
saliency audit were sufficient; architectural rebuilding was neither needed
nor authorized.
```

Consequences:

```text
The shaded F(Delta,q) surface is 41.89% wider and 34.87% taller than in v3;
its labels use the actual data-to-display transform and the longest leader is
44.94 px. Right-column gaps are both 20 px. All routes remain outside card
interiors, acquisition annotations clear selected candidates, and final lint
passes with 0 hard failures. The topology triptych and all physical/numerical
definitions remain unchanged. Human freeze approval is still required; no
manuscript LaTeX or Figures 2--4 were modified.
```

## D61 — Make the figure engine the mandatory single source of truth (2026-08-23)

Decision:

```text
All new or materially modified manuscript figures must load the complete
figures/figure_engine constitution and a figure-specific contract before
drawing. The preflight record must retain the rule hashes and human gate.
Generic validation must hard-fail common geometry, typography, registered
contrast, vector-integrity, and scientific-text violations. Figure-specific
audits remain additive, and an automated pass never approves a human gate.
```

Reason:

```text
The successive Figure 1 passes showed that local corrections are reliable only
when scientific hierarchy, style tokens, measured geometry, and validation are
separate and globally consistent. Ad-hoc prompts and duplicated YAML rules did
not provide a reproducible basis for Figures 2--4.
```

Consequences:

```text
Canonical rules now live only in figures/figure_engine/constitution and
styles; root-level legacy rule files are compatibility pointers. Contracts and
examples support framework, phase-map, topology, convergence, trajectory, 3D,
and slice-atlas figures. Figure 1 v5 builders now require engine preflight, but
the approved artwork was not regenerated in this integration pass. Physical
and numerical definitions remain governed by MODEL_SPEC.md and
docs/NUMERICS_SPEC.md. GATE_FIG1_V5_APPROVAL remains HUMAN_WAIT.
```

## D62 — Require negative controls and preserve the Fig.1 regression failure (2026-08-23)

Decision:

```text
Treat the figure engine as a tested software framework. Every practical hard
rule must have a negative fixture that fails for the intended reason and a
corrected control that passes. Maintain a machine-readable enforcement
registry and name human-review-only rules explicitly. Use unchanged Figure 1
v5 as the first regression benchmark. Do not weaken the 4-px text-border rule,
edit the regression expectation, or redraw Figure 1 merely to obtain a pass.
```

Reason:

```text
A preflight hash proves that rules were read but does not prove they constrain
generation. The expanded self-test demonstrated real constraint capability and
found a previously missed Figure 1 defect: reliability-row-1 has only 3.25 px
right clearance against the 4-px minimum. Preserving the failure is more useful
than silently tuning the rule or artwork around it.
```

Consequences:

```text
The invalid fixture fails on all 15 expected conditions, the corrected control
passes, and 17 unit tests pass. Figure 1 v5 now fails one of 45 checks by the
known regression signature; its SVG/PDF/PNG remain unchanged. Legacy builders
that only preflight but retain local constants are recorded as adapter debt.
Phase 3 Figure 1 engine regeneration requires separate user authorization.
GATE_FIG1_V5_APPROVAL remains HUMAN_WAIT.
```

## D63 — Regenerate Figure 1 as a contract-driven engine benchmark (2026-08-23)

Decision:

```text
Create figures/fig1_engine/fig1_engine_v1 as a new review candidate. Preserve
the frozen v5 files and physics assets, but require the new master, workflow
ribbons, reliability card, metadata, measured geometry, and validation evidence
to be generated from the figure engine. Do not auto-pass the human gate.
```

Reason:

```text
The framework self-check proved that the rules constrain generation and found
a real v5 border-clearance defect. A new benchmark was required to demonstrate
that the engine can repair a known failure without weakening the rule or
recomputing scientific content.
```

Consequences:

```text
The v5 regression is corrected from 3.25 px to 7.25 px declared right
clearance, with an independent browser bbox pass. The primary physical card is
60.25% of registered card area, generic lint passes 45/45, specific lint passes
13/13, and 18 unit tests pass. The PDF is exactly 178.00 x 120.15 mm and color
and grayscale renders were inspected. Frozen v5 hashes remain unchanged.
GATE_FIG1_ENGINE_V1_APPROVAL remains HUMAN_WAIT; Figures 2--4 are unchanged.
```

## D64 — Refine Figure 1 scientific assets without changing the engine (2026-08-23)

Decision:

```text
Freeze fig1_engine_v1 and create fig1_engine_v2 as a traceable scientific-
asset refinement. Preserve the canvas, card architecture, claims, physical and
topological definitions, and all canonical engine rules. Refine only the
thermodynamic landscape/FFLO branch markers, BdG-oracle internal hierarchy,
physical-observation readout, and allowed layered workflow-arrow profile.
```

Reason:

```text
Engine v1 proved constraint, reproducibility, and validation capability, but
its central landscape remained plot-like, the free-energy oracle output was
not dominant enough, the observation card resembled a checklist, and the
workflow connectors remained visually mechanical. These are asset-quality
problems rather than framework deficiencies.
```

Consequences:

```text
The central landscape occupies 31.32% of its card and the central card ranks
first in visual mass. The free-energy minimap is larger than the Hamiltonian
and spectrum subobjects, cFFLO/tFFLO remain separate representative branches,
and tFFLO remains a gapped bulk plus in-gap boundary state. Generic lint passes
45/45, specific lint 19/19, and engine tests 18/18. A duplicate v2 manifest
entry exposed by lint was corrected without altering validator logic.
GATE_FIG1_ASSET_REFINEMENT_APPROVAL remains HUMAN_WAIT; Figure 2 is blocked.
```

## D65 — Activate versioned Visual Narrative enforcement before Fig.1 v3 artwork (2026-08-25)

Decision:

```text
Activate visual_narrative_rules.yaml as a canonical Figure Engine rule. Require
its fields for schema v2 and explicit schema-v1 opt-in while preserving old
schema-v1 contracts. Resolve the primary visual object to a real scientific
subobject, enforce Fig.1 scientific-content fill ratio as a hard check, and
separate hard failures, diagnostic warnings, and human-review judgments. After
the complete health check passes, create only a Fig.1 v3 proposal package and
stop at GATE_FIG1_V3_PROPOSAL.
```

Reason:

```text
Aggregate card area allowed a visually small landscape to pass, and saliency
proxies cannot establish first fixation or aesthetic success. A versioned,
executable narrative contract plus negative controls is required before the
engine can guide another artwork revision without reverting to ad-hoc prompts.
```

Consequences:

```text
Fourteen canonical rule files are loaded and hashed; 24 unit tests pass; the
invalid fixture produces all 17 expected detections and the corrected control
passes. The frozen engine-v1 fill-ratio failure is retained as a regression
signal, and engine-v1 artwork hashes are unchanged. The v3 proposal passes its
technical proposal checks but contains no artwork. GATE_FIG1_V3_PROPOSAL is
HUMAN_WAIT and must not be inferred from validator output.
```

## D66 — Build Fig.1 engine-v3 after proposal approval and retain the final human gate (2026-08-26)

Decision:

```text
Record GATE_FIG1_V3_PROPOSAL as approved and generate a separate
figures/fig1_engine_v3 review candidate. Preserve the v2 card architecture,
frozen scientific definitions, topology semantics, and peripheral assets.
Rebuild only the central asset so the real tightly cropped 3D surface satisfies
the 0.65 scientific-content fill-ratio rule. Stop at
GATE_FIG1_ENGINE_V3_FINAL_APPROVAL.
```

Reason:

```text
The proposal passed the versioned narrative contract, but its scientific
content requirement could only be verified on actual artwork. A separate
candidate allows the engine to demonstrate that the true surface—not its parent
card or an inflated plot bbox—occupies the required space without overwriting
the frozen v1/v2 benchmarks.
```

Consequences:

```text
The measured surface fill ratio is 0.716878. The vector PDF is exactly
178.00 x 120.15 mm; Fig.1-specific validation passes 20/20 and canonical lint
has zero hard failures. Two saliency/contrast proxy warnings and the human
first-fixation judgment remain open by design. Twenty-four engine tests pass,
and v1/v2 hashes remain unchanged. The candidate is not final-approved and does
not authorize Figure 2 or manuscript integration.
```

## D67 — Isolate the compact thermodynamic-landscape correction behind an asset gate (2026-08-26)

Decision:

```text
Do not redraw or reassemble Figure 1. Build exactly two deterministic,
schematic thermodynamic-landscape candidates in a dedicated asset-only gate.
Represent Normal, Uniform SC, and one parent FFLO sector as representative
basins across parameter space; retain the separate topology triptych as the
only cFFLO/tFFLO interpretation. Require actual foreground-mask geometry and a
negative/control validator self-test before human review.
```

Reason:

```text
The engine-v3 central object passes its earlier fill-ratio rule but still reads
as a narrow diagonal ribbon. This is a scientific-asset composition problem,
not authority to change the oracle, topology semantics, surrounding workflow,
or exact physics. A local gate allows the terrain itself to be tested without
overwriting a validated master.
```

Consequences:

```text
Candidate A (compact hero) and Candidate B (restrained five-level contour
floor) pass every hard geometry check. Both retain one warning because their
internal whitespace is below the hard maximum but above the preferred target.
The elongated negative fixture is rejected, the compact control passes, no
exact BdG calculation was run, and the frozen engine-v3 SVG/PDF hashes are
unchanged. GATE_COMPACT_THERMODYNAMIC_LANDSCAPE_ASSET remains HUMAN_WAIT with
no selected candidate; full-Fig.1 integration is unauthorized.
```

## D68 — Integrate approved Candidate A as Figure 1 engine-v4 (2026-08-26)

Decision:

```text
Record the compact-landscape asset gate as approved with Candidate A selected.
Create a separate figures/fig1_engine_v4 candidate and replace only the upper
thermodynamic landscape plus its immediate local labels and axes. Preserve the
v3 topology row, seven peripheral cards, workflow ribbons, canvas, and claims.
Stop at GATE_FIG1_ENGINE_V4_APPROVAL.
```

Reason:

```text
Candidate A passed the isolated compact-terrain contract and provides the
intended dominant physical manifold. Integrating it locally tests the approved
asset in the complete figure while protecting previously audited scientific
semantics and making all unrelated regions regression-verifiable.
```

Consequences:

```text
The real surface fill ratio is 0.662770 and passes the 0.65 threshold. Generic
and Fig.1-specific validation have zero hard failures; 24 engine tests pass;
the 178.00 x 120.15 mm vector PDF and grayscale render pass print inspection.
Topology and seven peripheral cards are pixel-identical to v3. The diagnostic
saliency proxy ranks landscape first and topology third, so final reading order
remains human review. No physics was recomputed, v3 hashes are unchanged, and
GATE_FIG1_ENGINE_V4_APPROVAL remains HUMAN_WAIT.
```

## D69 — Freeze Fig.1 engine-v4 and route Figures 2-4 through strict-data contracts (2026-08-27)

Decision:

```text
Record figures/fig1_engine_v4 as SCIENTIFIC_FREEZE. Extend the existing Figure
Engine with a schema-v3 strict-data panel router and provenance-first gates.
Build the Figure 2 review package from frozen artifacts only, then stop at
GATE_FIG2_HUMAN_SCIENCE_REVIEW. Before that human approval, create only data
manifests, panel contracts, and readiness reports for Figures 3 and 4.
```

Reason:

```text
Data figures require source hashes, exact/derived semantics, transformation
authorization, trust masks, and raw convergence validation before visual
narrative or style. The evidence archive supports an exact-sample Figure 2,
but does not contain final dense-monitor or thermodynamic boundary-coordinate
artifacts. Inferring smooth boundaries would violate the frozen-evidence rule.
Stage IV diagnostic continuous surfaces likewise cannot be promoted to formal
3D phase/topology surfaces while strict convergence gates remain open.
```

Consequences:

```text
Figure 2 uses 7434 endpoint samples, the true 512-call cold start, complete
iterations 0-34, raw convergence metrics, and a separate 129-call hard-risk
audit layer. No interpolation, smoothing, resampling, or boundary extraction
is used. Figure Engine lint passes; Fig.2-specific checks pass 21/21; PNG
rendering is deterministic. Figure 3 is contract-ready with the k=6 caveat.
Figure 4 is contract-ready-with-caveat and continuous surfaces remain blocked.
No new physics calculation, manuscript LaTeX merge, or external retrieval is
authorized. GATE_FIG2_HUMAN_SCIENCE_REVIEW is HUMAN_WAIT.
```

## D70 — Freeze Fig.2 and generate strict-data Fig.3/Fig.4 review candidates (2026-08-27)

Decision:

```text
Record the user's Figure 2 approval as SCIENTIFIC_FREEZE with a hash manifest.
Release the existing Figure 3 and Figure 4 render authorizations. Build both
figures from frozen recovered artifacts only and stop at their separate human
science-review gates. Keep Figure 4's continuous 3D surface unauthorized.
```

Reason:

```text
Figure 2 passed both automated and human science review. Stage III contains a
complete frozen endpoint plus archived k=8 boundary/convergence evidence.
Stage IV supports an exact-observation demonstration and explicit fixed-mu
slices, but not formal 3D topology closure. Rendering those evidence classes
separately preserves the approved manuscript scope without upgrading a
diagnostic surface into a physical result.
```

Consequences:

```text
Figure 2 artwork is immutable under figures/fig2_strict_data/
fig2_scientific_freeze.yaml. Figure 3 uses 3213 trusted FFLO labels (2290
cFFLO, 923 tFFLO), the archived iteration-18 k=8 contour, raw convergence
traces, and the visible k=6 sensitivity result. Figure 4 uses 7081 exact 3D
observations, four evenly spaced explicit bins from the 20-row fixed-mu audit,
and raw failed convergence metrics. For Figure 4, FFLO-only topology counts
are 3145 cFFLO and 1445 tFFLO; the archived 3265/1466 totals include 141
uniform-SC samples and are not relabeled as FFLO. No physics, interpolation,
smoothing, RBF/KNN surface, or boundary re-extraction was run. Both figures
pass Figure Engine hard checks and remain at human science-review gates.
```

## D71 — Approve and freeze Fig.3/Fig.4; pass the overall figure gate (2026-08-27)

Decision:

```text
Record the user's explicit approval of Figure 3 and Figure 4. Freeze Figure 3
without changing its caveats and freeze Figure 4 with its strict nonclosure
caveat. Pass GATE_FIGURE_APPROVAL as approve_with_edits, where the required
edits are permanent caveat constraints rather than artwork changes.
```

Consequences:

```text
All four manuscript figures are now scientifically frozen. Figure 3 and
Figure 4 each have a ten-artifact hash manifest and their generator refuses to
overwrite frozen artwork. The Results-by-figure LangGraph may now resume.
External Introduction retrieval, new physics, and final LaTeX integration
remain unauthorized. The next graph stop must therefore be
INTRO_CITATION_LOOP_GATE after Results, Methods, Supplement, and Discussion
drafting from frozen evidence.
```

## D72 — Add an independent Results human-review gate before Methods (2026-08-27)

Decision:

```text
Resume the existing active_learning_prl_v1 SQLite thread after the approved
Figure gate, generate Results/captions and their audits from the graph's
results_by_figure_writer node, and stop at a new minimal-compatible
GATE_RESULTS_HUMAN_REVIEW. Do not enter Methods until Results and captions are
explicitly approved by the user.
```

Reason:

```text
The existing graph contained a Results node but it only validated provisional
paragraph metadata and routed directly to Methods after Figure approval. The
writing prompt requires a real evidence-traced Results package and a distinct
human scientific freeze. Automated claim, numerical, and wording audits cannot
substitute for that decision.
```

Consequences:

```text
The graph now has 16 nodes. The Results package is stored under
output/pdf/manuscript_results_review_v1/ and the same thread checkpoint is
exported as checkpoint_results_review.json. Current state is
GATE_RESULTS_HUMAN_REVIEW=HUMAN_WAIT, results_claims_frozen=false, with Methods,
Supplement, Discussion, Introduction retrieval, and all later nodes unexecuted.
Figure 1--4 hashes are unchanged and no physics was recomputed.
```

## D73 — Reopen only the Fig. 4 panel-C text after rendered-artwork consistency audit (2026-08-27)

Decision:

```text
Accept the user's overall Results/caption approval, execute the authorized
Methods/SM/Discussion/Outlook drafting, but reopen the Results freeze only for
the two Fig. 4 panel-C text fragments. Stop at a new human correction gate.
Do not alter Figure 4 artwork and do not proceed to external Introduction
retrieval.
```

Reason:

```text
The approved final Figure 4 hash resolves to the three-slice manuscript figure
configured at mu/t = -0.35, 0.55, 1.25. The approved Results/caption package
instead inherited the four-slice centers -0.45, 0.15, 0.85, 1.45 from an
earlier strict-data prototype. Scientific correctness and figure/text
traceability require a targeted human-reviewed correction even though the main
higher-dimensional-exploration claim and strict nonclosure caveat are intact.
```

Consequences:

```text
GATE_RESULTS_FIG4_CAPTION_CORRECTION is HUMAN_WAIT. The proposed edit changes
"four" to "three" and substitutes the three centers actually visible in the
approved artwork; no data, physics, counts, convergence metric, artwork, or
claim changes. Intermediate main/SM PDFs are marked as layout previews pending
the correction. The same SQLite thread is checkpointed at
checkpoint_fig4_caption_correction_gate.json. All 19 manuscript workflow tests
pass and external retrieval remains false.
```

## D74 — Apply the approved Fig. 4 amendment and refreeze Results (2026-08-27)

Decision:

```text
Apply only the approved Figure 4 panel-C text correction: replace four slices
with three and use the centers -0.35, 0.55, and 1.25. Rebuild the Results,
main-manuscript review, and Supplemental Material PDFs; refreeze the Results
hashes; pass GATE_RESULTS_FIG4_CAPTION_CORRECTION; and stop at the separate
INTRO_CITATION_LOOP_GATE.
```

Consequences:

```text
Figure 4 artwork, frozen data, phase/topology counts, convergence metrics, and
physics are unchanged. The Results packet now binds to the already approved
final three-slice artwork rather than the preserved four-slice prototype.
Results/artwork/source consistency passes, 20 manuscript tests pass, and
current_workflow_state reports
awaiting_introduction_retrieval_permission. External scientific-literature
retrieval remains false.
```

## D75 — Treat journal architecture and editorial threshold separately (2026-08-27)

Status: superseded as active positioning guidance by D76; retained only as the
historical comparison that prompted the user's format clarification.

Decision:

```text
Record that the current four-figure manuscript is structurally closer to a
Nature Portfolio Article than to a PRL Letter. Do not infer from that structure
that it presently meets the Nature Physics or Nature Communications threshold.
Recommend Communications Physics for the current evidence package. Preserve a
PRL option only as a separate compressed branch centered on Stage III. Require
new authorization before any scientific computation intended to raise the
Nature Portfolio ceiling.
```

Reason:

```text
The evidence chain benefits from Article length, but Stage IV is strictly
nonclosed, Stage V establishes no acquisition superiority, Stage VI is Outlook
only, and the present demonstrations use one physical model. Official journal
criteria distinguish broad/exceptional physics impact from a significant
advance for a specialized physics community.
```

## D76 — Decouple PRL-style formatting from journal selection (2026-08-27)

Decision:

```text
Record target_journal=undecided. Retain the PRL/REVTeX two-column appearance as
an author-preferred working format only. Keep the complete Figures 1--4
scientific narrative and all approved caveats; do not perform journal-specific
compression, expansion, or template conversion at this stage.
```

Reason:

```text
The user has no publication preference and simply prefers PRL typography. A
typesetting preference must not silently become a scientific-scope decision or
submission recommendation.
```

Consequences:

```text
The intermediate manuscript title and metadata now state that the PRL-style
layout is author-preferred and journal-neutral. The historical
active_learning_prl_v1 run ID is retained for provenance. No physics, figures,
claims, gates, or external-retrieval permissions changed.
```

## D77 — Make Figure Engine freezes the sole manuscript figure authority (2026-08-28)

Decision:

```text
Bind manuscript Fig. 1 to figures/fig1_engine_v4 and Figs. 2--4 to their
strict-data scientific freezes. Reject the legacy manuscript_figures/final
main-figure bundle as a compilation source. Require source and staged PDF
hashes to match the registered freezes before compilation, and require the
LaTeX recorder to confirm that all four staged PDFs were embedded.
```

Reason:

```text
The downstream compiler hard-coded legacy candidate paths and bypassed the
authoritative Figure Engine registry. This also caused the approved four-slice
strict-data Fig. 4 to be replaced by a legacy three-slice artwork. File
existence and caption checks alone could not detect the provenance failure.
```

Consequences:

```text
D73/D74 and GATE_RESULTS_FIG4_CAPTION_CORRECTION are retained as historical
records but superseded. Fig. 4 text again uses four slices at mu/t = -0.45,
0.15, 0.85, and 1.45 with half-width 0.025. No physics or artwork is changed.
GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR passes; the Manuscript LangGraph rebuilds
Results, Methods, SM, Discussion, and Outlook and stops at the unchanged
INTRO_CITATION_LOOP_GATE.
```

## D78 — Add a non-destructive PRL-integrated publication profile (2026-08-28)

Decision:

```text
Create active_learning_prl_integrated_v1 as a child publication-profile run of
active_learning_prl_v1. Use a continuous Introduction -> integrated
Results/Methods -> Discussion structure, insert the four canonical frozen
figures near first substantive use, and retain their frozen captions in the
first candidate. Keep Stage VI only as the final Discussion paragraph. Do not
modify the parent run, frozen figures, physical definitions, or evidence
registries.
```

Reason:

```text
The author prefers the compact PRL visual format but has not selected a target
journal. A separate profile allows genuine method/result integration and
inline figures without destroying the audited intermediate manuscript or
turning a formatting preference into a venue or scope claim.
```

Consequences:

```text
The candidate compiles to six pages and is explicitly marked overlength by an
internal planning estimate. Citation gaps remain provisional because external
retrieval is unauthorized. Caption-condensation options are proposals only.
The new run stops at GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW=HUMAN_WAIT;
neither scientific freeze nor publication freeze is granted automatically.
```

## D79 — Retain the Fig. 2--4 single-column layout direction but reject direct scaling as final (2026-08-28)

Decision:

```text
Keep the four-page layout trial in which Fig. 1 is double-column and Figs. 2--4
are single-column. Do not replace the current reviewed candidate or refreeze
the manuscript. Treat the trial as evidence that the single-column narrative
works, but do not adopt the existing 178 mm figure PDFs by direct scaling.
Any production use requires separately generated 86.8 mm Figure Engine assets
with at least 7 pt internal text at placed size.
```

Reason:

```text
The trial improves manuscript flow and reduces the candidate from six pages to
four without float or LaTeX defects. However, the 0.4875 scale factor reduces
registered 7 pt internal text to about 3.41 pt. Several small convergence,
sensitivity, slice, and nonclosure annotations carry scientific caveats and
cannot be treated as decorative or moved out of legibility.
```

Consequences:

```text
The single-column trial remains a non-authoritative layout proof. Frozen data,
captions, claims, figures, and the original HUMAN_WAIT gate do not change. A
future Figure Engine pass may simplify secondary labels and redistribute
annotations, but may not remove the Stage III sensitivity or Stage IV
finite-domain/nonclosure evidence.
```

## D80 — Execute verified Introduction retrieval in an isolated cited branch (2026-08-28)

Decision:

```text
After explicit authorization, create active_learning_prl_integrated_cited_v1
as a child of the PRL-integrated candidate. Require a verified source registry,
claim-to-citation matrix, BibTeX bibliography, citation-coverage audit, compiled
provenance audit, and a separate human manuscript gate. Use external literature
only for background and precedent; preserve frozen Results, figures, caveats,
and physical definitions.
```

Reason:

```text
The core workflow contained only a pre-retrieval planning node and a blocking
citation stub, while the integrated branch bypassed them and retained a
provisional uncited opening. A publication candidate needs an executable
retrieval/synthesis/audit path, but automated coverage cannot replace human
scientific judgment about source balance or claim scope.
```

Consequences:

```text
The cited branch contains 12 verified references for six scoped background
claims and compiles to six pages with clean citation, font, float, and figure-
provenance audits. The original integrated candidate remains byte-identical.
The new branch stops at GATE_CITED_INTRODUCTION_MANUSCRIPT_HUMAN_REVIEW=
HUMAN_WAIT; no scientific or publication freeze is inferred. Native single-
column Fig. 2--4 production remains a separate unresolved asset gate.
```

## D81 — Expand the cited Introduction to a tiered 50-reference survey (2026-08-28)

Decision:

```text
Create active_learning_prl_integrated_cited_v2 as a child of cited_v1. Set an
admissible bibliography range of 45--55 with a target of 50. Preserve strict
primary/authoritative support for strong claims, while allowing verified
Crossref DOI metadata to admit contextual survey references. Forbid search-
snippet-only citations, uncited bibliography padding, and use of external
literature as evidence for this project's Results.
```

Reason:

```text
The 12-reference cited_v1 branch established traceability but was too narrow
for the requested physics and computational background. A tiered policy
broadens platform, altermagnetic, topology, and scientific-ML context without
pretending that every contextual source received the same claim-level audit as
the core authorities.
```

Consequences:

```text
The v2 Introduction contains 50 unique cited sources mapped to 12 bounded claim
clusters: 12 T1 core/bibliographically cross-checked sources and 38 T2 Crossref
DOI-metadata sources. All 50 are cited and defined; none is bibliography-only.
Frozen Results and figure hashes remain unchanged. The seven-page candidate
passes citation, compile, font, caveat, provenance, and every-page-content
checks, but remains at the separate HUMAN_WAIT manuscript gate.
```

## D82 — Remove threshold-based Stage-IV closure claims from Figure 4 (2026-08-31)

Decision:

```text
Retire Figure 4(d) from the current manuscript candidate. Do not use the
registered Stage-IV operational cutoffs as physical convergence criteria.
Keep Figure 4(a)--(c) as finite-domain exact observations, and move archived
iteration histories to a Supplemental diagnostic figure without cutoff lines
or PASS/FAIL labels.
```

Reason:

```text
The map-change, proxy-surface displacement, coverage, surprise, and component
criteria diagnose a particular reconstruction and sampling procedure. Their
cutoffs were not calibrated to a thermodynamic, spectral, or topological
physical scale. Raw histories remain informative, but cutoff crossings do not
authorize a physical closure statement.
```

Consequences:

```text
Figure Engine now requires an explicit scientific or numerical basis before a
diagnostic cutoff line may be drawn, and diagnostic-only mode forbids pass/fail
labels. Figure 4 v2 contains panels (a)--(c) only. Supplemental Figure S1
contains six raw histories without cutoff lines. The main and Supplemental
texts were regenerated through a versioned LangGraph child run. The physical
model, exact datasets, Figures 1--3, and 50-source bibliography did not change.
All new artifacts remain at human science/visual review gates.
```

## D83 — Separate public narrative from reproducibility evidence (2026-08-31)

Decision:

```text
Continue from the Stage-IV revision branch with a publication layer that keeps
the main text physics-first and moves implementation, audit, and dense
numerical evidence into a self-contained Supplement. Report active-learning
efficiency only as exact-location accounting unless a matched wall-time or
candidate-wide boundary baseline is present.
```

Reason:

```text
The exported evidence establishes 6,880 unique exact evaluations from a
77,361-location candidate grid, corresponding to 8.893% sampling and an
11.244-fold exact-location reduction. It does not preserve a matched exhaustive
wall-time baseline or a candidate-wide boundary denominator. Internal gate and
provenance vocabulary obscured the physics narrative but remained useful for
the machine-readable audit trail.
```

Consequences:

```text
The LangGraph publication layer adds visibility routing, public-language and
number-density checks, method-evidence ingestion, efficiency accounting, and a
Supplement reproducibility gate. The Supplement reconstructs the actual oracle,
surrogate, acquisition, stopping, topology, and environment details from code
and frozen artifacts. Four method/efficiency diagrams pass the Figure Engine.
Figures 1--4 remain byte-identical. No physics was recomputed and no external
retrieval was performed. The continuation stops at an explicit human gate.
```

```text
GATE_PUBLIC_MANUSCRIPT_AND_SUPPLEMENT_HUMAN_REVIEW = HUMAN_WAIT
MAIN_PAGES = 8
SUPPLEMENT_PAGES = 9
COMPILED_REVIEW_PACKAGE_PAGES = 18
REFERENCES = 50
NEW_PHYSICS = false
EXTERNAL_RETRIEVAL = false
```

## D84 — Add scientific depth without changing frozen physics (2026-09-03)

Decision:

```text
Continue from active_learning_public_reproducible_v4 with a minimal LangGraph
scientific-depth layer. Add a compact model, joint (Delta,q) minimization,
thermodynamic phase logic, gapped Pfaffian applicability, figure-specific
physical interpretation, and a four-function Discussion. Preserve the 50-source
Introduction, frozen Figures 1--4, exact labels, and public-language budget.
```

Reason:

```text
The parent manuscript was reproducible but read as an extended abstract. The
missing material was physical organization and method logic, not additional
counts, thresholds, provenance vocabulary, or workflow inventories.
```

Consequences:

```text
The main text gains 553 workflow-normalized words and passes the Results-depth,
minimum-method-kernel, Discussion-synthesis, unsupported-interpretation, and
main/Supplement consistency audits. The implemented objective is denoted
F_impl after a code-level PASS_WITH_NOTATION_FIX audit. Stage-I and Stage-II
count semantics are separated. The readable full-width layout retains a known
four-page figure-gallery limitation; the single-column fallback is rejected for
print readability. The run stops at a new HUMAN_WAIT gate.
```

## D85 — Use single-column placement for main-text Figures 2--4 (2026-09-03)

Decision:

```text
By explicit human instruction, keep Fig. 1 as a full-width figure and insert
Figs. 2--4 as ordinary one-column REVTeX floats. Use legal per-result float
barriers to keep each figure near its scientific discussion. Preserve the
frozen figure binaries rather than silently creating new artwork.
```

Reason:

```text
The full-width layout separated all Results prose from four consecutive figure
pages. The single-column placement materially improves compiled proximity:
Fig. 2 and Fig. 3 share the page of their discussions, and Fig. 4 is within one
page of its first discussion.
```

Consequences:

```text
The selected six-page manuscript has no float warning, overfull box, or
undefined reference. Direct scaling reduces the 178-mm registered assets to
one-column width, so internal text readability remains an explicit human-review
caveat. No scientific text, caption, reference, Supplement, data, label, or
figure binary changes. The layout remains at HUMAN_WAIT.
```
