# Project Summary

Last updated: 2026-07-03

## 1. Project Identity

This is a computational physics research project for the one-dimensional
altermagnetic FFLO topological superconducting phase diagram.

The project has two connected goals:

1. Compute reliable exact BdG phase-diagram data using CUDA/H100 resources.
2. Use active learning to refine phase boundaries with fewer exact BdG calls.

The machine-learning model is not treated as a physics replacement. It is used
as a scheduler that decides where exact BdG calculations are most valuable.

## 2. Current Scientific Target

The current phase diagram is parameterized by

```text
x-axis: k_B T / t
y-axis: J_A / t
```

The primary observables are

```text
Delta_opt
q_opt
eta
I_c^+
I_c^-
```

The first-stage phase labels are

```text
normal:
    Delta_opt < DELTA_EPS

uniform_SC:
    Delta_opt >= DELTA_EPS and abs(q_opt) < Q_EPS

FFLO:
    Delta_opt >= DELTA_EPS and abs(q_opt) >= Q_EPS
```

The active-learning refinement currently focuses on

```text
normal / superconducting boundary
uniform_SC / FFLO boundary
eta sign boundary
strong-diode boundary
high-J_A q-sensitive region
```

Topological labels such as trivial FFLO, topological FFLO, and edge-mode checks
remain a later milestone unless a robust invariant is integrated into the
pointwise exact oracle.

## 3. Canonical Documentation

Project-level agent instructions require these memory documents:

```text
docs/PROJECT_SUMMARY.md
docs/MODEL_SPEC.md
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/report_qa/
```

Current canonical model details are in

```text
MODEL_SPEC.md
```

`docs/MODEL_SPEC.md` is a pointer to that root-level file to avoid maintaining
two independent copies of the Hamiltonian and physical definitions.

Use this reading order before non-trivial code changes:

```text
1. AGENTS.md
2. docs/PROJECT_SUMMARY.md
3. MODEL_SPEC.md
4. docs/NUMERICS_SPEC.md
5. docs/DECISIONS.md
6. target source files
```

## 4. Important Files and Directories

Core exact solvers and plotting:

```text
eta_phase_diagram_cuda.py
tfflo_1d_cuda.py
plot_eta_phase_diagram.py
finite_T_phase_diagram.m
```

Warm-start exact dataset:

```text
eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/
eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz
```

Active-learning package:

```text
ml_phase/
    config.py
    dataset_builder.py
    labels.py
    models.py
    acquisition.py
    exact_oracle.py
    hpc.py
    active_refine.py
    evaluate.py
    plot_active_learning.py
    report_builder.py
```

GBU/H100 helper scripts:

```text
00_check_env_gbu.sh
00b_check_h100_cuda.sh
01_prepare_candidate_shards.sh
02_submit_h100_oracle.sh
03_merge_h100_shards.sh
04_build_report.sh
scripts/slurm_active_refine.sh
scripts/slurm_exact_oracle_array.sh
RUN_ORDER_GBU_HPC.md
```

Local copy of HPC outputs:

```text
PD_ML/
PD_ML/ML_Phase_hpc/active_runs/
PD_ML/ML_Phase_hpc/figures/
PD_ML/ML_Phase_hpc/reports/
```

Report-oriented Q&A notes:

```text
docs/report_qa/
```

This directory stores explanatory question-and-answer notes that may be reused
in the final report, paper draft, thesis text, or presentation narrative.

## 5. Warm-Start Dataset

The current warm-start dataset contains an exact BdG grid with

```text
n_JA = 156
n_kT = 138
total exact points = 21528
kT range roughly: 0.01 to 0.5
JA range roughly: 0.01 to 1.2
```

Expected arrays:

```text
kT_vec
JA_vec
q_vec
eta_matrix
q_opt_matrix
delta_opt_matrix
ic_plus_matrix
ic_minus_matrix
delta0
```

Matrix convention:

```text
row = JA index
col = kT index
```

This dataset is the correct starting point for active learning. Starting the
model from no data would waste exact BdG calls rediscovering the known coarse
phase structure.

## 6. Exact High-Performance Phase-Diagram Workflow

The exact CUDA/BdG workflow is

```text
1. choose a (kT, JA) grid;
2. build the BdG Hamiltonian;
3. scan q and Delta;
4. minimize free energy;
5. extract q_opt and Delta_opt;
6. compute current response;
7. compute I_c^+, I_c^- and eta;
8. assign phase labels;
9. save npz/csv outputs;
10. generate figures.
```

The finite-temperature boundary from `finite_T_phase_diagram.m` is used as a
reference or soft mask. It should not replace exact BdG labels.

Plotting convention:

```text
Do not show kT < 0.
Use LaTeX-style labels when producing publication figures.
Keep top and right plot borders visible if matching previous figure style.
Use thinner phase-boundary lines than the original embedded plotting code.
```

## 7. Active-Learning Refinement Workflow

The active-learning workflow is

```text
warm-start exact npz
    -> flatten supervised dataset
    -> train regression ensemble and phase classifier
    -> generate dense candidate grid
    -> predict observables and uncertainty
    -> compute acquisition score
    -> select diverse boundary-like points
    -> partition points into H100 shards
    -> run exact oracle on each shard
    -> merge exact shard outputs
    -> append trusted exact results
    -> retrain
    -> repeat until boundary convergence
```

The cluster loop driver supports continuation from an existing trusted dataset:

```text
START_ITER=0 starts from the warm-start dataset and produces dataset_iter001.
START_ITER=1 continues from dataset_iter001 and produces later datasets.
```

The current acquisition score combines

```text
classification uncertainty
regression uncertainty
Delta-boundary score
q-boundary score
eta-boundary score
predicted-gradient score
diversity score
```

The refined specification now additionally requires

```text
q-window risk score
high-JA extrapolation/risk score
Delta-refinement risk score
```

because the earlier fixed q range is not sufficient for all high-JA points.

## 8. q-Range Correction Required

Important physical issue:

```text
As J_A increases, q_opt also grows.
For J_A/t > 1.2, q_opt can grow rapidly.
The original q range may no longer contain the true free-energy minimum.
```

Required numerical behavior:

```text
1. exact_oracle.py must record q_min, q_max, n_q, q_opt, q_edge_hit;
2. if q_opt is near q_min or q_max, the point is not a trusted final label;
3. rerun with an expanded q window or mark the point as low-confidence;
4. active learning should prioritize this region but not blindly learn from
   q-truncated results;
5. acquisition.py should include q-edge-risk, especially for JA > 1.2.
```

This is one of the most important pending implementation corrections.

## 9. Delta Refinement Required

Important numerical issue:

```text
Near the normal/SC boundary, Delta_opt is small and the free-energy minimum is
shallow. A coarse Delta grid can visibly shift the inferred phase boundary.
```

Required numerical behavior:

```text
1. detect points with Delta_opt close to DELTA_EPS;
2. detect small free-energy difference between Delta = 0 and Delta_opt;
3. refine Delta locally near the minimum;
4. record Delta_refinement_level and free_energy_gap_to_normal;
5. do not append ambiguous normal/SC labels silently.
```

This correction should be implemented before interpreting a production
high-resolution active-learning boundary as final physics.

## 10. HPC / GBU Cluster Notes

Relevant GBU cluster facts from the user run:

```text
H100 partition: NV_H100
CUDA module: compiler/cuda/cuda-12.8.1
Python env used remotely: /public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
torch.cuda.is_available() can be False on login nodes
CUDA should be checked inside an NV_H100 job
```

Preferred remote environment setup:

```bash
export PROJECT_DIR=$PWD
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
export CONDA_ENV=
export EXCLUDE_NODES=gpuh01
```

Avoid relying on `conda activate` inside SLURM scripts because the cluster
reported

```text
CondaError: Run 'conda init' before 'conda activate'
```

The scripts should call `${PYTHON_BIN}` directly when possible.

The first completed 128-point H100 test found a driver mismatch on `gpuh01`.
The automatic loop scripts therefore exclude `gpuh01` by default for both
candidate-generation jobs and exact-oracle array jobs. Clear `EXCLUDE_NODES`
only after verifying the node's driver and PyTorch/CUDA compatibility.

## 11. Standard HPC Run Order

Environment check:

```bash
bash 00_check_env_gbu.sh
```

Current production active-learning loop from the warm-start dataset:

```bash
export RUN_ID=active_boundary_loop_512x50_r0075
export START_ITER=0
export N_ITERS=50
export POINTS_PER_ITER=512
export WORLD_SIZE=8
export N_ENSEMBLE=5
export REG_EPOCHS=240
export CLS_EPOCHS=240
export BATCH_SIZE=512
export EXCLUDE_NODES=gpuh01
bash hpc_active_loop.sh
```

Current dense-grid acquisition radius:

```text
diversity_min_dist = existing_min_dist = 0.0075
```

## 12. Completed Work in Current Thread

Implemented or prepared:

```text
1. standalone phase-diagram plotting code was discussed and corrected;
2. Word document guidance was translated into ML_Guidance.md, although that
   file appears encoding-corrupted locally and should be regenerated if needed;
3. active-learning plan was written in Active_Learning_Phase_Boundary_Refinement_Plan.md;
4. ml_phase package was created for dataset building, models, acquisition,
   exact oracle, HPC sharding, plotting, evaluation, and report building;
5. GBU/H100 upload scripts were created;
6. H100 script assumptions were corrected for NV_H100 and CUDA module naming;
7. report_builder.py was patched so missing optional figures do not break
   pdflatex and PNGs are converted to RGB when possible;
8. MODEL_SPEC.md was expanded with q-window and Delta-refinement requirements;
9. this docs/PROJECT_SUMMARY.md was created as the primary project memory;
10. code structure was updated so exact-oracle outputs, datasets, acquisition
    scores, plots, and shard merge helpers can carry q-window and
    Delta-refinement risk metadata before the multi-iteration loop is built;
11. `QDELTA_REFINEMENT_EXECUTION_PLAN.md` was added for the remaining
    q-expansion, Delta-refinement, trusted-filtering, and active-loop work;
12. `package_hpc_upload.ps1` and `hpc_one_click_submit.sh` were added for a
    source-focused H100 upload bundle and one-command cluster smoke workflow;
13. `QDELTA_TARGET_LOGIC_CODE_REWRITE_PLAN.md` was added to specify the target
    exact-oracle rewrite: normal-state q-not-applicable handling, SC-only q
    expansion, Delta-boundary refinement, trusted filtering, and rerun reasons;
14. `ml_phase/exact_oracle.py` was rewritten to implement pointwise q/Delta
    confirmation logic: normal-state q is marked not-applicable, SC q-edge
    points can expand q and rerun, Delta-ambiguous points can rescan a refined
    Delta interval, and each point records trusted/unresolved status metadata;
15. `ml_phase/hpc.py` now writes both `exact_merged_iterXXX.npz` and
    `exact_trusted_iterXXX.npz`, plus a reasoned `rerun_points.csv`;
16. `scripts/slurm_exact_oracle_array.sh` now enables q expansion and Delta
    refinement by default through environment-controlled flags;
17. `scripts/dev_check_qdelta_logic.py` was added as a lightweight sanity check
    for normal q-not-applicable handling, q expansion config, Delta refinement
    config, and trusted/rerun split logic;
18. `positive_delta_gap` was added to the exact-oracle Delta logic so that
    stable `Delta_opt=0` normal points are not mislabeled as unresolved merely
    because their free-energy gain relative to the `Delta=0` baseline is zero;
19. `ml_phase.append_trusted` was added to append only trusted H100 exact
    outputs into the next active-learning dataset;
20. `ml_phase.active_refine` now supports `--resume-dataset` and
    `--start-iteration`, allowing candidate generation from the latest trusted
    dataset instead of always restarting from the warm-start npz;
21. `hpc_active_loop.sh` was added as the recommended cluster driver for
    multi-iteration active learning: candidate generation, H100 exact oracle,
    shard merge, trusted append, and next-iteration resume;
22. `hpc_active_loop.sh` now supports `START_ITER` so an already completed
    `iter000` can be followed by automatic iterations from `dataset_iter001`;
23. H100 submission helpers now default to `EXCLUDE_NODES=gpuh01` because the
    previous cluster run showed a CUDA driver mismatch on that node;
24. a 25-iteration 128-point automatic active-learning run was analyzed under
    `hpc_upload_qdelta_20260510_143031/ML_Phase_128_loop24`;
25. Delta-unresolved points are now interpreted as finite-resolution
    normal/SC boundary-band data when `Delta_opt=0` and the best positive
    Delta state is not lower than the normal state by more than the adopted
    energy tolerance;
26. `ml_phase.hpc` now derives `delta_boundary_band_normal`,
    `training_eligible_exact`, and `needs_rerun_exact` during shard merge,
    writes `merge_summary_iterXXX.json`, and keeps boundary-band normal points
    eligible for training;
27. `ml_phase.append_trusted` now appends `training_eligible_exact` points and
    reports clean trusted versus boundary-band appended counts;
28. `ml_phase.active_refine` now suppresses previously identified boundary-band
    coordinates during candidate selection and writes
    `selection_diagnostics.json`;
29. `ml_phase.report_builder` now reports the latest appended dataset size and
    separates clean trusted, boundary-band normal, training-eligible, and
    true rerun-required counts;
30. a 10-iteration continuation run was analyzed under
    `hpc_upload_qdelta_20260511_115659/ML_Phase_128_looptail10`, validating
    boundary-band training eligibility and exposing repeated selection of
    already-known dataset points as the next active-learning bottleneck;
31. `ml_phase.active_refine` now hard-excludes existing exact coordinates
    during candidate selection using 4-decimal rounded `(kT, JA)` keys, applies
    the same 4-decimal cooldown to recent selected and boundary-band points,
    and writes candidate-pool exclusion counts to `selection_diagnostics.json`;
32. a 3-iteration 128-point continuation validation was analyzed under
    `hpc_upload_qdelta_20260511_162837/ML_Phase_128_verify_delta`, confirming
    that 4-decimal existing-data exclusion greatly reduces repeated selected
    points against the current exact dataset;
33. `ml_phase.active_refine` also hard-excludes candidates whose normalized
    distance to any existing exact point is below `existing_min_dist`, matching
    the selected-selected diversity radius. The original validation used
    0.015; the current production-upload default is 0.0075;
34. the first active-learning flowchart and its generator were removed after
    review because the layout mixed workflow responsibilities and was visually
    crowded. `scripts/plot_active_learning_flowchart_v2.py` now generates a
    replacement six-stage clockwise workflow schematic that separates the
    learning dataset, ML training, acquisition, HPC exact BdG oracle, quality
    gate, and append/diagnostics stages. The old
    `active_learning_flowchart.png/pdf` outputs were deleted; the current
    outputs are `active_learning_flowchart_v2.png/pdf` plus a JSON summary.
35. `scripts/plot_ml_training_architecture.py` now generates a dedicated
    ML-training architecture schematic for the active-learning model. The
    figure shows the exact training data, stratified validation split,
    standardization, two parallel MLP ensembles, LaTeX loss definitions,
    ensemble uncertainty products, training parameters, and downstream
    validation metrics. On 2026-05-13 the deep-network panels were redrawn
    using representative neuron circles and fully connected line diagrams
    instead of compact layer rectangles, while preserving the true
    `2 -> 64 -> 64 -> out_dim` architecture. The current outputs are
    `ml_training_architecture.png/pdf` plus `ml_training_architecture_summary.json`
    under
    `hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/ml_training_architecture/`.
36. `scripts/plot_active_learning_flowchart_v2.py` was revised to use
    LaTeX-style symbols for the dataset, acquisition score, uncertainty terms,
    and exact-output variables. The exact-computation block is now labeled
    "HPC Exact BdG Oracle" rather than H100, so the figure describes the
    workflow generically without tying the schematic to one GPU model.
37. `docs/report_qa/` was added as the dedicated directory for explanatory
    project Q&A that may later be incorporated into the final report. Its
    README defines the recommended `YYYYMMDD_topic.md` naming convention and a
    question/short-answer/technical-notes/report-use structure. `AGENTS.md`
    now requires future report-useful Q&A to be saved or summarized there
    without overriding canonical model, numerics, or decision documents.
    The first note,
    `docs/report_qa/20260513_ml_training_active_learning_qa.md`, records the
    discussion of why the workflow uses separate regression and phase
    classifier ensembles, how classifier uncertainty enters acquisition, what
    the quality gate does, why RMSE values are validation diagnostics rather
    than a summed joint training loss, and why continuous observables and phase
    labels use different loss functions. It also records the caveat that the
    dense-grid acquisition radius is useful for ordinary candidate selection
    but too crude for explicit boundary-bracket midpoint refinement. It also
    records the early-stop rule added on 2026-05-13 for active-learning loops
    whose unique data growth stalls.
38. `hpc_active_loop.sh`, `ml_phase/append_trusted.py`,
    `ml_phase/active_refine.py`, and `ml_phase/config.py` now support
    low-new-point early stopping. Append summaries record `input_samples`,
    `output_samples`, and `new_unique_samples_added`. The HPC loop stops by
    default if no candidates are selected, if zero new unique samples are
    appended, or if fewer than 8 new unique samples are appended for 2
    consecutive iterations. The local Python active-refinement loop uses
    matching config fields and writes `local_append_summary.json` for local
    exact runs.
39. The next HPC upload package was prepared for a warm-start production run:
    `hpc_active_loop.sh` now defaults to `START_ITER=0`, `N_ITERS=50`,
    `POINTS_PER_ITER=512`, `WORLD_SIZE=8`, `N_ENSEMBLE=5`,
    `REG_EPOCHS=240`, `CLS_EPOCHS=240`, and `BATCH_SIZE=512`. The ordinary
    dense-grid acquisition radius was halved to
    `diversity_min_dist = existing_min_dist = 0.0075`.
40. `package_hpc_upload.ps1` was rerun after the 512x50/r0.0075 changes,
    producing `hpc_upload_qdelta_20260513_120556.tar.gz`. The staged upload
    directory contains the warm-start npz, revised `hpc_active_loop.sh`,
    revised `ml_phase/config.py`, and updated numerical/decision docs.
41. `report_active_learning_r0015_note/` was added as an English LaTeX
    research-note folder for the completed `r=0.015` active-learning stage.
    The note explains the exact BdG warm-up calculation, phase labels, MLP
    surrogate training, acquisition function, quality gate, `r=0.015`
    selection rule, rechecked boundary counts, and limitations. It uses five
    curated figures: the original exact phase diagram, the active-learning
    main-boundary result, the combined eta phase diagram, the active-learning
    workflow, and the ML-training architecture.
```

Observed remote run:

```text
active_boundary_h100_smoke: smoke workflow ran through 01/02/03/04
active_boundary_h100_v1: 128-point workflow was reported as complete
```

Local copied outputs include figures and a compiled report under

```text
PD_ML/ML_Phase_hpc/
```

## 13. Current Results Interpretation

The latest analyzed 128-point active-learning result contains 25 automatic
iterations, `iter000` through `iter024`, with output in

```text
hpc_upload_qdelta_20260510_143031/ML_Phase_128_loop24
```

The run completed the full automatic loop repeatedly:

```text
train/select -> H100 exact oracle -> merge -> trusted filter -> append -> next iteration
```

Aggregate numerical summary:

```text
selected exact calls: 3200
merged exact points: 3200
trusted exact points: 3053
rerun/boundary-band points: 147
initial samples: 21528
latest dataset_iter025 samples: 23214
net sample increase: 1686
q expanded and confirmed: 21
q unresolved: 0
Delta refined: 2574
Delta unresolved before boundary-band interpretation: 147
```

The acquisition function concentrates points near

```text
1. the high-JA low-kT boundary descending from large JA;
2. the broad FFLO/normal or superconducting boundary around JA/t ~ 0.6-0.8;
3. the low-JA, low-kT eta/Delta-sensitive region;
4. the high-kT, lower-JA tail where the boundary bends downward.
```

The uncertainty map confirms that the classifier is most uncertain near phase
interfaces, while the acquisition map further expands the target region by
including predicted gradients, diversity, and boundary scores.

The q-window correction is no longer the dominant numerical problem in this
run:

```text
q-edge-hit total: 0
q-unresolved total: 0
q-expanded confirmed total: 21
```

The dominant issue is the normal/SC finite-resolution boundary band. The 147
Delta-unresolved entries all have

```text
delta_boundary_unresolved;max_delta_refinement_reached
```

Most have `Delta_opt=0` and `positive_delta_gap` in the `1e-9` to `1e-8`
range. This means the exact oracle did not find a positive-Delta state with a
resolvable condensation-energy gain. At the adopted energy scale, these should
be interpreted as normal-side or normal/SC boundary-band points rather than as
generic solver failures.

The 25-iteration result supports the active-learning goal by showing that the
surrogate scheduler repeatedly targets physically meaningful boundary regions,
that q-window risk is handled by the exact oracle, and that remaining
uncertainty is localized to the normal/SC boundary-band interpretation.

The 10-iteration continuation after boundary-band semantics produced

```text
iter025 through iter034
selected exact calls: 1280
clean trusted points: 1215
boundary-band normal points: 63
training-eligible points: 1278
true rerun-required points: 2
latest dataset_iter035 samples: 23587
net sample increase over dataset_iter025: 373
q unresolved: 0
```

This confirms that most previously "Delta-unresolved" points are now handled as
finite-resolution normal/SC boundary-band normal data. The remaining true rerun
points are positive-Delta boundary cases near

```text
kT/t ~= 0.074667
J_A/t ~= 1.19
```

The main new bottleneck was candidate efficiency: many selected points were
already present in the dataset. This has now been addressed at the acquisition
stage by hard-excluding existing exact coordinates rounded to 4 decimals and by
cooling down recent selected and boundary-band coordinates with the same
rounding convention.

The 3-iteration validation after this exclusion change produced

```text
iter035 through iter037
selected exact calls: 384
merged exact points: 384
training-eligible exact points: 378
clean trusted exact points: 350
boundary-band normal points: 28
rerun-required points after boundary-band interpretation: 6
dataset_iter035 samples: 23587
dataset_iter038 samples: 23957
net sample increase: 370
already existing coordinates by independent rounded/exact check:
    iter035: 4
    iter036: 3
    iter037: 1
q unresolved: 0
q expanded and confirmed: 3
Delta refined: 322
Delta unresolved: 34
Delta unresolved requiring further rerun: 6
```

This validates the acquisition-side fix in practice: the previous 82-97
repeated existing coordinates per 128-point iteration dropped to 1-4, and the
net unique dataset increase is now close to the number of selected points.
However, `selection_diagnostics.json` reported `already_in_dataset_rounded=0`
for the same iterations, so the next code pass should fix the diagnostic and
close the remaining high-JA existing-coordinate leak.

The follow-up code change now uses the same normalized radius for selected to
existing-exact exclusion:

```text
previous validation setting:
existing_min_dist = diversity_min_dist = 0.015

current production-upload setting:
existing_min_dist = diversity_min_dist = 0.0075
```

An offline candidate-pool check using `dataset_iter038` showed that the finite
candidate pool remains above the 128-point batch size after the stronger
exclusion:

```text
finite candidates before existing-distance exclusion: 30130
finite candidates after existing-distance exclusion: 1366
```

The boundary-extraction step from the interrupted work session was rechecked
on 2026-05-12 using

```text
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42
```

Important provenance result:

```text
dataset_iter039.npz
dataset_iter040.npz
dataset_iter041.npz
dataset_iter042.npz
```

all have the same array-content hash and contain 24083 samples. Thus
`dataset_iter042` is a postprocessing/empty-append alias of `dataset_iter039`,
not evidence for three further successful H100 exact iterations.

The default boundary extraction was reproduced exactly:

```text
n_exact_points: 24083
phase counts: normal=7011, uniform_SC=405, FFLO=16667
boundary segments: 8090
eta_zero: 7787
normal_sc: 122
strong_diode: 70
uniform_fflo: 111
confidence: high=5643, medium=2410, low=37
```

Independent checks found:

```text
kT_negative_points: 0
nonfinite observable points: 0
phase-threshold mismatches: 0
boundary predicate failures: 0
boundary interpolation outside segment: 0
needs_rerun_exact: 0
q_unresolved: 0
delta_boundary_band_normal: 110
```

Parameter sensitivity showed that `eta_zero` remains overwhelmingly numerous
across tested bin widths, while `normal_sc` and `uniform_fflo` are more
sensitive to `kt_bin_width`. The archived all-boundary
`targeted_refinement_points.csv` should therefore not be used directly for the
next H100 run, because eta-zero segments can dominate the refinement budget.

The recheck also found that applying a dense-grid acquisition exclusion radius
to explicit boundary-bracket midpoint targets can reject prioritized midpoint
candidates. This is expected because a boundary midpoint is close to the two
exact points that define the bracket. Future boundary refinement needs a
boundary-specific target policy instead of blindly reusing the dense-grid
candidate exclusion radius.

Recheck outputs:

```text
docs/BOUNDARY_EXTRACTION_RECHECK_REPORT.md
scripts/recheck_boundary_extraction.py
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/boundaries/recheck_iter042_default/
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/boundaries/recheck_iter042_sensitivity/
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/boundaries/recheck_iter042_audit/
```

A combined visualization was then generated to overlay the rechecked boundary
data and active-learning additions on top of the original warm-start exact
phase diagram:

```text
scripts/plot_phase_boundary_overlay.py
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/phase_boundary_overlay/original_phase_with_rechecked_boundaries_and_new_data.png
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/phase_boundary_overlay/original_phase_with_rechecked_boundaries_and_new_data.pdf
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/phase_boundary_overlay/original_phase_boundary_overlay_summary.json
```

This plot uses the original 21528-point warm-start phase labels as the
background and marks the 2555 active-learning/new exact coordinates separately
from the original grid. It also marks the 110 new finite-resolution
normal/SC boundary-band points with a distinct symbol. Boundary overlays use
the rechecked `normal_sc`, `uniform_fflo`, `strong_diode`, and `eta_zero`
segment CSVs. The `eta_zero` layer is intentionally faint because it is a
dense response-function sign boundary rather than a thermodynamic phase
boundary.

For clearer reading, a simplified plot set was added:

```text
scripts/plot_clean_phase_and_eta_maps.py
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/clean_phase_eta_maps/clean_phase_main_boundaries_new_nearby.png
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/clean_phase_eta_maps/enhanced_warm_start_eta_with_original_boundaries.png
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/clean_phase_eta_maps/clean_phase_eta_maps_summary.json
```

The simplified phase plot keeps only the original phase background,
`normal_sc` boundary, `uniform_fflo` boundary, and new exact points within a
normalized distance 0.025 of those two main boundaries. It marks 1247 of the
2555 new exact points as near the main phase boundaries, including all 110
finite-resolution boundary-band normal points.

The enhanced eta plot combines the original warm-start eta matrix with the
active-learning exact results. The warm-start background and the 2555 new exact
points use the same signed-power color scale,
`sign(eta) * abs(eta)**0.45`, so weak diode-efficiency regions remain visible.
The plot uses active-learning extracted `normal_sc` and `uniform_fflo`
boundaries, while retaining only the old cFFLO/tFFLO reference curves because
the topological distinction was not revalidated by the active-learning exact
oracle. The color transform is only for visualization; it does not change the
stored eta values or any physical labels.

A third reading-oriented eta plot was added:

```text
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/clean_phase_eta_maps/all_exact_eta_with_revised_boundaries.png
hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/clean_phase_eta_maps/all_exact_eta_with_revised_boundaries.pdf
```

This figure does not mark new points separately. It colors all 24083 exact
points from the current active-learning dataset by eta on the same signed-power
scale. It uses active-learning extracted `normal_sc` and `uniform_fflo`
boundaries, while retaining the old cFFLO/tFFLO reference curves only as
topological-boundary placeholders.

The project now has a dedicated Q&A memory folder:

```text
docs/report_qa/
```

Use it for explanations that are likely to become report text, especially
questions about the active-learning logic, ML model interpretation, quality
gate, acquisition function, phase-boundary interpretation, and numerical
caveats.

Current entries:

```text
docs/report_qa/README.md
docs/report_qa/20260513_ml_training_active_learning_qa.md
```

## 14. Pending Implementation Priorities

Highest priority:

```text
1. implement a boundary-specific targeted-refinement policy that quota-limits
   eta_zero and prioritizes normal_sc plus uniform_fflo brackets.
2. decide and document the boundary-midpoint distance rule, because
   dense-grid existing-distance rules are not appropriate for all bracket
   midpoint targets.
3. add explicit boundary displacement diagnostics and extract a
   publication-facing normal/SC boundary line with uncertainty band.
4. add cooldown or accounting for true rerun-required coordinates so repeated
   positive-Delta ambiguous points do not reappear across later iterations.
```

Next priority:

```text
1. run a moderate larger-budget validation only after existing-point exclusion
   and boundary diagnostics are implemented;
2. create robust topological labels only after exact invariant/OBC checks are
   integrated;
3. benchmark seconds per point on H100 for different n_q/n_delta settings if
   higher-resolution Delta studies become scientifically necessary;
4. add restart/resume logic for partial shard completion;
5. improve LaTeX report with scientific interpretation sections.
```

## 15. Work Rules for Future Agents

Before non-trivial changes:

```text
1. read AGENTS.md;
2. read this file;
3. read MODEL_SPEC.md;
4. read docs/NUMERICS_SPEC.md and docs/DECISIONS.md;
5. inspect target source files;
6. summarize model assumptions and planned edits before modifying code.
```

Do not silently change

```text
Hamiltonian conventions
phase-label thresholds
axis conventions
q or Delta search conventions
HPC partition assumptions
trusted dataset semantics
```

Do not commit or push unless explicitly requested. Never commit credentials,
private keys, `.env` files, or large generated outputs unless explicitly
requested.

## 16. Latest Figure Additions

The ML-training architecture figure added on 2026-05-13 is an explanatory
schematic only. It does not change `ml_phase/models.py`, the physical labels,
the loss functions, or the acquisition logic.

The plotted training structure matches the current Torch implementation:

```text
input: x = (k_B T/t, J_A/t)
regression ensemble: 2 -> 64 -> 64 -> 5
classification ensemble: 2 -> 64 -> 64 -> 3
n_ensemble: 5
optimizer: Adam
regression loss: MSELoss on standardized targets
classification loss: CrossEntropyLoss on phase labels
```

The RMSE quantities shown in the figure are validation diagnostics. They are
not summed into a joint training loss; the regression ensemble and classifier
ensemble are trained separately with their respective losses.

The script writes a JSON summary with training parameters and a commit hash
when the working directory is inside a Git repository. In the current local
copy, Git metadata is unavailable, so the summary records `commit: null`.

## 17. 2026-05-13 ML_Phase_512 Result Audit

The new HPC-returned result directory was inspected:

```text
hpc_upload_qdelta_20260513_120556/ML_Phase_512
```

The run ID is:

```text
active_boundary_loop_512x50_r0075
```

This run uses the revised dense-grid exclusion radius:

```text
diversity_min_dist = 0.0075
existing_min_dist = 0.0075
points_per_iter = 512
world_size = 8
enable_early_stop = true
min_new_points_per_iter = 8
max_low_append_iters = 2
```

The returned result contains three available active-learning iteration
directories, `iter000` through `iter002`, plus datasets through
`dataset_iter003.npz`. It did not produce a 50-loop sequence in the returned
local copy. The append history is:

```text
dataset_iter000.npz: 21528 samples, warm start
dataset_iter001.npz: 22034 samples, +506 training-eligible points
dataset_iter002.npz: 22527 samples, +493 training-eligible points
dataset_iter003.npz: 22527 samples, +0 training-eligible points
```

`dataset_iter002.npz` and `dataset_iter003.npz` are byte-identical. The final
available dataset therefore contains 22527 exact/training samples, not a
larger 50-loop accumulated dataset. Phase counts recomputed from the active
thresholds are:

```text
normal:     5520
uniform_SC: 155
FFLO:       16852
```

Basic consistency checks on the latest dataset:

```text
all inspected arrays finite: yes
phase labels match Delta/q threshold recomputation: yes, 0 mismatches
rounded (kT, JA) duplicates at 4 decimals: 0
q_unresolved count in latest dataset: 0
q_expanded count in latest dataset: 34
delta_unresolved count in latest dataset: 125
delta_refined count in latest dataset: 457
boundary-band normal count in latest dataset: 125
```

Per-iteration exact merge summary:

```text
iter000: selected 512, merged 512, training eligible 506, rerun required 6
iter001: selected 512, merged 512, training eligible 493, rerun required 19
iter002: selected 3,   merged 3,   training eligible 0,   rerun required 3
```

The final `iter002` selected only three points. All three were Delta-refined
but still Delta-unresolved and flagged as rerun-required, so they were not
appended to the trusted/training dataset. This is the practical stop condition
for the current result: the candidate pool left only three eligible points
after existing-distance, recent-selection, and boundary-band exclusion, and
none passed the quality gate for training append.

The auto-generated active-learning report was checked and revised:

```text
hpc_upload_qdelta_20260513_120556/ML_Phase_512/reports/active_learning_phase_boundary_report.tex
hpc_upload_qdelta_20260513_120556/ML_Phase_512/reports/active_learning_phase_boundary_report.pdf
```

Report-generation source files changed:

```text
ml_phase/report_builder.py
report/active_learning_phase_boundary_report.tex
```

The report now includes:

```text
latest dataset phase counts
latest append status
per-iteration selected/merged/training-eligible/rerun-required table
neutral "HPC" wording instead of "Multi-H100"
split acquisition formula to avoid overfull equation layout
```

The report PDF was regenerated with `pdflatex`. The final log has no fatal
errors, LaTeX errors, undefined references, or overfull/underfull warnings
from the report content. The local directory is not a Git repository, so
`git status --short` reports `fatal: not a git repository`.

Current interpretation:

```text
1. The 512-point larger run successfully added 999 training-eligible exact
   points over warm start before exhausting the current ordinary dense-grid
   candidate pool.
2. The lack of 50 completed loops is not itself a physics result; it reflects
   acquisition/exclusion/quality-gate saturation under the current radius and
   candidate grid.
3. The three final rerun-required points should be handled explicitly before
   using them as evidence for a shifted SC-normal boundary.
```

Recommended next steps:

```text
1. Do not interpret `iter002` selected-point geometry as a new phase-boundary
   shape; those points did not enter the trusted dataset.
2. If more refinement is needed, change the acquisition candidate pool or use a
   boundary-specific refinement strategy instead of simply requesting more
   loops.
3. Add cooldown/accounting for rerun-required coordinates so repeated
   unresolved boundary points do not consume later iterations.
4. Use `dataset_iter003.npz` as the latest available dataset only because it is
   identical to `dataset_iter002.npz`; scientifically, the last accepted
   append occurred at `dataset_iter002.npz`.
```

## 18. 2026-05-13 Continuous-Learning Plan

A planning document was added for optional checkpoint-based continuous learning
of the active-learning MLP surrogates:

```text
docs/continuous_learning_plan.md
```

Current behavior is unchanged:

```text
Each active-learning iteration initializes fresh MLP ensembles and retrains on
the full cumulative accepted dataset. Neural-network weights are not saved in
the current production outputs.
```

The plan proposes a staged implementation:

```text
1. save model checkpoints first while preserving full_retrain as the default;
2. store regression/classifier ensemble weights, scalers, class mapping, and
   dataset hashes;
3. add optional warm_start_weights mode only after checkpoint reproduction is
   validated;
4. compare continuous learning against the full-retraining baseline before any
   production use.
```

No physical definitions, phase-label thresholds, exact-oracle rules, or
current active-learning behavior were changed by this documentation update.

The plan was expanded with three related active-learning design clarifications:

```text
1. Reports should show cumulative selected and cumulative accepted
   active-learning points, not only the latest iteration, because a saturated
   late iteration can hide the useful accepted data from earlier iterations.
2. Radius exclusion should be treated as an acquisition-control scale, not a
   physical convergence criterion. After one radius saturates, boundary-focused
   refinement should use extracted boundary brackets and a smaller local
   boundary radius rather than simply stopping.
3. High-JA warm-start labels should not be treated as fully trusted boundary
   evidence, because the original fixed q window was known to be insufficient
   for rapidly shifting q at JA/t >~ 1.2. ML-predicted high-JA boundaries still
   require exact validation with q-window expansion.
```

The same document now also contains a detailed boundary-focused active-learning
workflow. The workflow separates:

```text
Stage 0: define scientific targets
Stage 1: build initial cumulative dataset
Stage 2: train or load ML surrogates
Stage 3: build separate candidate pools
Stage 4: acquisition and quota-based point selection
Stage 5: exact BdG oracle evaluation
Stage 6: trusted append and boundary extraction
Stage 7: stopping/radius-schedule decisions
Stage 8: report and figure updates
```

The updated stopping logic requires agreement between:

```text
1. exact boundary stability,
2. exact-oracle quality flags,
3. boundary-focused ML error/loss convergence,
4. acquisition saturation at the current radius.
```

ML loss/error convergence is recorded as an auxiliary stopping criterion, not
as a standalone physical convergence criterion.

## 19. 2026-05-13 ML_Phase_512 Report Figure Execution and Upload

The first report/diagnostic step from the boundary-focused workflow was
implemented for the returned 512-point warm-start run.

New script:

```text
scripts/plot_cumulative_active_points.py
```

The script reads a run directory and generates cumulative active-learning
point diagnostics:

```text
figures/active_boundary_loop_512x50_r0075_cumulative_selected_points.png
figures/active_boundary_loop_512x50_r0075_cumulative_accepted_points.png
figures/active_boundary_loop_512x50_r0075_cumulative_points_summary.json
```

For the current `ML_Phase_512` result, the cumulative summary is:

```text
selected points across recorded iterations: 1027
accepted/training-eligible points: 999
boundary-band normal points in accepted data: 125
rerun-required points: 28
```

The active-learning report builder and LaTeX template were updated so the
report includes the cumulative selected/accepted figure pair:

```text
ml_phase/report_builder.py
report/active_learning_phase_boundary_report.tex
hpc_upload_qdelta_20260513_120556/ML_Phase_512/reports/active_learning_phase_boundary_report.tex
hpc_upload_qdelta_20260513_120556/ML_Phase_512/reports/active_learning_phase_boundary_report.pdf
```

The report PDF was rebuilt with `pdflatex`; the final log contains no fatal
errors, LaTeX errors, undefined references, or overfull/underfull warnings from
the report content.

A package manifest was added inside the result directory:

```text
hpc_upload_qdelta_20260513_120556/ML_Phase_512/reports/package_manifest_20260513.md
```

The updated `ML_Phase_512` result directory was compressed as:

```text
hpc_upload_qdelta_20260513_120556_ML_Phase_512_checked_20260513.zip
```

Archive size:

```text
32731849 bytes
```

The archive was copied to the requested OneDrive location:

```text
E:/Onedrive/OneDrive - The University of Hong Kong - Connect/GBU_SC/Fu_FFLO/hpc_upload_qdelta_20260513_120556_ML_Phase_512_checked_20260513.zip
```

Follow-up correction:

```text
The Windows zip archive is not the preferred HPC transfer format because it
preserved Windows-style path separators and caused unzip issues on the cluster.
The result directory was repackaged with tar using Unix-style archive paths:

hpc_upload_qdelta_20260513_120556_ML_Phase_512_checked_20260513.tar.gz

This tar.gz archive was uploaded to the same OneDrive folder:

E:/Onedrive/OneDrive - The University of Hong Kong - Connect/GBU_SC/Fu_FFLO/hpc_upload_qdelta_20260513_120556_ML_Phase_512_checked_20260513.tar.gz
```

Current implementation status:

```text
Done:
    cumulative selected/accepted report figures for the current 512-run
    report PDF regenerated
    package manifest added
    result archive created and uploaded to OneDrive

Not yet done:
    checkpoint saving
    warm_start_weights continuous-learning mode
    parsing boundary-displacement diagnostics as a hard HPC-loop stop
```

## 20. 2026-05-14 Clean Warm-Start Runnable HPC Package

The previous `ML_Phase_512_checked` archives were result packages, not clean
runnable source packages. A new package was created specifically for starting a
fresh active-learning run from the warm-start exact BdG data.

New run wrapper:

```text
run_from_warmup_512.sh
```

New package README:

```text
hpc_run_readme.md
```

The wrapper starts from the warm-start exact dataset and unsets
`RESUME_DATASET`. Default run parameters:

```text
RUN_ID=active_boundary_warmup_512x50_hybrid_r00375
START_ITER=0
N_ITERS=50
POINTS_PER_ITER=512
WORLD_SIZE=8
N_ENSEMBLE=5
REG_EPOCHS=240
CLS_EPOCHS=240
BATCH_SIZE=512
BOUNDARY_REFINEMENT_MODE=hybrid
BOUNDARY_LOCAL_MIN_DIST=0.00375
```

Runnable archive:

```text
hpc_upload_qdelta_warmup_512x50_r0075_20260514.tar.gz
```

The archive contains source code, scripts, docs, report template, and the
warm-start `.npz`. It does not contain previous `ML_Phase_512` result outputs.

Uploaded location:

```text
E:/Onedrive/OneDrive - The University of Hong Kong - Connect/GBU_SC/Fu_FFLO/hpc_upload_qdelta_warmup_512x50_r0075_20260514.tar.gz
```

Cluster run command:

```bash
tar -xzf hpc_upload_qdelta_warmup_512x50_r0075_20260514.tar.gz
cd hpc_upload_qdelta_warmup_512x50_r0075_20260514
chmod +x run_from_warmup_512.sh
bash run_from_warmup_512.sh
```

Important caveat:

```text
The original 2026-05-14 clean package predated the boundary-local selection
implementation. The current source tree now contains hybrid boundary-local
selection, but a new tar.gz should be built before uploading this revised code
to the cluster. Checkpoint warm-start training is still not implemented.
```

## 21. 2026-05-14 Report Note: Losses vs Acquisition Function

The academic note was updated to clarify the distinction between the two MLP
training losses and the acquisition function:

```text
report_active_learning_r0015_note/active_learning_r0015_note.tex
report_active_learning_r0015_note/active_learning_r0015_note.pdf
```

Added content:

```text
1. Regression MSE loss and classification cross-entropy loss are optimized
   neural-network training objectives.
2. The acquisition function is evaluated after training and ranks uncomputed
   candidate points; it does not update neural-network weights.
3. The acquisition score components were explained:
   classifier uncertainty, regression uncertainty, Delta-boundary proximity,
   q-boundary proximity, eta-zero proximity, gradient score, diversity score,
   q-window risk, Delta-refinement risk, and extrapolation risk.
4. A table of current default acquisition weights was added:
   w_cls_uncertainty=1.0,
   w_reg_uncertainty=0.8,
   w_delta_boundary=1.0,
   w_q_boundary=1.0,
   w_eta_boundary=0.7,
   w_gradient=0.7,
   w_diversity=0.3,
   w_q_edge_risk=0.8,
   w_delta_refine_risk=0.8,
   w_extrapolation=0.4.
5. The note states explicitly that these weights are fixed hyperparameters in
   the current implementation, not trainable neural-network parameters.
```

The note PDF was rebuilt with `pdflatex`; the final log has no fatal errors,
LaTeX errors, undefined references, or overfull/underfull warnings from the
updated content.

## 22. 2026-05-14 Candidate Grid and Acquisition Q&A

The report Q&A memory was expanded with explanations of the active-learning
candidate grid and selection logic:

```text
docs/report_qa/20260513_ml_training_active_learning_qa.md
```

Added Q&A topics:

```text
1. difference between ML training losses and the acquisition function;
2. why acquisition is a post-training candidate-ranking rule rather than a
   trainable loss;
3. whether active-learning point selection is random;
4. how the finite dense candidate grid is generated;
5. why predicting all candidate-grid points with the MLP is cheap compared
   with exact BdG calculations;
6. how boundary-oriented selection arises from acquisition terms, not from a
   direct boundary-position training loss.
```

Important recorded details:

```text
candidate grid:
    n_kt_candidates = 241
    n_ja_candidates = 321
    total candidate coordinates = 77361

candidate mask:
    kT >= 0
    JA >= JA_min
    JA <= boundary_ja(kT) + finite_t_band_width

finite_t_band_width:
    0.08

selected exact batch:
    earlier runs used 128 points/iteration
    later production configuration uses 512 points/iteration
```

## 23. 2026-05-14 Report Note: Candidate Grid and Selection Logic

Q&A material about candidate grids and selection logic was incorporated into
the academic note:

```text
report_active_learning_r0015_note/active_learning_r0015_note.tex
report_active_learning_r0015_note/active_learning_r0015_note.pdf
```

Added report content:

```text
1. the dense candidate grid is finite, not a continuous plane;
2. current candidate-grid dimensions are 241 x 321 = 77361 coordinates;
3. MLP prediction on all candidate-grid points is cheap because it is only a
   small-network forward pass, not exact BdG minimization;
4. only selected batches such as 128 or 512 points are sent to the exact BdG
   oracle;
5. a physics-aware finite-temperature mask restricts admissible candidates
   using JA <= JA_ref(kT) + 0.08;
6. selection is not random sampling, but a deterministic ranking procedure for
   fixed trained ensemble outputs and fixed acquisition configuration;
7. boundary-directed behavior comes from acquisition terms, not from direct
   boundary-position training loss.
```

The note PDF was rebuilt with `pdflatex`; the final log has no fatal errors,
LaTeX errors, undefined references, or overfull/underfull warnings from the
updated content.

## 24. 2026-05-14 Boundary-Local Hybrid Refinement Implementation

The boundary-focused active-learning plan is now partially implemented in the
selection layer.

Files changed:

```text
ml_phase/config.py
ml_phase/active_refine.py
ml_phase/extract_phase_boundaries.py
ml_phase/report_builder.py
report/active_learning_phase_boundary_report.tex
hpc_active_loop.sh
scripts/slurm_active_refine.sh
run_from_warmup_512.sh
hpc_run_readme.md
MODEL_SPEC.md
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/continuous_learning_plan.md
```

Implemented behavior:

```text
1. ActiveLearningConfig now supports boundary_refinement_mode = off | hybrid |
   local.
2. The Python default remains off for backward-compatible direct calls.
3. The current HPC wrapper defaults to hybrid for the next boundary-refinement
   production run.
4. Hybrid mode extracts exact boundary brackets each iteration using
   ml_phase.extract_phase_boundaries.
5. Boundary-local midpoint targets are selected for normal_sc and
   uniform_fflo brackets with boundary_local_min_dist = 0.00375.
6. Boundary midpoint targets are not rejected by the ordinary global
   existing_min_dist = 0.0075 rule; exact duplicate-coordinate exclusion still
   applies.
7. Remaining point budget is filled from high-JA q-risk and ordinary global
   acquisition candidates.
8. The selection layer writes selected_points_by_pool.csv,
   candidate_scores_boundary.csv, candidate_scores_global.csv, boundary
   summaries, and boundary displacement diagnostics where available.
9. The report builder now includes the refinement mode, boundary-local radius,
   selected counts by pool, boundary segment counts, and boundary displacement
   summary.
```

Default hybrid quotas:

```text
normal_sc: 0.44
uniform_fflo: 0.31
high_ja_q_risk: 0.125
global: 0.125
eta_zero: 0.0
strong_diode: 0.0
```

Validation performed:

```text
python -m compileall ml_phase scripts

python -m ml_phase.active_refine \
  --warm-start <warm-start npz> \
  --run-id boundary_refine_smoke2 \
  --mode hpc \
  --dry-run \
  --iterations 1 \
  --points-per-iter 16 \
  --world-size 2 \
  --n-ensemble 1 \
  --reg-epochs 1 \
  --cls-epochs 1 \
  --batch-size 4096 \
  --boundary-refinement-mode hybrid \
  --output-root ML_Phase_boundary_smoke2

python -m ml_phase.extract_phase_boundaries \
  --dataset ML_Phase_boundary_smoke2/active_runs/boundary_refine_smoke2/dataset_iter000.npz \
  --output-dir ML_Phase_boundary_smoke2/active_runs/boundary_refine_smoke2/iter000/boundaries_check \
  --kt-bin-width 0.005 \
  --max-local-spacing 0.035 \
  --max-refinement-points 16 \
  --output-root ML_Phase_boundary_smoke2

python -m ml_phase.report_builder \
  --run-id boundary_refine_smoke2 \
  --run-root ML_Phase_boundary_smoke2/active_runs \
  --output ML_Phase_boundary_smoke2/reports/active_learning_phase_boundary_report.tex

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory ML_Phase_boundary_smoke2/reports \
  ML_Phase_boundary_smoke2/reports/active_learning_phase_boundary_report.tex
```

Smoke-run result:

```text
points_per_iter = 16
selected_by_pool: boundary=12, high_ja_q_risk=2, global=2
selected_by_boundary_type: normal_sc=8, uniform_fflo=4
boundary segments extracted from warm-start data:
    eta_zero=5626
    normal_sc=88
    strong_diode=52
    uniform_fflo=94
standalone boundary extractor targeted points with max_refinement_points=16:
    n_targeted_refinement_points=16
```

Known unresolved issues:

```text
1. Boundary displacement diagnostics are available from iteration 1 onward,
   because iteration 0 has no previous boundary set for comparison.
2. The shell scripts could not be syntax-checked locally because WSL/bash is
   unavailable on this Windows machine.
3. The HPC loop still uses new-unique-sample early stopping as the hard stop;
   boundary displacement is recorded for diagnostics but is not yet parsed by
   hpc_active_loop.sh as a hard stop condition.
4. Checkpoint-based continuous learning remains unimplemented.
```

## 25. 2026-05-14 Clean Boundary-Hybrid Warm-Start Package

A new clean runnable HPC archive was created for the boundary-local hybrid
active-learning code.

Archive:

```text
hpc_upload_qdelta_warmup_512x50_hybrid_r00375_20260514.tar.gz
```

Top-level extracted directory:

```text
hpc_upload_qdelta_warmup_512x50_hybrid_r00375_20260514/
```

Default run behavior:

```text
RUN_ID=active_boundary_warmup_512x50_hybrid_r00375
START_ITER=0
N_ITERS=50
POINTS_PER_ITER=512
WORLD_SIZE=8
BOUNDARY_REFINEMENT_MODE=hybrid
BOUNDARY_LOCAL_MIN_DIST=0.00375
```

The wrapper unsets `RESUME_DATASET`, so the run starts from the warm-start
exact BdG dataset rather than previous 128-point or 512-point active-learning
outputs.

Cluster commands:

```bash
tar -xzf hpc_upload_qdelta_warmup_512x50_hybrid_r00375_20260514.tar.gz
cd hpc_upload_qdelta_warmup_512x50_hybrid_r00375_20260514
bash run_from_warmup_512.sh
```

Packaging checks:

```text
1. tar listing confirms the archive has a single top-level directory.
2. run_from_warmup_512.sh inside the archive sets POINTS_PER_ITER=512,
   START_ITER=0, WORLD_SIZE=8, and BOUNDARY_REFINEMENT_MODE=hybrid.
3. hpc_active_loop.sh inside the archive defaults to N_ITERS=50 and
   POINTS_PER_ITER=512.
4. The warm-start npz is included.
5. The archive listing did not contain previous ML_Phase_512 results, smoke
   outputs, active_runs/iter outputs, dataset_iter001+ files, exact_shard
   files, or exact_merged files.
```

## 26. 2026-05-14 Q&A: Acquisition vs Boundary Midpoint Refinement

The report Q&A memory was expanded:

```text
docs/report_qa/20260513_ml_training_active_learning_qa.md
```

Added explanation:

```text
1. Manual hybrid quotas allocate exact-call budget among scientific tasks.
2. ML acquisition still ranks dense-grid high-JA q-risk and global exploration
   candidates.
3. Boundary-local midpoint refinement shrinks already confirmed exact
   phase-boundary brackets and is not meant to replace ML exploration.
4. Acquisition alone is useful for discovery, but it does not guarantee
   bisection-like reduction of a specific exact boundary bracket.
5. Raw midpoint targets are a conservative first local-refinement step; future
   improvements may use boundary-normal or local-curve-aware targets when
   bracket geometry is sparse or awkward.
```

## 27. 2026-05-14 Active-Learning Workflow Web Draft

A static explanatory webpage was added to clarify the current active-learning
logic before further algorithmic changes:

```text
docs/active_learning_workflow_web/index.html
docs/active_learning_workflow_web/styles.css
```

Purpose:

```text
1. Present the full active-learning loop from warm-start exact data through
   ML training, point selection, exact BdG oracle, quality gate, append, and
   retraining.
2. Separate dense-grid ML acquisition from boundary-local midpoint refinement.
3. State that dense-grid acquisition is based on trained MLP predictions on
   the finite 241 x 321 candidate grid.
4. State that boundary-local midpoints are based on accepted exact brackets,
   not ML-predicted phase labels.
5. Show the current manual hybrid quotas:
   normal_sc=44%, uniform_fflo=31%, high_ja_q_risk=12.5%,
   global=12.5%, eta_zero=0%, strong_diode=0%.
6. Record current limits: boundary displacement is diagnostic only, automatic
   multi-stage radius shrinking is not implemented, and checkpoint
   warm-start training remains unimplemented.
```

This is an explanatory draft for project discussion. It does not change
physical definitions, phase thresholds, active-learning code, HPC packaging, or
canonical model/numerics decisions.

Follow-up revisions on 2026-05-14:

```text
1. AGENTS.md now requires project webpages, reports, slides, workflow diagrams,
   and other explanatory material to write mathematical formulas in LaTeX
   notation rather than ASCII pseudo-formulas.
2. The workflow webpage now loads MathJax and uses LaTeX notation for the BdG
   Hamiltonian, free-energy minimization, MLP losses, acquisition score, and
   midpoint definition.
3. A separate warm-up exact BdG calculation section was added. It explains the
   Hamiltonian, the scan over Delta and q, the current warm-up grid
   dimensions, and the rough scan complexity
   N_kT N_JA N_Delta N_q N_k d^3.
4. The ML section now explains in more detail how trained MLP predictions are
   converted into acquisition-score terms, then masked, sorted, and selected.
5. The wording around fixed hybrid quotas was rewritten to say directly that
   the fractions decide how many of the 512 selected points come from each
   pool; they are not learned during neural-network training.
```

Strategy-rethink update on 2026-05-14:

```text
1. The workflow webpage now marks boundary-local midpoint point selection as a
   legacy/current-code path to remove rather than the desired final strategy.
2. The "What Is Not Yet Automatic" section was replaced by a "Next to do"
   section focused on replacing the selection logic.
3. The page now states that midpoint selection conflicts with the intended
   active-learning goal when it consumes a large fixed fraction of exact-call
   budget, because it is geometric bisection rather than ML-guided selection.
4. The role of radius was reframed as sampling-density and redundancy control,
   not as a physical convergence criterion.
5. Four executable redesign options were added:
   pure ML acquisition with adaptive radius,
   learned boundary-probability field,
   learned utility/gain model,
   and batch-level information-diversity selection.
6. The same strategy discussion was recorded for report reuse in
   docs/report_qa/20260514_active_learning_selection_strategy_rethink.md.
```

## 28. 2026-05-14 Acquisition-Only Selection Rewrite

The midpoint-removal reconstruction plan has been implemented in the active
learning selection path.

Major files changed:

```text
ml_phase/acquisition.py
ml_phase/active_refine.py
ml_phase/config.py
ml_phase/extract_phase_boundaries.py
ml_phase/report_builder.py
hpc_active_loop.sh
run_from_warmup_512.sh
scripts/slurm_active_refine.sh
scripts/dev_check_acquisition_only.py
MODEL_SPEC.md
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/report_qa/20260514_active_learning_selection_strategy_rethink.md
docs/active_learning_workflow_web/index.html
report/active_learning_phase_boundary_report.tex
hpc_run_readme.md
```

Important implementation decisions:

```text
1. selected_points.csv is now generated only from dense-grid ML acquisition.
2. boundary_refinement_mode=diagnostic extracts exact boundary segments for
   diagnostics but does not generate midpoint candidates.
3. legacy boundary_refinement_mode=hybrid and local now raise:
   "Midpoint-based selection has been disabled. Use ML-guided acquisition
   instead."
4. exact duplicate coordinates remain hard-excluded by the 4-decimal rounded
   key rule.
5. the previous rigid existing-distance exclusion has been replaced in the
   main selection score by a soft observation-repulsion factor R_obs.
6. within-batch diversity is now a soft batch-repulsion factor R_batch rather
   than a hard minimum-distance fallback.
7. response-boundary terms are computed and reported as A_response diagnostics
   but are not included in A0_main by default.
```

The corrected acquisition score is organized as:

```text
A_phase:
    classifier uncertainty mix
    regression uncertainty for Delta and q
    Delta-boundary proximity
    superconducting-gated q-boundary proximity
    Delta/q gradient score

A_numerical:
    superconducting-gated q-window risk

A_explore:
    extrapolation risk multiplied by uncertainty

A_response:
    eta-zero proximity, eta gradient, and response uncertainty diagnostics

A0_main = A_phase + A_numerical + A_explore
```

Validation performed:

```text
python -m compileall ml_phase scripts

python scripts/dev_check_acquisition_only.py

python -m ml_phase.active_refine \
  --warm-start eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz \
  --run-id acquisition_only_smoke \
  --mode hpc \
  --dry-run \
  --iterations 1 \
  --points-per-iter 16 \
  --world-size 2 \
  --n-ensemble 1 \
  --reg-epochs 1 \
  --cls-epochs 1 \
  --batch-size 4096 \
  --boundary-refinement-mode diagnostic \
  --output-root ML_Phase_acquisition_only_smoke

python -m ml_phase.active_refine ... --boundary-refinement-mode hybrid

python -m ml_phase.report_builder \
  --run-id acquisition_only_smoke \
  --run-root ML_Phase_acquisition_only_smoke/active_runs \
  --output ML_Phase_acquisition_only_smoke/reports/active_learning_phase_boundary_report.tex

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory ML_Phase_acquisition_only_smoke/reports \
  ML_Phase_acquisition_only_smoke/reports/active_learning_phase_boundary_report.tex
```

Smoke-run result:

```text
selected points: 16
selection_source: acquisition for all selected rows
selected_by_pool: acquisition=16
n_targeted_refinement_points: 0
boundary segments still extracted from warm-start data:
    eta_zero=5626
    normal_sc=88
    strong_diode=52
    uniform_fflo=94
legacy hybrid mode rejected with the intended ValueError
```

Current project state:

```text
The next warm-start 512-point active-learning package should use
RUN_ID=active_boundary_warmup_512x50_acquisition_only and
BOUNDARY_REFINEMENT_MODE=diagnostic. It will start from warm-up exact data,
train on the cumulative exact dataset each iteration, select all new exact
points by ML-guided acquisition, run the exact BdG oracle, quality-gate the
results, append training-eligible points, and repeat.
```

Known unresolved issues:

```text
1. The new acquisition-only strategy still uses hand-chosen component weights.
2. The utility/gain model option has not been implemented.
3. The boundary-displacement diagnostic is not yet parsed as a hard HPC loop
   stopping criterion.
4. Checkpoint or warm-start neural-network training remains unimplemented.
5. The legacy hybrid/local code path has been rejected at runtime but older
   documentation sections remain as historical, superseded decisions.
```

Next recommended steps:

```text
1. Package a clean acquisition-only warm-start 512x50 HPC tarball.
2. Run a short cluster smoke test before a full 50-loop job.
3. Compare selected-point maps against the previous hybrid midpoint run.
4. Track duplicate rate, appended unique points, boundary displacement, and
   validation metrics per exact call.
5. Consider a learned utility/gain model only after collecting enough
   acquisition-only iteration logs.
```

## 29. 2026-05-14 StopController for Acquisition-Only Loop

The acquisition-only loop now has a convergence-oriented stop controller.

Major files changed:

```text
ml_phase/stop_controller.py
ml_phase/active_refine.py
hpc_active_loop.sh
scripts/dev_check_stop_controller.py
Full_reconstruct_plan.md
MODEL_SPEC.md
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/report_qa/20260514_active_learning_stop_controller.md
hpc_run_readme.md
```

Important implementation decisions:

```text
1. The selector may keep finding soft-repulsion candidates, so candidate
   availability is no longer the main stop criterion.
2. Each candidate-generation iteration writes
   monitor_predictions_iterXXX.npz containing the fixed monitor grid,
   predicted phase map, candidate mask, A0_main, and corrected score.
3. selected_points_by_pool.csv now records predicted_phase_before_exact for
   every selected point.
4. ml_phase.stop_controller runs after exact shard merge and trusted append in
   hpc_active_loop.sh.
5. Stop metrics are written to iterXXX/stop_metrics_iterXXX.json,
   stop_state.json, and stop_metrics_history.json.
6. Eta-zero response boundaries and topology are excluded from the current
   main stop rule.
7. The convergence rule requires at least five of seven conditions to pass,
   with q-edge/rerun and main-boundary coverage as mandatory gates.
```

Stop metrics:

```text
phase_map_change
boundary_shift_normal_sc
boundary_shift_uniform_fflo
label_surprise_rate
selected_A0_ratio
q_edge_trigger_rate
rerun_required_rate
boundary_coverage_p95
```

Default stop configuration:

```text
min_iterations = 5
patience = 4
max_iterations = 50, or END_ITER in hpc_active_loop.sh
map_tol = 0.002
boundary_shift_tol = 1.0 * dense_grid_spacing_norm
surprise_tol = 0.05
selected_A0_ratio_tol = 0.15
qedge_rate_tol = 0.01
rerun_rate_tol = 0.01
coverage_tol = 1.5 * dense_grid_spacing_norm
```

Validation performed:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_acquisition_only.py
python scripts/dev_check_stop_controller.py

python -m ml_phase.active_refine \
  --warm-start eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz \
  --run-id stop_controller_smoke \
  --mode hpc \
  --dry-run \
  --iterations 1 \
  --points-per-iter 16 \
  --world-size 2 \
  --n-ensemble 1 \
  --reg-epochs 1 \
  --cls-epochs 1 \
  --batch-size 4096 \
  --boundary-refinement-mode diagnostic \
  --output-root ML_Phase_stop_controller_smoke
```

Smoke-run result:

```text
monitor_predictions_iter000.npz was written with 77361 monitor-grid points.
selected_points_by_pool.csv includes predicted_phase_before_exact.
The toy stop-controller tests verify:
    stable main boundaries stop after patience,
    selectable candidates do not prevent convergence stop,
    eta-only response changes do not block stop,
    high q-edge trigger rate prevents stop,
    poor boundary coverage prevents stop.
```

Known unresolved issues:

```text
1. The stop controller is integrated into the HPC loop after exact merge and
   append. Local non-HPC active_refine mode still mainly serves smoke testing.
2. Boundary shift is currently measured on predicted monitor-grid phase
   boundaries. Exact extracted boundary diagnostics are still saved
   separately.
3. The selected_A0 baseline is based on early selected A0 means; future work
   may replace this with a calibrated expected-gain model.
```

## 30. 2026-05-14 Warm-Up 512x50 Acquisition-Only Stop Package

A clean runnable HPC package was prepared for the current recommended
production run.

Archive:

```text
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514.tar.gz
```

Top-level extracted directory:

```text
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/
```

Uploaded copy:

```text
E:\Onedrive\OneDrive - The University of Hong Kong - Connect\GBU_SC\Fu_FFLO\hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514.tar.gz
```

Default run behavior:

```text
RUN_ID=active_boundary_warmup_512x50_acquisition_only
START_ITER=0
N_ITERS=50
POINTS_PER_ITER=512
WORLD_SIZE=8
BOUNDARY_REFINEMENT_MODE=diagnostic
ENABLE_STOP_CONTROLLER=1
STOP_MIN_ITERATIONS=5
STOP_PATIENCE=4
```

The wrapper unsets `RESUME_DATASET`, so the run starts from the warm-start
exact BdG data rather than previous 128-point or 512-point active-learning
outputs. All selected exact-call points come from ML-guided dense-grid
acquisition; boundary extraction is diagnostic only.

Package contents:

```text
ml_phase/ source files
scripts/ source files
docs/ memory and explanatory docs
report/active_learning_phase_boundary_report.tex
warm-start exact npz
hpc_active_loop.sh
run_from_warmup_512.sh
run_from_warmup_512_background.sh
hpc_run_readme.md
core exact-solver and plotting source files
```

Packaging checks:

```text
1. run_from_warmup_512.sh inside the archive sets START_ITER=0,
   POINTS_PER_ITER=512, N_ITERS=50, WORLD_SIZE=8, and
   BOUNDARY_REFINEMENT_MODE=diagnostic.
2. hpc_active_loop.sh inside the archive defaults to ENABLE_STOP_CONTROLLER=1,
   STOP_MIN_ITERATIONS=5, and STOP_PATIENCE=4.
3. The warm-start npz is included.
4. The archive listing did not contain previous active_runs outputs,
   dataset_iter001+ files, exact_shard files, exact_merged files, `ML_Phase`
   result directories, `__pycache__`, or `id_rsa`.
5. A background wrapper was added so the cluster-side controlling loop can be
   launched with `nohup` and monitored through a pid file and log file.
```

## 31. 2026-05-15 r=0.015 Note Uses Vector PDF Figures

The English research note now inserts vector PDF figures instead of PNG
figures where matching PDF sources exist.

Files changed:

```text
report_active_learning_r0015_note/active_learning_r0015_note.tex
report_active_learning_r0015_note/figures/fig01_original_exact_phase_diagram.pdf
report_active_learning_r0015_note/figures/fig02_active_learning_main_boundaries.pdf
report_active_learning_r0015_note/figures/fig03_combined_eta_phase_diagram.pdf
report_active_learning_r0015_note/figures/fig04_active_learning_workflow.pdf
report_active_learning_r0015_note/figures/fig05_ml_training_architecture.pdf
```

Implementation:

```text
1. Copied the existing PDF counterparts of the five report figures into the
   note's local figures directory with matching fig01--fig05 names.
2. Updated all five \includegraphics calls from .png to .pdf.
3. Kept the older PNG files as local fallback assets, but they are no longer
   used by the LaTeX note.
```

Validation:

```text
powershell -ExecutionPolicy Bypass -File .\build_note.ps1
pdftoppm -png -r 120 -f 4 -l 6 report_active_learning_r0015_note/active_learning_r0015_note.pdf tmp/pdfs/active_learning_r0015_note_fig_pages/page
pdftoppm -png -r 120 -f 10 -l 11 report_active_learning_r0015_note/active_learning_r0015_note.pdf tmp/pdfs/active_learning_r0015_note_fig_pages/page
```

The LaTeX log confirms that all five figures are read as PDF graphics, and
rendered preview pages 4, 5, 6, 10, and 11 show the figures without missing
assets or obvious clipping.

## 32. 2026-05-16 512x50 Acquisition-Only Result Check and Report Repair

The new cluster output under

```text
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ML_Phase_512_AL/
```

was inspected.

Data status:

```text
run_id = active_boundary_warmup_512x50_acquisition_only
iterations present = iter000 through iter049
selected points = 512 per iteration
merged exact points = 512 per iteration
accepted/training exact points across iterations = 24507
rerun-required exact points across iterations = 1093
final reported exact samples = 45082
final stop reason = max_iterations
```

The run completed the configured 50 iterations, but it did not satisfy the
convergence stop rule.  The final StopController state is

```text
stop_reason = max_iterations
convergence_pass = false
passed conditions = 5 of 7
mandatory gates passed = false
rerun_required_rate = 0.056640625 > 0.01
selected_A0_ratio = 0.440246967 > 0.15
```

Thus the main predicted phase boundaries are largely stable, but the
quality-gate/rerun rate and selected-acquisition-score saturation criteria
still indicate useful or unresolved exact work.

Local data caveat and repair:

```text
dataset_iter000.npz through dataset_iter050.npz in the local copied result are
not readable by numpy and raise BadZipFile.  The matching CSV files are present
and readable, and all per-iteration exact shard, exact_merged, exact_training,
and exact_trusted npz files were checked as readable.
```

The local `dataset_iterXXX.npz` files were rebuilt from their matching CSV
files, restoring the standard active-learning arrays:

```text
x
y_reg
y_phase
y_eta_sign
y_strong_diode
```

After rebuilding, all 51 dataset npz files load successfully, with
`dataset_iter050.npz` containing 45082 samples.

Report repair:

```text
scripts/plot_cumulative_active_points.py
ml_phase/report_builder.py
```

were updated so cumulative plotting and report generation fall back to
`dataset_iterXXX.csv` when the matching `dataset_iterXXX.npz` file is
unreadable.  The cumulative selected-points plot was also changed to use a
continuous iteration colorbar instead of one legend entry per iteration, so
the 50-loop result remains readable.

The missing cumulative figures were regenerated:

```text
ML_Phase_512_AL/figures/active_boundary_warmup_512x50_acquisition_only_cumulative_selected_points.png
ML_Phase_512_AL/figures/active_boundary_warmup_512x50_acquisition_only_cumulative_accepted_points.png
ML_Phase_512_AL/figures/active_boundary_warmup_512x50_acquisition_only_cumulative_points_summary.json
```

The generated LaTeX report now compiles successfully:

```text
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ML_Phase_512_AL/reports/active_learning_phase_boundary_report.pdf
```

Validation commands:

```text
python -m compileall ml_phase scripts
python scripts/plot_cumulative_active_points.py --run-dir <ML_Phase_512_AL active run> --output-dir <ML_Phase_512_AL figures>
python -m ml_phase.report_builder --run-id active_boundary_warmup_512x50_acquisition_only --run-root <ML_Phase_512_AL active_runs> --output <ML_Phase_512_AL report tex>
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
pdftoppm -png -r 120 -f 3 -l 6 <report pdf> tmp/pdfs/ml_phase_512_al_report/page
```

Rendered preview pages 4--6 were inspected and show the summary tables and
figures without missing-image errors or clipping.

## 33. 2026-05-16 Display Sync for Acquisition Report and Trace Fields

The active-learning report and trace outputs were synchronized with the current
acquisition-only logic.  This was a display-layer repair only: no physics
definition, acquisition score formula, exact-oracle rule, or stop criterion was
changed.

Files changed:

```text
report/active_learning_phase_boundary_report.tex
ml_phase/report_builder.py
ml_phase/active_refine.py
ml_phase/acquisition.py
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/report/active_learning_phase_boundary_report.tex
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ml_phase/report_builder.py
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ml_phase/active_refine.py
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ml_phase/acquisition.py
```

Implementation decisions:

```text
1. The old hard-coded report formulas
   U_c(x), U_r(x), and S(x)
   were replaced by the current decomposition
   A_phase + A_numerical + A_explore = A0_main,
   with A_response shown separately as a diagnostic-only score.
2. The report now explains the final greedy ranking score
   A_select = A0_main * R_obs * R_batch
   using LaTeX notation, consistent with the current code path.
3. report_builder now reads the latest stop_metrics_iterXXX.json and
   stop_state.json so the PDF records stop status, stop reason, passed
   conditions, patience counter, and the main convergence metrics.
4. candidate_scores.csv, monitor_predictions_iterXXX.npz, and future
   selected_points_by_pool.csv outputs now expose synchronized field names and
   aliases such as
   cls_uncertainty_mix, B_q_SC, E_q_SC, E_ext_uncertain, P_SC,
   U_delta, U_q, U_eta, U_ic_plus, and U_ic_minus.
5. The same display fixes were mirrored into the packaged HPC upload copy so
   local code and packaged report logic do not drift apart again.
```

Current artifact status:

```text
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ML_Phase_512_AL/reports/active_learning_phase_boundary_report.pdf
```

now compiles with the updated acquisition description and a dedicated stop
diagnostics section.  The current report still spans eight pages because of the
figure block near the end, but the previous wrong acquisition formula is no
longer present.

Validation:

```text
python -m compileall ml_phase hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ml_phase
python -m ml_phase.report_builder --run-id active_boundary_warmup_512x50_acquisition_only --run-root ML_Phase_512_AL/active_runs --template report/active_learning_phase_boundary_report.tex --output ML_Phase_512_AL/reports/active_learning_phase_boundary_report.tex
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
rg -n "A_{0,\\mathrm{main}}|Stop reason: max_iterations" ML_Phase_512_AL/reports/active_learning_phase_boundary_report.tex
```

Known remaining limitation:

```text
Historical per-iteration CSV outputs produced before this display-sync patch
are not backfilled automatically.  The new trace fields are guaranteed for
future outputs generated by the updated code, while the already completed
2026-05-14 run mainly benefits from the repaired report text and future
reproducibility.
```

## 34. 2026-05-17 Active-Learning Report Layout Polish

The rebuilt 512x50 acquisition-only report was checked visually after the
display-sync repair.

Files changed:

```text
ml_phase/report_builder.py
hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ml_phase/report_builder.py
docs/PROJECT_SUMMARY.md
```

Summary:

```text
1. Stop diagnostics now read numerical values from the nested
   stop_metrics_iterXXX.json["metrics"] object instead of printing N/A.
2. Stop condition and metric summaries now use readable labels such as
   "normal/SC shift" and "q-edge/rerun rates" instead of raw underscored keys.
3. The packaged report builder now uses the same dataset npz-or-csv fallback
   as the root report builder.
4. The packaged report builder now compresses long per-iteration tables to the
   first five and last eight rows, with an omitted-iteration marker, preventing
   the 50-row table from crowding the page.
```

Validation:

```text
python -m compileall ml_phase hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ml_phase

python -m ml_phase.report_builder \
  --run-id active_boundary_warmup_512x50_acquisition_only \
  --run-root ML_Phase_512_AL/active_runs \
  --template report/active_learning_phase_boundary_report.tex \
  --output ML_Phase_512_AL/reports/active_learning_phase_boundary_report.tex

pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex

pdftoppm -png -r 120 -f 1 -l 7 \
  hpc_upload_qdelta_warmup_512x50_acquisition_only_stop_20260514/ML_Phase_512_AL/reports/active_learning_phase_boundary_report.pdf \
  tmp/pdfs/ml_phase_512_al_report_check/page
```

Current state:

```text
The package report now compiles to a seven-page PDF with the current
acquisition-only formula, a populated stop-diagnostics table, and a compact
iteration summary.  The LaTeX log no longer reports overfull or underfull
layout warnings for the checked build.
```

## 35. 2026-05-17 Discovery-Mode Active Learning

The active-learning workflow now supports discovery mode as a separate path
from warm-start refinement.

Files changed:

```text
ml_phase/config.py
ml_phase/acquisition.py
ml_phase/active_refine.py
ml_phase/stop_controller.py
ml_phase/report_builder.py
report/active_learning_phase_boundary_report.tex
hpc_active_loop.sh
scripts/slurm_active_refine.sh
run_from_warmup_512.sh
scripts/dev_check_acquisition_only.py
scripts/dev_check_stop_controller.py
scripts/dev_check_discovery_mode.py
MODEL_SPEC.md
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/report_qa/20260517_discovery_mode_active_learning.md
```

Implementation decisions:

```text
1. Added run_mode = discovery | refinement.
2. Added candidate_domain_mode = full | prior_band.
3. discovery mode starts from random_grid exact seed points and forbids
   finite_t_band_width / finite-T prior candidate-band pruning.
4. refinement mode keeps the warm-start or resume-dataset workflow and may
   explicitly use candidate_domain_mode=prior_band.
5. discovery iter000 writes random seed selected points with
   selection_source=random_seed and does not train ML before exact labels
   exist.
6. later discovery iterations use stochastic acquisition sampling with dynamic
   R_batch, not deterministic fixed-region quotas.
7. A0_main remains the current acquisition decomposition:
   A_phase + A_numerical + A_explore.
8. StopController now uses five main phase-map / thermodynamic-boundary
   conditions and requires at least four to pass over patience. selected_A0,
   q-edge, and rerun rates remain diagnostics and cleanup warnings.
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_discovery_mode.py
python scripts/dev_check_acquisition_only.py
python scripts/dev_check_stop_controller.py
python scripts/dev_check_qdelta_logic.py

python -m ml_phase.active_refine \
  --run-id discovery_seed_smoke \
  --mode hpc \
  --iterations 1 \
  --initial-seed-size 8 \
  --batch-size-max 4 \
  --world-size 2 \
  --output-root ML_Phase_discovery_smoke \
  --n-ensemble 1 \
  --reg-epochs 1 \
  --cls-epochs 1 \
  --batch-size 16

python -m ml_phase.active_refine \
  --run-id refinement_prior_band_smoke \
  --run-mode refinement \
  --candidate-domain-mode prior_band \
  --finite-t-band-width 0.08 \
  --selection-mode topk \
  --warm-start <warm-start npz> \
  --mode hpc \
  --iterations 1 \
  --points-per-iter 8 \
  --world-size 2 \
  --output-root ML_Phase_discovery_smoke \
  --n-ensemble 1 \
  --reg-epochs 1 \
  --cls-epochs 1 \
  --batch-size 512 \
  --boundary-refinement-mode diagnostic
```

Current state:

```text
Discovery mode can generate an initial random exact-call seed batch without a
warm-start dataset. In the smoke run, all 77361 dense-grid points were
available under candidate_domain_mode=full, and the selected random seed
included high-JA points.

The old warm-start refinement path remains available through
run_mode=refinement, candidate_domain_mode=prior_band, finite_t_band_width=0.08.
```

Known unresolved issues:

```text
1. Multi-seed aggregation is represented in the configuration design but not
   yet implemented as an automatic outer loop.
2. Hidden ground truth is recorded as configuration metadata only; a dedicated
   offline discovery benchmark evaluator still needs to be added.
3. Existing HPC upload archives are not automatically regenerated by this code
   change.
```

## 36. 2026-05-17 Discovery-Mode HPC Package Wrapper

Files changed:

```text
run_discovery_512x50.sh
run_discovery_512x50_background.sh
package_hpc_upload.ps1
hpc_run_readme.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a one-command discovery-mode HPC launcher and a nohup background wrapper.
The launcher starts from random exact seed points, uses the full rectangular
candidate domain, disables warm-start input and finite-T prior-band pruning,
and defaults to INITIAL_SEED_SIZE=512, BATCH_SIZE_MAX=256, N_ITERS=100,
WORLD_SIZE=8, and SELECTION_MODE=stochastic.

The packaging script now creates a discovery archive named
hpc_upload_qdelta_discovery_512seed_256x50_<timestamp>.tar.gz and does not copy
the warm-start exact dataset into the bundle. The discovery package includes
only the discovery launcher entry points to avoid accidentally starting the
older warm-start refinement workflow from this archive. The refinement launcher
remains available in the working tree through run_from_warmup_512.sh.
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_discovery_mode.py
python scripts/dev_check_acquisition_only.py
python scripts/dev_check_stop_controller.py
python scripts/dev_check_qdelta_logic.py
```

Known limitation:

```text
Local bash syntax checks could not be run on Windows because WSL has no
installed distribution in the current environment. The shell scripts are
intended to be executed on the Linux HPC login node.
```

## 37. 2026-05-19 HPC Discovery Orchestration Hardening

Files changed:

```text
hpc_active_loop.sh
scripts/recover_active_iter.sh
docs/PROJECT_SUMMARY.md
```

Summary:

```text
The active-learning HPC orchestration was hardened after a discovery run
stalled at iter016 even though Slurm array elements 51322_0..51322_7 had all
completed and all exact shard files were present.

hpc_active_loop.sh now requires an explicit RUN_ID in discovery mode, prints a
preflight summary before any sbatch submission, supports DRY_RUN=1, uses a
PROJECT_DIR/RUN_ID flock lock, records per-iteration status.json, skips already
completed candidate/exact/merge/append/stop stages on resume, and treats
N_ITERS as the number of iterations to run from START_ITER rather than as a
global total.

wait_for_job now checks both squeue and sacct. For Slurm array jobs it parses
array elements such as 51322_0..51322_7 while ignoring .batch/.extern records.
Exact-oracle waiting also uses shard-file completion as a safeguard when sacct
has no failure state but accounting is delayed.

scripts/recover_active_iter.sh was added to recover one completed exact-oracle
iteration from shard files without resubmitting exact BdG jobs or overwriting
an existing next dataset by default.
```

Operational notes:

```text
For the current discovery run, the intended resume shape is:

RUN_ID=active_boundary_discovery_512seed_256x50
START_ITER=17
N_ITERS=83
STOP_MAX_ITERATIONS=100

This means the next loop starts at iter017 and targets iter000..iter099 in
total. The wrapper should be run with nohup on the login node so that Slurm
subtasks can finish and the main loop can still perform merge, append, and
stop-controller steps.
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_discovery_mode.py
python scripts/dev_check_acquisition_only.py
python scripts/dev_check_stop_controller.py
python scripts/dev_check_qdelta_logic.py
```

Known limitation:

```text
Local shell syntax checks and fake Slurm tests could not be executed in the
current Windows environment because bash resolves to WSL and no WSL
distribution is installed. Run `bash -n hpc_active_loop.sh` and
`bash -n scripts/recover_active_iter.sh` on the Linux HPC login node before
resuming production.
```

## 38. 2026-05-19 Discovery Run Selection Diagnostics

Files changed:

```text
docs/report_qa/20260519_discovery_selection_diagnostics.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
The latest discovery-mode output under
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed
was inspected.  The correct run directory is
active_boundary_discovery_512seed_256x50.  A mistaken old-default run directory
also exists in the same output tree and should not be mixed into the analysis.

The discovery run stopped at iter065 with stop_reason =
converged_main_phase_boundaries and produced dataset_iter066.npz with 17061
exact samples.  The last selected batch contains 256 acquisition_stochastic
points, but its distance distribution is not more boundary-focused than a
random full-grid sample.  Only 44/256 selected points are within normalized
distance 0.05 of the extracted normal/SC or uniform-SC/FFLO boundaries.
```

Interpretation:

```text
The last batch looks random because the current A0_main score remains broad
and high away from the extracted thermodynamic boundaries, especially in
high-J_A normal regions.  The issue is therefore not the random initial seed,
but the current acquisition/sampling policy failing to localize late-stage
selection on the main phase boundaries.
```

Next recommended step:

```text
Add per-iteration selected-point distance-to-main-boundary diagnostics, then
revise discovery acquisition or stochastic sampling so that phase-boundary
focus increases after the global phase map stabilizes.
```

## 39. 2026-05-19 Discovery Exact Phase Map Report Figure

Files changed:

```text
scripts/plot_discovery_exact_phase_map.py
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/reports/active_learning_phase_boundary_report.tex
docs/PROJECT_SUMMARY.md
```

Generated outputs:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/figures/active_boundary_discovery_512seed_256x50_exact_phase_map.pdf
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/figures/active_boundary_discovery_512seed_256x50_exact_phase_map.png
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/figures/active_boundary_discovery_512seed_256x50_exact_phase_map.json
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/reports/active_learning_phase_boundary_report.pdf
```

Summary:

```text
Generated a color exact-data phase map from dataset_iter066.npz and inserted
the PDF figure into the discovery active-learning report.  The figure uses
exact y_phase labels for normal, uniform SC, and FFLO points, and overlays the
final extracted normal/SC and uniform-SC/FFLO boundary points from iter065.

The final exact dataset contains 17061 points:
normal = 13339
uniform SC = 282
FFLO = 3440
```

Validation:

```text
python scripts/plot_discovery_exact_phase_map.py ...
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
pdftoppm -png -r 140 -f 6 -l 6 active_learning_phase_boundary_report.pdf report_page
```

Visual check:

```text
Rendered page 6 of the report. The new exact phase map is visible, uses the
expected colors and boundary markers, and fits on the page.
```

## 40. 2026-05-19 Discovery Report Display and Metric Wording Fixes

Files changed:

```text
ml_phase/report_builder.py
ml_phase/plot_active_learning.py
report/active_learning_phase_boundary_report.tex
scripts/dev_check_report_builder.py
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/reports/active_learning_phase_boundary_report.tex
```

Generated outputs:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/figures/active_boundary_discovery_512seed_256x50_cumulative_progress.png
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/reports/active_learning_phase_boundary_report.pdf
```

Summary:

```text
The discovery report was corrected at the reporting layer only.  No acquisition,
selection, StopController, exact BdG oracle, or neural-network training logic
was changed.

The acquisition text now switches by selection_mode.  For stochastic discovery
runs it describes A_select as the sampling-weight base and reports the
probability p_i proportional to A_select^gamma, with dynamic R_batch updates
after each draw.  The old "final greedy selection score" wording is no longer
used for stochastic runs.

Iteration-count reporting now distinguishes the latest completed iteration,
the number of completed active-learning iterations, and the final dataset.
For the current run this is iter065, 66 completed iterations, and
dataset_iter066.npz.

The previous blank Figure 4 was caused by tiny placeholder PNGs being inserted
when cumulative figures were absent.  report_builder now generates a real
cumulative progress figure from selected_points, merge summaries, and dataset
sizes, or prints an explicit unavailable reason instead of a blank caption.

Boundary F1 remains N/A because hidden-ground-truth boundary evaluation has not
been implemented for this report.  The learning curve now omits the Boundary F1
legend whenever all boundary_f1 entries are missing or NaN.  The report also
explains that Section 8 diagnostic maximum boundary displacement is not the
same quantity as the StopController boundary-shift metric in Section 9.
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_report_builder.py
python -m ml_phase.report_builder --run-id active_boundary_discovery_512seed_256x50 --run-root hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/active_runs --template report/active_learning_phase_boundary_report.tex --output hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/reports/active_learning_phase_boundary_report.tex
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
pdftoppm -png -r 120 -f 8 -l 9 active_learning_phase_boundary_report.pdf report_check_page
```

Visual check:

```text
Rendered pages 8 and 9.  Figure 4 now shows the cumulative progress plot, and
Figure 5 no longer includes a Boundary F1 legend.
```

## 41. 2026-05-19 Discovery Active-Pool Acquisition Sampling

Files changed:

```text
ml_phase/config.py
ml_phase/acquisition.py
ml_phase/active_refine.py
ml_phase/plot_active_learning.py
ml_phase/report_builder.py
hpc_active_loop.sh
scripts/slurm_active_refine.sh
scripts/dev_check_discovery_mode.py
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Discovery-mode acquisition sampling was changed from broad corrected-score
sampling to A0_main-gated stochastic sampling.  The new logic first builds a
high-information active pool from A0_main using a quantile threshold and a
relative-to-p95 threshold, then samples only within that pool with weights
proportional to (A0_main * R_obs * R_batch)^gamma.

R_obs is now intentionally mild by default:
observation_repulsion_floor = 0.5 and observation_repulsion_length = 0.02.
It is a sampling-weight downweight near existing observations, not the main
criterion for whether a point is worth computing.  R_batch remains a stronger
within-batch anti-clustering factor.

Discovery batches are now adaptive.  Early iterations can relax the active-pool
quantile to satisfy a minimum exploratory batch, while later iterations do not
force selection to fill batch_size_max when the high-information pool is
smaller or exhausted.

Per-iteration diagnostics now include active-pool size, active-pool thresholds,
threshold relaxation, effective sample size, selected A0 concentration, R_obs
statistics, and selected-to-predicted-boundary distances versus a random
active-pool baseline.

Report plotting now overlays predicted boundaries on phase/acquisition maps and
can generate selection-source and selection-focus diagnostic figures when the
new diagnostics are available.
```

Why it matters:

```text
The previous discovery run's late selected points looked close to random
full-domain coverage.  The new policy keeps stochasticity but confines it to
ML/acquisition-defined high-information candidates, so randomness explores
within an informative pool rather than across the entire dense grid.
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_discovery_mode.py
python scripts/dev_check_acquisition_only.py
python scripts/dev_check_stop_controller.py
python scripts/dev_check_report_builder.py
python scripts/dev_check_qdelta_logic.py
```

Known unresolved issue:

```text
The new boundary-focus diagnostics will be fully visible only in newly produced
discovery runs. Existing reports generated before this change do not contain
the per-iteration active-pool and selected-to-boundary diagnostic fields.
```

## 42. 2026-05-19 Report Section for Exact-Oracle Numerical Safeguards

Files changed:

```text
report/active_learning_phase_boundary_report.tex
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/reports/active_learning_phase_boundary_report.tex
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a "Numerical Accuracy Safeguards" section to the active-learning report
template and regenerated the current discovery report.  The section clarifies
that the exact-data phase map is built from accumulated exact BdG results, not
from a pre-assumed phase boundary or ML-only prediction.

The new text explains the q-window refinement logic, including q-edge detection,
q-window expansion for superconducting points, q-unresolved/rerun-required
metadata, and why normal-state q is treated as not applicable.  It also explains
Delta refinement near the normal/SC boundary through the comparison between the
best positive-Delta state and the Delta=0 normal state, including the
finite-resolution boundary-band interpretation.
```

Validation:

```text
python -m ml_phase.report_builder --run-id active_boundary_discovery_512seed_256x50 --run-root hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/active_runs --template report/active_learning_phase_boundary_report.tex --output hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/reports/active_learning_phase_boundary_report.tex
python scripts/dev_check_report_builder.py
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
```

Current generated report:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260517_125005/ML_Phase_512_seed/reports/active_learning_phase_boundary_report.pdf
```

## 43. 2026-05-19 Fresh Discovery HPC Package After Active-Pool Sampling Update

Files changed:

```text
run_discovery_512x50.sh
hpc_run_readme.md
package_hpc_upload.ps1
docs/PROJECT_SUMMARY.md
```

Generated package:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535.tar.gz
```

Summary:

```text
Created a fresh clean runnable HPC package for starting discovery active
learning from scratch after the A0_main active-pool sampling update.  The
package does not include the previous ML_Phase_512_seed result directory or
warm-start exact data.

The discovery launcher now exposes the new defaults:
ACTIVE_POOL_QUANTILE=0.90
ACTIVE_POOL_REL_TO_P95=0.3
ACTIVE_POOL_MIN_QUANTILE=0.70
ACTIVE_SELECTION_MIN_ITERATIONS=5
SAMPLING_POWER=2.0
BATCH_SIZE_MIN_BEFORE_MIN_ITER=64
BATCH_SIZE_MIN_AFTER_MIN_ITER=0
N_ITERS=100
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_discovery_mode.py
python scripts/dev_check_acquisition_only.py
python scripts/dev_check_stop_controller.py
python scripts/dev_check_report_builder.py
python scripts/dev_check_qdelta_logic.py
./package_hpc_upload.ps1
tar -tzf hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535.tar.gz
```

Run command on HPC:

```text
tar -xzf hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535.tar.gz
cd hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535
chmod +x run_discovery_512x50.sh run_discovery_512x50_background.sh
bash run_discovery_512x50_background.sh
```

## 44. 2026-05-20 Discovery Run Analysis for 512 Seed / 256 Batch Active Pool

Files reviewed:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter
docs/report_qa/20260520_discovery_iter_analysis.md
```

Summary:

```text
Checked the latest discovery-mode active-learning output.  The run completed
iterations 0-49 and produced dataset_iter050.npz with 12970 exact samples.
It stopped by max_iterations rather than convergence.  The latest
StopController pass count was 2/5: label_surprise_rate and uniform/FFLO
boundary shift passed, while phase_map_change, normal/SC shift, and boundary
coverage did not all satisfy tolerance.

The exact oracle diagnostics in the last batch were clean for q-window and
rerun logic: q_edge_hit=0, q_expanded=0, q_unresolved=0, needs_rerun_exact=0.
The main remaining ambiguity was Delta boundary-band metadata.

The stochastic sampler is acquisition-biased but still not strongly boundary
focused.  The last active pool contained about 49k candidates, selected
A0_main mean was about 1.28 times the unseen mean, and selected boundary-band
fraction was about 7.0% versus a random baseline of about 4.7%.  This explains
why cumulative selected points still visually resemble broad stochastic
coverage.  The likely reason is that the active-pool OR rule with
active_pool_rel_to_p95=0.3 is too permissive.
```

Next recommended steps:

```text
Tighten active-pool construction before another full discovery run.  Candidate
options are: quantile-only gating, a stricter combination rule, increasing
active_pool_rel_to_p95 to roughly 0.7-0.8, raising active_pool_quantile to 0.95,
or increasing sampling_power only after the active pool is demonstrably narrow.
```

## 45. 2026-05-20 Discovery Default Iteration Count Increased to 100

Files changed:

```text
run_discovery_512x50.sh
hpc_active_loop.sh
hpc_run_readme.md
package_hpc_upload.ps1
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Changed the default active-learning iteration count from 50 to 100 for the
discovery launcher and the generic HPC active-loop fallback.  The run can still
be overridden with N_ITERS=<value>.  The README and packaging console message
were updated to advertise N_ITERS=100.
```

Validation:

```text
rg confirmed the current launcher/readme/package default is N_ITERS=100.
bash -n could not be run locally because this Windows environment has no WSL
distribution installed.
```

## 46. 2026-05-20 Selection Region and Component Diagnostics Added to Discovery Report

Files changed:

```text
ml_phase/report_builder.py
report/active_learning_phase_boundary_report.tex
scripts/dev_check_report_builder.py
docs/PROJECT_SUMMARY.md
```

Generated for the latest discovery result:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/active_runs/active_boundary_discovery_512seed_256x50/iter*/selection_region_diagnostics_iter*.json
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/active_runs/active_boundary_discovery_512seed_256x50/iter*/selection_region_diagnostics_iter*.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/figures/active_boundary_discovery_512seed_256x50_selection_region_fractions.png
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/figures/active_boundary_discovery_512seed_256x50_active_pool_region_fractions.png
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/figures/active_boundary_discovery_512seed_256x50_selection_score_concentration.png
hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/figures/active_boundary_discovery_512seed_256x50_iter049_selected_regions.png
```

Summary:

```text
Added report-time diagnostics that classify hard-unseen candidates, active-pool
candidates, selected points, and same-size random baseline samples by predicted
normal interior, SC interior, and main-boundary band.  Main boundaries include
only normal/SC and uniform-SC/FFLO boundaries; eta-zero and strong-diode
response boundaries are intentionally excluded.

The report now includes a "Selection Region and Component Diagnostics" section
with latest region-distribution and component-attribution tables, time-series
plots for selected/active-pool region fractions, score concentration diagnostics,
and a latest selected-by-region spatial plot.  The diagnostics are generated
from existing monitor_predictions_iterXXX.npz and selected_points_by_pool.csv
files; they do not alter acquisition, stochastic sampling, StopController, or
the exact BdG oracle.

For existing results where candidate_scores_global.csv is empty, hard-unseen
candidates are reconstructed from the dense grid and dataset_iterXXX exact
coordinates using the existing 4-decimal duplicate-key convention.
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_report_builder.py
python scripts/dev_check_discovery_mode.py
python -m ml_phase.report_builder --run-id active_boundary_discovery_512seed_256x50 --run-root hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/active_runs --template report/active_learning_phase_boundary_report.tex --output hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/reports/active_learning_phase_boundary_report.tex
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
```

Latest diagnostic result:

```text
For iter049, selected points are dominated by predicted normal interior:
selected normal interior fraction = 0.851562,
selected SC interior fraction = 0.078125,
selected boundary-band fraction = 0.0703125.

The same-size random baseline had boundary-band fraction = 0.0273438, so the
selection is acquisition-biased but still not strongly boundary-focused.  The
component table indicates selected normal-interior points have high A_phase and
a strong Delta-boundary contribution, suggesting that broad Delta-transition
tails or boundary-like Delta scores inside predicted normal regions should be
inspected before changing the acquisition formula.
```
## 47. 2026-05-20 Boundary-Focused Discovery Acquisition Sharpening

Files changed:

```text
ml_phase/config.py
ml_phase/acquisition.py
ml_phase/active_refine.py
ml_phase/report_builder.py
report/active_learning_phase_boundary_report.tex
scripts/slurm_active_refine.sh
scripts/dev_check_discovery_mode.py
hpc_active_loop.sh
run_discovery_512x50.sh
hpc_run_readme.md
package_hpc_upload.ps1
MODEL_SPEC.md
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/report_qa/20260520_phase_boundary_sharp_acquisition.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Implemented the Phase_boundary_sharp_plan.md engineering plan for discovery
active learning.  The exact BdG oracle, neural-network architecture,
phase-label definitions, StopController main convergence logic, warm-start
refinement path, midpoint selection state, and prior-band logic were not
changed.

The discovery acquisition now suppresses false deep-normal Delta-boundary
signals by replacing raw B_delta in A_phase with
B_delta_gated = B_delta_raw * U_NS, where U_NS = 4 * P_normal * P_SC.  The raw
and gated values, plus P_normal/P_uniform/P_FFLO/P_SC/U_NS/U_UF, are saved for
diagnostics.

The active pool is now built from A0_for_pool rather than raw A0_main.  The
default pool rule is max_threshold:
threshold = max(quantile(A0_for_pool), active_pool_rel_to_p95 * p95), with
active_pool_rel_to_p95=0.7 and a scheduled active-pool fraction cap.  The
previous loose OR pool rule remains only as an explicit legacy option.

Discovery mode now applies a soft high-confidence phase-interior penalty,
piecewise sampling-power annealing, and piecewise exploration-weight annealing.
Defaults are gamma 1.5 -> 2.5 -> 4.0 and w_ext 0.15 -> 0.08 -> 0.03 at
iterations 0/10/30.  R_obs remains mild and does not decide active-pool
membership.

The selected-point CSV, candidate-score CSV, monitor_predictions npz, selection
diagnostics, and report component-attribution table now carry A0_main_raw,
A0_for_pool, B_delta_raw, U_NS, B_delta_gated, B_q_raw, B_q_gated,
interior_penalty, high_confidence_interior, current sampling power, and current
exploration weight fields when produced by new runs.
```

Why it matters:

```text
The previous 2026-05-19 discovery run selected many predicted normal-interior
points because raw B_delta stayed high in deep normal regions.  The new
selection policy keeps stochastic acquisition but confines it to a sharper,
score-defined high-information pool and makes the Delta-boundary term depend
on actual normal/SC classifier competition.
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_discovery_mode.py
python scripts/dev_check_acquisition_only.py
python scripts/dev_check_stop_controller.py
python scripts/dev_check_report_builder.py
python scripts/dev_check_qdelta_logic.py
python -m ml_phase.report_builder --run-id active_boundary_discovery_512seed_256x50 --run-root hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/active_runs --template report/active_learning_phase_boundary_report.tex --output hpc_upload_qdelta_discovery_512seed_256x50_20260519_185535/ML_Phase_512_Iter/reports/active_learning_phase_boundary_report.tex
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
```

Known unresolved issue:

```text
The regenerated report for the existing 20260519 run can show N/A for new
diagnostic columns because that run was produced before B_delta_raw/U_NS/
A0_for_pool were saved.  A new discovery run is needed to evaluate whether the
sharpened active pool actually increases selected boundary-band fraction and
reduces predicted-normal-interior selection.
```

Next recommended steps:

```text
Package and run a fresh discovery job with the sharpened defaults.  After the
first 10-20 iterations, inspect active_pool_fraction,
selected_boundary_band_fraction, selected_normal_interior_fraction,
B_delta_raw/U_NS/B_delta_gated attribution, and N_eff/active_pool_size before
spending a full 100-loop allocation.
```

## 48. 2026-05-21 Final Exact Eta Map Added to Discovery Report

Files changed:

```text
ml_phase/report_builder.py
report/active_learning_phase_boundary_report.tex
scripts/dev_check_report_builder.py
docs/PROJECT_SUMMARY.md
```

Generated for the latest discovery result:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/figures/active_boundary_discovery_512seed_256x50_exact_eta_revised_boundaries.pdf
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/figures/active_boundary_discovery_512seed_256x50_exact_eta_revised_boundaries.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/figures/active_boundary_discovery_512seed_256x50_exact_eta_revised_boundaries.json
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/figures/active_boundary_discovery_512seed_256x50_final_exact_boundaries/
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/active_learning_phase_boundary_report.pdf
```

Summary:

```text
Added a report-time final exact diode-efficiency map colored by eta on the
signed-power color scale.  The figure uses the final dataset_iter020.npz from
the 20260520 discovery run and re-extracts normal/SC and uniform-SC/FFLO
boundaries from that final dataset, rather than reusing the iter019 boundary
cache.  The old cFFLO/tFFLO finite-T curves are retained only as topology
reference curves, with the caption explicitly noting that topology is not
revalidated by the current pointwise exact oracle.

The figure is inserted immediately after the cumulative progress figure in the
Active-Learning Results section.  The report builder prefers the PDF figure for
LaTeX inclusion and also writes PNG plus JSON metadata for quick inspection.
```

Why it matters:

```text
The report now connects the active-learning phase-boundary convergence to a
direct physical observable map.  It also avoids the off-by-one ambiguity in the
previous boundary cache by deriving the displayed revised boundaries from the
actual final dataset.
```

Validation:

```text
python -m compileall ml_phase scripts
python scripts/dev_check_report_builder.py
python -m ml_phase.report_builder --run-id active_boundary_discovery_512seed_256x50 --run-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/active_runs --template report/active_learning_phase_boundary_report.tex --output hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/active_learning_phase_boundary_report.tex
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
pdflatex -interaction=nonstopmode -halt-on-error active_learning_phase_boundary_report.tex
```

Current project state:

```text
The 20260520 sharpened discovery run stopped after iter019 with
stop_reason=converged_main_phase_boundaries and final dataset_iter020.npz.
Final exact phase counts are normal=1609, uniform_SC=648, FFLO=2850.  The last
selection diagnostic shows boundary-band focusing improved strongly relative
to random baseline: selected boundary-band fraction 0.503906 versus random
baseline 0.011719.
```

Known unresolved issues:

```text
The final dataset still contains delta-boundary-band metadata
(delta_unresolved/delta_boundary_ambiguous count 633).  These points are
training-eligible boundary-band data under the current numerical rule, but
they should be discussed explicitly when making physical claims about the
sharp normal/SC transition.
```

Next recommended steps:

```text
Use the new exact eta map together with the selection diagnostics to decide
whether another discovery run is needed.  If the current result is sufficient,
start drafting the report narrative around exact-call efficiency, boundary
convergence, and remaining Delta-resolution boundary-band limitations.
```

## 49. 2026-05-21 Boundary and Anomaly Audit Report

Files changed:

```text
scripts/boundary_anomaly_audit.py
docs/PROJECT_SUMMARY.md
```

Generated for the latest discovery result:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/boundary_and_anomaly_audit_report.md
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/audit_tables/boundary_audit_summary.json
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/audit_tables/high_JA_boundary_kink_points.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/audit_tables/eta_positive_high_JA_points.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/audit_tables/delta_ambiguous_points.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/audit_tables/rerun_required_points.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/audit_tables/old_topology_reference_nearby_points.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/reports/audit_figures/
```

Summary:

```text
Added an offline boundary-and-anomaly audit script for the 20260520 discovery
run.  The script reads the final dataset_iter020.npz, final extracted
normal/SC and uniform/FFLO boundaries, all selected-point traces, and the
latest exact-oracle merge output.  It writes markdown, CSV/JSON tables, and PNG
diagnostic figures without modifying acquisition, selection, StopController,
the neural networks, or the exact BdG oracle.

The audit explicitly separates final-dataset metadata from latest exact-oracle
metadata, because rerun-required points are not appended into the final
training dataset.  This preserves the latest exact-oracle facts:
delta_ambiguous=56, delta_unresolved=56, rerun_required=26 for iter019.
```

Key audit results:

```text
High-JA normal/SC kink subset:
    count=342, trusted=296, delta_ambiguous=46, delta_unresolved=46,
    boundary_band_normal=46, rerun_required=0, q_edge_hit=0, q_expanded=79.

JA>1.25 and eta>0 subset:
    count=33, eta>0.02 count=29, trusted=33, clean_response=33,
    delta_ambiguous=0, rerun_required=0, q_edge_hit=0, q_expanded=7.
    These points should be treated as a physics-check target rather than
    dismissed as numerical artifacts by current metadata alone.

Delta ambiguous / rerun-required subset:
    total unique Delta-ambiguous/unresolved points including latest oracle
    rerun-only points = 659.  Rerun-required points = 26, all latest-oracle
    Delta-unresolved points not appended as final trusted data.  524/659
    ambiguous points and 24/26 rerun-required points lie within the configured
    near-normal/SC-boundary distance threshold.

Old cFFLO/tFFLO reference-near subset:
    count=166, trusted=156, delta_ambiguous=10, eta>0 count=5.  This is only a
    response correlation with old topology-reference curves; no pointwise
    topology oracle was evaluated.
```

Validation:

```text
python -m compileall scripts/boundary_anomaly_audit.py
python scripts/boundary_anomaly_audit.py --ml-phase-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1 --run-id active_boundary_discovery_512seed_256x50
```

Known unresolved issues:

```text
The high-JA eta>0 points are metadata-clean under the current audit, so their
origin remains a physics/numerics question.  Next checks should inspect these
33 rows directly, verify q-window and current-response definitions pointwise,
and consider a topology oracle if they correlate with old topology-reference
curves.
```

Next recommended steps:

```text
Use the audit CSVs to define a small targeted exact rerun set:
1. the 26 rerun-required latest-oracle points;
2. selected high-JA normal/SC kink points with Delta ambiguity;
3. the 33 clean JA>1.25, eta>0 points for a focused response and q-window check.
```

## 50. 2026-05-21 Independent q-window/Delta Numerical Audit Harness

Files changed:

```text
scripts/numerical_audit_qwindow_delta.py
docs/PROJECT_SUMMARY.md
```

Generated audit-only folder:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/
```

Summary:

```text
Prepared an independent numerical audit harness for the 20260520 discovery
run.  The harness selects suspicious points from the previous
boundary-and-anomaly audit tables, writes q-window and strict-Delta rerun
input tables, and generates Slurm helper scripts.  It does not modify
active_runs, dataset_iter020.npz, acquisition, selection, StopController, NN
architecture, or the production exact oracle workflow.

The audit folder is self-contained: it includes its own copy of
numerical_audit_qwindow_delta.py under audit/scripts so that Slurm jobs can run
from the uploaded package directory without depending on the local Windows
workspace path.
```

Prepared subsets:

```text
eta_positive_high_JA_selected.csv: 33 points with JA > 1.25 and eta > 0.
high_JA_kink_delta_selected.csv: 46 high-JA normal/SC-kink points with Delta
    ambiguity, Delta-unresolved, or boundary-band-normal metadata.
rerun_required_selected.csv: 26 latest-oracle rerun-required points.
clean_control_selected.csv: 20 clean trusted high-JA kink control points.
combined_audit_points.csv: 123 unique coordinates across all roles.
```

Validation:

```text
python -m compileall scripts/numerical_audit_qwindow_delta.py
python scripts/numerical_audit_qwindow_delta.py setup --ml-phase-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1 --run-id active_boundary_discovery_512seed_256x50
python ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/numerical_audit_qwindow_delta.py --help
    (run from the uploaded package directory)
```

Known unresolved issues:

```text
The audit calculations have not been launched yet.  The q-window rerun will
test whether the 33 high-JA eta-positive points remain positive when the
response-level q-window includes superconducting branch endpoints and Ic+/Ic-
positions away from q-window edges.  The strict-Delta rerun will test whether
the high-JA normal/SC kink and ambiguous boundary-band points are stable under
tighter near-zero-Delta refinement.
```

Next recommended steps:

```text
On the HPC package directory, enter
ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1, submit the q-window and
Delta audit arrays, then run scripts/collect_results.sh to generate
qwindow_comparison.csv, delta_refine_comparison.csv, figures, and
reports/numerical_audit_qwindow_delta_report.md.
```

## 51. 2026-05-22 Slurm Line-Ending Fix for Numerical Audit Scripts

Files changed:

```text
scripts/numerical_audit_qwindow_delta.py
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/submit_qwindow_array.sh
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/submit_delta_array.sh
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/collect_results.sh
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Fixed the audit harness generator so generated text files, especially Slurm
shell scripts, are written with Unix LF line endings on Windows.  This resolves
the HPC sbatch error:

    Batch script contains DOS line breaks (\r\n)

The fix adds a small LF-only text writer and regenerates the existing audit
folder helper scripts.  No active-learning logic, acquisition function,
StopController, NN code, exact BdG oracle, or original run data were modified.
```

Validation:

```text
python -m compileall scripts/numerical_audit_qwindow_delta.py
python scripts/numerical_audit_qwindow_delta.py setup --ml-phase-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1 --run-id active_boundary_discovery_512seed_256x50

Line-ending byte check:
collect_results.sh CRLF=0
submit_delta_array.sh CRLF=0
submit_qwindow_array.sh CRLF=0

python ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/numerical_audit_qwindow_delta.py --help
    (run from the uploaded package directory)
```

Next recommended steps:

```text
Re-upload or sync the regenerated
ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/*.sh files to the
HPC location.  If the HPC directory is named ML_Phase instead of
ML_Phase_512_seed_v1, place the numerical_audit_qwindow_delta_v1 folder under
ML_Phase and run the same sbatch commands from that folder.
```

## 52. 2026-05-22 Slurm Submit-Directory Fix for Numerical Audit Scripts

Files changed:

```text
scripts/numerical_audit_qwindow_delta.py
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/submit_qwindow_array.sh
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/submit_delta_array.sh
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1/scripts/collect_results.sh
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Fixed another HPC portability issue in the audit Slurm helpers.  Slurm copies
batch scripts to a spool directory before execution, so using dirname "$0"
inside the job incorrectly resolved AUDIT_ROOT to a path under
/opt/gridview/slurm/spool/slurmd.  The helpers now use SLURM_SUBMIT_DIR as the
default AUDIT_ROOT and only fall back to pwd outside Slurm.  PROJECT_DIR is
then derived from AUDIT_ROOT/../.. unless explicitly provided.

This resolves errors such as:

    python: can't open file '/opt/gridview/slurm/spool/slurmd/scripts/numerical_audit_qwindow_delta.py'

The fix only changes audit job orchestration scripts.  It does not change
physics definitions, active-learning selection, acquisition, StopController,
NN architecture, or exact BdG oracle logic.
```

Validation:

```text
python -m compileall scripts/numerical_audit_qwindow_delta.py
python scripts/numerical_audit_qwindow_delta.py setup --ml-phase-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1 --run-id active_boundary_discovery_512seed_256x50

Line-ending byte check remains clean:
collect_results.sh CRLF=0
submit_delta_array.sh CRLF=0
submit_qwindow_array.sh CRLF=0

The regenerated submit_qwindow_array.sh now sets:
AUDIT_ROOT="${AUDIT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${AUDIT_ROOT}/../.." && pwd)}"
```

Next recommended steps:

```text
Re-upload/sync the regenerated numerical_audit_qwindow_delta_v1/scripts/
directory to the HPC run folder, then submit from inside
ML_Phase/numerical_audit_qwindow_delta_v1 or
ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1.
```

## 53. 2026-05-22 Numerical Audit Result Check and Report Compilation

Files changed/generated:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result/reports/numerical_audit_qwindow_delta_report.md
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result/reports/numerical_audit_qwindow_delta_report.tex
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result/reports/numerical_audit_qwindow_delta_report.pdf
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result/tables/audit_result_check_summary.json
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Checked the returned numerical_audit_qwindow_delta_v1_result folder and compiled
an updated audit report.  The Delta-refinement rerun returned all 8 shards and
produced 92 comparison rows.  The q-window response audit is partial: only
ranks 4-7 returned results, giving 32 rows for 16/33 high-JA positive-eta input
points.  Ranks 0-3 failed with CUDA driver/PyTorch CUDA initialization errors
on the HPC nodes, so q-window conclusions must be treated as incomplete.
```

Key results:

```text
q-window input points: 33
q-window covered points: 16
q-window rows: 32 / expected 66
q-window completed class counts: q_window_artifact=28, response_stable_positive=4
q-window missing point indices: 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,32
failed q-window logs: slurm-56983_0.out through slurm-56983_3.out

Delta rows: 92
Delta phase changed: 5
Delta new strict phase counts: normal=57, FFLO=30, boundary_ambiguous=4, uniform_SC=1
Delta trusted exact new: 88 / 92
Delta q_expanded_new: 4
Delta q_unresolved_new: 0
```

Validation:

```text
Generated audit_result_check_summary.json from qwindow_comparison.csv,
delta_refine_comparison.csv, and Slurm logs.
Compiled numerical_audit_qwindow_delta_report.tex with pdflatex.
The final PDF was written successfully as a 4-page report, and the LaTeX log has
no fatal alignment errors after table correction.
```

Known unresolved issues:

```text
The q-window response audit must be rerun for the missing 17 point indices on a
GPU node/environment with compatible NVIDIA driver and PyTorch CUDA versions, or
with a CPU/compatible fallback.  Until then, only the completed 16-point subset
can be interpreted.  The Delta refinement audit is usable as a completed
small-target numerical check.
```

## 54. 2026-05-22 Complete q-window????蹌? Audit and Report

Files changed/generated:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/tables/qwindow_comparison.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/tables/qwindow_comparison_complete_of004.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/tables/qwindow_level_stability_by_point.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/tables/qwindow_positive_eta_local_extrema_diagnostics.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/tables/audit_result_check_summary_complete.json
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/figures/qwindow_eta_sign_stability_complete.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/figures/eta_positive_high_JA_before_after_complete.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/figures/qwindow_two_level_eta_stability.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/figures/qwindow_positive_eta_qextrema_shift.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/reports/numerical_audit_qwindow_delta_report_complete.md
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/reports/numerical_audit_qwindow_delta_report_complete.tex
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2/reports/numerical_audit_qwindow_delta_report_complete.pdf
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Analyzed the????蹌? q-window result in numerical_audit_qwindow_delta_v1_result2.
The????蹌? used a world_size=4 q-window run whose rank000-rank003 summaries cover
all 33 high-JA positive-eta input points with two expansion levels per point.
The old partial world_size=8 q-window summaries were left in the directory but
are not used as the canonical complete comparison table.
```

Key q-window results:

```text
q-window input points: 33
q-window rows: 66
unique q-window points: 33
response-valid rows: 66
q-window class counts: q_window_artifact=57, response_stable_positive=9
positive eta rows after expanded q-window: 9
positive eta unique points: 6
positive in both expansion levels: 3

Density/status counts over all 33 points:
eta_nonpositive_after_audit: 27
sign_changes_with_window_or_density: 3
q_extremum_location_unstable: 2
stable_under_two_level_audit: 1
```

Interpretation:

```text
The q-window width is sufficient: all completed rows pass the response-level
window validity test.  The residual positive-eta points are not yet robust
physical claims.  Three positive-any points change eta sign between the two
expansion levels, two keep positive eta but have unstable Ic extremum locations,
and only one point passes the current two-level stability screen.  The remaining
positive eta should be treated as q-density / response-extraction unresolved.
```

Delta audit status retained:

```text
Delta rows: 92
Delta phase changed: 5
Delta new strict phase counts: normal=57, FFLO=30, boundary_ambiguous=4, uniform_SC=1
Delta still boundary ambiguous: 4
```

Validation:

```text
Generated complete q-window tables and figures from qwindow_summary_rank000_of004
through qwindow_summary_rank003_of004.
Compiled numerical_audit_qwindow_delta_report_complete.tex with pdflatex.
The PDF was written successfully as a 5-page report.  The LaTeX log contains
minor overfull hbox warnings only, no fatal errors.
```

Next recommended steps:

```text
Run a focused q-density convergence audit on the six positive-any q-window
points, keeping the expanded q-window fixed and increasing n_q to 3200, 6400,
and possibly 12800.  Point 17 is the only currently two-level-stable candidate
and should be included as the highest-priority control for this density test.
```

## 55. 2026-05-22 Fixed-window q-density Audit Harness

Files changed/generated:

```text
scripts/numerical_audit_qdensity.py
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added an audit-only fixed-window q-density convergence harness for the six
residual positive-eta candidates from the complete q-window audit.  The script
uses the expand_1.0 q-window from qwindow_comparison_complete_of004.csv as the
fixed q range and recomputes each point at nq=3200, 6400, and 12800.  It does
not append to active-learning datasets and does not modify production
acquisition, StopController, NN training, or exact BdG oracle workflow.
```

Prepared q-density input points:

```text
points = 11, 13, 17, 20, 21, 25
fixed q-window = [-pi, 2.375] from the expand_1.0 q-window audit rows
nq levels = 3200, 6400, 12800
curve-save points = 11 and 21
```

Generated folder structure:

```text
numerical_audit_qdensity_v1/config/qdensity_config.json
numerical_audit_qdensity_v1/input_points/qdensity_positive_eta_points.csv
numerical_audit_qdensity_v1/scripts/submit_qdensity_array.sh
numerical_audit_qdensity_v1/scripts/collect_qdensity_results.sh
numerical_audit_qdensity_v1/scripts/numerical_audit_qdensity.py
numerical_audit_qdensity_v1/reports/qdensity_convergence_report.md
```

Validation:

```text
python -m compileall scripts/numerical_audit_qdensity.py
python scripts/numerical_audit_qdensity.py setup --source-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qwindow_delta_v1_result2
python ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/scripts/numerical_audit_qdensity.py --help
    (run from the uploaded package root)

Line endings:
submit_qdensity_array.sh CRLF=0
collect_qdensity_results.sh CRLF=0

HPC safeguard:
submit_qdensity_array.sh includes #SBATCH --exclude=gpu01.
```

Next recommended steps:

```text
Upload/sync numerical_audit_qdensity_v1/ to the HPC ML_Phase directory, submit
scripts/submit_qdensity_array.sh, then run scripts/collect_qdensity_results.sh.
Return the generated tables, curves, figures, and report for interpretation.
```

## 56. 2026-05-23 Fixed-window q-density Audit Results

Files changed/generated:

```text
scripts/numerical_audit_qdensity.py
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/config/qdensity_config.json
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/scripts/numerical_audit_qdensity.py
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/tables/qdensity_convergence.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/tables/qdensity_summary_by_point.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/tables/qdensity_check_summary.json
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/figures/eta_vs_nq.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/figures/Ic_vs_nq.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/figures/qIc_shift_vs_nq.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/reports/qdensity_convergence_report.md
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/reports/qdensity_convergence_report.tex
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/reports/qdensity_convergence_report.pdf
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Collected the returned fixed-window q-density audit for points
11, 13, 17, 20, 21, and 25 at nq=3200, 6400, and 12800 using the fixed
expand_1.0 q-window.  The q-density collect/report logic was updated to use the
requested five report groups, to add a per-row classification column, to add the
eta_change_from_previous_nq alias required by the report request, and to write a
LaTeX/PDF report in addition to the Markdown report.
```

Key results:

```text
Expected point/nq rows: 18
Collected point/nq rows: 18
Status counts: ok=18

density-converged large positive eta: 0
weak near-zero positive eta: 0
sign-changing artifact: 4
q-extremum-location unstable: 1
unresolved: 1

sign-changing artifact points: 11, 13, 17, 20
q-extremum-location unstable point: 25
unresolved point: 21
```

Interpretation:

```text
None of the six residual positive-eta candidates passes the fixed-window
q-density convergence screen.  Four candidates change eta sign between
nq=6400 and nq=12800.  Point 25 remains eta=+1 at all tested densities but its
critical-current extremum location shifts by more than the two-fine-dq
tolerance.  Point 21 remains unresolved because it fails one or more convergence
tolerances despite positive eta at nq=6400 and nq=12800.
```

Known unresolved issue:

```text
The returned run saved full response curves and top-5 extrema only for points
11 and 21.  Point 25 was requested for full curve output but is missing
point0025_nq3200/6400/12800 response.npz and top_extrema.csv files because the
submitted config still had curve_save_points=[11, 21].  The local script and
config have now been updated to curve_save_points=[11, 21, 25] for any future
rerun of point 25 curves.
```

Validation:

```text
python -m compileall scripts/numerical_audit_qdensity.py
python scripts/numerical_audit_qdensity.py collect --audit-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1
pdflatex -interaction=nonstopmode -halt-on-error qdensity_convergence_report.tex

The PDF compiled successfully as a 3-page report.  The report explicitly states
that point 25 curve files are missing from the returned data.
```

## 57. 2026-05-23 Response-curve Pathology Audit Harness

Files changed/generated:

```text
scripts/response_curve_pathology_audit.py
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added an audit-only response-curve pathology script for points 21 and 25 at
nq=6400 and nq=12800.  The audit checks whether saturated eta=1 is caused by a
real diode response or by near-zero critical-current branches / extremum
extraction artifacts.  It saves full q response curves when rerun on HPC and
produces branch-separated I(q) plots, top-10 branch extrema tables, response
summary tables, and Markdown/LaTeX/PDF reports.
```

Current local analysis:

```text
point 21 nq=6400:
    Ic_plus = 7.3796178e-05
    Ic_minus = 0
    |Ic_plus| + |Ic_minus| = 7.3796178e-05 < 1e-4
    eta_denominator_unreliable = true
    branch_near_zero = true

point 21 nq=12800:
    Ic_plus = 2.5707789e-04
    Ic_minus = 0
    eta_denominator_unreliable = false by the absolute denominator threshold
    branch_near_zero = true

Interpretation:
    point 21 eta=1 is not allowed as a positive-eta claim because the Ic_minus
    branch is numerically zero in both nq=6400 and nq=12800 curves.
```

Known unresolved issue:

```text
Point 25 curve files are still missing for nq=6400 and nq=12800, so point 25
cannot yet be audited at the response-curve level.  The new helper script
scripts/submit_response_pathology_curves.sh under response_curve_pathology_audit
can rerun exactly these four point/nq curve tasks and uses
#SBATCH --exclude=gpuh01.

The first point-25 rerun attempt returned slurm-59162_2.out and
slurm-59162_3.out but failed before calculation with
ModuleNotFoundError: No module named 'numerical_audit_qdensity'.  The import path
has been fixed in scripts/response_curve_pathology_audit.py and in the audit
folder script copy so it can find the parent qdensity scripts directory on HPC.
Rerun array tasks 2-3 after uploading/syncing the fixed script.
```

Validation:

```text
python -m compileall scripts/response_curve_pathology_audit.py
python scripts/response_curve_pathology_audit.py setup --qdensity-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1
python scripts/response_curve_pathology_audit.py analyze --qdensity-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1 --audit-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit
pdflatex -interaction=nonstopmode -halt-on-error response_curve_pathology_report.tex

The response pathology PDF compiled successfully as a 3-page report containing
the two available point-21 branch plots.
```

## 58. 2026-05-23 Local RTX 4090 Response-curve Rerun for Point 25

Files changed/generated:

```text
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/curves/point0025_nq6400_response.npz
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/curves/point0025_nq12800_response.npz
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/raw_outputs/response_pathology_summary_rank002_of004.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/raw_outputs/response_pathology_summary_rank003_of004.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/tables/response_pathology_summary.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/tables/response_pathology_top10_extrema.csv
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/figures/point0025_nq6400_response_pathology.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/figures/point0025_nq12800_response_pathology.png
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/reports/response_curve_pathology_report.md
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/reports/response_curve_pathology_report.tex
hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1/numerical_audit_qdensity_v1/response_curve_pathology_audit/reports/response_curve_pathology_report.pdf
docs/PROJECT_SUMMARY.md
```

Summary:

```text
The missing point-25 response curves were computed locally on the RTX 4090
instead of rerunning on HPC.  The nq=6400 curve took about 15 minutes; the
nq=12800 curve took about 30 minutes.  Both tasks completed without CUDA OOM
and wrote audit-only curve files under response_curve_pathology_audit.
```

Key results:

```text
point 25 nq=6400:
    Ic_plus = 2.0669398e-04
    Ic_minus = 0
    |Ic_plus| + |Ic_minus| = 2.0669398e-04
    eta = 1
    eta_denominator_unreliable = false
    branch_near_zero = true
    positive_eta_allowed = false

point 25 nq=12800:
    Ic_plus = 3.7732059e-05
    Ic_minus = 0
    |Ic_plus| + |Ic_minus| = 3.7732059e-05 < 1e-4
    eta = 1
    eta_denominator_unreliable = true
    branch_near_zero = true
    positive_eta_allowed = false
```

Interpretation:

```text
Point 25 is not a robust positive-eta signal.  Like point 21, its eta=1 comes
from a near-zero / zero critical-current branch rather than a stable finite
diode response.  The report classifies the point-25 curves as eta
ill-conditioned, with branch_near_zero=true at both nq=6400 and nq=12800.
```

Validation:

```text
nvidia-smi confirmed local NVIDIA GeForce RTX 4090 with CUDA available through
torch 2.10.0+cu126.

python scripts/response_curve_pathology_audit.py run --rank 2 --world-size 4
python scripts/response_curve_pathology_audit.py run --rank 3 --world-size 4
python scripts/response_curve_pathology_audit.py analyze --qdensity-root ... --audit-root ...
pdflatex -interaction=nonstopmode -halt-on-error response_curve_pathology_report.tex

The final response pathology PDF compiled successfully as a 5-page report with
point-21 and point-25 branch plots.  The LaTeX log has only overfull-box
warnings from long pathology labels.
```

## 59. 2026-05-23 Consolidated Numerical Reliability Audit Report

Files changed/generated:

```text
report_numerical_reliability_audit_20260523/numerical_reliability_audit.tex
report_numerical_reliability_audit_20260523/numerical_reliability_audit.pdf
report_numerical_reliability_audit_20260523/figures/
report_numerical_reliability_audit_20260523/tables/
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Created a consolidated English LaTeX note for the current discovery-run
numerical reliability work.  The note integrates the discovery phase diagram,
the Delta-refinement and high-JA boundary audit, the response-level q-window
audit, the fixed-window q-density convergence audit, and the final response
curve pathology audit for points 21 and 25.
```

Main report conclusions:

```text
The thermodynamic free-energy phase criterion is internally consistent within
the scanned q and Delta window, but high-JA/low-T phase-boundary reliability
still depends on whether the q-window covers all relevant FFLO branches.

The high-JA positive-eta candidates are not robust diode-response evidence.
Most disappear or change sign after q-window and q-density audits.  The
remaining eta=1 cases, including points 21 and 25, are small-denominator or
branch-near-zero pathologies with Ic_minus=0.

The next calculation should be a phase-boundary q-window and branch-minimum
audit that saves F_min(q), extracts multiple local minima, and checks whether
expanded q windows move the high-JA normal/SC boundary.
```

Validation:

```text
pdflatex -interaction=nonstopmode -halt-on-error numerical_reliability_audit.tex
pdflatex -interaction=nonstopmode -halt-on-error numerical_reliability_audit.tex

The report compiled successfully as a 10-page PDF.  The LaTeX log contains only
minor overfull-box warnings, no fatal errors.
```

OneDrive:

```text
The report folder should be synced to:
E:/Onedrive/OneDrive - The University of Hong Kong - Connect/GBU_SC/Fu_FFLO/report_numerical_reliability_audit_20260523
```

## 60. 2026-05-23 Report Synchronization and ChatGPT Handoff Protocol

Files changed/generated:

```text
AGENTS.md
report_numerical_reliability_audit_20260523/numerical_reliability_audit.md
report_numerical_reliability_audit_20260523/decision_log.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a project-level report synchronization protocol to AGENTS.md.  Future
report-generation tasks must not rely on PDF alone; they should also produce a
Markdown report with the same scientific content, key CSV tables, important
figures, and a short decision_log.md for ChatGPT handoff.
```

Current report synchronization status:

```text
The consolidated numerical reliability audit now has:

report_numerical_reliability_audit_20260523/numerical_reliability_audit.pdf
report_numerical_reliability_audit_20260523/numerical_reliability_audit.tex
report_numerical_reliability_audit_20260523/numerical_reliability_audit.md
report_numerical_reliability_audit_20260523/decision_log.md
report_numerical_reliability_audit_20260523/tables/
report_numerical_reliability_audit_20260523/figures/
```

Why it matters:

```text
The Markdown and decision-log companions make the report usable by ChatGPT and
other text-first tools without relying on PDF parsing.  The decision log records
the main conclusion: current high-JA positive-eta points are not robust diode
evidence, while the next major uncertainty is a thermodynamic phase-boundary
q-window and branch-minimum audit.
```

## 61. 2026-05-23 Phase q-window and Delta-refinement Audit Workflow

Files changed/generated:

```text
scripts/phase_qwindow_delta_refinement_audit.py
report_phase_qwindow_delta_refinement_v1/
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Implemented an audit-only production-oriented workflow for high-JA, low-T
normal/SC boundary-sensitive points.  The new script prepares input manifests,
HPC helper scripts, q-window expansion scans, near-zero Delta refinement runs,
collection tables, figures, Markdown/PDF reports, and a decision_log.md.
```

Prepared input counts:

```text
qwindow_sensitive_points = 342
delta_sensitive_points = 96
clean_control_points = 20
combined_unique_points = 345
```

Scientific decision:

```text
The free-energy phase criterion is not changed.  A lower-free-energy
positive-Delta superconducting state is enough for basic SC classification
within the scanned window.  Expanded q-window scans are used to test branch
identity, q_opt stability, boundary robustness, and topology readiness.
Near-zero Delta refinement is used only for tolerance-sensitive normal/SC
boundary points.
```

Generated report scaffold:

```text
report_phase_qwindow_delta_refinement_v1/phase_qwindow_delta_refinement.md
report_phase_qwindow_delta_refinement_v1/phase_qwindow_delta_refinement.tex
report_phase_qwindow_delta_refinement_v1/phase_qwindow_delta_refinement.pdf
report_phase_qwindow_delta_refinement_v1/decision_log.md
report_phase_qwindow_delta_refinement_v1/input_points/
report_phase_qwindow_delta_refinement_v1/scripts/
```

Validation:

```text
python -m compileall scripts/phase_qwindow_delta_refinement_audit.py
python scripts/phase_qwindow_delta_refinement_audit.py self-test
python scripts/phase_qwindow_delta_refinement_audit.py setup --ml-phase-root hpc_upload_qdelta_discovery_512seed_256x50_20260520_210207/ML_Phase_512_seed_v1 --report-root report_phase_qwindow_delta_refinement_v1
pdflatex -interaction=nonstopmode -halt-on-error phase_qwindow_delta_refinement.tex

The pending report PDF compiled successfully.  The generated SLURM scripts use
LF line endings and exclude gpuh01.
```

## 62. 2026-05-25 Phase q-window and Delta-refinement Audit Results

Files changed/generated:

```text
hpc_phase_qwindow_delta_refinement_20260525/report_phase_qwindow_delta_refinement_v1/
report_phase_qwindow_delta_refinement_v1/phase_qwindow_delta_refinement.md
report_phase_qwindow_delta_refinement_v1/phase_qwindow_delta_refinement.tex
report_phase_qwindow_delta_refinement_v1/phase_qwindow_delta_refinement.pdf
report_phase_qwindow_delta_refinement_v1/decision_log.md
report_phase_qwindow_delta_refinement_v1/tables/
report_phase_qwindow_delta_refinement_v1/figures/
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Collected the completed HPC phase-side numerical audit and generated an
enhanced Markdown/PDF report.  The audit checks high-JA, low-temperature
boundary-sensitive points using expanded free-energy q-window scans and
near-zero Delta refinement.  It remains audit-only and does not modify the
active-learning dataset, acquisition logic, StopController, neural-network
surrogate, or exact-oracle definitions.
```

Key numerical results:

```text
q-window expansion:
    345 unique points
    690 total expansion rows
    all rows status=ok
    expand_0p5_width: 121 lower-branch rows, 78 phase changes
    expand_1p0_width: 94 lower-branch rows, 78 phase changes

Phase changes:
    78 previously normal points become FFLO after expanded q-window scans.
    These points lie at kBT/t = 0 to 0.0583333 and JA/t = 1.331625 to 1.411125.
    New q_opt values lie roughly from -1.173 to -1.026 and are not at the
    expanded q-window edge.
    New Delta_opt values are positive, from about 0.0105 to 0.0617.
    New DeltaF_min values are negative, down to about -1.76e-4.

Near-zero Delta refinement:
    69 points refined
    all rows valid
    1 point changed, from normal to boundary_ambiguous
    no robust new superconducting point was created by Delta refinement alone

Branch-resolved output:
    5520 local-minimum rows saved
    at expand_1p0_width, 231 points have at least two low-energy local minima
    and 228 points have at least three low-energy local minima
```

Interpretation:

```text
The audit validates the workflow strategy, not the old high-JA boundary labels.
Discovery active learning located the boundary-sensitive region, response-level
eta anomalies were correctly separated from thermodynamic phase robustness, and
audit-only q-window expansion exposed the dominant phase-side numerical issue.

The old high-JA normal/SC boundary should not be treated as final.  Expanded
free-energy q-window scans reveal a missing FFLO branch for a substantial subset
of boundary-sensitive points.  The near-zero Delta refinement remains useful as
a guardrail for boundary-band ambiguity, but it is not the dominant correction
mechanism in this run.
```

Validation:

```text
python hpc_phase_qwindow_delta_refinement_20260525/report_phase_qwindow_delta_refinement_v1/scripts/phase_qwindow_delta_refinement_audit.py collect --report-root hpc_phase_qwindow_delta_refinement_20260525/report_phase_qwindow_delta_refinement_v1
pdflatex -interaction=nonstopmode -halt-on-error phase_qwindow_delta_refinement.tex

The enhanced report compiled successfully as a 4-page PDF with five figures.
The report and companion Markdown/decision log were synchronized into the root
report_phase_qwindow_delta_refinement_v1/ directory.
```

Next recommended step:

```text
Update the production exact phase-side q-window policy for high-JA,
low-temperature boundary points, then rerun or version-correct the affected
normal/SC boundary labels.  Keep Delta refinement as a boundary-band guardrail.
Use the saved low-energy local-minimum table as a candidate manifest for later
branch-resolved topology calculations.
```

## 61. 2026-05-25 Robust Oracle Rollout (Code Integration)

Files changed:

```text
ml_phase/exact_oracle.py
ml_phase/active_refine.py
ml_phase/config.py
scripts/slurm_exact_oracle_array.sh
scripts/robust_oracle_smoke_check.py
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Integrated a production-oriented robust exact-oracle mode (oracle_mode=robust_al)
into the active-learning path without changing the thermodynamic phase
criterion or acquisition strategy. The robust mode adds adaptive q-window
expansion diagnostics, local low-energy minima refinement boxes, and
near-zero-Delta guardrail metadata per exact point.
```

Implementation decisions:

```text
1) Keep legacy oracle path fully available for backward compatibility.
2) Route robust mode through exact_oracle dispatcher; no rank-to-rank coupling.
3) Persist branch-candidate tables per point for later topology analysis.
4) Pass ORACLE_MODE through SLURM array script to avoid hidden mode mismatch.
5) Default active_refine CLI to discovery-scale values:
   iterations=100, points_per_iter=256.
```

Current state:

```text
Code-level integration is complete and py_compile checks pass for the modified
modules. Runtime validation still requires cluster smoke runs on selected
high-JA boundary-sensitive points, followed by a new discovery rerun under a
new run_id using robust_al mode.
```

Known unresolved items:

```text
1) Full runtime smoke benchmarks were not completed locally because exact-point
   CPU execution is too slow for practical local validation.
2) Performance-cost calibration (per-point walltime and per-iteration budget)
   must be measured on NV_H100 before launching the full rerun.
```

Recommended next actions:

```text
1) Run scripts/robust_oracle_smoke_check.py on a small audit subset in HPC.
2) Compare robust vs legacy phase changes and q-expansion metadata.
3) Launch a new discovery run with oracle_mode=robust_al and a new run_id.
4) Generate md+pdf+csv report package for the rerun.
```

## 62. 2026-05-25 Acquisition Profile A/B Packaging

Files changed:

```text
ml_phase/config.py
ml_phase/acquisition.py
ml_phase/active_refine.py
hpc_active_loop.sh
scripts/slurm_active_refine.sh
scripts/check_acquisition_profiles.py
scripts/package_acquisition_profile_hpc.py
scripts/compare_acquisition_profiles.py
scripts/export_active_run_report_md.py
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added configurable acquisition profiles for controlled discovery-mode A/B runs:
`full` (baseline) and `simple_phase` (phase-focused simplified score). The
exact-oracle side remains `robust_al`; thermodynamic phase criterion is
unchanged.
```

Implementation details:

```text
1) acquisition_profile now flows from config -> active_refine CLI -> Slurm
   launcher -> hpc_active_loop env.
2) full profile keeps existing A0_main composition.
3) simple_phase uses phase-focused score terms and excludes numerical-risk and
   response terms from A0_main while retaining diagnostics.
4) selected point metadata now records acquisition_profile.
5) Added smoke regression script to verify full-formula consistency and simple
   profile field behavior.
6) Added a packaging script that emits two standalone HPC package directories
   with code snapshots, manifest, launcher scripts, and decision log.
```

Smoke validation:

```text
python scripts/check_acquisition_profiles.py

Result:
- full profile formula regression: pass
- simple profile output check: pass
- both profiles produce non-empty selected batches on deterministic mock input
```

Artifacts generated:

```text
hpc_packages/robust_oracle_full_acquisition/
hpc_packages/robust_oracle_simple_phase_acquisition/
```

## 63. 2026-05-25 Self-Contained HPC Package Fix

Files changed/generated:

```text
scripts/package_acquisition_profile_hpc.py
hpc_packages/hpc_upload_robust_oracle_acq_compare_20260525/
hpc_upload_robust_oracle_acq_compare_20260525.tar.gz
docs/PROJECT_SUMMARY.md
```

Summary:

```text
The first acquisition-profile packages were code-snapshot packages, not
self-contained run directories. On HPC this caused missing-path failures such
as missing hpc_active_loop.sh and scripts/slurm_active_refine.sh when the
uploaded folder layout differed from the local hpc_packages/ layout.

The packaging script now emits a self-contained run directory:

hpc_upload_robust_oracle_acq_compare_20260525/

The package root directly contains hpc_active_loop.sh, complete ml_phase code,
needed Slurm/helper scripts, eta_phase_diagram_cuda.py, tfflo_1d_cuda.py,
report template, docs, and two launchers:

run_full_acquisition.sh
run_simple_phase_acquisition.sh
```

Validation:

```text
python -m py_compile on key package modules passed.
Shell scripts were normalized to Unix LF line endings.
Old generated ml_phase active_runs/datasets/figures/reports were excluded.
The final tarball is:

hpc_upload_robust_oracle_acq_compare_20260525.tar.gz
```

Recommended HPC usage:

```text
tar -xzf hpc_upload_robust_oracle_acq_compare_20260525.tar.gz
cd hpc_upload_robust_oracle_acq_compare_20260525
bash run_full_acquisition.sh
bash run_simple_phase_acquisition.sh
```

## 64. 2026-05-26 Robust Oracle Slurm Walltime Adjustment

Files changed:

```text
scripts/slurm_exact_oracle_array.sh
hpc_packages/hpc_upload_robust_oracle_acq_compare_20260525/scripts/slurm_exact_oracle_array.sh
hpc_upload_robust_oracle_acq_compare_20260525/scripts/slurm_exact_oracle_array.sh
```

Summary:

```text
The first robust-oracle acquisition comparison run reached iter000 exact
oracle submission but both full and simple acquisition jobs timed out after the
default 4 hour Slurm walltime. Partial shards showed approximately 450/512
exact seed points completed, so the failure was walltime/cost rather than
candidate generation or acquisition logic.

The exact oracle array walltime was increased from 04:00:00 to 12:00:00 in the
source Slurm script and in the existing self-contained package copies. This is
a pragmatic run-control adjustment; it does not change the thermodynamic phase
criterion, acquisition formula, or robust oracle numerical method.
```

Current state:

```text
The failed iter000 partial outputs should not be resumed as START_ITER=1
because dataset_iter001.npz was never produced. Rerun with a fresh run_id or
clear the failed active-run directory before restarting START_ITER=0.
```

## 65. 2026-06-01 Robust-Oracle Run Audit Report

Files generated:

```text
hpc_upload_robust_oracle_acq_compare_20260525_12h/ML_Phase_512_20260601/reports/robust_oracle_run_audit_20260601/
  robust_oracle_run_audit_20260601.md
  robust_oracle_run_audit_20260601.tex
  robust_oracle_run_audit_20260601.pdf
  decision_log.md
  tables/*.csv
  figures/*.png
tmp/generate_robust_oracle_run_audit.py
tmp/plot_computed_delta_redblue_maps.py
```

Summary:

```text
Generated a standalone diagnostic report for the latest robust-oracle
full-vs-simple acquisition run.  The report explains why the control loop did
not generate the normal final report, why late selected points concentrate in
the upper-right high-kT/high-JA region, and why training loss convergence did
not trigger StopController convergence.
```

Findings:

```text
1) Both acquisition profiles completed active-learning post-processing through
   iter031.  Iter032 exact shards exist, but iter032 was not merged/appended or
   stop-checked because the login-node control loop disappeared while waiting
   on the exact array.
2) The latest complete datasets contain no normal labels:
   full: 841 samples = 215 uniform_SC + 626 FFLO;
   simple_phase: 1028 samples = 359 uniform_SC + 669 FFLO.
3) After approximately iter014-iter015, new training-eligible appended samples
   drop to zero.  Late exact points are mostly delta_boundary_ambiguous /
   rerun_required and therefore do not update the training set.
4) The upper-right selected-point concentration is interpreted as a blind spot
   created by trusted-filter rejection, not as physical discovery.
5) The run should not be used to rank full vs simple acquisition until stable
   normal labels can enter the training dataset.
```

Recommended next actions:

```text
Recover iter032 for both profiles from existing shards; then fix robust
oracle/trusted-filter handling of stable normal points so only genuinely
near-boundary ambiguous points are excluded.  Add a blocked-stop condition for
repeated zero training-eligible appends before launching another long
acquisition comparison.
```

Follow-up figure:

```text
Added a red-blue exact-point phase-map style figure:

figures/computed_delta_redblue_map_comparison.png

The figure uses all currently computed exact shard points, de-duplicated by
latest coordinate, and colors them by Delta_opt using a blue-to-red gradient.
Blue corresponds to near-normal Delta_opt=0 points, while red corresponds to
larger superconducting order.  The figure is inserted into the Markdown and
PDF audit report.
```

Additional eta-map follow-up:

```text
Added eta-colored exact-point maps with boundary overlays:

figures/computed_eta_boundary_map_comparison.png
figures/computed_eta_magnitude_expected_boundaries_comparison.png
figures/computed_eta_magnitude_normal_sc_only_comparison.png

The signed eta map colors all currently computed exact shard points by eta,
clipped to [-1,1].  The eta-magnitude map colors points by |eta|.  Both maps
initially overlaid finite-q threshold diagnostics, but the report now treats
those curves as non-physical diagnostics rather than true uniform/FFLO phase
boundaries: for J_A>0 a nonzero optimal q is physically expected.  The
phase-safe eta-magnitude figure suppresses the finite-q diagnostic and overlays
only the expected normal/SC boundary from the earlier discovery run.  The
current normal/SC boundary is not available from the robust-oracle run because
the latest complete training datasets contain no normal labels.
```

## 66. 2026-06-01 Robust-Oracle Normal-Label Root-Cause Audit

Files generated:

```text
hpc_upload_robust_oracle_acq_compare_20260525_12h/ML_Phase_512_20260601/reports/robust_oracle_root_cause_audit/
  root_cause_audit.md
  root_cause_audit.tex
  root_cause_audit.pdf
  decision_log.md
  tables/*.csv
  figures/*.png
tmp/robust_oracle_root_cause_audit.py
```

Summary:

```text
Performed a read-only root-cause audit of the robust-oracle runs after normal
labels disappeared from the active-learning datasets.  No acquisition,
StopController, exact-oracle production code, or active-run data were modified.
```

Finding:

```text
The normal labels disappear before append.  Raw robust-oracle shards for
iter031/iter032 contain many Delta_opt=0 normal-like points, but the robust
adaptive-box path marks them boundary_ambiguous because positive_delta_gap is
tested against free_energy_ambiguity_tol=1e-6.  Under the older positive-gap
rule, points with positive_delta_gap > 1e-8 were stable normal.  In the broken
runs, 252/254 normal-like points per audited late batch fall into
(1e-8, 1e-6], so they are stable by the old rule but ambiguous by the new
robust rule.  Since trusted_exact requires not boundary_ambiguous, these
points are written with trusted_exact=0 and training_eligible_exact=0.
```

Decision:

```text
Do not start mini AL or full A/B runs until a minimal patch separates
phase_label, confidence_state, training_eligible_exact, and rerun_required.
Stable normal points must be allowed to append with q_status=not_applicable and
q_unresolved=false.
```

## 67. 2026-06-01 Robust-Oracle Label-Closure Patch and HPC Package

Files changed:

```text
ml_phase/exact_oracle.py
ml_phase/hpc.py
scripts/check_robust_oracle_label_closure.py
scripts/package_acquisition_profile_hpc.py
docs/NUMERICS_SPEC.md
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
```

Files generated:

```text
reports/robust_oracle_label_closure_validation/
  label_closure_validation.md
  label_closure_validation.tex
  label_closure_validation.pdf
  decision_log.md
  tables/existing_shard_validation.csv
  tables/existing_shard_validation.json
  tables/synthetic_case_validation.csv
  tables/synthetic_case_validation.json

hpc_packages/hpc_upload_robust_oracle_label_closure_acq_compare_20260601/
hpc_packages/robust_oracle_full_acquisition/
hpc_packages/robust_oracle_simple_phase_acquisition/
hpc_upload_robust_oracle_label_closure_acq_compare_20260601.tar.gz
```

Summary:

```text
Implemented the minimal production patch requested after the robust-oracle
normal-label audit.  The patch separates thermodynamic phase labels from
confidence state, training eligibility, and rerun status.  Stable normal
points with Delta_opt=0 and positive_delta_gap > positive_delta_gap_tol are
now trusted, training-eligible normal labels.  Boundary-band normal points
with 0 <= positive_delta_gap <= positive_delta_gap_tol can also enter the
training set with boundary_ambiguous metadata.  q_status=not_applicable for
normal points is not treated as q_unresolved.
```

Important implementation decision:

```text
The thermodynamic phase criterion was not changed.  The acquisition function,
neural network, StopController, and response-side eta logic were not changed.
The fix is limited to exact-oracle metadata and merge/trusted-filter semantics
so that numerically stable normal labels are not rejected before append.
```

Validation:

```text
Command:
python -m py_compile ml_phase\exact_oracle.py ml_phase\hpc.py ml_phase\append_trusted.py scripts\check_robust_oracle_label_closure.py

Command:
python scripts\check_robust_oracle_label_closure.py

Result:
The validation script reprocessed existing broken full/simple robust-oracle
shards from iter031 and iter032.  Each audited late batch contained 252 stable
normal points.  After the patch, the late-batch training-eligible counts become
254 rather than zero, q_unresolved remains zero for stable normals, and all
synthetic stable-normal / boundary-band-normal / clean-SC / solver-failed
checks pass.
```

HPC package:

```text
The new self-contained package uses new run IDs:

active_boundary_discovery_robust_oracle_full_acq_label_closed_v1
active_boundary_discovery_robust_oracle_simple_acq_label_closed_v1

The package should first be used for a 3-5 iteration mini active-learning run.
Full 100-iteration A/B comparison should wait until the mini run confirms:
normal_count > 0, normal labels append each round, rerun-required fraction is
not persistently near one, and selected points do not collapse into the
upper-right blind spot.
```

Known unresolved issue:

```text
The patch has local shard-level validation only.  It has not yet been verified
on HPC through a fresh mini active-learning loop with the robust exact oracle.
Do not use it for a long A/B acquisition comparison until that mini run passes.
```

## 68. 2026-06-02 Robust-Oracle Acquisition A/B Mini-Run Report

Files generated:

```text
hpc_upload_robust_oracle_label_closure_acq_compare_20260601/ML_Phase_512_20260601/reports/robust_oracle_acquisition_ab_mini_report/
  robust_oracle_acquisition_ab_mini_report.md
  robust_oracle_acquisition_ab_mini_report.tex
  robust_oracle_acquisition_ab_mini_report.pdf
  decision_log.md
  tables/*.csv
  figures/*.png
tmp/generate_ab_mini_report.py
```

Report-only file changed:

```text
ml_phase/report_builder.py
```

Summary:

```text
Generated a complete comparison report for the 5-iteration robust-oracle
mini active-learning runs:

active_boundary_discovery_robust_oracle_full_acq_label_closed_v1
active_boundary_discovery_robust_oracle_simple_acq_label_closed_v1

The report compares run identity/fairness, acquisition formula roles, label
closure, robust-oracle q-window and Delta-refinement metadata, dataset growth,
boundary-focusing diagnostics, acquisition component attribution, walltime, and
historical context relative to the previous working discovery run.
```

Main findings:

```text
1) The label-closure bug is fixed for this mini run.  Full reaches
   dataset_iter005 with 1460 samples = 566 normal + 89 uniform_SC + 805 FFLO.
   Simple-phase reaches dataset_iter005 with 1411 samples = 776 normal +
   9 uniform_SC + 626 FFLO.
2) Every completed iteration appends training-eligible exact points; there is
   no zero-append regime.
3) q_unresolved remains zero in the audited mini-run datasets, and
   boundary-band normal is treated as training-eligible metadata rather than
   solver failure.
4) Runtime config and selected-point diagnostics confirm simple_phase actually
   used the simple acquisition branch with A_numerical=0.  The older
   single-profile report printed a full-profile acquisition-weight template for
   simple_phase; this was a report-builder display error.
5) The report-builder change is report-only.  It changes the displayed
   acquisition formula summary for simple_phase and does not change acquisition
   calculation, exact oracle, StopController, training, tolerance, or Slurm
   workflow.
6) Full acquisition has stronger latest-iteration boundary enrichment in the
   five-iteration mini run; simple_phase has not yet beaten full.
7) Exact-oracle walltime was 13:32:30 for full and 14:19:01 for simple_phase.
   Current logs do not contain point-level runtime decomposition by base scan,
   q expansion, Delta refinement, or local-box refinement.
```

Next recommended steps:

```text
Add robust-oracle runtime instrumentation before optimizing performance.
Prefer an intermediate 10-15 iteration validation before spending a full
100-iteration A/B acquisition-comparison budget.
```

## 69. 2026-06-02 Robust-Oracle Acquisition A/B Mini-Run Report v2

Files generated:

```text
hpc_upload_robust_oracle_label_closure_acq_compare_20260601/ML_Phase_512_20260601/reports/robust_oracle_acquisition_ab_mini_report_v2/
  robust_oracle_acquisition_ab_mini_report_v2.md
  robust_oracle_acquisition_ab_mini_report_v2.tex
  robust_oracle_acquisition_ab_mini_report_v2.pdf
  decision_log.md
  tables/*.csv
  figures/*.png
tmp/generate_ab_mini_report_v2.py
```

Summary:

```text
Regenerated the robust-oracle full-vs-simple acquisition mini-run comparison
as a new v2 report without overwriting the v1 report.  This was a report-only
task: no acquisition function, exact oracle, tolerance, StopController, Slurm
script, or active-learning iteration was changed or launched.
```

Additions over v1:

```text
The v2 report adds acquisition formula and weight comparison tables, explicit
runtime instrumentation gap records, per-iteration robust-oracle q-window and
Delta-refinement summaries, boundary-type distribution split into normal/SC
and uniform-SC/FFLO boundary fractions, selected-point boundary-type maps, and
uniform-SC sample-count analysis.
```

Follow-up completion after checking the original report prompt:

```text
The v2 generator now also writes a complete runtime_instrumentation_gaps.csv
with why_needed/current_availability/recommended_instrumentation_location/
priority columns, a uniform_sc_coverage_summary.csv table, and active-pool plus
selected-point acquisition component attribution split by normal interior, SC
interior, normal/SC boundary, uniform-SC/FFLO boundary, and overlap where the
logs allow it.  The Markdown report now follows the requested 17-section
structure and includes Q1-Q11 status answers in the executive summary.
```

Main findings:

```text
1) The mini-run remains a partially controlled A/B comparison: initial random
   points, candidate grid, robust oracle, batch size, and world size match, but
   Slurm node allocation differs and internal solver-stage timings are not
   instrumented.
2) Label closure remains confirmed: both profiles append normal / uniform_SC /
   FFLO labels and no completed iteration has zero append.
3) Runtime traces confirm simple_phase used the simplified acquisition branch
   with A_numerical=0.  This remains a report/runtime verification; the
   acquisition logic was not changed.
4) Boundary-type splitting changes the interpretation of aggregate focusing:
   simple_phase has strong latest normal/SC boundary focus, while full covers
   the uniform_SC/FFLO boundary and therefore has broader phase-boundary
   coverage.
5) The final uniform_SC sample gap is scientifically important: full has 89
   uniform_SC samples and simple_phase has 9.  This should be tracked as a
   first-class metric in the next 10-15 iteration validation.
6) The exact walltime difference should not be attributed to acquisition-score
   computation because per-point runtime decomposition by base scan, q
   expansion, Delta refinement, local boxes, and branch-candidate output is
   still missing.
```

Validation:

```text
Generated the v2 report and compiled the PDF with pdflatex.  The LaTeX log has
no overfull warnings; only the harmless "No author given" warning remains.
Rendered the PDF pages with pdftoppm and visually checked representative pages
to confirm figures fit within the page text width.
```
## 68. 2026-06-02 Optional incremental q-window expansion and timing instrumentation

Files changed/generated:

```text
ml_phase/exact_oracle.py
scripts/slurm_exact_oracle_array.sh
scripts/compare_incremental_qexpansion_regression.py
scripts/run_qwindow_incremental_benchmark.py
scripts/build_qwindow_incremental_performance_report.py
scripts/package_acquisition_profile_hpc.py
tests/test_incremental_q_expansion.py
tests/test_qexpansion_fallback.py
tests/test_qgrid_alignment.py
tests/test_qscan_cache_merge.py
tests/test_rank_summary_schema.py
tests/test_timing_fields.py
docs/QWINDOW_INCREMENTAL_REFACTOR_PLAN.md
docs/QWINDOW_INCREMENTAL_CALL_GRAPH.md
docs/QWINDOW_INCREMENTAL_DECISION_LOG.md
```

Summary:

```text
Added an optional robust_incremental exact-oracle mode for q-window expansion.
The existing robust_al mode remains the full-rescan baseline.  The incremental
path scans only newly exposed left/right q strips, merges them into a q-scan
cache, and explicitly records fallback_full_rescan_used/reason if it cannot use
the strip path.  Per-point and rank-level timing/workload counters are now
saved for future performance reports.
```

Important constraints preserved:

```text
No thermodynamic phase criterion, Delta tolerance, stable-normal admission,
acquisition, StopController, topology, eta-response, random seed, or AL batch
logic was changed by this performance refactor.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py scripts\compare_incremental_qexpansion_regression.py scripts\run_qwindow_incremental_benchmark.py scripts\build_qwindow_incremental_performance_report.py scripts\package_acquisition_profile_hpc.py
python -m pytest tests\test_qscan_cache_merge.py tests\test_qgrid_alignment.py tests\test_timing_fields.py tests\test_incremental_q_expansion.py tests\test_qexpansion_fallback.py tests\test_rank_summary_schema.py -q

Result: 6 passed.
```

Next recommended steps:

```text
Run scripts/compare_incremental_qexpansion_regression.py on representative
high-JA/low-T correction points and stable-normal controls before using
robust_incremental in any AL mini-run.  Then run
scripts/run_qwindow_incremental_benchmark.py on HPC and summarize exact shard
JSONs with scripts/build_qwindow_incremental_performance_report.py.
```

## 69. 2026-06-02 HPC package for robust incremental q-window validation

Files generated:

```text
hpc_packages/hpc_upload_robust_incremental_qwindow_20260602_v3/
hpc_packages/hpc_upload_robust_incremental_qwindow_20260602_v3.tar.gz
```

Summary:

```text
Created a self-contained HPC upload package that uses oracle_mode=robust_incremental
and INCREMENTAL_Q_EXPANSION_FLAG=--enable-incremental-q-expansion.  The package
contains ml_phase, scripts, docs, tests, eta_phase_diagram_cuda.py,
tfflo_1d_cuda.py, hpc_active_loop.sh, launch scripts for full and simple-phase
acquisition, and a report-only performance collector.
```

Validation:

```text
Inside the package:

python -m py_compile ml_phase\exact_oracle.py scripts\compare_incremental_qexpansion_regression.py scripts\run_qwindow_incremental_benchmark.py scripts\build_qwindow_incremental_performance_report.py
python -m pytest tests\test_qscan_cache_merge.py tests\test_qgrid_alignment.py tests\test_timing_fields.py tests\test_incremental_q_expansion.py tests\test_qexpansion_fallback.py tests\test_rank_summary_schema.py -q

Result: 6 passed.
```

Recommended HPC first step:

```text
Upload hpc_upload_robust_incremental_qwindow_20260602_v3.tar.gz, extract it,
then run N_ITERS=1 bash run_full_incremental.sh and
N_ITERS=1 bash run_simple_incremental.sh as smoke tests before any longer run.
```

Update:

```text
The first HPC smoke attempt failed during candidate generation with Slurm
FAILED:2:0 because ml_phase.active_refine argparse still allowed only
oracle_mode in {legacy, robust_al}.  The launch scripts correctly passed
robust_incremental, but active_refine rejected it before candidate artifacts
were generated.  Fixed active_refine.py to accept robust_incremental as a
pass-through exact-oracle mode.

The next HPC smoke attempt failed with FAILED:1:0 because ml_phase.config
validate_config still allowed only oracle_mode in {legacy, robust_al}.  Fixed
config.py so ActiveLearningConfig validation also accepts robust_incremental.
```

## 70. 2026-06-03 Robust incremental q-window A/B mini-run local report

Files generated:

```text
hpc_upload_robust_incremental_qwindow_20260602_v3/build_incremental_qwindow_ab_report.py
hpc_upload_robust_incremental_qwindow_20260602_v3/ML_Phase_512_Speed_20260602/reports/robust_incremental_qwindow_ab_report/
```

Summary:

```text
Generated a local report from downloaded HPC results for the five-iteration
robust_incremental q-window mini-runs.  The report is audit-only and did not
rerun active learning, exact BdG calculations, Slurm jobs, acquisition, or
StopController logic.  The HPC-side final report failed only because the active
loop tried to read missing report/active_learning_phase_boundary_report.tex
after numerical outputs had already been written.
```

Main results:

```text
Current robust_incremental full acquisition final dataset:
1461 total = 565 normal + 99 uniform_SC + 797 FFLO.

Current robust_incremental simple-phase final dataset:
1399 total = 773 normal + 8 uniform_SC + 618 FFLO.

Five-iteration exact walltime:
current full      = 11:52:07
current simple    = 12:12:25
previous full     = 13:31:50
previous simple   = 14:18:17

Relative to the previous label-closure robust-oracle mini-run, the incremental
q-window package reduced exact walltime by about 12% for full acquisition and
about 15% for simple-phase acquisition.
```

Detailed timing interpretation:

```text
Current exact shards contain point-level timing fields.  Summed over points,
local_refinement dominates compute-seconds in both profiles, followed by base
scan, q expansion, and Delta refinement.  Previous label-closure runs do not
contain the same point-level timing fields, so stage-by-stage speedup relative
to the previous run cannot be claimed from current artifacts.
```

Next recommended steps:

```text
Fix the missing report template/path in the HPC active-loop package, or make the
active loop call the local report-only collector after final iteration.  Do not
change acquisition or exact-oracle physics based on this report-generation
failure.
```

Update:

```text
Generated a more complete LaTeX v2 report:

hpc_upload_robust_incremental_qwindow_20260602_v3/ML_Phase_512_Speed_20260602/reports/robust_incremental_qwindow_ab_report_latex_v2/

The v2 report includes expanded narrative sections, explicit horizontal
full-vs-simple comparison, vertical walltime comparison to the previous
label-closure mini-run, detailed current exact-solver stage timing, a
do-not-claim list, Markdown/PDF companions, 16 CSV tables, and 12 PNG figures.
The PDF was built with pdflatex and the final log did not report overfull boxes,
undefined references, fatal errors, or missing figures.
```

## 71. 2026-06-03 Local-refinement static code audit report

Files generated:

```text
report/build_local_refinement_static_audit.py
report/local_refinement_static_audit/local_refinement_static_audit.tex
report/local_refinement_static_audit/local_refinement_static_audit.pdf
report/local_refinement_static_audit/local_refinement_static_audit.md
report/local_refinement_static_audit/decision_log.md
report/local_refinement_static_audit/tables/
report/local_refinement_static_audit/figures/
```

Summary:

```text
Generated a report-only LaTeX code audit based on
hpc_upload_robust_incremental_qwindow_20260602_v3/Code_review.md.  The local
checkout displayed the Chinese prose in that file as mojibake, but the readable
code blocks specified a local-refinement static audit.  The audit reads
ml_phase/exact_oracle.py, eta_phase_diagram_cuda.py, current robust-incremental
exact shard timing metadata, and branch candidate CSV files.  It does not modify
production numerical logic, acquisition, tolerances, StopController, or Slurm.
```

Main conclusions:

```text
Local refinement is confirmed as the dominant current point-level compute cost.
The robust path already has a max_refined_minima top-k cap.  Current metadata is
point-level, not local-box-level, so safe top-k tightening, energy-window
pruning, branch reuse, adaptive boxes, or GPU batching should wait for
logging-only box-level instrumentation and fixed-point regression.  Branch reuse
is not active in the current exact oracle because local_refinement_reused_count
is written as zero.
```

Validation:

```text
pdflatex was run twice on local_refinement_static_audit.tex.  The final PDF has
7 pages.  The final LaTeX log did not report overfull boxes, undefined
references, fatal errors, or missing figures.  The report directory contains 21
CSV tables and 12 PNG figures.
```

## 72. 2026-06-03 Root-level artifact reorganization

Files generated/changed:

```text
scripts/organize_root_artifacts_20260603.ps1
project_history/README.md
project_history/root_reorganization_manifest_20260603.csv
project_history/root_level_inventory_after_20260603.csv
project_history/remaining_root_items_20260603.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Created a first-level archive folder named project_history/ and moved historical
generated artifacts out of the repository root.  The organization is grouped by
improvement stage rather than by raw file type: qdelta warmup uploads,
discovery runs, phase q-window/Delta refinement, robust-oracle acquisition
comparison, label-closure A/B, incremental q-window validation, smoke runs,
historical reports, raw exact data, plans/runbooks, legacy analysis, external
docs, and local temporary/cache files.
```

Preserved in root:

```text
Core source and canonical project files were intentionally left in place:
ml_phase/, scripts/, tests/, docs/, AGENTS.md, MODEL_SPEC.md,
eta_phase_diagram_cuda.py, tfflo_1d_cuda.py, hpc_active_loop.sh,
hpc_one_click_submit.sh, run_*.sh, report/, reports/, and other likely active
entry points.
```

Known unresolved cleanup:

```text
Six root items could not be moved by normal or elevated PowerShell Move-Item and
remain in the root.  They are listed in
project_history/remaining_root_items_20260603.md.  The blocked items are large
recent HPC result directories and id_rsa; the failure appears to be local
permission, sync, or file-lock related.  No forced deletion or ACL rewrite was
performed.
```

Next recommended steps:

```text
If a fully clean root is required, close any OneDrive/sync/indexing processes or
move the blocked directories manually in Explorer, then update
project_history/root_reorganization_manifest_20260603.csv.  Do not move core
source directories or launch entry points without first checking import paths
and HPC scripts.
```

## 73. 2026-06-03 Local-refinement refactor Stage 0 scaffold

Files generated/changed:

```text
docs/LOCAL_REFINEMENT_REFACTOR_MASTER_PLAN.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_CALL_GRAPH.md
ml_phase/report_builder.py
hpc_active_loop.sh
scripts/build_local_refinement_regression_points.py
scripts/run_local_refinement_fixed_point_regression.py
scripts/compare_local_refinement_variants.py
tests/test_report_template_path.py
tests/test_local_refinement_regression_scaffold.py
reports/local_refinement_refactor/stage_00_baseline/
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Started the two-phase/local-refinement optimization goal with Stage 0 baseline
freeze.  Stage 0 records the robust_incremental baseline, creates fixed-point
regression scaffolds, adds report-template path guardrails, and separates final
report-generation failure from numerical active-loop completion.  It does not
change the thermodynamic phase criterion, Delta tolerances, stable-normal
admission, acquisition, StopController, exact-oracle local-refinement selection,
or Slurm submission logic.
```

Baseline state:

```text
oracle_mode = robust_incremental
max_refined_minima = 6
basin clustering = off
energy pruning = off
branch reuse = off
adaptive box = off
GPU batching = off
Hamiltonian cache = off
```

Validation:

```text
python -m py_compile ml_phase\report_builder.py scripts\build_local_refinement_regression_points.py scripts\run_local_refinement_fixed_point_regression.py scripts\compare_local_refinement_variants.py
python -m pytest tests\test_report_template_path.py tests\test_local_refinement_regression_scaffold.py -q

Result: 3 passed.

Generated reports/local_refinement_refactor/stage_00_baseline/fixed_point_regression_points.csv
with 32 points: 4 each for stable normal interior, boundary-band normal, clean
uniform_SC, clean FFLO, previous normal-to-FFLO correction, q-edge risk,
rerun-required, and near-degenerate/Delta-ambiguous categories.

Dry-run regression manifest was written under
reports/local_refinement_refactor/stage_00_baseline/regression_dry_run/.
```

Known unresolved issue:

```text
The exact fixed-point regression has not been run on the intended GPU/CUDA
environment.  Local bash syntax validation of hpc_active_loop.sh was also not
available because WSL has no installed Linux distributions on this machine.
```

Next recommended steps:

```text
Run scripts/run_local_refinement_fixed_point_regression.py without --dry-run on
the target GPU/CUDA environment, then compare any future local-refinement
variants against that baseline with scripts/compare_local_refinement_variants.py.
Only after the baseline fixed-point regression passes should Stage 1 box-level
instrumentation begin.
```

## 74. 2026-06-03 Local-refinement refactor Stage 1 plan

Files generated/changed:

```text
reports/local_refinement_refactor/stage_01_instrumentation/plan.md
reports/local_refinement_refactor/stage_01_instrumentation/decision_log.md
reports/local_refinement_refactor/stage_01_instrumentation/tables/local_box_timing_schema.csv
reports/local_refinement_refactor/stage_01_instrumentation/tables/local_refinement_summary_schema.csv
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Planned Stage 1 box-level instrumentation before editing exact-oracle code.
The plan identifies the existing insertion point in _confirm_one_point_robust:
after refine_targets are selected and around each local _run_scan_with_normal
box call.  Stage 1 is explicitly logging-only and must not change
refine_targets, local box bounds, local grid sizes, final minimum selection,
phase criteria, Delta tolerances, acquisition, StopController, or active
learning append behavior.
```

Planned outputs:

```text
performance/iterXXX_local_box_timing_rankYYY_ofZZZ.csv
performance/iterXXX_local_refinement_summary_rankYYY_ofZZZ.json
reports/local_refinement_refactor/stage_01_instrumentation/tables/
```

Required regression:

```text
With --enable-local-box-instrumentation enabled on the Stage 0 fixed-point set:
phase_candidate, trusted_exact, training_eligible_exact, q_unresolved,
delta_unresolved, and rerun_required must match the Stage 0 baseline exactly.
q_opt, Delta_opt, and DeltaF must match within existing floating tolerances.
```

Current status:

```text
Stage 1 is planned but not implemented.  The next action is to add the feature
flag and logging-only CSV writer, then run unit tests and fixed-point
baseline-vs-instrumented regression.
```

Update:

```text
Stage 1 local implementation is now complete behind an explicit feature flag.
The exact oracle accepts --enable-local-box-instrumentation and optional
--local-box-output-file.  When enabled, it writes local-box timing/effectivity
rows and a local-refinement summary JSON; when disabled, no local-box
instrumentation file is produced.  scripts/slurm_exact_oracle_array.sh passes
LOCAL_BOX_INSTRUMENTATION_FLAG through to the exact-oracle CLI, defaulting to
disabled.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py scripts\run_local_refinement_fixed_point_regression.py
python -m pytest tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_report_template_path.py -q

Result: 5 passed.

python scripts\run_local_refinement_fixed_point_regression.py --points-file reports\local_refinement_refactor\stage_00_baseline\fixed_point_regression_points.csv --output-dir reports\local_refinement_refactor\stage_01_instrumentation\regression_dry_run --dry-run --enable-local-box-instrumentation

Result: dry-run passed.
```

Important constraint:

```text
Stage 1 only logs existing local-refinement boxes.  It does not alter
refine_targets, local box bounds, local grid sizes, local_scan calls, final
minimum selection, phase labels, trusted flags, training eligibility, or rerun
logic.
```

Remaining required validation:

```text
Run the Stage 0 baseline exact fixed-point regression and the Stage 1
instrumented fixed-point regression on the target GPU/CUDA environment, then
compare them with scripts/compare_local_refinement_variants.py.  Stage 2 basin
clustering should not start until phase/trusted/training/rerun flags match
exactly.
```

## 75. 2026-06-03 Local-refinement refactor Stage 1 HPC package

Files generated/changed:

```text
scripts/package_local_refinement_refactor_hpc.py
hpc_packages/local_refinement_refactor_stage01_instrumentation/
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Prepared the Stage 1 fixed-point GPU regression package for the local-box
instrumentation gate.  The package includes the fixed-point CSV, exact-oracle
code, regression runner, comparison script, Slurm submit wrappers, manifest,
and Stage 0/1 planning docs.  It is fixed-point regression only: it does not
run active learning and does not append training data.
```

Important implementation decision:

```text
The package builder excludes historical ml_phase run-output directories
active_runs, datasets, figures, hpc_jobs, models, and reports from both the
runnable ml_phase copy and the code snapshot.  This keeps the HPC artifact
focused on source code and fixed-point regression inputs rather than old
generated outputs.
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_refactor_hpc.py
python scripts\package_local_refinement_refactor_hpc.py
python -m py_compile hpc_packages\local_refinement_refactor_stage01_instrumentation\ml_phase\exact_oracle.py hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\run_local_refinement_fixed_point_regression.py hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\compare_local_refinement_variants.py
python scripts\run_local_refinement_fixed_point_regression.py --points-file fixed_points\fixed_point_regression_points.csv --output-dir reports\local_refinement_refactor\stage_01_instrumentation\regression_dry_run_package --dry-run --enable-local-box-instrumentation
```

Result:

```text
Package archive created at
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.
The rebuilt archive is about 348 KB and does not contain ml_phase historical
run-output directories, __pycache__, or the package dry-run output.  Package
dry-run passed with 32 fixed points across 8 categories and did not run exact
oracle calculations.
```

Known unresolved issue:

```text
The required target GPU/CUDA exact regression is still pending.  Stage 2 basin
clustering must not begin until the Stage 0 baseline and Stage 1 instrumented
fixed-point GPU outputs compare with zero mismatch in phase_candidate,
trusted_exact, training_eligible_exact, q_unresolved, delta_unresolved, and
rerun_required.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz to
the target HPC project directory, run scripts/submit_stage0_baseline_regression.sh
and scripts/submit_stage1_instrumented_regression.sh, then run
scripts/compare_stage1_regression.sh after both jobs finish.
```

## 76. 2026-06-03 Local-refinement Stage 1 gate verifier

Files generated/changed:

```text
scripts/verify_local_refinement_stage1_gate.py
tests/test_local_refinement_stage1_gate.py
scripts/package_local_refinement_refactor_hpc.py
reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented/stage1_gate_status.json
reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented/stage1_gate_status.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added an artifact-level verifier for the Stage 1 fixed-point GPU regression
gate.  The verifier checks that baseline and instrumented exact regression
CSV/manifest files exist, comparison outputs exist, the instrumented local-box
timing CSV exists and has required columns, the expected 32 fixed points are
covered, required classification/status fields have zero mismatches, and
q_opt/Delta_opt/DeltaF differences are within configured verifier thresholds.
```

Important implementation decision:

```text
The verifier does not recompute exact BdG physics and does not define new phase
or Delta criteria.  It only verifies whether the required Stage 0/1 regression
artifacts prove logging-only equivalence.  The floating-difference thresholds
are command-line verifier thresholds for comparing two completed runs, not
changes to production numerical tolerances.
```

Validation:

```text
python -m py_compile scripts\verify_local_refinement_stage1_gate.py scripts\package_local_refinement_refactor_hpc.py scripts\compare_local_refinement_variants.py
python -m pytest tests\test_local_refinement_stage1_gate.py tests\test_local_refinement_regression_scaffold.py -q

Result: 3 passed.

python scripts\verify_local_refinement_stage1_gate.py

Result: expected nonzero exit because required GPU regression artifacts are
missing.  The verifier wrote stage1_gate_status.json/md with status=fail and
the missing file list.
```

HPC package update:

```text
Rebuilt hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.
The package now includes scripts/verify_local_refinement_stage1_gate.py and
scripts/verify_stage1_gate.sh.  The README instructs running
compare_stage1_regression.sh followed by verify_stage1_gate.sh after both
GPU regression jobs complete.
```

Current project state:

```text
Stage 0 and Stage 1 remain locally complete but GPU-regression-pending.
Stage 2 remains pending and must not start until stage1_gate_status.json reports
status=pass for the returned target GPU/CUDA exact regression artifacts.
```

Next recommended steps:

```text
Run the Stage 0 and Stage 1 fixed-point exact regression jobs on HPC, run
scripts/compare_stage1_regression.sh, then run scripts/verify_stage1_gate.sh.
Only if the verifier reports status=pass should Stage 2 basin-clustering
planning begin.
```

## 77. 2026-06-03 Local-refinement Stage 1 HPC return collector

Files generated/changed:

```text
scripts/collect_local_refinement_stage1_outputs.py
scripts/package_local_refinement_refactor_hpc.py
tests/test_local_refinement_stage1_gate.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
reports/local_refinement_refactor/stage_01_instrumentation/decision_log.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
```

Summary:

```text
Added a cross-platform Python collector for Stage 1 HPC regression return
artifacts and a package-level Bash wrapper.  The collector runs the Stage 1 gate
verifier, writes return_bundle_metadata/return_manifest.json and
missing_paths.txt, and creates
local_refinement_refactor_stage1_regression_results.tar.gz containing available
logs, regression outputs, comparison outputs, gate status, package manifest,
README, and fixed-point CSV.
```

Important implementation decision:

```text
The collector is evidence packaging only.  It does not submit Slurm jobs, rerun
exact calculations, compare variants differently, or change the Stage 2 entry
criterion.  It intentionally continues when the gate fails so incomplete HPC
attempts still return diagnostic logs and missing-path metadata.
```

Validation:

```text
python -m py_compile scripts\collect_local_refinement_stage1_outputs.py scripts\verify_local_refinement_stage1_gate.py scripts\package_local_refinement_refactor_hpc.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q

Result: 3 passed.

python scripts\collect_local_refinement_stage1_outputs.py --archive local_refinement_refactor_stage1_regression_results_test.tar.gz
    (run inside hpc_packages/local_refinement_refactor_stage01_instrumentation)

Result: collector succeeded with gate_status=fail because GPU outputs are still
missing, and the test archive contained README.md, RUN_MANIFEST.json,
stage1_gate_status.json/md, return_bundle_metadata, and the fixed-point CSV.
```

HPC package update:

```text
Rebuilt hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.
The package now includes scripts/collect_local_refinement_stage1_outputs.py and
scripts/collect_stage1_regression_outputs.sh.  The README instructs running:

bash scripts/compare_stage1_regression.sh
bash scripts/verify_stage1_gate.sh
bash scripts/collect_stage1_regression_outputs.sh
```

Current project state:

```text
Stage 0 and Stage 1 are still locally complete but GPU-regression-pending.
The current gate status remains fail because exact GPU artifacts are missing.
Stage 2 remains pending and must not start until the returned target GPU/CUDA
artifacts make stage1_gate_status.json report status=pass.
```

Next recommended steps:

```text
Upload the rebuilt Stage 1 package, run the two fixed-point GPU regression
Slurm jobs, run compare_stage1_regression.sh, run verify_stage1_gate.sh, then
run collect_stage1_regression_outputs.sh and return
local_refinement_refactor_stage1_regression_results.tar.gz for local audit.
```

## 78. 2026-06-03 Local-refinement Stage 1 workflow submitter

Files generated/changed:

```text
scripts/package_local_refinement_refactor_hpc.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
reports/local_refinement_refactor/stage_01_instrumentation/decision_log.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
```

Summary:

```text
Extended the Stage 1 HPC package generator to write
scripts/submit_stage1_regression_workflow.sh and
scripts/slurm_stage1_postprocess.sh.  The workflow submitter launches the Stage
0 baseline fixed-point regression and Stage 1 instrumented fixed-point
regression, then submits the postprocess job with an afterok dependency on both
exact jobs.  The postprocess job runs compare_stage1_regression.sh,
verify_stage1_gate.sh, and collect_stage1_regression_outputs.sh.
```

Important implementation decision:

```text
This is Slurm orchestration only.  It does not change exact-oracle physics,
local-box instrumentation behavior, comparison logic, pass criteria,
acquisition, StopController, Delta tolerances, or active-learning append logic.
If the HPC scheduler requires a partition for the postprocess job, use
POSTPROCESS_SBATCH_ARGS, for example
POSTPROCESS_SBATCH_ARGS="--partition=NV_H100".
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_refactor_hpc.py scripts\collect_local_refinement_stage1_outputs.py scripts\verify_local_refinement_stage1_gate.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q

Result: 3 passed.

python scripts\package_local_refinement_refactor_hpc.py

Inspected the generated package scripts and archive contents.  The archive
contains submit_stage1_regression_workflow.sh and slurm_stage1_postprocess.sh,
and does not contain return-bundle test artifacts, __pycache__, or historical
ml_phase run-output directories.
```

Current project state:

```text
Stage 0 and Stage 1 remain locally complete but GPU-regression-pending.
The current gate status remains fail because target GPU exact artifacts are
missing.  Stage 2 remains pending until the returned target GPU/CUDA run makes
stage1_gate_status.json report status=pass.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz,
extract it on HPC, run bash scripts/submit_stage1_regression_workflow.sh, then
return local_refinement_refactor_stage1_regression_results.tar.gz after the
dependent postprocess job completes.
```

## 79. 2026-06-03 Local-refinement Stage 1 return-bundle importer

Files generated/changed:

```text
scripts/import_local_refinement_stage1_results.py
tests/test_local_refinement_stage1_gate.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
reports/local_refinement_refactor/stage_01_instrumentation/decision_log.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a local importer for returned Stage 1 HPC regression bundles.  The
importer extracts local_refinement_refactor_stage1_regression_results.tar.gz
into a staging directory under
reports/local_refinement_refactor/stage_01_instrumentation/imported_results/
and reruns the Stage 1 artifact-level gate verifier against the extracted
bundle.
```

Important implementation decision:

```text
The importer does not extract returned HPC files into the repository root and
does not promote imported files into canonical local reports automatically.
This avoids overwriting local README/manifest/log/report files and keeps failed
or incomplete HPC attempts isolated for audit.
```

Validation:

```text
python -m py_compile scripts\import_local_refinement_stage1_results.py scripts\collect_local_refinement_stage1_outputs.py scripts\verify_local_refinement_stage1_gate.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q

Result: 4 passed.
```

Current project state:

```text
No returned Stage 1 GPU regression archive is present locally.  The current
canonical gate status remains fail because baseline/instrumented GPU exact
artifacts are missing.  Stage 2 remains pending until an imported or canonical
stage1_gate_status.json reports status=pass.
```

Next recommended steps:

```text
After the HPC workflow returns
local_refinement_refactor_stage1_regression_results.tar.gz, run:

python scripts/import_local_refinement_stage1_results.py local_refinement_refactor_stage1_regression_results.tar.gz

Then inspect the imported stage1_gate_status.json before deciding whether Stage
2 basin-clustering planning may start.
```

## 80. 2026-06-03 Local-refinement Stage 1 package validator

Files generated/changed:

```text
scripts/validate_local_refinement_hpc_package.py
scripts/package_local_refinement_refactor_hpc.py
tests/test_local_refinement_stage1_gate.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
reports/local_refinement_refactor/stage_01_instrumentation/package_validation.json
```

Summary:

```text
Added an automated package-integrity validator for the Stage 1 HPC regression
package.  The validator checks the extracted package directory and tar.gz
archive for required files, required RUN_MANIFEST keys, README workflow
commands, shell script shebangs and LF line endings, and absence of transient
return/import metadata, returned result archives, __pycache__, and historical
ml_phase run-output directories.
```

Important implementation decision:

```text
The validator is packaging QA only.  It does not run exact calculations,
compare physics outputs, change pass criteria, or modify active-learning logic.
It formalizes the manual archive/content checks used before uploading the Stage
1 package to HPC.
```

Validation:

```text
python -m py_compile scripts\validate_local_refinement_hpc_package.py scripts\package_local_refinement_refactor_hpc.py scripts\verify_local_refinement_stage1_gate.py scripts\collect_local_refinement_stage1_outputs.py scripts\import_local_refinement_stage1_results.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q

Result: 5 passed.

python scripts\package_local_refinement_refactor_hpc.py
python scripts\validate_local_refinement_hpc_package.py

Result: package validation status=pass, directory_file_count=93,
archive_file_count=93, failures=[].
```

Current project state:

```text
The Stage 1 HPC package is structurally validated and ready for upload.  The
Stage 1 scientific/behavioral gate is still pending because no target GPU/CUDA
exact regression artifacts have returned.  Stage 2 remains pending until the
Stage 1 gate verifier reports status=pass on returned GPU results.
```

Next recommended steps:

```text
Upload the validated package archive, run the Stage 1 regression workflow on
HPC, return the generated result bundle, import it locally with
scripts/import_local_refinement_stage1_results.py, and inspect the imported
stage1_gate_status.json.
```

## 81. 2026-06-03 Local-refinement Stage 1 package checksum sidecars

Files generated/changed:

```text
scripts/package_local_refinement_refactor_hpc.py
scripts/validate_local_refinement_hpc_package.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
reports/local_refinement_refactor/stage_01_instrumentation/decision_log.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.sha256
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.metadata.json
reports/local_refinement_refactor/stage_01_instrumentation/package_validation.json
```

Summary:

```text
Added portable SHA256 and metadata sidecars for the Stage 1 fixed-point HPC
package.  The package validator now recomputes the archive hash and fails if
the .sha256 or .metadata.json sidecar is missing or inconsistent with the
tarball.
```

Important implementation decision:

```text
This is upload/package QA only.  It does not change exact-oracle physics,
local-box instrumentation behavior, comparison pass criteria, acquisition,
StopController, Delta/q tolerances, or active-learning append logic.
```

Validation:

```text
python -m py_compile scripts\validate_local_refinement_hpc_package.py scripts\package_local_refinement_refactor_hpc.py scripts\verify_local_refinement_stage1_gate.py scripts\collect_local_refinement_stage1_outputs.py scripts\import_local_refinement_stage1_results.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q

Result: 5 passed.

python scripts\package_local_refinement_refactor_hpc.py
python scripts\validate_local_refinement_hpc_package.py

Result: package validation status=pass, directory_file_count=93,
archive_file_count=93, checksum sidecars verified, failures=[].
```

Current project state:

```text
The Stage 1 upload package now includes integrity sidecars and is structurally
validated.  The Stage 1 scientific/behavioral gate is still pending because no
target GPU/CUDA exact regression artifacts have returned.  Stage 2 remains
pending until the Stage 1 gate verifier reports status=pass on returned GPU
results.
```

Next recommended steps:

```text
Upload the archive plus sidecars, run sha256sum -c on HPC, execute
bash scripts/submit_stage1_regression_workflow.sh, return the generated result
bundle, import it locally, and inspect the imported stage1_gate_status.json
before Stage 2 planning.
```

## 82. 2026-06-03 Local-refinement Stage 1 HPC handoff note

Files generated/changed:

```text
docs/report_qa/20260603_local_refinement_stage1_hpc_handoff.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a report-QA handoff note for the Stage 1 fixed-point HPC regression
gate.  The note records the three upload files, the HPC checksum command, the
workflow submit command, the returned archive name, the local import command,
and the exact Stage 1 pass criteria.
```

Important implementation decision:

```text
This is documentation and handoff material only.  It does not start Stage 2,
does not change exact-oracle physics, does not alter instrumentation behavior,
and does not change the Stage 1 gate requirement.
```

Current project state:

```text
The Stage 1 package is ready for upload with checksum sidecars.  The Stage 1
scientific gate still reports status=fail because target GPU/CUDA exact
regression artifacts have not returned.  Stage 2 remains pending.
```

Next recommended steps:

```text
Upload the archive and sidecars, verify sha256sum on HPC, run the workflow
submitter, return local_refinement_refactor_stage1_regression_results.tar.gz,
and import it locally before any Stage 2 planning.
```

## 83. 2026-06-03 Local-refinement Stage 1 runtime preflight

Files generated/changed:

```text
scripts/preflight_local_refinement_stage1_hpc.py
scripts/package_local_refinement_refactor_hpc.py
scripts/validate_local_refinement_hpc_package.py
tests/test_local_refinement_stage1_gate.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/PROJECT_SUMMARY.md
reports/local_refinement_refactor/stage_01_instrumentation/stage1_runtime_preflight_local_package.json
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.sha256
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz.metadata.json
reports/local_refinement_refactor/stage_01_instrumentation/package_validation.json
```

Summary:

```text
Added a Stage 1 extracted-package runtime preflight.  It checks required package
paths, RUN_MANIFEST invariants, the 32-row fixed-point CSV, syntax of key
Python scripts without writing __pycache__, and records a torch/CUDA visibility
snapshot.  The Stage 1 workflow submitter now runs this preflight before
submitting the baseline and instrumented Slurm jobs.
```

Important implementation decision:

```text
The preflight is a package/runtime sanity check only.  It does not run exact BdG
calculations, does not compare physics outputs, does not change pass criteria,
and does not unblock Stage 2.  On a login node, CUDA visibility is recorded but
not required unless the preflight is explicitly called with --require-cuda.
```

Validation:

```text
python -m py_compile scripts\preflight_local_refinement_stage1_hpc.py scripts\package_local_refinement_refactor_hpc.py scripts\validate_local_refinement_hpc_package.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q

Result: 6 passed.

python scripts\package_local_refinement_refactor_hpc.py
python scripts\preflight_local_refinement_stage1_hpc.py --package-root hpc_packages\local_refinement_refactor_stage01_instrumentation --output-json reports\local_refinement_refactor\stage_01_instrumentation\stage1_runtime_preflight_local_package.json
python scripts\validate_local_refinement_hpc_package.py

Result: preflight status=pass and package validation status=pass.
```

Current project state:

```text
Stage 1 now has upload integrity checks and extracted-package runtime preflight
checks.  The scientific/behavioral gate still waits for target GPU/CUDA exact
baseline and instrumented regression artifacts.  Stage 2 remains pending.
```

Next recommended steps:

```text
Upload the rebuilt archive and sidecars, verify sha256sum on HPC, extract the
package, run the preflight, submit the workflow, return the result bundle, and
import it locally before any Stage 2 planning.
```

## 84. 2026-06-03 Local-refinement goal-run audit summary

Files generated/changed:

```text
scripts/audit_local_refinement_refactor_goal_run.py
reports/local_refinement_refactor_goal_run/goal_run_summary.md
reports/local_refinement_refactor_goal_run/goal_run_summary.tex
reports/local_refinement_refactor_goal_run/goal_run_summary.pdf
reports/local_refinement_refactor_goal_run/goal_run_audit_summary.json
reports/local_refinement_refactor_goal_run/decision_log.md
reports/local_refinement_refactor_goal_run/tables/stage_status.csv
reports/local_refinement_refactor_goal_run/tables/evidence_matrix.csv
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a goal-run audit generator for the local-refinement refactor.  It reads
the current Stage 1 gate status, package validation summary, and runtime
preflight summary, then writes a Markdown/LaTeX/PDF goal-run summary, a
decision log, and CSV tables for stage status and evidence.
```

Important implementation decision:

```text
The audit is evidence reporting only.  It does not change physics, numerical
methods, package behavior, instrumentation, or active-learning logic.  It
explicitly records that Stage 2 remains pending until the Stage 1 target
GPU/CUDA regression gate passes.
```

Validation:

```text
python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py
python scripts\audit_local_refinement_refactor_goal_run.py
pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex

Result: audit status=blocked_by_stage1_gpu_gate; stage1_gate_status=fail;
package_validation_status=pass; preflight_status=pass.  The PDF compiled
successfully as a 1-page report.
```

Current project state:

```text
The goal-run report now has machine-readable evidence that the refactor is not
complete and that the active blocker is the missing Stage 1 GPU equivalence
gate.  Stage 2, basin clustering, selective refinement, pruning, branch reuse,
adaptive boxes, GPU batching, and Hamiltonian cache remain pending.
```

Next recommended steps:

```text
Run the Stage 1 fixed-point regression workflow on HPC, return and import the
result bundle, then rerun scripts/audit_local_refinement_refactor_goal_run.py
to update the goal-run evidence before planning Stage 2.
```

## 85. 2026-06-03 Goal-run audit figure output

Files generated/changed:

```text
scripts/audit_local_refinement_refactor_goal_run.py
reports/local_refinement_refactor_goal_run/figures/stage_gate_status.png
reports/local_refinement_refactor_goal_run/goal_run_summary.md
reports/local_refinement_refactor_goal_run/goal_run_summary.tex
reports/local_refinement_refactor_goal_run/goal_run_summary.pdf
reports/local_refinement_refactor_goal_run/goal_run_audit_summary.json
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Extended the goal-run audit generator to write a PNG stage-gate status figure
and embed it in both the Markdown and LaTeX/PDF summaries.  The figure makes the
current state visually explicit: Stage 0 and Stage 1 are local-complete but GPU
regression pending, while Stages 2-7 remain pending.
```

Important implementation decision:

```text
This is report synchronization only.  It does not alter the Stage 1 gate, does
not start Stage 2 planning, and does not change physics, numerical methods,
instrumentation, package execution, or active-learning behavior.
```

Validation:

```text
python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py
python scripts\audit_local_refinement_refactor_goal_run.py
pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex

Result: goal_run_summary.pdf compiled successfully and
figures/stage_gate_status.png exists.  The report still records
status=blocked_by_stage1_gpu_gate.
```

Current project state:

```text
The goal-run report now satisfies the report handoff requirement for Markdown,
PDF, CSV tables, decision log, and PNG figure evidence.  The scientific blocker
is unchanged: Stage 1 target GPU/CUDA exact regression outputs are still
missing, so Stage 2 remains pending.
```

## 86. 2026-06-03 Goal-run report protocol verifier

Files generated/changed:

```text
scripts/verify_local_refinement_goal_run_report.py
reports/local_refinement_refactor_goal_run/goal_run_report_validation.json
docs/report_qa/20260603_local_refinement_stage1_hpc_handoff.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a standalone verifier for the local-refinement refactor goal-run report.
It checks that the Markdown, LaTeX, PDF, decision log, CSV tables, and PNG
figure exist and are non-empty, that the stage table covers Stages 0-7, that
Stage 2 remains pending, and that the audit summary agrees with the source
Stage 1 gate, package validation, and runtime preflight JSON files.
```

Important implementation decision:

```text
The verifier is report-protocol QA only.  It does not change any physical
definition, numerical method, acquisition rule, instrumentation path, or HPC
workflow.  It deliberately keeps the distinction between package/preflight
success and the unresolved Stage 1 target GPU/CUDA equivalence gate.
```

Validation:

```text
python -m py_compile scripts\verify_local_refinement_goal_run_report.py scripts\audit_local_refinement_refactor_goal_run.py
python scripts\verify_local_refinement_goal_run_report.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q
python scripts\validate_local_refinement_hpc_package.py
python scripts\preflight_local_refinement_stage1_hpc.py --package-root hpc_packages\local_refinement_refactor_stage01_instrumentation --output-json reports\local_refinement_refactor\stage_01_instrumentation\stage1_runtime_preflight_local_package.json
python scripts\audit_local_refinement_refactor_goal_run.py
pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex

Result: verifier status=pass with no errors; pytest reported 6 passed; the
refreshed Stage 1 HPC package validation and extracted-package preflight both
reported status=pass; goal_run_summary.pdf compiled successfully.
```

Current project state:

```text
The goal-run report now has a machine-readable validation companion file.  The
main blocker is unchanged: Stage 1 gate status is still fail because the target
GPU fixed-point baseline/instrumented artifacts are missing.  Stage 2 remains
pending and has not been planned or implemented.
```

Next recommended steps:

```text
Upload the refreshed Stage 1 HPC package and sidecars, verify the archive
checksum on HPC, run the extracted-package preflight and fixed-point workflow,
return the result bundle, and import it locally before any Stage 2 planning.
```

## 87. 2026-06-03 Stage 1 HPC package import-path fix

Files generated/changed:

```text
scripts/run_local_refinement_fixed_point_regression.py
scripts/preflight_local_refinement_stage1_hpc.py
tests/test_local_refinement_stage1_gate.py
docs/report_qa/20260603_local_refinement_stage1_import_path_fix.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Fixed the Stage 1 HPC fixed-point runner so package-root modules remain
importable when the job is launched as
python scripts/run_local_refinement_fixed_point_regression.py from the extracted
package root.  The runner now inserts its repository/package root into
sys.path before importing eta_phase_diagram_cuda and ml_phase.exact_oracle.
The package preflight now performs isolated import checks for those modules so
this failure mode is caught before Slurm exact jobs are submitted.  The import
check disables bytecode writing so preflight does not create `__pycache__`
files inside the extracted package.
```

Important implementation decision:

```text
This is an execution-path repair only.  It does not alter the physical model,
phase criterion, q-window policy, Delta-refinement policy, fixed-point set,
local-box instrumentation semantics, or Stage 1 pass criteria.
```

Validation:

```text
python -m py_compile scripts\run_local_refinement_fixed_point_regression.py scripts\preflight_local_refinement_stage1_hpc.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q
python scripts\validate_local_refinement_hpc_package.py
python scripts\preflight_local_refinement_stage1_hpc.py --package-root hpc_packages\local_refinement_refactor_stage01_instrumentation --output-json reports\local_refinement_refactor\stage_01_instrumentation\stage1_runtime_preflight_local_package.json
python scripts\validate_local_refinement_hpc_package.py

Result: syntax checks passed; pytest reported 7 passed, including a
package-style subprocess test that executes the fixed-point runner from
scripts/ with package-root stub modules.  Package validation passed before and
after extracted-package preflight; preflight reported import_check_count=2 for
eta_phase_diagram_cuda and ml_phase.exact_oracle.
```

Current project state:

```text
The reported HPC ModuleNotFoundError is fixed locally and covered by tests.
The previous uploaded/extracted Stage 1 package should be replaced with a
rebuilt package.  Stage 1 still requires target GPU/CUDA baseline and
instrumented exact-regression outputs before the gate can pass.
```

Next recommended steps:

```text
Rebuild and revalidate the Stage 1 HPC package, upload the refreshed archive
and sidecars to HPC, rerun preflight, submit the fixed-point workflow, return
the result bundle, and import it locally.  Stage 2 remains pending.
```

## 88. 2026-06-03 Stage 1 package-root path hardening

Files generated/changed:

```text
scripts/run_local_refinement_fixed_point_regression.py
scripts/compare_local_refinement_variants.py
scripts/verify_local_refinement_stage1_gate.py
scripts/collect_local_refinement_stage1_outputs.py
scripts/preflight_local_refinement_stage1_hpc.py
scripts/package_local_refinement_refactor_hpc.py
tests/test_local_refinement_stage1_gate.py
docs/report_qa/20260603_local_refinement_stage1_import_path_fix.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Hardened the Stage 1 HPC package so runtime entry points resolve relative paths
against the extracted package root, which is inferred from the script location,
not from the caller's current working directory.  The fixed-point runner,
comparison script, gate verifier, collector, and preflight now follow this
package-root convention.  Generated Slurm exact and postprocess scripts also
derive PROJECT_DIR from their own location under scripts/ instead of relying on
$PWD.
```

Important implementation decision:

```text
The package should be self-contained after extraction.  Runtime scripts may use
absolute paths only when explicitly supplied by the caller; otherwise package
relative paths such as fixed_points/ and reports/ are interpreted inside the
extracted package.  This change is execution-path hardening only and does not
alter physics definitions, numerical methods, fixed-point data, local-box
instrumentation, or Stage 1 gate criteria.
```

Validation:

```text
python -m py_compile scripts\run_local_refinement_fixed_point_regression.py scripts\compare_local_refinement_variants.py scripts\verify_local_refinement_stage1_gate.py scripts\collect_local_refinement_stage1_outputs.py scripts\preflight_local_refinement_stage1_hpc.py scripts\package_local_refinement_refactor_hpc.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q

Result: syntax checks passed; pytest reported 7 passed.  The fixed-point runner
test now launches the package script from an external working directory and
verifies that outputs are written inside the package rather than under the
external cwd.
```

Current project state:

```text
The Stage 1 package no longer depends on local/HPC directory layout matching,
provided the archive is extracted and the package's own scripts are used.
Stage 1 still needs target GPU/CUDA baseline and instrumented exact regression
outputs before the gate can pass.  Stage 2 remains pending.
```

Next recommended steps:

```text
Rebuild and revalidate the Stage 1 HPC package, upload the refreshed archive
and sidecars, rerun package preflight on HPC, submit the fixed-point workflow,
return the result bundle, and import it locally before any Stage 2 planning.
```

## 89. 2026-06-03 Stage 1 writable RUN_ROOT support

Files generated/changed:

```text
scripts/run_local_refinement_fixed_point_regression.py
scripts/compare_local_refinement_variants.py
scripts/verify_local_refinement_stage1_gate.py
scripts/collect_local_refinement_stage1_outputs.py
scripts/preflight_local_refinement_stage1_hpc.py
scripts/package_local_refinement_refactor_hpc.py
tests/test_local_refinement_stage1_gate.py
docs/report_qa/20260603_local_refinement_stage1_hpc_handoff.md
docs/report_qa/20260603_local_refinement_stage1_import_path_fix.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Fixed the HPC permission failure where Stage 1 package scripts attempted to
create logs/ and reports/ inside a non-writable extracted package directory.
The package now separates PACKAGE_ROOT from RUN_ROOT: PACKAGE_ROOT is the
extracted package directory used for code and fixed_points/, while RUN_ROOT is
the writable output directory used for logs/, reports/, and the returned result
archive.  If RUN_ROOT is not set, scripts write under the package root when it
is writable, otherwise they fall back to SCRATCH, TMPDIR, or HOME.
```

Important implementation decision:

```text
The Stage 1 package remains self-contained for inputs and code, but outputs no
longer require write permission in the package directory.  This is an HPC
execution-permission fix only.  It does not change exact-oracle physics,
fixed-point inputs, q-window or Delta policies, local-box instrumentation, or
Stage 1 gate criteria.
```

Validation:

```text
python -m py_compile scripts\run_local_refinement_fixed_point_regression.py scripts\compare_local_refinement_variants.py scripts\verify_local_refinement_stage1_gate.py scripts\collect_local_refinement_stage1_outputs.py scripts\preflight_local_refinement_stage1_hpc.py scripts\package_local_refinement_refactor_hpc.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q
python scripts\package_local_refinement_refactor_hpc.py
python scripts\validate_local_refinement_hpc_package.py
RUN_ROOT=<external writable dir> python hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\preflight_local_refinement_stage1_hpc.py --package-root hpc_packages\local_refinement_refactor_stage01_instrumentation --output-json reports\local_refinement_refactor\stage_01_instrumentation\stage1_runtime_preflight_runroot_probe.json
RUN_ROOT=<external writable dir> python hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\run_local_refinement_fixed_point_regression.py --points-file fixed_points\fixed_point_regression_points.csv --output-dir reports\local_refinement_refactor\stage_01_instrumentation\regression_dry_run_runroot_probe --dry-run --enable-local-box-instrumentation
python scripts\validate_local_refinement_hpc_package.py

Result: syntax checks passed; pytest reported 8 passed; package validation
reported status=pass; RUN_ROOT preflight and RUN_ROOT dry-run both wrote to the
external writable run directory; package validation still passed after the
probes.
```

Current project state:

```text
The Stage 1 package should now run even when the extracted package directory is
read-only, as long as RUN_ROOT points to a writable directory.  Stage 1 still
needs target GPU/CUDA baseline and instrumented exact regression outputs before
the gate can pass.  Stage 2 remains pending.
```

Next recommended steps:

```text
Upload the refreshed archive and sidecars, export RUN_ROOT to a writable HPC
directory, run preflight with --package-root . --run-root "$RUN_ROOT", submit
the workflow, return "$RUN_ROOT"/local_refinement_refactor_stage1_regression_results.tar.gz,
and import the returned bundle locally before any Stage 2 planning.
```

## 90. 2026-06-03 Stage 1 gpuh01 exclusion and CUDA runtime probe

Files generated/changed:

```text
scripts/package_local_refinement_refactor_hpc.py
scripts/validate_local_refinement_hpc_package.py
docs/report_qa/20260603_local_refinement_stage1_cuda_node_exclusion.md
docs/report_qa/20260603_local_refinement_stage1_hpc_handoff.md
reports/local_refinement_refactor/stage_01_instrumentation/regression_summary.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Applied the existing project decision to exclude gpuh01 to the Stage 1
local-refinement HPC package.  The generated Stage 0 baseline and Stage 1
instrumented GPU exact-regression Slurm scripts now include
#SBATCH --exclude=gpuh01.  They also run a CUDA runtime tensor-allocation probe
before calling the fixed-point exact oracle, so driver mismatches fail early in
the job log rather than inside the solver.
```

Important implementation decision:

```text
This is scheduling/runtime protection only.  It does not change exact-oracle
physics, fixed-point inputs, q-window or Delta policies, local-box
instrumentation, comparison tolerances, or Stage 1 gate criteria.  The package
validator now fails if either GPU exact Slurm script lacks the gpuh01 exclusion
or CUDA runtime probe.
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_refactor_hpc.py scripts\validate_local_refinement_hpc_package.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q
python scripts\package_local_refinement_refactor_hpc.py
python scripts\validate_local_refinement_hpc_package.py
python scripts\preflight_local_refinement_stage1_hpc.py --package-root hpc_packages\local_refinement_refactor_stage01_instrumentation --run-root . --output-json reports\local_refinement_refactor\stage_01_instrumentation\stage1_runtime_preflight_local_package.json
python scripts\validate_local_refinement_hpc_package.py

Result: syntax checks passed; pytest reported 8 passed; package validation
reported status=pass and confirmed the generated GPU Slurm scripts contain the
gpuh01 exclusion and CUDA runtime probe; preflight reported status=pass.
```

Current project state:

```text
The Stage 1 package should no longer be scheduled on gpuh01 by default.  Stage
1 still needs target GPU/CUDA baseline and instrumented exact regression
outputs from compatible nodes before the gate can pass.  Stage 2 remains
pending.
```

Next recommended steps:

```text
Upload the refreshed archive and sidecars, export RUN_ROOT to a writable HPC
directory, verify sha256sum, rerun preflight, submit the workflow, return the
result bundle from RUN_ROOT, and import it locally before any Stage 2 planning.
```

## 91. 2026-06-03 Downloaded Stage 1 package inspection

Files generated/changed:

```text
docs/report_qa/20260603_local_refinement_stage1_downloaded_package_check.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Inspected the downloaded
local_refinement_refactor_stage01_instrumentation directory after the HPC queue
became empty.  The downloaded folder is PACKAGE_ROOT only; it does not contain
the returned result archive or pointwise CSV outputs because those were written
to RUN_ROOT.  The top-level Slurm logs in the downloaded package record that
baseline and instrumented GPU fixed-point exact jobs completed and the
postprocess job ran compare, gate verification, and collection.
```

Evidence from downloaded Slurm outputs:

```text
slurm-70684.out: baseline exact regression wrote outputs under
/public_hw/home/sci_bfu/local_refinement_refactor_stage1_run/reports/local_refinement_refactor/stage_00_baseline/regression_gpu_baseline

slurm-70685.out: instrumented exact regression wrote outputs under
/public_hw/home/sci_bfu/local_refinement_refactor_stage1_run/reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_instrumented

slurm-70686.out: postprocess reported n_common_points=32, no missing or extra
candidate points, flag_mismatch_count=0, max q/Delta/DeltaF differences all
0.0, local_box_rows=192, and gate_status=pass.
```

Important implementation decision:

```text
No code or physical definitions were changed during this inspection.  The
current evidence indicates the Stage 1 HPC gate passed on the cluster, but the
local repository still needs the returned archive imported before Stage 2
planning should start.
```

Current project state:

```text
The required return archive is expected at:
/public_hw/home/sci_bfu/local_refinement_refactor_stage1_run/local_refinement_refactor_stage1_regression_results.tar.gz

The local downloaded PACKAGE_ROOT does not contain this archive.  Local Stage 1
canonical reports remain unimported until that RUN_ROOT archive is downloaded.
```

Next recommended steps:

```text
Download
/public_hw/home/sci_bfu/local_refinement_refactor_stage1_run/local_refinement_refactor_stage1_regression_results.tar.gz
to the repository root, then run:

python scripts/import_local_refinement_stage1_results.py local_refinement_refactor_stage1_regression_results.tar.gz

After local import confirms status=pass, update the goal-run audit and begin
Stage 2 planning.
```

## 92. 2026-06-03 Downloaded Stage 1 result import and gate confirmation

Files generated/changed:

```text
scripts/import_local_refinement_stage1_results.py
tests/test_local_refinement_stage1_gate.py
local_refinement_refactor_stage01_instrumentation/imported_results/stage1_regression_results/
docs/report_qa/20260603_local_refinement_stage1_downloaded_package_check.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
The returned Stage 1 RUN_ROOT archive was found in the downloaded package tree:
local_refinement_refactor_stage01_instrumentation/local_refinement_refactor_stage1_run/local_refinement_refactor_stage1_regression_results.tar.gz

It was imported locally under:
local_refinement_refactor_stage01_instrumentation/imported_results/stage1_regression_results

The local import verified the Stage 1 gate report and confirmed status=pass.
Generated/imported outputs were kept inside the downloaded package tree rather
than the repository-root reports/ directory.
```

Gate evidence:

```text
expected_points=32
baseline_rows=32
candidate_rows=32
comparison_rows=32
local_box_rows=192
missing_files=[]
failures=[]
n_common_points=32
n_missing_in_candidate=0
n_extra_in_candidate=0
flag_mismatch_count=0
max_deltaf_abs_diff=0.0
max_q_opt_abs_diff=0.0
max_delta_opt_abs_diff=0.0
mismatch_points_empty=true

baseline exact runtime: 7342.084829706699 s
instrumented exact runtime: 7339.3942006491125 s
baseline phase_counts: normal=8, SC=24
instrumented phase_counts: normal=8, SC=24
trusted_count=12
training_eligible_count=16
rerun_required_count=16
```

Important implementation decision:

```text
The Stage 1 result importer now inserts the repository root into sys.path when
executed directly as python scripts/import_local_refinement_stage1_results.py,
fixing the direct-script ModuleNotFoundError for imports from scripts/.

The importer default output location was changed from the repository-root
reports/local_refinement_refactor/stage_01_instrumentation/imported_results
tree to imported_results/ next to the returned archive.  Callers can still
override this with --import-root, but the default now follows the downloaded
package/RUN_ROOT locality rule.

These are execution-path and output-location fixes only.  They do not change
the physical model, q/Delta policies, exact fixed-point set, local-box
instrumentation semantics, comparison tolerances, or Stage 1 pass criteria.
```

Validation:

```text
python scripts/import_local_refinement_stage1_results.py local_refinement_refactor_stage01_instrumentation\local_refinement_refactor_stage1_run\local_refinement_refactor_stage1_regression_results.tar.gz --import-root local_refinement_refactor_stage01_instrumentation\imported_results --extract-dir local_refinement_refactor_stage01_instrumentation\imported_results\stage1_regression_results
python -m py_compile scripts\import_local_refinement_stage1_results.py tests\test_local_refinement_stage1_gate.py
python -m pytest tests\test_local_refinement_stage1_gate.py -q

Result: local import returned gate_status=pass with no missing files or
failures; pytest reported 9 passed, including direct CLI importer coverage and
default archive-adjacent import-root coverage.
```

Current project state:

```text
Stage 1 is locally confirmed as passed from the returned target GPU/CUDA
baseline-vs-instrumented fixed-point regression bundle.  Stage 2 remains
unimplemented but is no longer blocked by the missing Stage 1 result import.

The root-level goal-run report was not rewritten during this check because the
current output-location rule is to keep generated/imported outputs inside the
downloaded package tree unless an explicit root report refresh is requested.
```

Next recommended steps:

```text
Plan Stage 2 from the locally confirmed Stage 1 gate evidence.  If a canonical
root-level goal-run report refresh is needed later, write it only after choosing
an explicit non-root or approved report output location.
```

## 93. 2026-06-03 Stage 2 basin-clustering local minimal implementation

Files generated/changed:

```text
ml_phase/exact_oracle.py
scripts/run_local_refinement_fixed_point_regression.py
tests/test_basin_clustering.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
reports/local_refinement_refactor/stage_02_basin_clustering/plan.md
reports/local_refinement_refactor/stage_02_basin_clustering/implementation_summary.md
reports/local_refinement_refactor/stage_02_basin_clustering/test_summary.md
reports/local_refinement_refactor/stage_02_basin_clustering/decision_log.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Started the post-Stage-1 local-refinement optimization work from
TwoPhase_Optimization.md.  Stage 2 was decomposed into basin identity,
mandatory-risk preservation, representative metadata, point-level counters, and
disabled-by-default feature-flag integration.

Added a pure `cluster_branch_candidates(...)` helper that merges duplicate
coarse local minima into basin representatives using coarse-grid q, Delta, and
DeltaF proximity.  Representatives aggregate mandatory-risk reasons so
global-best, edge-risk, Delta-near-epsilon, and near-degenerate basins are not
silently removed by clustering.
```

Important implementation decision:

```text
Stage 2 basin clustering is feature-flagged off by default.  The baseline
robust_incremental local-box path remains available and unchanged unless
enable_basin_clustering is explicitly set.

This stage does not alter thermodynamic phase labels, q-window policy,
Delta-refinement policy, stable-normal admission, acquisition formula,
StopController behavior, candidate-domain strategy, eta-response policy, or
topology policy.  It also does not introduce pruning, branch reuse, adaptive
boxes, GPU batching, Hamiltonian caching, or a production active-learning
restart.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py scripts\run_local_refinement_fixed_point_regression.py tests\test_basin_clustering.py
python -m pytest tests\test_basin_clustering.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: syntax checks passed; targeted tests reported 15 passed; full local
test suite reported 23 passed.
```

Current project state:

```text
Stage 0 and Stage 1 are completed with imported target GPU/CUDA Stage 1 gate
evidence.  Stage 2 is locally minimal-complete with synthetic tests and CLI/API
feature flags, but no Stage 2 GPU fixed-point variant has been run yet.

No new HPC package was generated during this stage; this follows the current
workflow preference to continue local minimal tests through later stages and
assemble the HPC package set after the local stage implementations are ready.
```

Next recommended steps:

```text
Proceed to Stage 3 mandatory-risk keep and selective refinement.  Keep
mandatory basins eligible regardless of ordinary caps, add synthetic tests for
global-best, edge-risk, Delta-near-epsilon, and near-degenerate preservation,
and avoid generating the combined HPC package set until the later local stages
are minimally tested.
```

## 94. 2026-06-03 Stage 3 selective-refinement local minimal implementation

Files generated/changed:

```text
ml_phase/exact_oracle.py
tests/test_selective_refinement.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
reports/local_refinement_refactor/stage_03_selective_refinement/plan.md
reports/local_refinement_refactor/stage_03_selective_refinement/implementation_summary.md
reports/local_refinement_refactor/stage_03_selective_refinement/test_summary.md
reports/local_refinement_refactor/stage_03_selective_refinement/decision_log.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Implemented the Stage 3 mandatory-risk keep and selective-refinement policy as
a disabled-by-default helper, `select_local_refine_targets(...)`.  In legacy
mode it preserves the previous local-refinement target-selection behavior and
total cap.  In selective mode it keeps mandatory-risk basins first and then
adds a limited number of ordinary basins.
```

Important implementation decision:

```text
Selective refinement is opt-in through enable_selective_refinement.  The
setting max_optional_refined_basins limits ordinary basin refinement, while
mandatory_basins_can_exceed_cap controls whether mandatory basins can exceed
the total cap.

This stage does not change thermodynamic phase labels, q-window policy,
Delta-refinement policy, basin clustering default state, acquisition formula,
StopController behavior, energy-window pruning, branch reuse, adaptive boxes,
GPU batching, Hamiltonian caching, or production active-learning behavior.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_basin_clustering.py tests\test_selective_refinement.py
python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: syntax checks passed; targeted tests reported 18 passed; full local
test suite reported 26 passed.
```

Current project state:

```text
Stages 2 and 3 are locally minimal-complete and feature-flagged off by default.
No Stage 2 or Stage 3 GPU fixed-point variant has been run yet.  No new HPC
package was generated during this stage.
```

Next recommended steps:

```text
Proceed to Stage 4 energy-window pruning.  Keep pruning opt-in, apply it only
to ordinary non-mandatory basins, record pruned reasons, and add synthetic tests
showing that mandatory basins are never pruned by the energy window.
```

## 95. 2026-06-03 Stage 4 energy-window pruning local minimal implementation

Files generated/changed:

```text
ml_phase/exact_oracle.py
scripts/run_local_refinement_fixed_point_regression.py
tests/test_energy_window_pruning.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
reports/local_refinement_refactor/stage_04_energy_pruning/plan.md
reports/local_refinement_refactor/stage_04_energy_pruning/implementation_summary.md
reports/local_refinement_refactor/stage_04_energy_pruning/test_summary.md
reports/local_refinement_refactor/stage_04_energy_pruning/decision_log.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Implemented the Stage 4 energy-window pruning policy as a disabled-by-default
helper, `mark_energy_window_pruning(...)`.  The helper marks only ordinary
non-mandatory basins as pruned when their energy above the global candidate
exceeds the configured pruning window.  The target selector skips pruned
ordinary basins.
```

Important implementation decision:

```text
Energy-window pruning is opt-in through energy_window_pruning_enabled and does
not apply to global-best, edge-risk, Delta-near-epsilon, or near-degenerate
basins.  Pruned rows carry pruned_reason=ordinary_above_energy_window.

This stage does not change thermodynamic phase labels, q-window policy,
Delta-refinement policy, acquisition formula, StopController behavior, branch
reuse, adaptive boxes, GPU batching, Hamiltonian caching, or production
active-learning behavior.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py scripts\run_local_refinement_fixed_point_regression.py tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py
python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: syntax checks passed; targeted tests reported 21 passed; full local
test suite reported 29 passed.
```

Current project state:

```text
Stages 2, 3, and 4 are locally minimal-complete and feature-flagged off by
default.  No Stage 2/3/4 GPU fixed-point variants have been run yet.  No new
HPC package was generated during this stage.
```

Next recommended steps:

```text
Proceed to Stage 5 branch-reuse prototype.  Keep reuse opt-in, require exact
configuration/signature matches, record reuse rejection reasons, and add
synthetic tests showing reuse is rejected when a lower-energy competing branch
appears.
```

## 96. 2026-06-03 Stage 5 branch-reuse decision prototype

Files generated/changed:

```text
ml_phase/exact_oracle.py
tests/test_branch_reuse.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
reports/local_refinement_refactor/stage_05_branch_reuse/plan.md
reports/local_refinement_refactor/stage_05_branch_reuse/implementation_summary.md
reports/local_refinement_refactor/stage_05_branch_reuse/test_summary.md
reports/local_refinement_refactor/stage_05_branch_reuse/decision_log.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Implemented the Stage 5 branch-reuse decision prototype as pure helpers:
`branch_reuse_signature(...)` and `evaluate_branch_reuse_candidate(...)`.
The prototype creates stable configuration signatures and returns explicit
reuse accept/reject decisions with named rejection reasons.
```

Important implementation decision:

```text
Stage 5 does not connect branch reuse to the production local-refinement loop.
Reuse is rejected if cached data are missing, solver or local-box signatures do
not match, q/Delta/energy values are outside tolerance, or a lower-energy
competing branch is present.  This avoids silent reuse before box-level
reuse/rejection diagnostics and GPU fixed-point regression exist.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_branch_reuse.py tests\test_energy_window_pruning.py tests\test_selective_refinement.py tests\test_basin_clustering.py
python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_branch_reuse.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: syntax checks passed; targeted tests reported 25 passed; full local
test suite reported 33 passed.
```

Current project state:

```text
Stages 2, 3, and 4 are locally minimal-complete and feature-flagged off by
default.  Stage 5 has local branch-reuse decision primitives and tests, but
production-loop reuse integration remains pending.  No new HPC package was
generated during this stage.
```

Next recommended steps:

```text
Proceed to Stage 6 adaptive local-box skeleton.  Record basin width and
curvature proxies with fixed boxes still as the default, add synthetic tests for
proxy computation, and keep adaptive boxes disabled until later GPU regression.
```

## 97. 2026-06-03 Stage 6 adaptive local-box skeleton

Files generated/changed:

```text
ml_phase/exact_oracle.py
tests/test_adaptive_local_box_skeleton.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
reports/local_refinement_refactor/stage_06_adaptive_box_skeleton/plan.md
reports/local_refinement_refactor/stage_06_adaptive_box_skeleton/implementation_summary.md
reports/local_refinement_refactor/stage_06_adaptive_box_skeleton/test_summary.md
reports/local_refinement_refactor/stage_06_adaptive_box_skeleton/decision_log.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Implemented the Stage 6 adaptive local-box skeleton as pure helpers:
`estimate_basin_geometry(...)` and `adaptive_local_box_half_widths(...)`.
Clustered basin representatives now carry diagnostic q width, Delta width,
DeltaF span, and curvature proxy fields.  Adaptive half-width suggestions are
bounded and default to fixed-box values when disabled.
```

Important implementation decision:

```text
Adaptive local boxes are diagnostics-only in this stage.  The production
exact-oracle local scan still uses fixed local box widths.  No adaptive-box
variant has been run on GPU.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_adaptive_local_box_skeleton.py tests\test_branch_reuse.py tests\test_energy_window_pruning.py tests\test_selective_refinement.py tests\test_basin_clustering.py
python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py tests\test_local_box_instrumentation.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_stage1_gate.py -q
python -m pytest tests -q

Result: syntax checks passed; targeted tests reported 28 passed; full local
test suite reported 36 passed.
```

Current project state:

```text
Stages 2, 3, and 4 are locally minimal-complete and feature-flagged off by
default.  Stage 5 and Stage 6 have local prototype/skeleton helpers and tests,
but production integration remains pending.  No new HPC package was generated
during this stage.
```

Next recommended steps:

```text
Proceed to Stage 7 GPU batching and Hamiltonian cache planning.  Keep it as a
planning/skeleton stage only: record tensor-shape requirements, cache
signature inputs, profiler hook locations, and future GPU validation gates
without changing production exact calculations.
```

## 98. 2026-06-03 Stage 7 GPU batching/cache and HPC package-set planning

Files generated/changed:

```text
reports/local_refinement_refactor/stage_07_hpc_packaging/plan.md
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Completed the Stage 7 planning skeleton for GPU local-box batching,
Hamiltonian cache signatures, profiler hook locations, future GPU validation
gates, and the combined HPC package set.
```

Important implementation decision:

```text
Stage 7 does not implement GPU batching or Hamiltonian caching.  No production
exact calculation changes were made.  The combined HPC package set remains
pending until variant-runner support can explicitly enable and validate the
Stage 2/3/4 variants and until Stage 5/6 integration diagnostics are designed.
```

Validation:

```text
Stage 7 added planning documents only.  The latest full executable validation
remains:

python -m pytest tests -q
Result: 36 passed.
```

Current project state:

```text
Stages 0 and 1 are complete with returned GPU/CUDA gate evidence.  Stages 2-4
are locally minimal-complete and default-off.  Stage 5 is a tested branch-reuse
decision prototype but not production-integrated.  Stage 6 is a tested adaptive
box geometry skeleton but not production-integrated.  Stage 7 planning is
complete.  The combined HPC package set has not yet been generated.
```

Next recommended steps:

```text
Add variant-runner support for explicit Stage 2 clustering, Stage 3 selective
refinement, and Stage 4 energy-pruning GPU fixed-point validations.  Then build
the combined HPC package set under hpc_packages/ for upload, keeping package
runtime outputs under each extracted package/RUN_ROOT directory.
```

## 99. 2026-06-03 Local-refinement variant runner and HPC package set

Files generated/changed:

```text
scripts/run_local_refinement_fixed_point_regression.py
scripts/package_local_refinement_variant_suite_hpc.py
tests/test_local_refinement_regression_scaffold.py
tests/test_local_refinement_variant_suite_package.py
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/DECISIONS.md
reports/local_refinement_refactor/stage_07_hpc_packaging/plan.md
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added explicit fixed-point regression variants for the runnable local
refinement optimization stages:

baseline
cluster_only
cluster_optional_k3
cluster_optional_k2
cluster_energy_window

The runner now resolves each variant to an explicit feature-flag dictionary and
writes that dictionary into regression_manifest.json for both dry-run and exact
execution.  The unintegrated cluster_energy_reuse variant is rejected with an
explicit error because Stage 5 branch reuse is still a decision prototype, not
a production exact-oracle loop feature.
```

Important implementation decision:

```text
The variant-suite package defaults runtime outputs to:

$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run

when RUN_ROOT is unset and the extracted package is writable.  Otherwise it
falls back to SCRATCH, TMPDIR, or HOME.  This avoids writing logs or reports to
the repository root or a non-writable login/current directory.
```

Generated HPC package:

```text
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
```

Package workflow:

```text
bash scripts/submit_variant_suite_regression_workflow.sh

Runs separate Slurm jobs for baseline, cluster_only, cluster_optional_k3,
cluster_optional_k2, and cluster_energy_window, excluding gpuh01 by default.
After all variants finish, the postprocess job compares every non-baseline
variant against baseline and collects:

$RUN_ROOT/local_refinement_refactor_variant_suite_results.tar.gz
```

Validation:

```text
python -m py_compile scripts\run_local_refinement_fixed_point_regression.py scripts\package_local_refinement_variant_suite_hpc.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_variant_suite_package.py
Result: passed

python -m pytest tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_stage1_gate.py -q
Result: 14 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports/local_refinement_refactor/variant_regression/preflight.json
Result: status=pass, fixed_point_count=32

python -m pytest tests -q
Result: 40 passed
```

Current project state:

```text
Stages 0 and 1 are complete with returned GPU/CUDA gate evidence.  Stages 2-4
are locally implemented and now have explicit runnable fixed-point regression
variants plus a generated HPC package.  The Stage 2/3/4 GPU variant runs are
still pending.  Stage 5 branch reuse and Stage 6 adaptive boxes remain
prototype/skeleton code and are not submitted as production variants.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_variant_suite.tar.gz to the
cluster, extract it, run scripts/submit_variant_suite_regression_workflow.sh,
and return $RUN_ROOT/local_refinement_refactor_variant_suite_results.tar.gz for
local import/check.
```

## 100. 2026-06-03 Local-refinement variant-suite return importer

Files generated/changed:

```text
scripts/import_local_refinement_variant_suite_results.py
scripts/package_local_refinement_variant_suite_hpc.py
tests/test_local_refinement_variant_suite_package.py
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
```

Summary:

```text
Added a local importer/checker for returned variant-suite HPC results:

scripts/import_local_refinement_variant_suite_results.py

The checker safely extracts the returned archive, verifies RUN_MANIFEST.json,
fixed-point count, every runnable variant's pointwise CSV and exact-run
manifest, and every baseline-vs-variant comparison summary.  It writes
variant_suite_gate_status.json/md under the extracted result tree and writes
latest_variant_suite_import_manifest.json under the import root.
```

Gate policy:

```text
Required variants:
baseline
cluster_only
cluster_optional_k3
cluster_optional_k2
cluster_energy_window

Required comparison pass conditions:
n_common_points = expected_points
n_missing_in_candidate = 0
n_extra_in_candidate = 0
flag_mismatch_count = 0
mismatch_points.csv empty
max_q_opt_abs_diff <= 1e-10
max_delta_opt_abs_diff <= 1e-10
max_deltaf_abs_diff <= 1e-8
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_variant_suite_hpc.py scripts\import_local_refinement_variant_suite_results.py tests\test_local_refinement_variant_suite_package.py
Result: passed

python -m pytest tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_regression_scaffold.py -q
Result: 7 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: rebuilt hpc_packages\local_refinement_refactor_variant_suite.tar.gz with importer included

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports/local_refinement_refactor/variant_regression/preflight.json
Result: status=pass, fixed_point_count=32

python -m pytest tests -q
Result: 42 passed
```

Current project state:

```text
The local side can now both generate the Stage 2/3/4 variant-suite upload
package and import/check the returned result archive.  Actual GPU fixed-point
variant results are still pending until the package is run on the cluster.
```

Next recommended steps:

```text
Upload and run the rebuilt
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.  After download,
verify the returned archive with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 110. 2026-06-04 Runbook-compatible HPC submit aliases

Files changed:

```text
scripts/package_local_refinement_refactor_hpc.py
scripts/package_local_refinement_variant_suite_hpc.py
scripts/package_local_refinement_upload_set.py
scripts/validate_local_refinement_hpc_package.py
scripts/preflight_local_refinement_stage1_hpc.py
tests/test_local_refinement_stage1_gate.py
tests/test_local_refinement_variant_suite_package.py
tests/test_local_refinement_upload_set.py
docs/DECISIONS.md
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
hpc_packages/local_refinement_refactor_stage01_instrumentation/
hpc_packages/local_refinement_refactor_stage01_instrumentation.tar.gz
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
reports/local_refinement_refactor_goal_run/
```

Summary:

```text
Added runbook-compatible submit aliases to the generated Stage 1 and
variant-suite HPC packages:

scripts/submit_local_refinement_fixed_point_regression.sh
scripts/submit_local_refinement_instrumented_benchmark.sh

The aliases only exec the existing package workflow scripts:

Stage 1 -> scripts/submit_stage1_regression_workflow.sh
Variant suite -> scripts/submit_variant_suite_regression_workflow.sh
```

Important implementation decision:

```text
No new one-iteration or intermediate active-learning submit scripts were
generated because those workflows are not validated in the current package
set.  The aliases do not change physical definitions, q/Delta safeguards,
feature flags, exact-oracle behavior, branch reuse status, adaptive-box
status, GPU batching, Hamiltonian cache, or performance-gate thresholds.
```

Upload policy:

```text
The upload-set RUN_ORDER now names the runbook-compatible fixed-point alias as
the required next command:

bash scripts/submit_local_refinement_fixed_point_regression.sh

inside the extracted local_refinement_refactor_variant_suite package. Runtime
outputs still default to the package-local RUN_ROOT:

$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_refactor_hpc.py scripts\package_local_refinement_variant_suite_hpc.py scripts\package_local_refinement_upload_set.py scripts\validate_local_refinement_hpc_package.py scripts\preflight_local_refinement_stage1_hpc.py tests\test_local_refinement_stage1_gate.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_upload_set.py
Result: passed

python -m pytest tests\test_local_refinement_stage1_gate.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_upload_set.py -q
Result: 15 passed

python scripts\package_local_refinement_refactor_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_stage01_instrumentation.tar.gz

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\validate_local_refinement_hpc_package.py --package-dir hpc_packages\local_refinement_refactor_stage01_instrumentation --archive hpc_packages\local_refinement_refactor_stage01_instrumentation.tar.gz
Result: status=pass

python hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\preflight_local_refinement_stage1_hpc.py --package-root hpc_packages\local_refinement_refactor_stage01_instrumentation --run-root hpc_packages\local_refinement_refactor_stage01_instrumentation\local_refinement_refactor_stage1_run --output-json hpc_packages\local_refinement_refactor_stage01_instrumentation\local_refinement_refactor_stage1_run\reports\local_refinement_refactor\stage_01_instrumentation\stage1_runtime_preflight.json
Result: status=pass, fixed_point_count=32

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor\variant_regression\preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass, package_count=2, required_next_package=local_refinement_refactor_variant_suite

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, upload_set_verify_status=pass

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=13

python -m pytest tests -q
Result: 68 passed
```

Current package metadata:

```text
local_refinement_refactor_stage01_instrumentation.tar.gz:
sha256 = 4e4dfb991c7253c14b1d788ca5452dd76fd3105f1f7b74b9a6927eb7afbff102
size   = 438361 bytes
package_file_count = 106

local_refinement_refactor_variant_suite.tar.gz:
sha256 = 0733afff92854a7c7b7f8efef02b97a84e93e02540e855ec1f93d9f964d4bc98
size   = 512630 bytes
package_file_count = 187

local_refinement_refactor_hpc_upload_set.tar.gz:
sha256 = 4a4556efba0cadb9d80a3dd69321e2476e8e5f15d828d9355e826a312bd0bbc4
size   = 958306 bytes
package_count = 2
```

Current project state:

```text
Stage 1 remains completed and imported locally as a pass.  Stage 2-4 remain
local-minimal-complete and package-ready, but GPU validation is still pending
until the returned variant-suite archive is imported with gate_status=pass,
import_status=pass, and performance_report_status=pass.  Stage 5/6 production
integration and Stage 7 production GPU batching/cache remain deferred.
```

## 110. 2026-06-04 Stage report completeness audit

Files changed:

```text
reports/local_refinement_refactor/stage_02_basin_clustering/regression_summary.md
reports/local_refinement_refactor/stage_03_selective_refinement/regression_summary.md
reports/local_refinement_refactor/stage_04_energy_pruning/regression_summary.md
reports/local_refinement_refactor/stage_05_branch_reuse/regression_summary.md
reports/local_refinement_refactor/stage_06_adaptive_box_skeleton/regression_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/regression_summary.md
reports/local_refinement_refactor/stage_report_completeness.json
scripts/audit_local_refinement_stage_reports.py
tests/test_local_refinement_stage_report_completeness.py
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
reports/local_refinement_refactor_goal_run/
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Completed the runbook-required stage report file set for all Stage 0-7
directories.  Stage 2-7 were missing regression_summary.md; each now records
the local regression evidence, the relevant GPU variant gate, current pending
status, and regression risks.

Added scripts/audit_local_refinement_stage_reports.py to check that every
stage directory has:

plan.md
implementation_summary.md
test_summary.md
regression_summary.md
decision_log.md

The audit writes a machine-readable companion file:

reports/local_refinement_refactor/stage_report_completeness.json
```

Important implementation decision:

```text
This is a documentation and audit-completeness change only.  It does not modify
the exact-oracle calculation, Hamiltonian, physical labels, q/Delta numerical
guardrails, acquisition logic, or StopController behavior.

Stage 2-4 remain GPU-variant pending.  Stage 5 and Stage 6 remain local
contracts only.  Stage 7 remains package-handoff ready, with production GPU
batching/cache still deferred until the necessary GPU equivalence gates exist.
```

Validation:

```text
python scripts\audit_local_refinement_stage_reports.py
Result: status=pass, stage_count=8, required_report_count=5, checked_file_count=40, missing_count=0

python -m py_compile scripts\audit_local_refinement_stage_reports.py tests\test_local_refinement_stage_report_completeness.py
Result: passed

python -m pytest tests\test_local_refinement_stage_report_completeness.py -q
Result: 1 passed

python -m pytest tests\test_local_refinement_stage_report_completeness.py tests\test_local_refinement_goal_run_audit.py -q
Result: 3 passed

python -m pytest tests -q
Result: 61 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor\variant_regression\preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass, package_count=2, required_next_package=local_refinement_refactor_variant_suite

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, variant_preflight_status=pass

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass
```

Current package metadata:

```text
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

The exact current archive hashes are intentionally kept in the sidecar metadata
files rather than duplicated here, because this project summary is copied into
the generated package and would otherwise make the archive hash self-referential.

Current project state:

```text
The Stage 0-7 report directories now satisfy the runbook file-completeness
requirement.  The latest variant-suite package includes the new Stage 2-7
regression summaries and is locally preflighted.  The active goal still waits
for the external GPU variant-suite return before Stage 2-4 can be marked
GPU-validated or before Stage 5-7 production integrations can be promoted.
```

Next recommended steps:

```text
Upload the refreshed hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
or hpc_packages/local_refinement_refactor_variant_suite.tar.gz, run the
variant-suite fixed-point regression on the target GPU/CUDA environment, and
return local_refinement_refactor_variant_suite_results.tar.gz for local import.
```

## 111. 2026-06-04 Runbook-named test matrix completion

Files changed:

```text
tests/test_mandatory_branch_keep.py
tests/test_feature_flag_baseline_equivalence.py
tests/test_local_refinement_runbook_test_matrix.py
tests/test_local_refinement_variant_suite_package.py
scripts/audit_local_refinement_runbook_tests.py
scripts/package_local_refinement_variant_suite_hpc.py
reports/local_refinement_refactor/runbook_test_matrix.json
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/regression_summary.md
reports/local_refinement_refactor_goal_run/
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Completed the runbook-named local-refinement test file matrix by adding:

tests/test_mandatory_branch_keep.py
tests/test_feature_flag_baseline_equivalence.py

The mandatory-branch test exercises the clustering -> energy-window pruning ->
selective-refinement chain and verifies that global-best, edge-risk,
Delta-near-epsilon, and near-degenerate mandatory branches are not pruned or
lost when mandatory branches are allowed to exceed the ordinary cap.

The feature-flag baseline test verifies that exact-oracle defaults and the
fixed-point regression baseline variant keep basin clustering, selective
refinement, and energy-window pruning disabled.

Added scripts/audit_local_refinement_runbook_tests.py, which writes:

reports/local_refinement_refactor/runbook_test_matrix.json
```

Important implementation decision:

```text
This is a local test/audit and packaging-consistency change only.  It does not
modify the exact-oracle solver, the Hamiltonian, thermodynamic phase labels,
q/Delta numerical guardrails, acquisition logic, or StopController behavior.

The variant-suite package now includes the new runbook test files and the audit
scripts they import, so package-local tests do not depend on missing repository
scripts.
```

Validation:

```text
python -m py_compile scripts\audit_local_refinement_runbook_tests.py tests\test_mandatory_branch_keep.py tests\test_feature_flag_baseline_equivalence.py tests\test_local_refinement_runbook_test_matrix.py
Result: passed

python -m pytest tests\test_mandatory_branch_keep.py tests\test_feature_flag_baseline_equivalence.py tests\test_local_refinement_runbook_test_matrix.py -q
Result: 6 passed

python scripts\audit_local_refinement_runbook_tests.py
Result: status=pass, expected_test_count=8, missing_count=0

python -m pytest tests\test_local_refinement_variant_suite_package.py tests\test_mandatory_branch_keep.py tests\test_feature_flag_baseline_equivalence.py tests\test_local_refinement_runbook_test_matrix.py -q
Result: 10 passed

python -m pytest tests -q
Result: 67 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor\variant_regression\preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, variant_preflight_status=pass

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass
```

Current package metadata:

```text
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

The exact current archive hashes are intentionally kept in the sidecar metadata
files rather than duplicated here, because this project summary is copied into
the generated package and would otherwise make the archive hash self-referential.

Current project state:

```text
The runbook-named local test matrix now exists and passes.  Stage 2-4 still
require the external GPU variant-suite return before promotion.  Stage 5/6
production integration and Stage 7 production GPU batching/cache remain
deferred until GPU equivalence evidence exists.
```

Next recommended steps:

```text
Upload the refreshed upload-set or variant-suite package, run the variant-suite
fixed-point GPU regression on the cluster, and import the returned archive
locally.
```

## 101. 2026-06-03 Local-refinement HPC upload-set handoff bundle

Files generated/changed:

```text
scripts/package_local_refinement_upload_set.py
tests/test_local_refinement_upload_set.py
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Summary:

```text
Added a top-level upload-set handoff bundle for the local-refinement refactor
HPC packages.  The bundle includes both current package archives and their
sidecars under archives/, plus UPLOAD_MANIFEST.json, README.md, RUN_ORDER.md,
and RETURN_CHECKLIST.md.
```

Current upload-set roles:

```text
local_refinement_refactor_stage01_instrumentation.tar.gz:
    completed_reference
    upload_priority = optional_reference

local_refinement_refactor_variant_suite.tar.gz:
    pending_gpu_validation
    upload_priority = required_next
```

Generated handoff package:

```text
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_upload_set.py tests\test_local_refinement_upload_set.py
Result: passed

python -m pytest tests\test_local_refinement_upload_set.py -q
Result: 1 passed

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python -m pytest tests\test_local_refinement_upload_set.py tests\test_local_refinement_variant_suite_package.py -q
Result: 4 passed

python -m pytest tests -q
Result: 43 passed
```

Current project state:

```text
The local upload handoff is now explicit and auditable.  The upload set marks
Stage 1 as completed reference and Stage 2/3/4 variant-suite GPU regression as
the required next cluster run.  Stage 5 branch reuse, Stage 6 adaptive boxes,
and Stage 7 GPU batching/cache remain not submitted as production variants.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz or directly
upload hpc_packages/local_refinement_refactor_variant_suite.tar.gz.  On the
cluster, run the variant-suite workflow and return
local_refinement_refactor_variant_suite_results.tar.gz for local import/check.
```

## 102. 2026-06-03 Goal-run audit refreshed after Stage 1 pass and variant package

Files generated/changed:

```text
scripts/audit_local_refinement_refactor_goal_run.py
scripts/verify_local_refinement_goal_run_report.py
tests/test_local_refinement_goal_run_audit.py
reports/local_refinement_refactor_goal_run/goal_run_summary.md
reports/local_refinement_refactor_goal_run/goal_run_summary.tex
reports/local_refinement_refactor_goal_run/goal_run_summary.pdf
reports/local_refinement_refactor_goal_run/goal_run_audit_summary.json
reports/local_refinement_refactor_goal_run/goal_run_report_validation.json
reports/local_refinement_refactor_goal_run/decision_log.md
reports/local_refinement_refactor_goal_run/tables/stage_status.csv
reports/local_refinement_refactor_goal_run/tables/evidence_matrix.csv
reports/local_refinement_refactor_goal_run/figures/stage_gate_status.png
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Refreshed the goal-run audit to use the imported Stage 1 GPU/CUDA pass evidence
instead of the old root-level Stage 1 missing-file report.  The audit now
records:

Stage 0: completed
Stage 1: completed
Stage 2: local_minimal_complete_gpu_variant_pending
Stage 3: local_minimal_complete_gpu_variant_pending
Stage 4: local_minimal_complete_gpu_variant_pending
Stage 5: prototype_local_complete_integration_pending
Stage 6: skeleton_local_complete_integration_pending
Stage 7: package_handoff_ready
```

Current audit conclusion:

```text
status = stage2_3_4_gpu_variant_pending
stage1_gate_status = pass
stage1_import_status = pass
variant_preflight_status = pass
variant_return_gate_status = pending
upload_set_status = present
```

Important implementation decision:

```text
The goal-run audit remains an evidence report only.  It does not change the
exact-oracle physics path, feature flags, q/Delta safeguards, active-learning
logic, branch reuse integration state, adaptive-box integration state, or HPC
package behavior.
```

Validation:

```text
python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py scripts\verify_local_refinement_goal_run_report.py tests\test_local_refinement_goal_run_audit.py
Result: passed

python -m pytest tests\test_local_refinement_goal_run_audit.py -q
Result: 1 passed

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass

python -m pytest tests -q
Result: 44 passed
```

Current project state:

```text
The goal-run report now agrees with the current worktree: Stage 1 passed from
the downloaded/imported target GPU bundle, Stages 2-4 are local-complete but
await GPU variant regression, and the upload-set handoff bundle is ready.
```

Next recommended steps:

```text
Run the variant-suite package on the cluster and import the returned
local_refinement_refactor_variant_suite_results.tar.gz.  Then rerun the
goal-run audit so the status can move from stage2_3_4_gpu_variant_pending to
stage2_3_4_gpu_variant_passed if the returned gate passes.
```

## 103. 2026-06-03 Variant-suite performance report builder

Files generated/changed:

```text
scripts/build_local_refinement_performance_report.py
scripts/package_local_refinement_variant_suite_hpc.py
tests/test_local_refinement_performance_report.py
tests/test_local_refinement_variant_suite_package.py
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Summary:

```text
Added scripts/build_local_refinement_performance_report.py to summarize
returned variant-suite runtime and local-box diagnostics.  The report builder
reads each variant's pointwise CSV, regression_manifest.json, and optional
local-box timing CSV, then writes runtime_summary.csv, local_box_summary.csv,
performance_summary.json, and performance_report.md.
```

Important implementation decision:

```text
The variant-suite postprocess now builds the performance report under
$RUN_ROOT/reports/local_refinement_refactor/variant_regression/performance_report
before collecting local_refinement_refactor_variant_suite_results.tar.gz.  This
keeps output inside the extracted package/run root and does not change any
physical criterion, numerical tolerance, feature flag, or exact-oracle path.
```

Generated packages:

```text
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json

hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Validation:

```text
python -m py_compile scripts\build_local_refinement_performance_report.py scripts\package_local_refinement_variant_suite_hpc.py tests\test_local_refinement_performance_report.py tests\test_local_refinement_variant_suite_package.py
Result: passed

python -m pytest tests\test_local_refinement_performance_report.py tests\test_local_refinement_variant_suite_package.py -q
Result: 5 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports/local_refinement_refactor/variant_regression/preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python -m pytest tests -q
Result: 47 passed
```

Current project state:

```text
The Stage 2/3/4 variant-suite upload package now returns both equivalence-gate
evidence and performance diagnostics.  Actual GPU fixed-point variant results
are still pending until the package is run on the cluster.
```

Next recommended steps:

```text
Upload and run the rebuilt variant-suite package or upload-set package.  After
download, import/check local_refinement_refactor_variant_suite_results.tar.gz
and inspect the returned performance_report/ directory before deciding which
optimization variants are ready for production integration.
```

## 104. 2026-06-03 Variant-suite importer requires performance companion files

Files generated/changed:

```text
scripts/import_local_refinement_variant_suite_results.py
tests/test_local_refinement_variant_suite_package.py
scripts/package_local_refinement_variant_suite_hpc.py
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Summary:

```text
Updated scripts/import_local_refinement_variant_suite_results.py so importing a
returned variant-suite archive now builds or refreshes the local-refinement
performance report after the physics-equivalence gate passes.
```

Important implementation decision:

```text
The importer preserves gate_status for the fixed-point equivalence check and
adds import_status for the combined import result.  A returned archive can have
gate_status=pass but import_status=fail if required performance companion files
cannot be generated.  This prevents a physics-equivalent return bundle from
silently missing runtime/local-box diagnostics.
```

Generated packages:

```text
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json

hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Validation:

```text
python -m py_compile scripts\import_local_refinement_variant_suite_results.py scripts\build_local_refinement_performance_report.py tests\test_local_refinement_variant_suite_package.py
Result: passed

python -m pytest tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_performance_report.py -q
Result: 6 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports/local_refinement_refactor/variant_regression/preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python -m pytest tests -q
Result: 47 passed
```

Current project state:

```text
The upload package and local import path now both enforce the report-sync
protocol for returned Stage 2/3/4 variant results: equivalence gate files and
performance companion files are both required for a successful import.
```

Next recommended steps:

```text
Run the rebuilt variant-suite package on the cluster and return
local_refinement_refactor_variant_suite_results.tar.gz.  Import with
scripts/import_local_refinement_variant_suite_results.py and check both
gate_status and import_status.
```

## 105. 2026-06-03 Variant-suite performance collector and stricter goal audit

Files generated/changed:

```text
scripts/collect_local_refinement_performance_report.sh
scripts/package_local_refinement_variant_suite_hpc.py
scripts/audit_local_refinement_refactor_goal_run.py
scripts/verify_local_refinement_goal_run_report.py
tests/test_local_refinement_variant_suite_package.py
tests/test_local_refinement_goal_run_audit.py
reports/local_refinement_refactor_goal_run/
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Summary:

```text
Added scripts/collect_local_refinement_performance_report.sh as the explicit
variant-suite performance collector named in the TwoPhase runbook.  The
generated Slurm postprocess now runs compare_variant_suite.sh, then the
performance collector, then collect_variant_suite_outputs.sh.
```

Important implementation decision:

```text
Goal-run audit completion for Stage 2-4 now requires all three returned
variant-suite statuses to pass:

gate_status = pass
import_status = pass
performance_report_status = pass

This prevents a physics-equivalent GPU return bundle from being reported as
complete if the runtime/local-box performance companion files are missing.
```

Output policy:

```text
The performance collector writes under RUN_ROOT.  In the generated upload
package, RUN_ROOT still defaults to
$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run when writable.  The
root checkout script also prefers the generated hpc_packages package run
directory instead of creating logs/ or reports/ directly in the repo root.
```

Validation:

```text
python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py scripts\verify_local_refinement_goal_run_report.py scripts\package_local_refinement_variant_suite_hpc.py tests\test_local_refinement_goal_run_audit.py tests\test_local_refinement_variant_suite_package.py
Result: passed

python -m pytest tests\test_local_refinement_goal_run_audit.py tests\test_local_refinement_variant_suite_package.py -q
Result: 6 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports/local_refinement_refactor/variant_regression/preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python -m pytest tests -q
Result: 48 passed

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass
```

Known local limitation:

```text
bash -n scripts\collect_local_refinement_performance_report.sh could not be
run on the Windows checkout because WSL has no installed Linux distribution.
The package preflight checks the collector is present; shell execution remains
a target-cluster validation item.
```

Current project state:

```text
Stage 1 remains complete.  Stages 2-4 remain local-minimal-complete and
package-ready, but GPU validation is still pending until the returned
variant-suite archive is imported with gate_status=pass, import_status=pass,
and performance_report_status=pass.
```

Next recommended steps:

```text
Upload and run the rebuilt variant-suite package or upload-set package.  After
download, import/check local_refinement_refactor_variant_suite_results.tar.gz
and inspect the imported performance_report directory before promoting any
optimization variant.
```

## 106. 2026-06-03 Stage 5/6 diagnostic contracts before production integration

Files changed:

```text
ml_phase/exact_oracle.py
tests/test_branch_reuse.py
tests/test_adaptive_local_box_skeleton.py
reports/local_refinement_refactor/stage_05_branch_reuse/implementation_summary.md
reports/local_refinement_refactor/stage_05_branch_reuse/test_summary.md
reports/local_refinement_refactor/stage_05_branch_reuse/decision_log.md
reports/local_refinement_refactor/stage_06_adaptive_box_skeleton/implementation_summary.md
reports/local_refinement_refactor/stage_06_adaptive_box_skeleton/test_summary.md
reports/local_refinement_refactor/stage_06_adaptive_box_skeleton/decision_log.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added local-only diagnostic contract helpers for Stage 5 branch reuse and
Stage 6 adaptive local boxes:

build_branch_reuse_cache_record(...)
build_branch_reuse_diagnostic_record(...)
build_adaptive_local_box_diagnostic_record(...)

These helpers define the records that future production integration must emit
before branch reuse or adaptive box suggestions can replace fixed local-box
scans.
```

Important implementation decision:

```text
The new helpers are not wired into _confirm_one_point_robust and do not change
which local boxes are refined, the fixed local-box half widths, feature-flag
defaults, physical labels, q/Delta tolerances, acquisition logic, or
StopController behavior.
```

Diagnostic fields now covered:

```text
Stage 5:
    cache validity
    point/branch/basin ids
    q-window level
    solver and local-box signatures
    reuse attempted/allowed flags
    explicit reuse rejection reason
    candidate and cached q/Delta/DeltaF
    q/Delta/energy differences
    lower competing branch energy

Stage 6:
    adaptive enabled flag
    default q/Delta half widths
    suggested q/Delta half widths
    min/max factors
    basin geometry inputs
    adaptive-box reason
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py
Result: passed

python -m pytest tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py -q
Result: 12 passed
```

Current project state:

```text
Stage 5/6 now have tested diagnostic record contracts, but production branch
reuse and adaptive-box integration remain pending.  The Stage 2-4 GPU
variant-suite return is still the main external validation blocker before any
optimization can be promoted.
```

Next recommended steps:

```text
Keep Stage 5/6 production integration deferred until Stage 2-4 GPU variant
results return.  After that, wire these diagnostic records into emitted
box-level outputs and create explicit branch-reuse/adaptive-box GPU regression
variants.
```

## 107. 2026-06-03 Stage 7 GPU batching/cache interface contracts

Files changed:

```text
ml_phase/exact_oracle.py
tests/test_gpu_batching_cache_skeleton.py
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added pure Stage 7 interface-contract helpers for future GPU-level
optimization:

build_local_box_batch_plan(...)
build_hamiltonian_cache_signature(...)
evaluate_hamiltonian_cache_candidate(...)
build_hamiltonian_cache_diagnostic_record(...)
build_local_box_profiler_event(...)

These helpers define auditable records for local-box batch dimensions,
q/Delta grid shape, cache signatures, cache hit/miss rejection reasons, tensor
construction locations, and profiler events.
```

Important implementation decision:

```text
The helpers are not wired into _confirm_one_point_robust and do not implement
or enable GPU batching, Hamiltonian caching, branch reuse, adaptive boxes, or
any production local-refinement behavior.  Physical definitions, q/Delta
guardrails, feature-flag defaults, acquisition logic, and StopController
behavior are unchanged.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_gpu_batching_cache_skeleton.py
Result: passed

python -m pytest tests\test_gpu_batching_cache_skeleton.py -q
Result: 6 passed

python -m pytest tests\test_gpu_batching_cache_skeleton.py tests\test_branch_reuse.py tests\test_adaptive_local_box_skeleton.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_goal_run_audit.py -q
Result: 24 passed

python -m pytest tests -q
Result: 59 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor\variant_regression\preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass
```

Current project state:

```text
Stage 7 now has tested local-only interface contracts for GPU batching and
Hamiltonian cache review.  Production GPU batching/cache integration remains
deferred.  Stage 2-4 GPU variant-suite validation is still pending until the
returned archive is imported with gate_status=pass, import_status=pass, and
performance_report_status=pass.
```

Next recommended steps:

```text
Upload and run the refreshed variant-suite/upload-set package on an allowed GPU
node excluding gpuh01.  Return
local_refinement_refactor_variant_suite_results.tar.gz, then import/check it
locally and inspect the generated performance_report directory before promoting
any optimization variant.
```

## 108. 2026-06-03 Stage 1 package-local RUN_ROOT alignment

Files changed:

```text
scripts/package_local_refinement_refactor_hpc.py
scripts/run_local_refinement_fixed_point_regression.py
scripts/compare_local_refinement_variants.py
scripts/verify_local_refinement_stage1_gate.py
scripts/collect_local_refinement_stage1_outputs.py
scripts/preflight_local_refinement_stage1_hpc.py
tests/test_local_refinement_stage1_gate.py
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Aligned the Stage 1 reference/instrumentation package with the package-local
runtime-output policy already used by the Stage 2-4 variant-suite package.
When RUN_ROOT is unset and the extracted Stage 1 package is writable, runtime
outputs now default to:

$PACKAGE_ROOT/local_refinement_refactor_stage1_run

instead of writing logs, reports, preflight JSON, return-bundle metadata, or
returned archives directly under the package source root.
```

Important implementation decision:

```text
Explicit RUN_ROOT still overrides the default.  If the package root is not
writable, scripts still fall back to SCRATCH, TMPDIR, or HOME.  This is only an
HPC output-location change and does not modify the exact-oracle calculation,
feature flags, q/Delta guardrails, physical labels, acquisition logic, or
StopController behavior.
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_refactor_hpc.py scripts\run_local_refinement_fixed_point_regression.py scripts\compare_local_refinement_variants.py scripts\verify_local_refinement_stage1_gate.py scripts\collect_local_refinement_stage1_outputs.py scripts\preflight_local_refinement_stage1_hpc.py tests\test_local_refinement_stage1_gate.py
Result: passed

python -m pytest tests\test_local_refinement_stage1_gate.py -q
Result: 10 passed

python -m pytest tests -q
Result: 60 passed

python scripts\package_local_refinement_refactor_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_stage01_instrumentation.tar.gz

python hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\validate_local_refinement_hpc_package.py --package-dir hpc_packages\local_refinement_refactor_stage01_instrumentation --archive hpc_packages\local_refinement_refactor_stage01_instrumentation.tar.gz
Result: status=pass

python hpc_packages\local_refinement_refactor_stage01_instrumentation\scripts\preflight_local_refinement_stage1_hpc.py --package-root hpc_packages\local_refinement_refactor_stage01_instrumentation --run-root hpc_packages\local_refinement_refactor_stage01_instrumentation\local_refinement_refactor_stage1_run --output-json hpc_packages\local_refinement_refactor_stage01_instrumentation\local_refinement_refactor_stage1_run\reports\local_refinement_refactor\stage_01_instrumentation\stage1_runtime_preflight.json
Result: status=pass, fixed_point_count=32
```

Current project state:

```text
Stage 1 and Stage 2-4 packages now both default to package-local run
directories.  Stage 2-4 GPU variant-suite validation is still pending until
the returned archive is imported with gate_status=pass, import_status=pass,
and performance_report_status=pass.
```

Next recommended steps:

```text
Upload and run the refreshed variant-suite/upload-set package on an allowed
GPU node excluding gpuh01.  Return
local_refinement_refactor_variant_suite_results.tar.gz, then import/check it
locally and inspect the generated performance_report directory before
promoting any optimization variant.
```

## 109. 2026-06-03 Downloaded Stage 1 result check and variant RUN_ROOT preflight alignment

Files changed:

```text
scripts/package_local_refinement_variant_suite_hpc.py
tests/test_local_refinement_variant_suite_package.py
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor_goal_run/
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Summary:

```text
Checked the downloaded Stage 1 return directory:

local_refinement_refactor_stage01_instrumentation

The imported Stage 1 gate passes.  Baseline, instrumented, and comparison CSVs
all contain 32 fixed points, the local-box timing table contains 192 rows, and
mismatch_points.csv is empty.  The maximum q_opt, Delta_opt, and DeltaF
differences between baseline and instrumented outputs are all zero.
```

Important implementation decision:

```text
The returned Stage 1 logs show that the old uploaded script wrote RUN_ROOT to:

/public_hw/home/sci_bfu/local_refinement_refactor_stage1_run

while PACKAGE_ROOT was:

/public_hw/home/sci_bfu/bkz/Fu_FFLO/local_refinement_refactor_stage01_instrumentation

This explains why the return archive was not immediately found under the
package directory on the cluster.  The current regenerated Stage 1 and
variant-suite packages now default runtime outputs to package-local run
directories when the extracted package is writable.

The variant-suite workflow preflight was also aligned with this policy: the
workflow now writes preflight JSON to

${RUN_ROOT}/reports/local_refinement_refactor/variant_regression/preflight.json

and the package test asserts this command path.  The preflight script itself
still resolves a relative --output-json against --run-root for local/manual
checks.

The rebuilt variant-suite package also includes a package-level code snapshot:

code_snapshot/ml_phase/exact_oracle.py
code_snapshot/scripts/run_local_refinement_fixed_point_regression.py

The package manifest now lists code_snapshot/ in package_layout, and package
preflight requires the core snapshot files plus config/variants.json.
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_variant_suite_hpc.py tests\test_local_refinement_variant_suite_package.py
Result: passed

python -m pytest tests\test_local_refinement_variant_suite_package.py -q
Result: 4 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor\variant_regression\preflight.json
Result: status=pass, fixed_point_count=32

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, variant_preflight_status=pass

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass

python -m pytest tests -q
Result: 67 passed

python -m py_compile tests\test_local_refinement_upload_set.py
Result: passed

python -m pytest tests\test_local_refinement_upload_set.py -q
Result: 1 passed
```

Latest validation after adding the upload-set verifier to the goal-run audit
evidence matrix:

```text
python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py scripts\verify_local_refinement_goal_run_report.py tests\test_local_refinement_goal_run_audit.py
Result: passed

python -m pytest tests\test_local_refinement_goal_run_audit.py -q
Result: 3 passed

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, upload_set_verify_status=pass

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass, package_count=2, required_next_package=local_refinement_refactor_variant_suite

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=13

python -m pytest tests -q
Result: 68 passed
```

Current package metadata:

```text
local_refinement_refactor_stage01_instrumentation.tar.gz:
sha256 = 4e4dfb991c7253c14b1d788ca5452dd76fd3105f1f7b74b9a6927eb7afbff102
size   = 438361 bytes
package_file_count = 106

local_refinement_refactor_variant_suite.tar.gz:
sha256 = 0733afff92854a7c7b7f8efef02b97a84e93e02540e855ec1f93d9f964d4bc98
size   = 512630 bytes
package_file_count = 187

local_refinement_refactor_hpc_upload_set.tar.gz:
sha256 = 4a4556efba0cadb9d80a3dd69321e2476e8e5f15d828d9355e826a312bd0bbc4
size   = 958306 bytes
package_count = 2
```

The upload-set archive now includes `verify_upload_set.py`, a standalone
bundle verifier intended to be run from the extracted upload-set directory
before nested archives are uploaded or extracted on the cluster.

Current project state:

```text
Stage 1 is completed and imported locally as a pass.  Stage 2-4 local variant
logic is packaged and locally preflighted, but the GPU variant-suite return is
still pending.  Stage 5/6 production integration and Stage 7 production GPU
batching/cache remain deferred until the Stage 2-4 GPU fixed-point variant
regression is returned and checked.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz, extract
the required variant-suite package, run:

bash scripts/submit_local_refinement_fixed_point_regression.sh

Then return:

$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run/local_refinement_refactor_variant_suite_results.tar.gz

and import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 112. 2026-06-04 Upload-set nested package verifier hardening

Files changed:

```text
scripts/package_local_refinement_upload_set.py
tests/test_local_refinement_upload_set.py
docs/DECISIONS.md
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
reports/local_refinement_refactor_goal_run/
```

Summary:

```text
Strengthened the generated upload-set verifier so it does not only compare
archive hashes and sidecars.  The verifier now opens each nested package
tarball and checks that the Stage 1 and variant-suite packages still contain:

README.md
RUN_MANIFEST.json
scripts/submit_local_refinement_fixed_point_regression.sh
scripts/submit_local_refinement_instrumented_benchmark.sh
the workflow scripts targeted by those aliases
RUN_MANIFEST.writable_run_root_env = RUN_ROOT
the expected package-local run directory suffix
the documented HPC command and return archive
```

Why it matters:

```text
This directly guards against the recent HPC handoff failure modes: missing
package-local imports, logs/reports writes to non-writable directories, and
confusion between package roots and runtime output roots.  It is a packaging
and audit-layer change only.  It does not change exact-oracle physics,
feature flags, q/Delta safeguards, branch reuse production status,
adaptive-box production status, GPU batching, or Hamiltonian cache behavior.
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_upload_set.py tests\test_local_refinement_upload_set.py
Result: passed

python -m pytest tests\test_local_refinement_upload_set.py -q
Result: 1 passed

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python -m pytest tests\test_local_refinement_stage1_gate.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_upload_set.py -q
Result: 15 passed

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass; both nested packages reported nested_package.status=pass

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, upload_set_verify_status=pass

python -m pytest tests\test_local_refinement_goal_run_audit.py -q
Result: 3 passed

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=13
```

Current package metadata:

```text
local_refinement_refactor_stage01_instrumentation.tar.gz:
sha256 = 4e4dfb991c7253c14b1d788ca5452dd76fd3105f1f7b74b9a6927eb7afbff102
size   = 438361 bytes
package_file_count = 106

local_refinement_refactor_variant_suite.tar.gz:
sha256 = 0733afff92854a7c7b7f8efef02b97a84e93e02540e855ec1f93d9f964d4bc98
size   = 512630 bytes
package_file_count = 187

local_refinement_refactor_hpc_upload_set.tar.gz:
sha256 = 4a4556efba0cadb9d80a3dd69321e2476e8e5f15d828d9355e826a312bd0bbc4
size   = 958306 bytes
package_count = 2
```

Current project state:

```text
Stage 1 remains complete and imported locally as a pass.  Stage 2-4 are still
local-minimal-complete and package-ready, but GPU validation remains pending
until the variant-suite return archive is imported with gate_status=pass,
import_status=pass, and performance_report_status=pass.  Stage 5/6 production
integration and Stage 7 production GPU batching/cache remain deferred.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster, run:

python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh

Return:

$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run/local_refinement_refactor_variant_suite_results.tar.gz

and import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 113. 2026-06-04 Goal-run audit exposes nested upload-set entrypoint evidence

Files changed:

```text
scripts/audit_local_refinement_refactor_goal_run.py
scripts/verify_local_refinement_goal_run_report.py
tests/test_local_refinement_goal_run_audit.py
reports/local_refinement_refactor_goal_run/
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Extended the goal-run audit so the upload-set nested package verifier result
is a first-class evidence-matrix row rather than being hidden inside the
generic upload-set verifier status.

The new evidence row records:

artifact = Upload-set nested package entrypoints
status = pass
nested_packages = 2
nested_pass = 2
alias_count = 4
missing_required_paths = 0
run_root_env_failures = 0
```

Why it matters:

```text
The final goal audit now explicitly proves that both nested HPC packages expose
the expected runbook submit aliases and package-local RUN_ROOT contract.  This
is stronger than saying only that verify_upload_set.py returned pass, and it
directly guards against the previous handoff failures involving missing
entrypoints, wrong workflow aliases, and runtime outputs written outside the
extracted package run directory.

This is an audit/reporting change only.  It does not modify the exact BdG
solver, thermodynamic phase definitions, q/Delta safeguards, local-refinement
feature flags, branch reuse production status, adaptive-box production status,
GPU batching, or Hamiltonian cache behavior.
```

Validation:

```text
python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py scripts\verify_local_refinement_goal_run_report.py tests\test_local_refinement_goal_run_audit.py
Result: passed

python -m pytest tests\test_local_refinement_goal_run_audit.py -q
Result: 3 passed

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, upload_set_nested_verify_status=pass

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=14
```

Current project state:

```text
Stage 1 remains complete and imported locally as a pass.  Stage 2-4 remain
local-minimal-complete and package-ready, but GPU validation is still pending
until the variant-suite return archive is imported with gate_status=pass,
import_status=pass, and performance_report_status=pass.  Stage 5/6 production
integration and Stage 7 production GPU batching/cache remain deferred.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster, run:

python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh

Return:

$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run/local_refinement_refactor_variant_suite_results.tar.gz

and import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 114. 2026-06-04 Variant-suite RUN_ROOT shell-output enforcement

Files changed:

```text
scripts/package_local_refinement_variant_suite_hpc.py
scripts/package_local_refinement_upload_set.py
tests/test_local_refinement_variant_suite_package.py
tests/test_local_refinement_upload_set.py
scripts/audit_local_refinement_refactor_goal_run.py
tests/test_local_refinement_goal_run_audit.py
docs/DECISIONS.md
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
reports/local_refinement_refactor_goal_run/
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
```

Summary:

```text
Fixed a remaining package-root output risk in the variant-suite collector.
collect_variant_suite_outputs.sh no longer writes return-bundle metadata to a
bare reports/local_refinement_refactor/... path.  It now writes metadata under:

$RUN_ROOT/reports/local_refinement_refactor/variant_regression/return_bundle_metadata

The upload-set verifier now scans all nested package shell scripts and fails
if any logs/reports output path is not expressed through RUN_ROOT.  The
goal-run evidence matrix reports shell_script_count=28 and
shell_policy_violations=0.
```

Why it matters:

```text
This directly addresses the previous HPC permission failure mode where mkdir
logs or mkdir reports could target a non-writable current directory.  The
final upload-set verifier now checks workflow scripts, Slurm scripts, submit
aliases, comparison helpers, collectors, and postprocess helpers before the
package is uploaded.  This is a packaging/runtime-output policy change only;
it does not change exact-oracle physics, q/Delta safeguards, feature flags,
branch reuse, adaptive boxes, GPU batching, or Hamiltonian cache behavior.
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_variant_suite_hpc.py scripts\package_local_refinement_upload_set.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_upload_set.py
Result: passed

python -m pytest tests\test_local_refinement_variant_suite_package.py -q
Result: 4 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python scripts\package_local_refinement_upload_set.py
Result: wrote hpc_packages\local_refinement_refactor_hpc_upload_set.tar.gz

python -m pytest tests\test_local_refinement_upload_set.py -q
Result: 1 passed

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor\variant_regression\preflight.json
Result: status=pass, fixed_point_count=32

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass; Stage 1 shell_script_count=11, violations=[]; variant-suite shell_script_count=17, violations=[]

python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py tests\test_local_refinement_goal_run_audit.py
Result: passed

python -m pytest tests\test_local_refinement_goal_run_audit.py -q
Result: 3 passed

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, upload_set_nested_verify_status=pass

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=14
```

Current package metadata:

```text
local_refinement_refactor_stage01_instrumentation.tar.gz:
sha256 = 4e4dfb991c7253c14b1d788ca5452dd76fd3105f1f7b74b9a6927eb7afbff102
size   = 438361 bytes
package_file_count = 106

local_refinement_refactor_variant_suite.tar.gz:
sha256 = bbb3eed19c353ecebc0bc36900cc9cad02e5297a1b88efa98d7813f472553785
size   = 516012 bytes
package_file_count = 187

local_refinement_refactor_hpc_upload_set.tar.gz:
sha256 = d1d5bd2d44d49bfc4baf9fcf2de3f84c1892c069f8b1bba49864df67d3ca4cb2
size   = 961767 bytes
package_count = 2
```

Current project state:

```text
Stage 1 remains complete and imported locally as a pass.  Stage 2-4 remain
local-minimal-complete and package-ready, but GPU validation is still pending
until the variant-suite return archive is imported with gate_status=pass,
import_status=pass, and performance_report_status=pass.  Stage 5/6 production
integration and Stage 7 production GPU batching/cache remain deferred.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster, run:

python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh

Return:

$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run/local_refinement_refactor_variant_suite_results.tar.gz

and import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 115. 2026-06-04 TwoPhase runbook completion audit

Files changed:

```text
scripts/audit_twophase_optimization_completion.py
tests/test_twophase_completion_audit.py
reports/local_refinement_refactor_goal_run/twophase_completion_audit.json
reports/local_refinement_refactor_goal_run/twophase_completion_audit.md
reports/local_refinement_refactor_goal_run/tables/twophase_completion_requirements.csv
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a requirement-by-requirement completion audit for the TwoPhase
Optimization runbook.  The audit reads the current local evidence instead of
assuming completion from partial progress.  It checks the runbook source,
project docs, Stage 0-7 report completeness, runbook-named tests, goal-run
report protocol validation, Stage 1 imported GPU gate, Stage 2-4 variant-suite
local preflight, upload-set verifier, nested RUN_ROOT shell-output policy, and
Stage 2-4 GPU return/import status.
```

Why it matters:

```text
The audit makes the current blocker explicit and machine-readable.  The
current status is pending_hpc, not complete: 8 requirements pass, 1 requirement
is pending_hpc, and 3 production-integration requirements are deferred until
the Stage 2-4 GPU variant-suite return gate passes.  This prevents the project
from silently treating local minimal tests or package readiness as full
runbook completion.
```

Validation:

```text
python -m py_compile scripts\audit_twophase_optimization_completion.py tests\test_twophase_completion_audit.py
Result: passed

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc, requirement_count=12, status_counts={pass:8, pending_hpc:1, deferred:3}

python -m pytest tests\test_twophase_completion_audit.py -q
Result: 1 passed

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass
```

Current project state:

```text
Stage 1 remains complete and imported locally as a pass.  Stage 2-4 remain
local-minimal-complete and package-ready, but the full TwoPhase runbook remains
pending_hpc until the variant-suite GPU return archive is imported with
gate_status=pass, import_status=pass, and performance_report_status=pass.
Stage 5 branch reuse production integration, Stage 6 adaptive local-box
production integration, and Stage 7 GPU batching/Hamiltonian cache production
integration remain deferred until that GPU gate passes.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster, run:

python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh

Return:

$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run/local_refinement_refactor_variant_suite_results.tar.gz

and import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 116. 2026-06-04 Upload-set GPU Slurm safety gate

Files changed:

```text
scripts/package_local_refinement_upload_set.py
tests/test_local_refinement_upload_set.py
scripts/audit_local_refinement_refactor_goal_run.py
tests/test_local_refinement_goal_run_audit.py
scripts/audit_twophase_optimization_completion.py
tests/test_twophase_completion_audit.py
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
reports/local_refinement_refactor_goal_run/goal_run_summary.md
reports/local_refinement_refactor_goal_run/goal_run_summary.tex
reports/local_refinement_refactor_goal_run/goal_run_summary.pdf
reports/local_refinement_refactor_goal_run/tables/evidence_matrix.csv
reports/local_refinement_refactor_goal_run/tables/twophase_completion_requirements.csv
reports/local_refinement_refactor_goal_run/twophase_completion_audit.json
reports/local_refinement_refactor_goal_run/twophase_completion_audit.md
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
docs/report_qa/20260604_local_refinement_gpu_slurm_policy.md
```

Summary:

```text
Strengthened the generated upload-set verifier so it scans GPU Slurm scripts
inside each nested package archive.  Any GPU script must now include
#SBATCH --exclude=gpuh01, run torch.empty(1, device="cuda"), and print
cuda_runtime_probe=pass.  The goal-run audit and TwoPhase completion audit now
surface gpu_script_count and gpu_policy_violations.
```

Why it matters:

```text
The previous HPC failure showed that gpuh01 has an NVIDIA driver too old for
the active PyTorch CUDA runtime.  The handoff now fails before submission if a
package can run GPU work on that known bad node or skip the CUDA runtime
allocation probe.  This is a packaging/HPC safety gate only and does not change
exact-oracle physics, numerical thresholds, feature flags, branch reuse,
adaptive boxes, GPU batching, or Hamiltonian cache behavior.
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_upload_set.py tests\test_local_refinement_upload_set.py scripts\audit_local_refinement_refactor_goal_run.py tests\test_local_refinement_goal_run_audit.py scripts\audit_twophase_optimization_completion.py tests\test_twophase_completion_audit.py
Result: passed

python -m pytest tests\test_local_refinement_upload_set.py -q
Result: 1 passed

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass; Stage 1 gpu_script_count=2, violations=[]; variant-suite gpu_script_count=5, violations=[]

python -m pytest tests\test_local_refinement_goal_run_audit.py tests\test_twophase_completion_audit.py -q
Result: 4 passed

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, upload_set_nested_verify_status=pass

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc, requirement_count=13, status_counts={pass:9, pending_hpc:1, deferred:3}

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=14
```

Current package metadata:

```text
local_refinement_refactor_stage01_instrumentation.tar.gz:
sha256 = 4e4dfb991c7253c14b1d788ca5452dd76fd3105f1f7b74b9a6927eb7afbff102
size   = 438361 bytes
package_file_count = 106

local_refinement_refactor_variant_suite.tar.gz:
sha256 = bbb3eed19c353ecebc0bc36900cc9cad02e5297a1b88efa98d7813f472553785
size   = 516012 bytes
package_file_count = 187

local_refinement_refactor_hpc_upload_set.tar.gz:
sha256 = dba7f2f833cd453e12b473c99ef99d0bdd53a20cc49a8af311d08f78526f27fb
size   = 962075 bytes
package_count = 2
```

Current project state:

```text
Stage 1 remains complete and imported locally as a pass.  Stage 2-4 remain
local-minimal-complete, package-ready, and upload-set verified with RUN_ROOT
and GPU Slurm safety policies.  The full TwoPhase runbook is still pending_hpc
until the variant-suite GPU return archive is imported with gate_status=pass,
import_status=pass, and performance_report_status=pass.  Stage 5 branch reuse
production integration, Stage 6 adaptive local-box production integration, and
Stage 7 GPU batching/Hamiltonian cache production integration remain deferred
until that GPU gate passes.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster, run:

python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh

Return:

$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run/local_refinement_refactor_variant_suite_results.tar.gz

and import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 117. 2026-06-04 Variant-suite return readiness checker

Files changed:

```text
scripts/check_local_refinement_variant_suite_return.py
tests/test_local_refinement_variant_suite_return_check.py
scripts/package_local_refinement_upload_set.py
tests/test_local_refinement_upload_set.py
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor_goal_run/
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Summary:

```text
Added scripts/check_local_refinement_variant_suite_return.py, a local readiness
checker for downloaded variant-suite HPC results.  It accepts a downloaded
return archive, extracted package/run directory, or parent directory; locates
local_refinement_refactor_variant_suite_results.tar.gz; checks the tarball for
required top-level result prefixes; scans log-like files for CUDA driver,
cuda_runtime_probe=fail, CUDA RuntimeError, Slurm failure, and gpuh01 patterns;
and emits a JSON decision plus the next importer command when ready.
```

Why it matters:

```text
Earlier handoff confusion included downloading a whole directory but not
knowing whether the expected return tarball existed or whether logs already
proved a CUDA/node failure.  The readiness checker does not replace the formal
importer or physics-equivalence gate, but it gives a quick local answer before
running the full import step.  The upload-set RETURN_CHECKLIST now includes
this readiness check before the importer command.
```

Validation:

```text
python -m py_compile scripts\check_local_refinement_variant_suite_return.py tests\test_local_refinement_variant_suite_return_check.py
Result: passed

python -m pytest tests\test_local_refinement_variant_suite_return_check.py -q
Result: 2 passed

python -m py_compile scripts\package_local_refinement_upload_set.py tests\test_local_refinement_upload_set.py scripts\check_local_refinement_variant_suite_return.py tests\test_local_refinement_variant_suite_return_check.py
Result: passed

python -m pytest tests\test_local_refinement_upload_set.py tests\test_local_refinement_variant_suite_return_check.py -q
Result: 3 passed

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass; Stage 1 gpu_script_count=2, violations=[]; variant-suite gpu_script_count=5, violations=[]

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, upload_set_nested_verify_status=pass

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc, requirement_count=13, status_counts={pass:9, pending_hpc:1, deferred:3}

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=14
```

Current package metadata:

```text
local_refinement_refactor_stage01_instrumentation.tar.gz:
sha256 = 4e4dfb991c7253c14b1d788ca5452dd76fd3105f1f7b74b9a6927eb7afbff102
size   = 438361 bytes
package_file_count = 106

local_refinement_refactor_variant_suite.tar.gz:
sha256 = bbb3eed19c353ecebc0bc36900cc9cad02e5297a1b88efa98d7813f472553785
size   = 516012 bytes
package_file_count = 187

local_refinement_refactor_hpc_upload_set.tar.gz:
sha256 = 366e95dc9afab16983d51ebc5a9735adf379d2d81d2ae40b04ebb0facc2e4091
size   = 962142 bytes
package_count = 2
```

Current project state:

```text
Stage 1 remains complete and imported locally as a pass.  Stage 2-4 remain
local-minimal-complete, package-ready, upload-set verified, and return-check
ready.  The full TwoPhase runbook is still pending_hpc until the variant-suite
GPU return archive is imported with gate_status=pass, import_status=pass, and
performance_report_status=pass.  Stage 5 branch reuse production integration,
Stage 6 adaptive local-box production integration, and Stage 7 GPU
batching/Hamiltonian cache production integration remain deferred until that
GPU gate passes.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster, run:

python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh

After downloading the run output, check readiness locally with:

python scripts/check_local_refinement_variant_suite_return.py <downloaded-return-directory-or-archive>

Then import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```


## 118. 2026-06-04 Return readiness checker audit integration

Files changed:

```text
scripts/audit_local_refinement_refactor_goal_run.py
scripts/verify_local_refinement_goal_run_report.py
scripts/audit_twophase_optimization_completion.py
tests/test_local_refinement_goal_run_audit.py
tests/test_twophase_completion_audit.py
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor_goal_run/goal_run_summary.md
reports/local_refinement_refactor_goal_run/goal_run_summary.tex
reports/local_refinement_refactor_goal_run/goal_run_summary.pdf
reports/local_refinement_refactor_goal_run/tables/evidence_matrix.csv
reports/local_refinement_refactor_goal_run/tables/twophase_completion_requirements.csv
reports/local_refinement_refactor_goal_run/twophase_completion_audit.json
reports/local_refinement_refactor_goal_run/twophase_completion_audit.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Integrated the variant-suite return readiness checker into the formal
goal-run evidence matrix and the TwoPhase completion requirement matrix.  The
goal-run report verifier now requires the evidence row
"Variant-suite return readiness checker", and the TwoPhase completion audit
now proves that the checker exists and that the upload-set RETURN_CHECKLIST
mentions both the readiness checker and the formal importer command.
```

Why it matters:

```text
The readiness checker is now a machine-checked handoff artifact rather than an
untracked helper script.  This prevents the project from silently losing the
local command that diagnoses "downloaded a whole directory but cannot find the
return archive" before the formal import/gate step.
```

Validation:

```text
python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py scripts\verify_local_refinement_goal_run_report.py scripts\audit_twophase_optimization_completion.py tests\test_local_refinement_goal_run_audit.py tests\test_twophase_completion_audit.py
Result: passed

python -m pytest tests\test_local_refinement_goal_run_audit.py tests\test_twophase_completion_audit.py -q
Result: 4 passed

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, evidence_matrix row_count=15

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc, requirement_count=14, status_counts={pass:10, pending_hpc:1, deferred:3}

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=15
```

Current project state:

```text
Stage 1 remains complete and imported locally as a pass.  Stage 2-4 remain
local-minimal-complete, package-ready, upload-set verified, and return-check
ready with formal audit evidence.  The full TwoPhase runbook is still
pending_hpc until the variant-suite GPU return archive is imported with
gate_status=pass, import_status=pass, and performance_report_status=pass.
Stage 5 branch reuse production integration, Stage 6 adaptive local-box
production integration, and Stage 7 GPU batching/Hamiltonian cache production
integration remain deferred until that GPU gate passes.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster, run:

python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh

After downloading the run output, check readiness locally with:

python scripts/check_local_refinement_variant_suite_return.py <downloaded-return-directory-or-archive>

Then import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 119. 2026-06-04 Variant-suite HPC status checker

Files changed:

```text
scripts/check_variant_suite_hpc_status.py
tests/test_variant_suite_hpc_status_check.py
scripts/package_local_refinement_variant_suite_hpc.py
tests/test_local_refinement_variant_suite_package.py
scripts/package_local_refinement_upload_set.py
tests/test_local_refinement_upload_set.py
docs/DECISIONS.md
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
reports/local_refinement_refactor_goal_run/
docs/PROJECT_SUMMARY.md
hpc_packages/local_refinement_refactor_variant_suite/
hpc_packages/local_refinement_refactor_variant_suite.tar.gz
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.sha256
hpc_packages/local_refinement_refactor_variant_suite.tar.gz.metadata.json
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
```

Summary:

```text
Added scripts/check_variant_suite_hpc_status.py and included it in the
variant-suite HPC package.  The checker reads RUN_ROOT logs, jobid files, and
the expected local_refinement_refactor_variant_suite_results.tar.gz archive.
It reports ready_to_return, failed_or_needs_log_review, or
pending_or_missing_return_archive.  With --query-scheduler it also queries
squeue/sacct when available.
```

Why it matters:

```text
This addresses the practical post-submit case where squeue shows no running or
pending jobs but the returned archive is not obvious.  The package now has a
cluster-side diagnostic command that can distinguish a ready archive from
missing postprocess output, CUDA runtime/old-driver failures, and failed Slurm
states.  The checker is diagnostic only; it does not change Slurm dependencies,
the exact-oracle calculation, feature flags, thresholds, or the formal import
gate.
```

Validation:

```text
python -m py_compile scripts\check_variant_suite_hpc_status.py tests\test_variant_suite_hpc_status_check.py scripts\package_local_refinement_variant_suite_hpc.py tests\test_local_refinement_variant_suite_package.py scripts\package_local_refinement_upload_set.py tests\test_local_refinement_upload_set.py
Result: passed

python -m pytest tests\test_variant_suite_hpc_status_check.py tests\test_local_refinement_variant_suite_package.py -q
Result: 6 passed

python scripts\package_local_refinement_variant_suite_hpc.py
Result: wrote hpc_packages\local_refinement_refactor_variant_suite.tar.gz

python hpc_packages\local_refinement_refactor_variant_suite\scripts\preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages\local_refinement_refactor_variant_suite --run-root hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor\variant_regression\preflight.json
Result: status=pass, fixed_point_count=32

python -m pytest tests\test_local_refinement_upload_set.py -q
Result: 1 passed

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass; nested packages pass; variant-suite support script present

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, upload_set_nested_verify_status=pass

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc, requirement_count=14, status_counts={pass:10, pending_hpc:1, deferred:3}

python -m pytest tests\test_variant_suite_hpc_status_check.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_upload_set.py tests\test_local_refinement_goal_run_audit.py tests\test_twophase_completion_audit.py -q
Result: 11 passed

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=15
```

Current package metadata:

```text
local_refinement_refactor_stage01_instrumentation.tar.gz:
sha256 = 4e4dfb991c7253c14b1d788ca5452dd76fd3105f1f7b74b9a6927eb7afbff102
size   = 438361 bytes
package_file_count = 106

local_refinement_refactor_variant_suite.tar.gz:
sha256 = ab90ea157af398a67132fc92a51b30118f18b3e4663bb0b973dd0bc998fe19fc
size   = 530366 bytes
package_file_count = 194

local_refinement_refactor_hpc_upload_set.tar.gz:
sha256 = f8b32b84953a3d27ea32d5343012aee22c9b99527b1569147e632a887cfdc87e
size   = 976711 bytes
package_count = 2
```

Current project state:

```text
Stage 1 remains complete and imported locally as a pass.  Stage 2-4 remain
local-minimal-complete, package-ready, upload-set verified, return-check ready,
and cluster-side status-check ready.  The full TwoPhase runbook is still
pending_hpc until the variant-suite GPU return archive is imported with
gate_status=pass, import_status=pass, and performance_report_status=pass.
Stage 5 branch reuse production integration, Stage 6 adaptive local-box
production integration, and Stage 7 GPU batching/Hamiltonian cache production
integration remain deferred until that GPU gate passes.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster, run:

python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh

If squeue has no visible jobs but the return archive is unclear, run:

python scripts/check_variant_suite_hpc_status.py --package-root . --run-root "$RUN_ROOT" --query-scheduler

After downloading the run output, check readiness locally with:

python scripts/check_local_refinement_variant_suite_return.py <downloaded-return-directory-or-archive>

Then import/check it locally with:

python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

## 120. Local Refinement Goal-Run Audit Includes HPC Status Checker Evidence

Date: 2026-06-04

Files changed:

```text
scripts/audit_local_refinement_refactor_goal_run.py
scripts/verify_local_refinement_goal_run_report.py
scripts/audit_twophase_optimization_completion.py
tests/test_local_refinement_goal_run_audit.py
tests/test_twophase_completion_audit.py
reports/local_refinement_refactor_goal_run/goal_run_summary.md
reports/local_refinement_refactor_goal_run/goal_run_summary.tex
reports/local_refinement_refactor_goal_run/goal_run_summary.pdf
reports/local_refinement_refactor_goal_run/tables/evidence_matrix.csv
reports/local_refinement_refactor_goal_run/tables/twophase_completion_requirements.csv
reports/local_refinement_refactor_goal_run/twophase_completion_audit.json
reports/local_refinement_refactor_goal_run/twophase_completion_audit.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
```

Summary:

```text
The formal goal-run audit now records the variant-suite HPC status checker as a
first-class evidence row.  The report verifier requires that row to pass and
checks that the checker exists in source, exists in the packaged variant suite,
is referenced by RUN_MANIFEST.json, and is documented in README.md.  The
TwoPhase completion audit now has an explicit requirement that the status
checker is packaged.
```

Why it matters:

```text
The post-submit diagnostic script is no longer only a convenience file inside
the upload package.  It is now part of the audited handoff contract, so a future
package or report regeneration fails if the status checker is missing from the
package or undocumented.  This closes the local evidence gap for the practical
case where squeue is empty but the return archive has not been found.
```

Validation:

```text
python -m py_compile scripts\audit_local_refinement_refactor_goal_run.py scripts\verify_local_refinement_goal_run_report.py scripts\audit_twophase_optimization_completion.py tests\test_local_refinement_goal_run_audit.py tests\test_twophase_completion_audit.py
Result: passed

python -m pytest tests\test_local_refinement_goal_run_audit.py tests\test_twophase_completion_audit.py -q
Result: 4 passed

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending, evidence_matrix row_count=16

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc, requirement_count=15, status_counts={pass:11, pending_hpc:1, deferred:3}

pdflatex -interaction=nonstopmode -halt-on-error goal_run_summary.tex
Result: goal_run_summary.pdf compiled successfully as a 2-page PDF, size=88999 bytes

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=16
```

Current project state:

```text
Stage 1 remains imported and passed.  Stage 2-4 remain locally implemented,
preflighted, upload-set verified, return-check ready, and cluster-side
status-check ready.  The TwoPhase objective is still pending_hpc because the
GPU variant-suite return archive has not been imported with gate_status=pass,
import_status=pass, and performance_report_status=pass.
```

Known unresolved issues:

```text
No local packaging, report, or audit blocker remains for the Stage 2-4 GPU
variant-suite handoff.  Completion is blocked only by the missing GPU return
archive/import result from the cluster run.  Stage 5-7 production integrations
remain deferred by design until that GPU gate passes.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz, run the
variant-suite package on an allowed GPU node, use
scripts/check_variant_suite_hpc_status.py on the cluster if squeue is empty and
the archive is unclear, then download and import
local_refinement_refactor_variant_suite_results.tar.gz locally.
```

## 121. Downloaded Stage 1 Directory Checked for Variant-Suite Return

Date: 2026-06-04

Files changed:

```text
reports/local_refinement_refactor_goal_run/downloaded_stage1_dir_variant_return_check.json
reports/local_refinement_refactor_goal_run/variant_run_root_return_check.json
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
The downloaded local_refinement_refactor_stage01_instrumentation directory and
the current local variant-suite run root were checked with
scripts/check_local_refinement_variant_suite_return.py.  Both checks returned
status=not_ready because local_refinement_refactor_variant_suite_results.tar.gz
is absent.  The Stage 1 downloaded directory scanned 11 log-like files and
found no fatal CUDA, Slurm, or old-driver log patterns.
```

Why it matters:

```text
This confirms that the locally downloaded directory contains the Stage 1
regression return/package contents, not the Stage 2-4 variant-suite GPU return
required by the TwoPhase completion gate.  The remaining blocker is therefore
not hidden in the downloaded Stage 1 directory.
```

Validation:

```text
python scripts\check_local_refinement_variant_suite_return.py local_refinement_refactor_stage01_instrumentation --output-json reports\local_refinement_refactor_goal_run\downloaded_stage1_dir_variant_return_check.json
Result: status=not_ready; missing local_refinement_refactor_variant_suite_results.tar.gz; files_scanned=11; fatal log matches=0

python scripts\check_local_refinement_variant_suite_return.py hpc_packages\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-json reports\local_refinement_refactor_goal_run\variant_run_root_return_check.json
Result: status=not_ready; missing local_refinement_refactor_variant_suite_results.tar.gz; files_scanned=0; fatal log matches=0

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass; package_count=2; nested packages pass
```

Current project state:

```text
The upload-set handoff package remains valid and ready.  No local
variant-suite GPU return archive is present yet.  The goal remains pending_hpc
until local_refinement_refactor_variant_suite_results.tar.gz is produced on
the cluster, downloaded, checked, and imported with the variant gate passing.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz, extract
and verify it on the cluster, run the variant-suite fixed-point regression on
an allowed GPU node, then return
local_refinement_refactor_variant_suite_results.tar.gz for local import.
```

## 122. Upload-Set Adds Top-Level Variant-Suite Run Helper

Date: 2026-06-04

Files changed:

```text
scripts/package_local_refinement_upload_set.py
tests/test_local_refinement_upload_set.py
docs/DECISIONS.md
reports/local_refinement_refactor/stage_07_hpc_packaging/implementation_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/test_summary.md
reports/local_refinement_refactor/stage_07_hpc_packaging/decision_log.md
hpc_packages/local_refinement_refactor_hpc_upload_set/
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.sha256
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.metadata.json
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added run_required_variant_suite.sh to the top level of the upload-set bundle.
The helper verifies the extracted upload set, extracts
archives/local_refinement_refactor_variant_suite.tar.gz if needed, cd's into
the variant-suite package, and runs the existing
scripts/submit_local_refinement_fixed_point_regression.sh alias.  The generated
upload-set verifier now checks that this helper is present, documented,
verifies first, extracts the required archive, submits the expected alias, and
contains no destructive delete command.
```

Why it matters:

```text
The next required step is an HPC handoff.  The helper reduces manual mistakes
in the verify/extract/cd/submit sequence while preserving the nested package's
validated workflow and RUN_ROOT output policy.  This is a packaging and
cluster-handoff change only, not a physics, numerics, feature-flag, or Slurm
dependency change inside the nested variant-suite package.
```

Validation:

```text
python -m py_compile scripts\package_local_refinement_upload_set.py tests\test_local_refinement_upload_set.py
Result: passed

python -m pytest tests\test_local_refinement_upload_set.py -q
Result: 1 passed

python hpc_packages\local_refinement_refactor_hpc_upload_set\verify_upload_set.py --upload-root hpc_packages\local_refinement_refactor_hpc_upload_set
Result: status=pass; top_level_handoff all required checks pass; nested packages pass

python scripts\audit_local_refinement_refactor_goal_run.py
Result: status=stage2_3_4_gpu_variant_pending

python scripts\audit_twophase_optimization_completion.py
Result: status=pending_hpc, requirement_count=15, status_counts={pass:11, pending_hpc:1, deferred:3}

python scripts\verify_local_refinement_goal_run_report.py
Result: status=pass, evidence_matrix row_count=16
```

Current package metadata:

```text
local_refinement_refactor_variant_suite.tar.gz:
sha256 = ab90ea157af398a67132fc92a51b30118f18b3e4663bb0b973dd0bc998fe19fc
size   = 530366 bytes
package_file_count = 194

local_refinement_refactor_hpc_upload_set.tar.gz:
sha256 = 1f41a14ecd565c0105518e4611c70996c6ef131017c9cc9b8f8f5351d2200f57
size   = 977568 bytes
package_count = 2
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  On the
cluster:

tar -xzf local_refinement_refactor_hpc_upload_set.tar.gz
cd local_refinement_refactor_hpc_upload_set
bash run_required_variant_suite.sh

If squeue has no visible jobs but the return archive is unclear:

cd local_refinement_refactor_variant_suite
python scripts/check_variant_suite_hpc_status.py --package-root . --run-root "${RUN_ROOT:-local_refinement_refactor_variant_suite_run}" --query-scheduler
```

## 123. Blocked Goal Checkpoint Written

Date: 2026-06-04

Files changed:

```text
docs/GOAL_CHECKPOINT.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Created docs/GOAL_CHECKPOINT.md as a handoff checkpoint while the TwoPhase
local-refinement goal is blocked.  The checkpoint records the original goal,
completed local modifications, modified-file and git-diff summaries, the
remaining Stage 2-4 GPU variant-suite return blocker, the exact external HPC
commands and expected output archive, and the pass/fail continuation paths.
```

Current project state:

```text
The goal remains blocked on the missing external GPU return archive
local_refinement_refactor_variant_suite_results.tar.gz.  The current upload
package is hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz with
sha256=1f41a14ecd565c0105518e4611c70996c6ef131017c9cc9b8f8f5351d2200f57.
```

## 124. Local-Refinement Refactor Major-Stage Report

Date: 2026-06-04

Files changed:

```text
project_history/reports/report_local_refinement_refactor_note/README.md
project_history/reports/report_local_refinement_refactor_note/build_note.ps1
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.tex
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.pdf
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.md
project_history/reports/report_local_refinement_refactor_note/decision_log.md
project_history/reports/report_local_refinement_refactor_note/tables/stage_status.csv
project_history/reports/report_local_refinement_refactor_note/tables/evidence_matrix.csv
project_history/reports/report_local_refinement_refactor_note/tables/twophase_completion_requirements.csv
project_history/reports/report_local_refinement_refactor_note/tables/report_summary_status.csv
project_history/reports/report_local_refinement_refactor_note/tables/upload_package_metadata.csv
project_history/reports/report_local_refinement_refactor_note/figures/stage_gate_status.png
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Created the second major-stage research note,
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.pdf,
using the previous active_learning_r0015_note report style.  The main text
summarizes the achieved local-refinement refactor, stage status, HPC handoff,
and current pending_hpc gate.  The appendix records the improvement path and
detours: instrumentation before pruning, import-path failure, RUN_ROOT
permission failure, gpuh01/CUDA runtime failure, squeue ambiguity, and the
top-level upload-set helper.
```

Validation:

```text
pdflatex -interaction=nonstopmode -halt-on-error local_refinement_refactor_note.tex
Result: PDF compiled, 7 pages

.\build_note.ps1
Result: PDF reproducibly rebuilt, 7 pages

pdftocairo -png -r 120 local_refinement_refactor_note.pdf tmp\pdf_render_cairo\local_refinement_refactor_note_page
Result: pages rendered; page 1, page 5, and page 7 visually inspected

pdftotext local_refinement_refactor_note.pdf tmp\local_refinement_refactor_note.txt
Result: text extraction checked with no placeholder question-mark sequences
```

Current project state:

```text
The report is explicit that the local-refinement optimization stage is
handoff-ready but externally blocked, not scientifically complete.  It records
the current upload set sha256
1f41a14ecd565c0105518e4611c70996c6ef131017c9cc9b8f8f5351d2200f57 and the
required pending return archive local_refinement_refactor_variant_suite_results.tar.gz.
```

## 125. Local-Refinement Report Visual Context Added

Date: 2026-06-04

Files changed:

```text
project_history/reports/report_local_refinement_refactor_note/README.md
project_history/reports/report_local_refinement_refactor_note/decision_log.md
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.tex
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.pdf
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.md
project_history/reports/report_local_refinement_refactor_note/figures/fig01_original_exact_phase_diagram.png
project_history/reports/report_local_refinement_refactor_note/figures/fig02_active_learning_main_boundaries.png
project_history/reports/report_local_refinement_refactor_note/figures/fig03_combined_eta_phase_diagram.png
project_history/reports/report_local_refinement_refactor_note/figures/fig04_active_learning_workflow.png
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added representative visual context to the second major-stage report.  The main
text now places the original exact phase diagram, active-learning boundary map,
combined eta phase diagram, and stage-gate status figure near the sections they
support.  The appendix now includes the active-learning workflow figure and
ties the later package-local import fixes, RUN_ROOT output policy, CUDA probe,
and return-archive checker to the exact-data return path.
```

Validation:

```text
.\build_note.ps1
Result: local_refinement_refactor_note.pdf rebuilt successfully, 11 pages.

pdftocairo -png -r 120 local_refinement_refactor_note.pdf tmp/pdf_render_cairo_local_refinement_note_figs/page
Result: rendered all 11 pages; pages 3, 4, 5, 7, and 9 visually inspected.

Select-String local_refinement_refactor_note.log for undefined references,
overfull boxes, fatal errors, and LaTeX errors
Result: no matches.

pdftotext local_refinement_refactor_note.pdf tmp/local_refinement_refactor_note_figs.txt
Result: text extracted; no placeholder question-mark sequences found.
```

Current project state:

```text
The report now makes the figure provenance explicit.  The old phase diagrams
are used as baseline and active-learning context, while the current
local-refinement stage remains handoff-ready but externally blocked on the
missing Stage 2-4 GPU variant-suite return archive.
```

## 126. Stage-Ordered Visual Report Rewrite

Date: 2026-06-05

Files changed:

```text
project_history/reports/report_local_refinement_refactor_note/README.md
project_history/reports/report_local_refinement_refactor_note/decision_log.md
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.tex
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.pdf
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.md
project_history/reports/report_local_refinement_refactor_note/figures/fig05_response_qwindow_stability.png
project_history/reports/report_local_refinement_refactor_note/figures/fig06_response_qdensity_eta.png
project_history/reports/report_local_refinement_refactor_note/figures/fig07_phase_qopt_shift.png
project_history/reports/report_local_refinement_refactor_note/figures/fig08_phase_change_map.png
project_history/reports/report_local_refinement_refactor_note/figures/fig09_deltaf_refinement.png
project_history/reports/report_local_refinement_refactor_note/figures/fig10_local_minima_candidates.png
docs/report_qa/20260605_stage_visual_report_structure.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Rewrote the second major-stage report as a compact stage-ordered visual
narrative.  The report now proceeds from exact warm-up baseline, to active
learning boundary concentration, to response numerical-audit downgrade, to
phase q-window/Delta audit correction, and finally to the local-refinement
stage-gate state.  New companion figures were copied from the numerical
reliability and phase q-window/Delta audit reports, using lowercase filenames
under the report figures directory.
```

Validation:

```text
.\build_note.ps1
Result: local_refinement_refactor_note.pdf rebuilt successfully, 8 pages.

pdftocairo -png -r 120 local_refinement_refactor_note.pdf tmp/pdf_render_cairo_local_refinement_stage_visual/page
Result: rendered all 8 pages; pages 3, 4, 5, and 7 visually inspected.

Select-String local_refinement_refactor_note.log for undefined references,
overfull boxes, fatal errors, LaTeX errors, and hyperref bookmark-token warnings
Result: no matches.

pdftotext local_refinement_refactor_note.pdf tmp/local_refinement_refactor_note_stage_visual.txt
Result: text extracted; no placeholder question-mark sequences found.
```

Current project state:

```text
The report now explains which visual changes came from which improvements:
active learning concentrated exact points near boundaries; response audits
downgraded unstable eta claims; phase q-window expansion moved high-JA labels;
Delta refinement and branch-candidate outputs became guardrails; and the
local-refinement refactor is presented as the engineering response to the
resulting robust-oracle cost.  The Stage 2-4 GPU variant-suite gate remains
pending external return and is not claimed as a completed result.
```

## 127. Added Label-Closed and Acquisition A/B Benchmark Figures to Report

Date: 2026-06-05

Files changed:

```text
project_history/reports/report_local_refinement_refactor_note/README.md
project_history/reports/report_local_refinement_refactor_note/decision_log.md
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.tex
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.pdf
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.md
project_history/reports/report_local_refinement_refactor_note/figures/fig11_ml_training_architecture.png
project_history/reports/report_local_refinement_refactor_note/figures/fig12_label_closed_exact_eta_boundaries.png
project_history/reports/report_local_refinement_refactor_note/figures/fig13_label_closed_selection_sources.png
project_history/reports/report_local_refinement_refactor_note/figures/fig14_label_closed_selection_focus.png
project_history/reports/report_local_refinement_refactor_note/figures/fig15_acq_full_selected_boundary_type.png
project_history/reports/report_local_refinement_refactor_note/figures/fig16_acq_simple_selected_boundary_type.png
docs/report_qa/20260605_stage_visual_report_structure.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added additional benchmark visuals requested during report review.  The report
now includes later label-closed active-learning figures corresponding to
active_learning_phase_boundary_report Fig. 8 and Fig. 9, the selection-focus
curve, the ML training architecture diagram, and full-vs-simple-phase
acquisition selected-by-boundary-type plots.  The simple-phase panel is the
mini-report Fig. 11 benchmark highlighted by the user.
```

Validation:

```text
.\build_note.ps1
Result: local_refinement_refactor_note.pdf rebuilt successfully, 10 pages.

pdftocairo -png -r 120 local_refinement_refactor_note.pdf tmp/pdf_render_cairo_local_refinement_benchmark_figs/page
Result: rendered all 10 pages; pages 3, 4, 5, and 6 visually inspected.

Select-String local_refinement_refactor_note.log for undefined references,
overfull boxes, fatal errors, LaTeX errors, and hyperref bookmark-token warnings
Result: no matches.

pdftotext local_refinement_refactor_note.pdf tmp/local_refinement_refactor_note_benchmark_figs.txt
Result: text extracted; no placeholder question-mark sequences found.
```

Current project state:

```text
The second major-stage report now contains a more complete visual benchmark
thread: initial exact phase map, active-learning boundary focusing, label-closed
active-learning result, ML/acquisition architecture and A/B profile comparison,
response audit downgrade, phase q-window correction, and local-refinement
engineering gate.  The report still does not claim the pending Stage 2-4 GPU
variant-suite result as complete.
```

## 128. Reworked Stage 2-4 Variant Suite as Point-Wise Slurm Array

Date: 2026-06-07

Files changed:

```text
scripts/run_local_refinement_variant_point.py
scripts/aggregate_local_refinement_variant_array_suite.py
scripts/package_local_refinement_variant_suite_hpc.py
scripts/check_variant_suite_hpc_status.py
scripts/package_local_refinement_upload_set.py output under hpc_packages/
tests/test_local_refinement_variant_suite_package.py
docs/DECISIONS.md
docs/report_qa/20260607_variant_array_suite_validation.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Replaced the Stage 2-4 variant-suite handoff design from five long serial
variant jobs to a standalone point-wise Slurm array package.  The new package
still recomputes baseline, cluster_only, cluster_optional_k3,
cluster_optional_k2, and cluster_energy_window from scratch, but each
variant/fixed-point pair is one independent array task.  Postprocess now uses
afterany and always attempts aggregation and return packaging, so timeout or
failed point tasks produce a diagnostic archive instead of leaving the suite in
DependencyNeverSatisfied with no return bundle.
```

Why it matters:

```text
The previous complete validation attempt showed baseline and cluster_only could
finish in about two hours, while the three more expensive optimization variants
timed out after 36 hours and produced no pointwise outputs.  The new design
keeps the same physics-equivalence gate but fixes the HPC granularity:
independent point outputs, baseline recomputation inside the same package,
variant-level aggregation compatible with the existing importer, explicit
task_status/missing_or_failed_tasks/equivalence_matrix summaries, and package
outputs rooted under the extracted package run directory.
```

Validation:

```text
python -m py_compile scripts/run_local_refinement_variant_point.py scripts/aggregate_local_refinement_variant_array_suite.py scripts/package_local_refinement_variant_suite_hpc.py scripts/check_variant_suite_hpc_status.py tests/test_local_refinement_variant_suite_package.py
Result: pass.

python -m pytest tests/test_local_refinement_variant_suite_package.py -q
Result: 5 passed.

python scripts/package_local_refinement_variant_suite_hpc.py
Result: rebuilt hpc_packages/local_refinement_refactor_variant_suite.tar.gz.

python hpc_packages/local_refinement_refactor_variant_suite/scripts/preflight_local_refinement_variant_suite_hpc.py --package-root hpc_packages/local_refinement_refactor_variant_suite --run-root hpc_packages/local_refinement_refactor_variant_suite/local_refinement_refactor_variant_suite_run --output-json hpc_packages/local_refinement_refactor_variant_suite/local_refinement_refactor_variant_suite_run/reports/local_refinement_refactor/variant_regression/preflight.json
Result: pass, fixed_point_count=32, task_count=160.

python scripts/package_local_refinement_upload_set.py
Result: rebuilt hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.

python hpc_packages/local_refinement_refactor_hpc_upload_set/verify_upload_set.py
Result: pass.
```

Current project state:

```text
The ready-to-upload handoff archive is
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz.  Its required
nested package is local_refinement_refactor_variant_suite.tar.gz with SHA256
61e6f4c97e2d7c3cbe565d6a8dd8ef4e59d85ce71ebbc7834c5cf3edf45afebd.  The outer
upload set SHA256 is e6c140d1889f1e77bb4403ecd95285e0cd7979dc12cc899fd3db6d48100622f8.  The nested
variant suite contains 160 array tasks (5 variants x 32 fixed points), excludes
gpuh01 by default, uses POINT_TIME=02:00:00 and MAX_CONCURRENT=32 by default,
and writes outputs under local_refinement_refactor_variant_suite_run.
```

Known unresolved issues:

```text
The new suite has been locally packaged and preflighted but not yet run on the
HPC GPUs.  If individual point tasks exceed POINT_TIME or fail numerically, the
return archive should still be produced, but the formal gate will fail until
missing_or_failed_tasks.csv is inspected and the corresponding point tasks are
rerun or explained.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz to the
cluster, extract it, and run bash run_required_variant_suite.sh.  If scheduler
capacity allows, keep MAX_CONCURRENT near 32 for the first full run; reduce it
only if queue policy or filesystem pressure requires it.
```

## 129. Inspected Returned Variant-Array Result and Generated Local Report

Date: 2026-06-08

Files changed:

```text
scripts/build_local_refinement_variant_array_return_report.py
project_history/reports/report_local_refinement_variant_array_return_20260608/README.md
project_history/reports/report_local_refinement_variant_array_return_20260608/decision_log.md
project_history/reports/report_local_refinement_variant_array_return_20260608/local_refinement_variant_array_return_report.md
project_history/reports/report_local_refinement_variant_array_return_20260608/local_refinement_variant_array_return_report.pdf
project_history/reports/report_local_refinement_variant_array_return_20260608/figures/*.png
project_history/reports/report_local_refinement_variant_array_return_20260608/tables/*.csv
docs/report_qa/20260608_variant_array_return_report.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Inspected the returned point-wise Stage 2-4 variant-suite archive under
local_refinement_refactor_hpc_upload_set and generated a compact local report
with PDF, Markdown, figures, tables, and decision log.  The suite returned a
diagnostic archive but did not pass the full gate: 88 of 160 point tasks
completed.  baseline and cluster_only completed all 32 fixed points and are
exactly equivalent on the comparison metrics.  cluster_optional_k3,
cluster_optional_k2, and cluster_energy_window each completed only 8 of 32
points and timed out on the hard-risk categories.
```

Why it matters:

```text
The result accepts the narrow cluster_only equivalence result but rejects the
more aggressive optimized-variant gate.  The failure mode is not an observed
physics mismatch on completed rows; it is missing hard-category coverage plus
a major performance regression.  On the eight common completed points, the
optimized variants take about 45 minutes on average versus about 3.4 minutes
for baseline/cluster_only and use about 85 local boxes versus about 6.
```

Validation:

```text
python -m py_compile scripts/build_local_refinement_variant_array_return_report.py
Result: pass.

python scripts/build_local_refinement_variant_array_return_report.py --return-root local_refinement_refactor_hpc_upload_set\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-dir project_history\reports\report_local_refinement_variant_array_return_20260608
Result: report regenerated with PDF, Markdown, figures, tables, README, and decision_log.md.

pdfinfo project_history\reports\report_local_refinement_variant_array_return_20260608\local_refinement_variant_array_return_report.pdf
Result: 3 pages, A4, PDF 1.4.

pdftoppm -png -r 120 project_history\reports\report_local_refinement_variant_array_return_20260608\local_refinement_variant_array_return_report.pdf tmp\local_refinement_variant_array_return_page
Result: rendered all three pages; visual inspection found no major clipping or overlap.

pdftotext project_history\reports\report_local_refinement_variant_array_return_20260608\local_refinement_variant_array_return_report.pdf tmp\local_refinement_variant_array_return_report.txt
Result: text extraction succeeded; no TODO, placeholder, or NaN markers found.
```

Current project state:

```text
The Stage 2-4 returned archive should be treated as a validation-failure
diagnostic report, not as a completed optimization proof.  cluster_only is the
only candidate variant accepted on the full 32-point fixed-point panel.  The
three optimized variants remain unresolved because their 72 missing tasks are
all in hard-risk categories.
```

Known unresolved issues:

```text
The optimized variants appear to over-expand local refinement, selecting about
85 local boxes on clean completed controls where baseline/cluster_only select
about 6.  Before another full rerun, inspect why optional cluster and energy
window controls are not pruning enough work.  The stale task JSON rows that
say running are interpreted as timed out because the Slurm err logs contain
time-limit cancellation messages.
```

Next recommended steps:

```text
Do not rerun baseline or cluster_only for this fixed-point gate unless the
panel changes.  Either rerun only the 72 missing optimized-variant tasks with
corrected checkpoint/time-limit strategy, or first run a smaller diagnostic
subset that records local-box selection decisions before timeout.
```

## 130. Audited Local-Refinement Target Construction Logic

Date: 2026-06-08

Files changed:

```text
scripts/debug_local_refinement_target_construction.py
reports/local_refinement_target_logic_audit/target_logic_audit.md
reports/local_refinement_target_logic_audit/target_logic_audit.pdf
reports/local_refinement_target_logic_audit/decision_log.md
reports/local_refinement_target_logic_audit/tables/*.csv
reports/local_refinement_target_logic_audit/figures/*.png
docs/report_qa/20260608_target_logic_audit.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Performed a report-only audit of the local-refinement target construction path
and returned variant-array outputs.  The helper does not call evaluate_points,
does not modify exact_oracle, does not modify acquisition, does not change
thermodynamic phase criteria or tolerances, and does not submit Slurm jobs.
It reads completed pointwise/local-box outputs plus source-code line evidence
to generate target logic audit tables, figures, Markdown, PDF, and decision
log under reports/local_refinement_target_logic_audit/.
```

Key findings:

```text
The local-box count of about 85 is confirmed to be completed refined local
boxes on completed rows, not raw candidates.  The selective variants overrun
because select_local_refine_targets keeps all mandatory basins first and the
current configs set mandatory_basins_can_exceed_cap=True.  K=3/K=2 only caps
ordinary optional basins.  Energy-window pruning is ordinary-only and pruned
zero rows on completed energy-window points, so it cannot reduce mandatory
overflow.  Clustering does run before mandatory selection; the failure is not
that mandatory targets bypass clustering entirely.
```

Known unresolved issues:

```text
The timed-out hard-risk optimized tasks have only startup JSON and no point
CSV, NPZ, or local-box timing CSV, so their exact target counts cannot be
reconstructed from current metadata.  Hard-risk target explosion is supported
by the completed clean controls and by code-path evidence, but not directly
proven for those timeout points.
```

Validation:

```text
python -m py_compile scripts/debug_local_refinement_target_construction.py
Result: pass.

python scripts/debug_local_refinement_target_construction.py --run-root local_refinement_refactor_hpc_upload_set\local_refinement_refactor_variant_suite\local_refinement_refactor_variant_suite_run --output-dir reports\local_refinement_target_logic_audit
Result: generated all requested CSV tables, seven figures, Markdown report,
PDF report, and decision_log.md.

pdfinfo reports\local_refinement_target_logic_audit\target_logic_audit.pdf
Result: 3 pages, A4, PDF 1.4.

pdftoppm -png -r 120 reports\local_refinement_target_logic_audit\target_logic_audit.pdf tmp\target_logic_audit_page
Result: rendered all three pages; visual inspection found readable tables and
figures with no major clipping.

pdftotext reports\local_refinement_target_logic_audit\target_logic_audit.pdf tmp\target_logic_audit.txt
Result: text extraction succeeded; no TODO, placeholder, or NaN markers found.
```

Next recommended steps:

```text
Do not directly rerun the 72 missing optimized-variant tasks.  First add a
true target-construction-only instrumented path that records raw candidates,
clustered basins, mandatory reasons, optional candidates, and final selected
targets before local-box scans.  Then test a rank-and-cap mandatory overflow
policy and rerun a bounded regression.
```

## 131. Implemented Middle-Plan Stage 1 Basin-Level Risk Annotation

Date: 2026-06-08

Files changed:

```text
ml_phase/exact_oracle.py
tests/test_basin_clustering.py
reports/local_refinement_refactor/stage_01_basin_level_risk/plan.md
reports/local_refinement_refactor/stage_01_basin_level_risk/implementation_summary.md
reports/local_refinement_refactor/stage_01_basin_level_risk/test_summary.md
reports/local_refinement_refactor/stage_01_basin_level_risk/dryrun_summary.md
reports/local_refinement_refactor/stage_01_basin_level_risk/decision_log.md
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/report_qa/20260608_basin_level_risk_refactor.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Implemented the Stage 1 step from
project_history/plans_and_runbooks/TwoPhase_Optimization_Middle.md.  Clustered
local-refinement basin representatives now carry explicit basin-level risk
flags and per-risk member counts.  Mandatory selection, energy-window pruning,
and local-box selection diagnostics now prefer basin-level risk metadata when
it exists, while unclustered baseline paths keep the old candidate-level
fallback.
```

Why it matters:

```text
The target-logic audit showed that optimized variants over-selected mandatory
targets.  Before adding rank-and-cap or hard total cap enforcement, the code
needs to make the unit of selection explicit.  This change ensures a basin
whose representative is ordinary but whose merged member is edge-risk or
Delta-near-epsilon remains explicitly marked and traceable at basin level.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py
Result: pass.

python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_mandatory_branch_keep.py -q
Result: 12 passed.

python -m pytest tests\test_basin_clustering.py tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_mandatory_branch_keep.py tests\test_local_refinement_regression_scaffold.py tests\test_local_refinement_variant_suite_package.py -q
Result: 21 passed.

python -m pytest tests -q
Result: 72 passed, 3 failed.  The failing tests are package/audit artifact
consistency checks unrelated to the basin-level risk functions:
variant_preflight_status missing, nested variant GPU script count 1 vs expected
5, and TwoPhase completion status incomplete vs pending_hpc.
```

Current project state:

```text
Middle-plan Stage 1 is locally implemented and documented.  It is an
auditability/correctness layer, not the target-count fix.  Stage 2 still needs
ranked mandatory selection and hard total cap enforcement before any rerun of
the 72 missing optimized tasks.
```

Known unresolved issues:

```text
Mandatory overflow remains unresolved because mandatory_basins_can_exceed_cap
is still unchanged.  The broader test suite also exposes stale or inconsistent
HPC package/audit artifacts that should be addressed before claiming full
repository-level validation.
```

Next recommended steps:

```text
Proceed to TwoPhase_Optimization_Middle.md Stage 2: ranked mandatory selection
and hard total cap.  Keep it feature-scoped, run target-construction-only
checks first, and do not submit GPU jobs until the target-count gate passes.
```

## 132. Implemented Middle-Plan Stage 2 Rank-and-Cap Target Selection

Date: 2026-06-08

Files changed:

```text
ml_phase/exact_oracle.py
scripts/run_local_refinement_fixed_point_regression.py
tests/test_selective_refinement.py
tests/test_local_refinement_regression_scaffold.py
reports/local_refinement_refactor/stage_02_rank_and_cap/
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/DECISIONS.md
docs/report_qa/20260608_rank_and_cap_refactor.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Implemented the Stage 2 step from
project_history/plans_and_runbooks/TwoPhase_Optimization_Middle.md.  Added an
opt-in `high_risk_overflow_policy = rank_and_cap` path, per-risk mandatory
basin caps, hard `max_total_refined_basins` enforcement for new variants, and
diagnostic metadata for selected and dropped target-construction rows.
```

Why it matters:

```text
The failed optimized variant-array runs showed that K=3/K=2 did not reduce
target count because mandatory-risk basins bypassed the cap.  The new
`rank_and_cap_*` variants preserve historical failed configurations while
providing a bounded target-construction path for the next dry-run gate.
```

Validation:

```text
python -m py_compile ml_phase\exact_oracle.py scripts\run_local_refinement_fixed_point_regression.py
Result: pass.

python -m pytest tests\test_selective_refinement.py tests\test_mandatory_branch_keep.py tests\test_energy_window_pruning.py tests\test_local_refinement_regression_scaffold.py -q
Result: 15 passed.

python -m pytest tests\test_feature_flag_baseline_equivalence.py tests\test_local_refinement_variant_suite_package.py tests\test_local_refinement_performance_report.py -q
Result: 10 passed.

python -m pytest tests\test_selective_refinement.py tests\test_mandatory_branch_keep.py tests\test_energy_window_pruning.py tests\test_local_refinement_regression_scaffold.py tests\test_feature_flag_baseline_equivalence.py -q
Result: 18 passed.
```

Current project state:

```text
Middle-plan Stage 2 is locally implemented and documented as a selector gate.
It has not yet been validated on all 32 fixed points.  No Slurm jobs were
submitted and no missing optimized tasks were rerun.
```

Known unresolved issues:

```text
The new policy still needs fixed-point target-construction dry-run evidence.
Energy-window effectiveness under rank-and-cap also remains pending until that
dry-run compares ordinary-pruned targets across risk categories.
```

Next recommended steps:

```text
Proceed to TwoPhase_Optimization_Middle.md Stage 3: target-construction dry-run
on the 32 fixed points for baseline, cluster_only, and the new
rank_and_cap_* variants.  Continue blocking expensive local-box or HPC reruns
until selected target counts pass the dry-run gate.
```

## 133. Verified Middle-Plan Stage 3 Ordinary Branch Policy Gate

Date: 2026-06-08

Files changed:

```text
tests/test_energy_window_pruning.py
reports/local_refinement_refactor/stage_03_ordinary_policy/
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md
docs/DECISIONS.md
docs/report_qa/20260608_ordinary_policy_refactor.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a local selector gate for the ordinary branch top-k and energy-window
policy under `rank_and_cap`.  The test confirms energy-window pruning removes
only ordinary high-energy basins and does not prune selected global-best,
edge-risk, Delta-near-epsilon, or near-degenerate basins.
```

Why it matters:

```text
The energy window should not be expected to fix mandatory target explosion.
It is an ordinary-branch pruning policy.  The runtime fix must come from
ranked mandatory caps, while energy-window effectiveness should be reported as
ordinary-pruned count.
```

Validation:

```text
python -m pytest tests\test_energy_window_pruning.py tests\test_selective_refinement.py -q
Result: 9 passed.

python -m pytest tests\test_mandatory_branch_keep.py tests\test_local_refinement_regression_scaffold.py tests\test_feature_flag_baseline_equivalence.py -q
Result: 10 passed.
```

Current project state:

```text
Middle-plan Stage 3 local ordinary-policy gate passes.  A real 32 fixed-point
target-construction dry-run is still required before any GPU local-box
regression.
```

Next recommended steps:

```text
Implement or run a target-construction-only dry-run for the 32 fixed points
using baseline, cluster_only, rank_and_cap_k3, rank_and_cap_k2, and
rank_and_cap_energy_window.  Do not submit full local-box GPU jobs until the
dry-run selected-target counts satisfy the cap.
```

## 134. Packaged 32-Point Target-Construction-Only HPC Gate

Date: 2026-06-08

Files changed:

```text
scripts/run_local_refinement_target_construction_point.py
scripts/aggregate_local_refinement_target_construction_dryrun.py
scripts/check_target_construction_dryrun_hpc_status.py
scripts/package_local_refinement_target_construction_dryrun_hpc.py
hpc_packages/local_refinement_target_construction_dryrun.tar.gz
hpc_packages/local_refinement_target_construction_dryrun/
docs/LOCAL_REFINEMENT_REFACTOR_STATUS.md
docs/DECISIONS.md
docs/report_qa/20260608_target_construction_hpc_package.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Created a complete HPC upload package for the 32 fixed-point
target-construction-only dry-run.  Each Slurm array task computes one fixed
point through coarse scan, q expansion, candidate detection, clustering,
risk annotation, energy-window marking, and final target selection, then stops
before local-box scans.  The task applies all five variants to the same shared
candidate set.
```

Why it matters:

```text
This is the required gate before any expensive GPU local-box regression.  It
tests whether `rank_and_cap_*` selected target counts satisfy the cap on the
actual 32 fixed points without repeating full local refinement or rerunning the
72 missing optimized tasks.
```

Validation:

```text
python -m py_compile scripts\run_local_refinement_target_construction_point.py scripts\aggregate_local_refinement_target_construction_dryrun.py scripts\check_target_construction_dryrun_hpc_status.py scripts\package_local_refinement_target_construction_dryrun_hpc.py
Result: pass.

python -m pytest tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_local_refinement_regression_scaffold.py -q
Result: 14 passed.

python scripts\package_local_refinement_target_construction_dryrun_hpc.py
Result: wrote hpc_packages\local_refinement_target_construction_dryrun.tar.gz.

Generated package preflight
Result: status=pass.

Generated package shell encoding check
Result: LF_OK.
```

Current project state:

```text
The target-construction dry-run package is ready to upload but has not been
submitted.  The package excludes gpuh01 in both Slurm scripts and workflow
submission defaults.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_target_construction_dryrun.tar.gz to HPC,
extract it, run bash scripts/submit_target_construction_dryrun_workflow.sh,
then download and inspect local_refinement_target_construction_dryrun_results.tar.gz.
Proceed to fixed-point GPU local-box regression only if the target-count gate
passes.
```

## 135. Checked Returned 32-Point Target-Construction Dry-Run

Date: 2026-06-08

Files changed:

```text
reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/target_construction_dryrun_return_check.md
reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/target_construction_dryrun_return_check.pdf
reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/decision_log.md
reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/tables/
reports/local_refinement_refactor/stage_04_target_construction_dryrun_return/figures/
docs/report_qa/20260608_target_construction_dryrun_return.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Inspected the returned HPC target-construction-only dry-run package under
local_refinement_target_construction_dryrun/.  The aggregate gate status is
pass: 32/32 points completed, 160 summary rows, 104015 candidate rows,
0 failed statuses, and 0 rank-and-cap gate failures.
```

Why it matters:

```text
The real 32 fixed-point construction gate confirms that rank_and_cap_k3,
rank_and_cap_k2, and rank_and_cap_energy_window prevent final selected-target
explosion before local-box scans.  Each capped variant selected at most 3
targets on every point, while baseline and cluster_only selected 6.
```

Validation:

```text
Returned gate JSON:
status=pass
expected_points=32
completed_points=32
failed_status_count=0
rank_and_cap_gate_failure_count=0

Log scan:
No Traceback, RuntimeError, ModuleNotFoundError, CUDA driver-too-old error,
CANCELLED, or TIMEOUT entries.  PyTorch preferred_linalg_library warnings were
present but nonfatal.

Per-point runtime:
mean wall_runtime_sec=36.65
min wall_runtime_sec=11.92
max wall_runtime_sec=55.89
local_box_scan=not_run
```

Current project state:

```text
The target-construction gate is complete and passing.  This does not establish
physics equivalence because no local-box scans were executed.  Energy-window
pruning had no effect in this returned set because ordinary branch count was
zero for all 32 points.
```

Next recommended steps:

```text
Proceed to a bounded local-box fixed-point regression for rank_and_cap_k3,
rank_and_cap_k2, and rank_and_cap_energy_window against baseline/cluster_only.
Do not rerun the old 72 missing optimized tasks from the pre-fix variant suite.
```

## 136. Added One-Pipeline Rank-and-Cap K3 Acceptance Workflow

Date: 2026-06-09

Files changed:

```text
scripts/run_local_refinement_rankcap_acceptance.py
scripts/submit_rankcap_acceptance_regression.sh
scripts/collect_rankcap_acceptance_report.sh
hpc_packages/local_refinement_rankcap_acceptance/
hpc_packages/local_refinement_rankcap_acceptance.tar.gz
reports/local_refinement_rankcap_acceptance/
docs/report_qa/20260609_rankcap_acceptance_pipeline.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a single rank-and-cap acceptance workflow for baseline robust_incremental
versus rank_and_cap_k3 only.  The workflow has Gate A target construction,
Gate B bounded local-box fixed-point regression, and Gate C runtime/workload
comparison, but does not split the work into new stage documents.
```

Why it matters:

```text
This is the required acceptance path before testing k2, energy-window,
branch reuse, or any active-learning validation.  It prevents the project from
running a broader test matrix before rank_and_cap_k3 has proven physics
equivalence and real local-refinement cost reduction against baseline.
```

Validation:

```text
python -m py_compile scripts\run_local_refinement_rankcap_acceptance.py
Result: pass.

python -m pytest tests\test_selective_refinement.py tests\test_energy_window_pruning.py tests\test_feature_flag_baseline_equivalence.py -q
Result: 12 passed.

Local collect result:
acceptance_status=pending_hpc_regression
gate_a_status=pass
max_rankcap_selected_targets=3
mandatory_overflow_points=32
physics_regression_status=pending on HPC

HPC package checks:
task_matrix variants: baseline=32, rank_and_cap_k3=32
shell_lf_ok=True
gpuh01 excluded in Slurm scripts.
```

Current project state:

```text
Gate A is satisfied using the returned 32 fixed-point target-construction
evidence.  Gate B and Gate C remain pending because real local-box regression
has not been run locally.  The current acceptance status is therefore
pending_hpc_regression, not pass, and the project should not enter one-
iteration AL validation yet.
```

Next recommended steps:

```text
Upload hpc_packages/local_refinement_rankcap_acceptance.tar.gz to HPC, extract
it, run bash scripts/submit_rankcap_acceptance_regression.sh from the package
root, then return local_refinement_rankcap_acceptance_results.tar.gz and rerun
the collect/import check locally.
```

## 137. Checked Returned Rank-and-Cap K3 Acceptance Results

Date: 2026-06-09

Files changed:

```text
scripts/run_local_refinement_rankcap_acceptance.py
hpc_packages/local_refinement_rankcap_acceptance/scripts/run_local_refinement_rankcap_acceptance.py
local_refinement_rankcap_acceptance_upload/local_refinement_rankcap_acceptance/scripts/run_local_refinement_rankcap_acceptance.py
local_refinement_rankcap_acceptance_upload/local_refinement_rankcap_acceptance/local_refinement_rankcap_acceptance_run/reports/local_refinement_rankcap_acceptance/
docs/report_qa/20260609_rankcap_acceptance_return.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Inspected the returned HPC rank-and-cap acceptance run.  Target construction
had 32/32 success JSONs and regression had 64/64 success JSONs.  The original
HPC collect job failed in lr_rc_collect-73799.out because the report-only CSV
field list omitted continuous baseline/rankcap columns.  Local inspection also
found that point_###_local_box_timing.csv files were being read as pointwise
regression rows.  The report-only collector now includes the missing columns
and reads only point_###.csv as regression point rows.
```

Why it matters:

```text
The fix allows already-returned HPC data to be collected without rerunning
physics.  It does not change the production exact oracle, phase criterion,
tolerances, acquisition, or active-learning loop.
```

Validation:

```text
Returned task status:
target JSONs: 32/32 success
regression JSONs: 64/64 success

Regenerated collect result:
acceptance_status=pass
gate_a_status=pass
gate_b_status=pass
gate_c_status=pass
expected_points=32
completed_points=32
max_rankcap_selected_targets=3
mandatory_overflow_points=32
phase_label_match_rate=1
trusted_exact_match_rate=1
training_eligible_exact_match_rate=1
q_unresolved_increased_count=0
delta_unresolved_increased_count=0
timeout_count=0
mismatch_point_count=0

Performance:
mean local boxes: 6 -> 2.75
mean local-refinement runtime: 189.767 s -> 86.9015 s
mean point total runtime: 234.194 s -> 132.314 s
local-box reduction: 54.17%
local-refinement runtime reduction: 54.21%
point total runtime reduction: 43.50%
```

Current project state:

```text
rank_and_cap_k3 has passed the one-pipeline 32 fixed-point acceptance test
against baseline robust_incremental.  The returned report is available under
local_refinement_rankcap_acceptance_upload/local_refinement_rankcap_acceptance/
local_refinement_rankcap_acceptance_run/reports/local_refinement_rankcap_acceptance/.
```

Next recommended steps:

```text
The user may decide whether to run one-iteration AL validation for the accepted
rank_and_cap_k3 path.  Do not infer that k2, energy-window, branch reuse,
Powell, GPU batching, Hamiltonian cache, mini AL, or full AL are already
validated.
```

## 138. Prepared One-Iteration AL Validation Package for Rank-and-Cap K3

Date: 2026-06-09

Files changed:

```text
ml_phase/exact_oracle.py
hpc_active_loop.sh
scripts/slurm_active_refine.sh
scripts/slurm_exact_oracle_array.sh
scripts/run_rankcap_k3_one_iter_validation.py
scripts/submit_rankcap_k3_one_iter_validation.sh
reports/rankcap_k3_one_iter_validation/preflight_check.md
reports/rankcap_k3_one_iter_validation/tables/preflight_check.csv
reports/rankcap_k3_one_iter_validation/tables/run_manifest.csv
hpc_packages/rankcap_k3_one_iter_validation/
hpc_packages/rankcap_k3_one_iter_validation.tar.gz
docs/report_qa/20260609_rankcap_k3_one_iter_validation_package.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Prepared a complete HPC upload package for exactly one active-learning
validation batch using the accepted rank_and_cap_k3 local-refinement path.
The wrapper runs iter000 as the random seed exact batch, appends
dataset_iter001, then runs iter001 as the only acquisition-selected batch,
appends dataset_iter002, and stops.  The workflow uses acquisition_profile=full,
oracle_mode=robust_incremental, incremental q expansion, basin clustering,
selective refinement, max_refined_minima=3, max_optional_refined_basins=3,
mandatory_basins_can_exceed_cap=False, and high_risk_overflow_policy=rank_and_cap.
It keeps k2, energy-window pruning, branch reuse, adaptive box, Powell, GPU
batching, Hamiltonian cache, acquisition changes, phase-criterion changes,
tolerance changes, and StopController changes disabled.
```

Why it matters:

```text
The previous 32 fixed-point acceptance test established paired physics
equivalence and speedup on fixed points.  This package is the next validation
step: it checks whether the same rank_and_cap_k3 oracle remains healthy inside
the real active-learning data flow, including dataset growth, label closure,
training eligibility, unresolved flags, phase coverage, local-box counts, and
rank/runtime summaries.
```

Validation:

```text
Preflight status: pass
rankcap acceptance prerequisite: pass
run_id collision check: pass
gpuh01 exclusion: configured as EXCLUDE_NODES=gpuh01
iteration semantics: iter000 seed + iter001 one acquisition batch, no iter002 acquisition
Python compile check: python -m py_compile ml_phase/exact_oracle.py scripts/run_rankcap_k3_one_iter_validation.py
Package Python compile check: python -m py_compile hpc_packages/rankcap_k3_one_iter_validation/ml_phase/exact_oracle.py hpc_packages/rankcap_k3_one_iter_validation/scripts/run_rankcap_k3_one_iter_validation.py
Bash encoding check: hpc_active_loop.sh and Slurm scripts are LF-only, no CRLF/NUL
Archive content check: required scripts, preflight files, and rankcap acceptance summary are present
Archive: hpc_packages/rankcap_k3_one_iter_validation.tar.gz
```

Current project state:

```text
The one-iteration AL validation has not yet been run on HPC.  Local work has
produced the upload package, preflight report, submit script, and collect/report
script.  validation_status remains pending_hpc_run until the returned archive is
checked.
```

Next recommended steps:

```text
Upload hpc_packages/rankcap_k3_one_iter_validation.tar.gz to the HPC, extract it,
run nohup bash scripts/submit_rankcap_k3_one_iter_validation.sh from the package
root, then return ML_Phase_512_Speed_20260602/rankcap_k3_one_iter_validation_results.tar.gz
for local inspection.  Do not start 3-5 iteration mini AL or full AL until this
one-iteration validation report passes.
```

## 139. Checked Returned Rank-and-Cap K3 One-Iteration AL Validation

Date: 2026-06-09

Files changed:

```text
rankcap_k3_one_iter_validation/reports/rankcap_k3_one_iter_validation/
rankcap_k3_one_iter_validation/ML_Phase_512_Speed_20260602/rankcap_k3_one_iter_validation_results.tar.gz
scripts/run_rankcap_k3_one_iter_validation.py
rankcap_k3_one_iter_validation/scripts/run_rankcap_k3_one_iter_validation.py
docs/report_qa/20260609_rankcap_k3_one_iter_validation_return.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Inspected the returned one-iteration AL validation package for rank_and_cap_k3.
The HPC run completed iter000 as the random seed exact batch and iter001 as the
only acquisition-selected exact batch.  All 16 exact shard JSONs/NPZ files were
present across the two iterations, merge and append completed, and the final
dataset reached dataset_iter002.  The local report-only collect step was rerun
from returned data to regenerate the validation markdown/PDF and returned archive.
No physical calculation was rerun.
```

Why it matters:

```text
This validates the accepted rank_and_cap_k3 local-refinement target construction
inside a real active-learning data flow, not only on the fixed 32-point
acceptance suite.  It confirms dataset growth, phase coverage, controlled rerun
rate, controlled unresolved flags, and reduced local-box workload for the one
acquisition batch.
```

Validation:

```text
run_id=active_boundary_discovery_rankcap_k3_one_iter_validation_v1
validation_status=pass
exact_shard_completion_status=confirmed
merge_append_status=confirmed
final_report_generation_status=confirmed
dataset samples: iter000=0, iter001=506, iter002=744
final phase counts: normal=377, uniform_SC=13, FFLO=354
iter001 selected points=256
iter001 training_eligible_count=238
training_eligible_fraction=0.929688
rerun_required_fraction=0.0703125
q_unresolved_count=0
delta_unresolved_count=0
mean/max local_boxes_refined_count=2.79297/3
mean local_refinement_runtime_sec=88.3859
mean point_total_runtime_sec=112.359
fallback_full_rescan_runtime_sec_sum=0 for iter000 and iter001
log scan: no traceback, OOM, CUDA initialization failure, timeout, or cancellation found
rank runtime imbalance ratio: iter000=1.0186, iter001=1.0654
```

Current project state:

```text
rank_and_cap_k3 has passed the one-iteration active-learning validation.  The
validation report is under rankcap_k3_one_iter_validation/reports/
rankcap_k3_one_iter_validation/.  A report-only wording issue was corrected so
the rank-imbalance question reports no material imbalance rather than implying
a serious imbalance.
```

Next recommended steps:

```text
It is reasonable to proceed to a 3-5 iteration mini AL validation for
rank_and_cap_k3 only.  Full-length AL remains cannot-determine from this single
iteration, and k2, energy-window, branch reuse, Powell, adaptive box, GPU
batching, and Hamiltonian cache remain unvalidated.
```

## 140. Packaged Rank-and-Cap K3 Five-Iteration Validation and Full Loop

Date: 2026-06-09

Files changed:

```text
scripts/run_rankcap_k3_active_loop_package.py
scripts/submit_rankcap_k3_5iter_validation.sh
scripts/submit_rankcap_k3_full_loop.sh
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
reports/rankcap_k3_5iter_validation/
reports/rankcap_k3_full_loop/
hpc_packages/rankcap_k3_5iter_validation/
hpc_packages/rankcap_k3_5iter_validation.tar.gz
hpc_packages/rankcap_k3_5iter_validation.tar.gz.sha256
hpc_packages/rankcap_k3_5iter_validation.tar.gz.metadata.json
hpc_packages/rankcap_k3_full_loop/
hpc_packages/rankcap_k3_full_loop.tar.gz
hpc_packages/rankcap_k3_full_loop.tar.gz.sha256
hpc_packages/rankcap_k3_full_loop.tar.gz.metadata.json
```

Summary:

```text
Created two independent rank_and_cap_k3 active-learning HPC upload packages.
The five-iteration package runs iter000 as the random exact seed and iter001
through iter005 as five acquisition-selected batches.  The full-loop package
starts from its own seed and defaults to 31 total iterations, i.e. seed plus
30 acquisition-selected batches.  Both packages include the code snapshot,
docs, tests, fixed-point acceptance evidence, one-iteration validation
evidence, submit wrappers, collection/report logic, manifests, return-archive
logic, and SHA256 sidecars.
```

Why it matters:

```text
The one-iteration validation established that rank_and_cap_k3 works for a
single active-learning batch.  The five-iteration package is the next safety
gate for repeated train/select/exact/merge/append behavior.  The full-loop
package is uploadable now but intentionally guarded with CONFIRM_FULL_LOOP=1,
so it can be staged on the cluster without being accidentally launched before
the five-iteration validation is reviewed.
```

Validation:

```text
python -m py_compile scripts/run_rankcap_k3_active_loop_package.py
python scripts/run_rankcap_k3_active_loop_package.py --mode preflight --package-kind 5iter --world-size 8
python scripts/run_rankcap_k3_active_loop_package.py --mode preflight --package-kind full --world-size 8
python scripts/run_rankcap_k3_active_loop_package.py --mode package --package-kind 5iter --world-size 8
python scripts/run_rankcap_k3_active_loop_package.py --mode package --package-kind full --world-size 8
package-internal preflight: pass for both packages
evidence in package: fixed-point rankcap acceptance pass; one-iteration AL validation pass
shell encoding check: all packaged .sh files are UTF-8, LF-only, no BOM, no CR
gpuh01 exclusion check: hpc_active_loop.sh passes EXCLUDE_NODES to both candidate and exact sbatch calls
5iter archive SHA256: 2aa8508861979ebcae3c49fe414767d0b58ff1e244365960ae8d7fabe37b64ca
full-loop archive SHA256: a45fb7cbba6f87313e0889faab679ac0749553321d98d05ce445294aaee9cd7f
```

Current project state:

```text
rank_and_cap_k3 is ready for a five-iteration closed-loop validation on HPC.
The full-loop package is also prepared as an independent package, but it
should not be launched until the five-iteration validation return report is
reviewed.  The full-loop submit wrapper requires CONFIRM_FULL_LOOP=1.
```

Next recommended steps:

```text
Upload hpc_packages/rankcap_k3_5iter_validation.tar.gz and
hpc_packages/rankcap_k3_full_loop.tar.gz.  Run only the five-iteration package
first.  After its returned report passes, decide whether to launch the full-loop
package by setting CONFIRM_FULL_LOOP=1 in the extracted full-loop package root.
```

## 2026-06-09 Second-Stage Report Diagram Update

Files changed:

```text
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.tex
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.md
project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.pdf
project_history/reports/report_local_refinement_refactor_note/README.md
project_history/reports/report_local_refinement_refactor_note/decision_log.md
project_history/reports/report_local_refinement_refactor_note/scripts/plot_free_energy_flow_and_system.py
project_history/reports/report_local_refinement_refactor_note/figures/fig17_free_energy_minimization_flow.png
project_history/reports/report_local_refinement_refactor_note/figures/fig18_fflo_altermagnetic_system_schematic.png
project_history/reports/report_local_refinement_refactor_note/tables/diagram_source_notes.csv
docs/report_qa/20260609_free_energy_flow_and_system_diagrams.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added two generated explanatory diagrams to the second major-stage report:
one conceptual FFLO + altermagnetic BdG system schematic and one code-path-
aware exact free-energy/local-refinement flowchart.  The TeX and Markdown
reports now introduce these figures before the stage-by-stage phase-diagram
narrative.  Companion metadata records that the figures are explanatory
material based on MODEL_SPEC.md, docs/NUMERICS_SPEC.md, and exact_oracle.py
function names.
```

Important implementation decision:

```text
The diagrams are report material only. They do not change the Hamiltonian,
phase thresholds, exact oracle, acquisition function, tolerance policy, or
local-refinement production behavior.
```

Validation:

```text
python project_history/reports/report_local_refinement_refactor_note/scripts/plot_free_energy_flow_and_system.py
powershell -ExecutionPolicy Bypass -File project_history/reports/report_local_refinement_refactor_note/build_note.ps1
pdftoppm -png -f 3 -l 4 project_history/reports/report_local_refinement_refactor_note/local_refinement_refactor_note.pdf tmp/pdfs/local_refinement_refactor_note_final_diagrams_v2
visual inspection of rendered pages 3 and 4: no clipping, overlap, missing figure, or black-square rendering issue
```

Next recommended steps:

```text
Use the new system schematic and free-energy flowchart as front-matter figures
when adapting the second-stage report into the larger final report or thesis
narrative.  Keep numerical-result claims separate from these explanatory
figures.
```

## 2026-06-10 Rankcap K3 Five-Iteration Validation Recheck

Files changed:

```text
scripts/recheck_rankcap_k3_5iter_validation.py
reports/rankcap_k3_5iter_validation_recheck/
docs/report_qa/20260610_rankcap_k3_5iter_validation_recheck.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Reviewed the returned rankcap_k3_5iter_validation package.  The original
package report marked validation_status=fail because it reported
max_local_boxes_refined_count=24.  A rank-level recheck showed this is a report
aggregation false positive: the collector merged local-box timing CSV files
without preserving rank, then grouped by iteration and rank-local point_id.
Recomputing from the raw iterXXX_local_box_timing_rankYYY_of008.csv files with
(iteration, rank, point_id) gives 1792 actual exact points, corrected max local
boxes = 3, corrected points above 3 = 0, and corrected validation_status=pass.
```

Why it matters:

```text
The five-iteration closed-loop validation supports rank_and_cap_k3: all shards,
merge, append, phase coverage, training-eligible growth, q/delta unresolved
checks, rerun fraction, tracebacks scan, mean local boxes, and corrected max
local boxes pass.  The package's numerical run is usable, but the active-loop
report collector has a rank-local point-id aggregation bug that would also
affect full-loop report status unless patched or rechecked.
```

Validation:

```text
python scripts/recheck_rankcap_k3_5iter_validation.py
pdftoppm -png -r 150 reports/rankcap_k3_5iter_validation_recheck/rankcap_k3_5iter_validation_recheck.pdf tmp/pdfs/rankcap_k3_5iter_validation_recheck_all
visual inspection of rendered pages 1-3: no clipping, overlap, missing figures, or unreadable tables
corrected gate table: all pass
corrected local boxes: unweighted mean 2.708333, weighted mean 2.659598, max 3
final dataset phase counts: normal=608, uniform_SC=102, FFLO=982
```

Current project state:

```text
rank_and_cap_k3 has now passed the five-iteration closed-loop validation after
correcting the report aggregation key.  The existing full-loop package should
not be judged by its unpatched report collector; patch the collector or run an
equivalent rank-level recheck on the returned full-loop data before making the
final acceptance decision.
```

Next recommended steps:

```text
Patch scripts/run_rankcap_k3_active_loop_package.py so local_box_rows preserves
rank/world_size from each timing filename and max local boxes is computed by
(iteration, rank, point_id), or by a globally unique point id.  Then repackage
or recheck the full-loop package before treating its validation_status as
authoritative.
```

## 2026-06-17 Rankcap K3 Full-Loop Enhanced Report

Files changed:

```text
scripts/build_rankcap_k3_full_loop_enhanced_report.py
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/full_loop_enhanced_report/
docs/report_qa/20260617_rankcap_k3_full_loop_enhanced_report.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Reviewed the returned rankcap_k3_full_loop package and generated an enhanced
full-loop report with phase diagrams, phase-evolution snapshots, learning
curves, local-box gate plots, runtime curves, rank-imbalance plots, and recent
acquisition overlays.  The original package report marked validation_status=fail
because it repeated the known rank-local point_id aggregation bug in merged
local-box timing rows.  Recomputing local-box counts from raw
iterXXX_local_box_timing_rankYYY_of008.csv files with (iteration, rank,
point_id) gives corrected_validation_status=pass.
```

Why it matters:

```text
The full-loop rank_and_cap_k3 run is accepted as a completed numerical run for
the second-stage report.  The corrected validation shows 8192 exact points,
corrected max local boxes = 3, points above 3 = 0, unweighted mean local boxes =
2.79296875, weighted mean local boxes = 2.7796630859375, and no traceback
matches.  The final dataset has 6880 samples: normal=1777, uniform_SC=715,
FFLO=4388.
```

Validation:

```text
python scripts/build_rankcap_k3_full_loop_enhanced_report.py
pdftoppm -png -r 140 rankcap_k3_full_loop/reports/rankcap_k3_full_loop/rankcap_k3_full_loop_enhanced.pdf tmp/pdfs/rankcap_k3_full_loop_enhanced_rerender_page
visual inspection of rendered pages 1-8: summary table, final phase diagram,
phase snapshots, learning curve, training/rerun curve, corrected local-box gate,
runtime curve, and recent acquisition overlay rendered without clipping or
missing figures
corrected gate table: all pass
```

Current project state:

```text
rank_and_cap_k3 has passed fixed-point, one-iteration, five-iteration, and
full-loop validation after rank-level report rechecks.  The active-loop report
collector still needs a code patch before future packages should treat
validation_status as authoritative without a separate recheck.
```

Next recommended steps:

```text
Use the enhanced full-loop report figures and tables in the second-stage report.
Patch the report collector's local-box aggregation key before running future
active-loop packages.
```

Addendum:

```text
The enhanced full-loop report was upgraded to a LaTeX source/PDF report and now
includes full-loop timing and surrogate machine-learning metric curves.  The
run contains 31 exact iterations: iter000 random seed plus 30 acquisition
batches.  Estimated wall time was 36.5713 h from active-loop lock timestamp to
collector summary timestamp, or 70.7832 min per exact iteration and 73.1426 min
per acquisition batch.  Exact-oracle wall time summed from per-iteration max
rank elapsed time was 34.6863 h, with mean acquisition exact-oracle wall time
65.5315 min.

Compared with the package's robust-incremental reference, rankcap_k3 reduced
mean local boxes from 6.0 to 2.79296875 (-53.4505%, 2.15x), mean
local-refinement runtime from 189.767 to 88.2856 sec/point (-53.4768%, 2.15x),
and mean point-total runtime from 234.194 to 117.285 sec/point (-49.9199%,
about 2.00x).  The package did not retain raw per-epoch training loss, so the
new ML curves plot available surrogate metrics from metrics_history.json
instead of claiming optimizer loss curves.
```

Additional generated files:

```text
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/rankcap_k3_full_loop_enhanced.tex
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_exact_walltime_curve.png
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_surrogate_metric_curves.png
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_phase_accuracy_reduction_curve.png
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/tables/enhanced_runtime_timing_summary.csv
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/tables/enhanced_exact_iteration_walltime.csv
rankcap_k3_full_loop/reports/rankcap_k3_full_loop/tables/enhanced_surrogate_metric_history.csv
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/full_loop_enhanced_report/
```

Additional validation:

```text
python -m py_compile scripts/build_rankcap_k3_full_loop_enhanced_report.py
python scripts/build_rankcap_k3_full_loop_enhanced_report.py
pdftoppm -png -r 140 rankcap_k3_full_loop/reports/rankcap_k3_full_loop/rankcap_k3_full_loop_enhanced.pdf tmp/pdfs/rankcap_k3_full_loop_latex_page
representative rendered LaTeX PDF pages visually inspected: summary, runtime,
validation gates, phase diagrams, local-box plots, exact wall-time curve, and
surrogate metric curves were readable
```

Convergence clarification:

```text
The rankcap_k3 full-loop result should be described as a successful full-loop
optimization validation, not as formal active-learning convergence.  The final
StopController state reports stop_reason=max_iterations, convergence_pass=false,
passed_condition_count=3, required_pass_count=4, patience_counter=0.  The final
phase-map and boundary-shift stability checks passed, but label_surprise_rate
and boundary_coverage_p95 did not satisfy the stop criteria.  A report note was
saved at docs/report_qa/20260617_rankcap_k3_full_loop_convergence.md.
```

Non-convergence analysis:

```text
Compared with the earlier 20-iteration q-delta discovery run, the rankcap_k3
full-loop run is not a like-for-like convergence test.  The earlier run stopped
at iteration 19 with stop_reason=converged_main_phase_boundaries after 4
consecutive main-boundary passes, but its label_surprise_rate condition was
still false.  The current rankcap_k3 run passed phase-map and boundary-shift
stability but failed label_surprise_rate and boundary_coverage_p95.  It also
showed much stronger q-expansion / hard-risk activity: q_expanded=2724/6880
versus 79/5107 in the earlier run, final q_edge_trigger_rate=0.66015625, and
final rerun_required_rate=0.36328125.  The likely bottleneck is now late-stage
active-learning acquisition/coverage behavior, not local-refinement cost.  A
report note was saved at
docs/report_qa/20260617_rankcap_k3_full_loop_nonconvergence_analysis.md.
```

Last-five stop-failure audit:

```text
A report-only audit of the rankcap_k3 full-loop last five acquisition-selected
batches was generated under
rankcap_k3_full_loop/reports/rankcap_k3_last5_stop_audit/ and mirrored into
rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/last5_selection_stop_audit/.
The audit covers iterations 26-30 and does not modify acquisition, exact oracle,
rankcap_k3, StopController, tolerances, or active-loop outputs.

The final StopController state remains stop_reason=max_iterations,
convergence_pass=false, passed_condition_count=3, required_pass_count=4.
The phase-map and boundary-shift stability checks passed, while label surprise
and boundary coverage failed.  Final label_surprise_rate was 0.18359375 versus
the 0.05 tolerance, and final boundary_coverage_p95 was
0.006588078458684216 versus the 0.00625 tolerance.

The last-five StopController label-surprise rates were reproduced exactly from
selected_points_by_pool.csv predictions and exact-batch Delta/q labels:
0.14453125, 0.1640625, 0.1328125, 0.17578125, and 0.18359375.  The mismatch is
dominated by selected points predicted as normal before exact evaluation but
labeled as FFLO after exact evaluation.  Final iteration diagnostics still show
high q-edge / rerun activity: q_edge_trigger_rate=0.66015625 and
rerun_required_rate=0.36328125.  The supported interpretation is that formal
non-convergence is now a late-stage acquisition / stop-metric issue, not a
rankcap_k3 local-refinement target-explosion issue.

Generated files include last5_selection_stop_audit.md,
last5_selection_stop_audit.pdf, decision_log.md, six CSV audit tables, and six
PNG figures.  A reusable report note was saved at
docs/report_qa/20260617_rankcap_k3_last5_stop_audit.md.
```

## 2026-06-17 Second-Stage Discussion Report

Files changed:

```text
scripts/build_second_stage_discussion_report.py
project_history/reports/report_second_stage_discussion/
docs/report_qa/20260617_second_stage_discussion_report.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Generated a discussion-ready second-stage report that organizes the full
project route from exact BdG warm-up phase diagrams to active learning,
numerical q-window / Delta / response audits, local-refinement target
construction diagnosis, rank_and_cap_k3 validation, full-loop results, and the
remaining StopController non-convergence issue.
```

Why it matters:

```text
The earlier local-refinement refactor note stopped before the rankcap
validation sequence was complete.  The new report updates the narrative to the
current state: rank_and_cap_k3 is accepted as a cost-controlled robust-oracle
optimization after fixed-point, one-iteration, five-iteration, and full-loop
validation, while formal active-learning convergence remains open due to
late-stage label surprise and boundary coverage.
```

Generated outputs:

```text
project_history/reports/report_second_stage_discussion/second_stage_discussion_report.md
project_history/reports/report_second_stage_discussion/second_stage_discussion_report.tex
project_history/reports/report_second_stage_discussion/second_stage_discussion_report.pdf
project_history/reports/report_second_stage_discussion/decision_log.md
project_history/reports/report_second_stage_discussion/tables/milestone_summary.csv
project_history/reports/report_second_stage_discussion/tables/validation_summary.csv
project_history/reports/report_second_stage_discussion/tables/speedup_summary.csv
project_history/reports/report_second_stage_discussion/tables/final_state_summary.csv
project_history/reports/report_second_stage_discussion/tables/figure_manifest.csv
project_history/reports/report_second_stage_discussion/figures/
```

Validation:

```text
python -m py_compile scripts/build_second_stage_discussion_report.py
python scripts/build_second_stage_discussion_report.py
pdflatex -interaction=nonstopmode second_stage_discussion_report.tex
pdftoppm -png -r 130 project_history/reports/report_second_stage_discussion/second_stage_discussion_report.pdf tmp/pdfs/second_stage_discussion_page
rendered PDF pages checked: title/roadmap, validation funnel, fixed-point
local-box figure, full-loop phase diagram, and last-five stop-failure
diagnostics were readable
```

Current project state:

```text
The second-stage discussion report is ready for project discussion.  It should
be used as a high-level narrative only; detailed numerical evidence remains in
the companion reports and CSV tables.  The next scientific decision is whether
formal convergence is required strongly enough to justify a separately planned
late-stage cleanup validation.
```

## 2026-06-17 Phase Boundary Stability Audit

Files changed:

```text
scripts/build_phase_boundary_stability_audit.py
reports/phase_boundary_stability_audit/
docs/report_qa/20260617_phase_boundary_stability_audit.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Generated a report-only audit of how StopController computes
phase_map_change, boundary_shift_normal_sc, boundary_shift_uniform_fflo,
label_surprise_rate, and boundary_coverage_p95 for the rankcap_k3 full-loop
run.  The audit does not modify acquisition, exact oracle, rankcap_k3,
StopController, tolerances, active-run artifacts, or Slurm state.
```

Main conclusions:

```text
phase_map_change is the changed-label fraction between adjacent saved
surrogate dense-grid phase_pred arrays, not an exact-dataset phase-map metric.

boundary_shift_normal_sc and boundary_shift_uniform_fflo extract predicted
phase-label crossing points from the monitor grid and use the p95 of
bidirectional normalized nearest-neighbor distances between current and
previous boundary point sets.

The final StopController phase-map and boundary-shift stability conditions
pass:
phase_map_change = 0.0006204676775119246 < 0.002
boundary_shift_normal_sc = 0.002604166666666674 < 0.004166666666666667
boundary_shift_uniform_fflo = 0.0 < 0.004166666666666667

Formal convergence still fails because label_surprise_rate =
0.18359375 > 0.05 and boundary_coverage_p95 =
0.006588078458684216 > 0.00625, leaving only 3/5 main conditions passed
where 4 are required.
```

Generated outputs:

```text
reports/phase_boundary_stability_audit/phase_boundary_stability_audit.md
reports/phase_boundary_stability_audit/phase_boundary_stability_audit.pdf
reports/phase_boundary_stability_audit/decision_log.md
reports/phase_boundary_stability_audit/tables/
reports/phase_boundary_stability_audit/figures/
```

Validation:

```text
python -m py_compile scripts/build_phase_boundary_stability_audit.py
python scripts/build_phase_boundary_stability_audit.py
pdflatex -interaction=nonstopmode phase_boundary_stability_audit.tex
pdftoppm -png -r 130 reports/phase_boundary_stability_audit/phase_boundary_stability_audit.pdf tmp/pdfs/phase_boundary_stability_audit/page
```

Current project state:

```text
The full-loop result should be described as having a stable main predicted
phase map and stable main predicted boundaries by StopController metrics, but
not as formally converged.  If formal convergence is required, the next
calculation should be a separately planned cleanup acquisition or
boundary-coverage validation rather than a threshold or StopController change.
```

## 2026-06-18 Surprise Decomposition and Cleanup Profile

Files changed:

```text
scripts/build_surprise_decomposition_audit.py
reports/surprise_decomposition_audit/
ml_phase/acquisition.py
ml_phase/config.py
ml_phase/active_refine.py
hpc_active_loop.sh
scripts/slurm_active_refine.sh
scripts/check_acquisition_profiles.py
docs/report_qa/20260618_surprise_cleanup_strategy.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Generated a report-only decomposition of late-stage label surprise in the
rankcap_k3 full-loop run, then added an opt-in `surprise_cleanup` acquisition
profile for a future bounded cleanup validation.  The default `full` and
`simple_phase` profiles are intentionally unchanged.
```

Main conclusions:

```text
The final non-convergence is driven by label surprise and slightly sparse
boundary coverage, not by local-refinement cap failure or main phase-map drift.

The dominant surprise channel is predicted normal -> exact FFLO:
normal_to_FFLO surprise count = 814
qedge_or_expanded_rate = 0.9963144963144963
rerun_rate = 0.9434889434889435

This supports a cleanup validation strategy that de-emphasizes q-edge-heavy
repeated selections and targets remaining surprise/coverage gaps after the
main phase map has stabilized.
```

Implementation decision:

```text
Added `acquisition_profile=surprise_cleanup` as an explicit opt-in profile.
It keeps phase/boundary uncertainty active, removes q-edge risk as a positive
numerical reward, applies a recorded q-edge penalty factor, and writes that
factor to candidate/monitor/selected artifacts.  No thermodynamic phase
criterion, exact oracle logic, rankcap_k3 logic, StopController threshold, or
default acquisition profile was changed.
```

Generated outputs:

```text
reports/surprise_decomposition_audit/surprise_decomposition_audit.md
reports/surprise_decomposition_audit/surprise_decomposition_audit.pdf
reports/surprise_decomposition_audit/decision_log.md
reports/surprise_decomposition_audit/tables/
reports/surprise_decomposition_audit/figures/
docs/report_qa/20260618_surprise_cleanup_strategy.md
```

Validation:

```text
python scripts/build_surprise_decomposition_audit.py
python scripts/check_acquisition_profiles.py
python -m py_compile ml_phase/acquisition.py ml_phase/config.py ml_phase/active_refine.py scripts/check_acquisition_profiles.py scripts/build_surprise_decomposition_audit.py
```

Current project state:

```text
The next optimization should be a one-batch cleanup validation from the
completed full-loop dataset using `ACQUISITION_PROFILE=surprise_cleanup` and a
new run_id.  It should be judged on whether label_surprise_rate decreases,
boundary_coverage_p95 improves, and q-edge/rerun-heavy selections no longer
dominate.  Do not rerun ordinary full-loop iterations before this cleanup
hypothesis is tested.
```

## 2026-06-18 Surprise Review Recheck

Files changed:

```text
scripts/build_surprise_review_recheck.py
reports/surprise_review_recheck/
docs/report_qa/20260618_surprise_review_recheck.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Generated a report-only recheck in response to a review of the surprise
decomposition report.  The recheck separates iter030, strict last-five, and
all-acquisition-iteration scopes, and audits the exact StopController
label_surprise_rate denominator.
```

Main conclusions:

```text
StopController label_surprise_rate counts all matched selected exact points.
It does not filter by trusted_exact, training_eligible_exact, rerun_required,
or q-expanded/q-edge status.

iter030:
selected = 256
surprise = 47
surprise_rate = 0.18359375
all surprises = predicted normal -> exact FFLO

strict last-five:
selected = 1280
surprise = 205
surprise_rate = 0.16015625
all surprises = predicted normal -> exact FFLO

strict last-five trusted/nonrerun surprise = 0 / 849 = 0
strict last-five rerun-required surprise = 205 / 341 = 0.6011730205278593
```

Interpretation:

```text
The late selected-batch surprise should not be treated as a global phase-map
error proxy.  The current blocker is concentrated in hard-risk rerun-required
normal-to-FFLO boundary points.  Current artifacts do not contain an
independent fixed-probe or random-control exact batch, so fixed-probe/control
surprise is currently cannot determine.
```

Generated outputs:

```text
reports/surprise_review_recheck/surprise_review_recheck.md
reports/surprise_review_recheck/surprise_review_recheck.pdf
reports/surprise_review_recheck/decision_log.md
reports/surprise_review_recheck/tables/
reports/surprise_review_recheck/figures/
docs/report_qa/20260618_surprise_review_recheck.md
```

Validation:

```text
python scripts/build_surprise_review_recheck.py
python -m py_compile scripts/build_surprise_review_recheck.py
```

Current project state:

```text
Before changing StopController or continuing full discovery, add a held-out
fixed-probe or random-control exact batch and report selected-batch surprise
in trusted/rerun/q-expanded strata.  This should determine whether formal
convergence should use all-selected surprise, trusted surprise, or a separate
held-out control surprise metric.
```

## 2026-06-18 Trusted Surprise Counterfactual

Files changed:

```text
scripts/build_trusted_surprise_counterfactual.py
reports/trusted_surprise_counterfactual/
docs/report_qa/20260618_trusted_surprise_counterfactual.md
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Generated a report-only counterfactual reconstruction of StopController
label-surprise convergence using existing full-loop artifacts.  The current
all-selected surprise definition was exactly reproduced, then alternative
non-rerun, training-eligible, trusted, hard-risk, trusted-boundary, and
trusted-interior surprise definitions were reconstructed without modifying
StopController or active-learning artifacts.
```

Main conclusions:

```text
current all-selected final surprise = 47 / 256 = 0.183594
final non-rerun surprise = 0 / 163 = 0.000000
final trusted surprise = 0 / 137 = 0.000000
final hard-risk surprise = 47 / 119 = 0.394958
last-five fraction of all surprise from rerun-required points = 1.000000
trusted-only counterfactual earliest stop iteration = 17
remaining blocker under trusted surprise = boundary_coverage_p95
recommended decision = Decision B, split formal surprise into trusted and
hard-risk layers
```

Generated outputs:

```text
reports/trusted_surprise_counterfactual/trusted_surprise_counterfactual.md
reports/trusted_surprise_counterfactual/trusted_surprise_counterfactual.pdf
reports/trusted_surprise_counterfactual/decision_log.md
reports/trusted_surprise_counterfactual/tables/
reports/trusted_surprise_counterfactual/figures/
docs/report_qa/20260618_trusted_surprise_counterfactual.md
```

Validation:

```text
python scripts/build_trusted_surprise_counterfactual.py
python -m py_compile scripts/build_trusted_surprise_counterfactual.py
pdftoppm rendered the first PDF page for a nonblank layout sanity check
```

Current project state:

```text
The evidence now supports treating all-selected surprise as an acquisition
difficulty diagnostic rather than the sole formal convergence gate.  The next
StopController design discussion should consider preserving the current field,
adding trusted and hard-risk surprise layers, requiring a denominator floor for
trusted surprise, and routing hard-risk rerun-required points to a numerical
reliability queue.  Boundary coverage remains the only late-stage blocker under
the trusted-surprise counterfactual.
```

## 2026-06-18 StopController Trusted-Surprise Implementation and HPC Package

Files changed:

```text
ml_phase/stop_controller.py
ml_phase/report_builder.py
hpc_active_loop.sh
scripts/recover_active_iter.sh
scripts/dev_check_stop_controller.py
scripts/replay_stopcontroller_surprise_modes.py
scripts/package_stopcontroller_trusted_surprise_hpc.py
tests/stopcontroller_surprise_helpers.py
tests/test_surprise_masks.py
tests/test_trusted_surprise_denominator.py
tests/test_stopcontroller_surprise_mode.py
tests/test_hardrisk_frontier_status.py
tests/test_rerun_independent_of_prediction.py
tests/test_backward_compatibility_all_selected.py
tests/test_stopcontroller_replay.py
docs/TRUSTED_SURPRISE_STOPCONTROLLER_PLAN.md
docs/TRUSTED_SURPRISE_STOPCONTROLLER_DECISION_LOG.md
docs/DECISIONS.md
docs/report_qa/20260618_trusted_surprise_stopcontroller.md
reports/stopcontroller_surprise_replay/
hpc_packages/stopcontroller_all_selected_baseline_snapshot*
hpc_packages/stopcontroller_trusted_surprise_v1*
```

Summary:

```text
Implemented explicit StopController surprise layers.  The default
`all_selected` mode preserves historical behavior.  The opt-in `trusted` mode
uses only trusted, training-eligible, non-rerun, q-resolved, and delta-resolved
exact labels for C4.  q-expanded points are still counted when otherwise
trusted.  Hard-risk surprise is recorded as a numerical-frontier diagnostic.
```

Important decisions:

```text
label_surprise_rate remains equal to label_surprise_all_selected
stop_surprise_mode defaults to all_selected
trusted_surprise_min_denominator defaults to 64
trusted_surprise_min_fraction defaults to 0.25
hard-risk active does not by itself block main_phase_converged
publication_ready remains false until the hard-risk frontier is closed
```

Validation:

```text
python -m py_compile ml_phase/stop_controller.py ml_phase/report_builder.py scripts/dev_check_stop_controller.py scripts/replay_stopcontroller_surprise_modes.py scripts/package_stopcontroller_trusted_surprise_hpc.py tests/stopcontroller_surprise_helpers.py tests/test_surprise_masks.py tests/test_trusted_surprise_denominator.py tests/test_stopcontroller_surprise_mode.py tests/test_hardrisk_frontier_status.py tests/test_rerun_independent_of_prediction.py tests/test_backward_compatibility_all_selected.py tests/test_stopcontroller_replay.py
python scripts/dev_check_stop_controller.py
python -m pytest tests/test_surprise_masks.py tests/test_trusted_surprise_denominator.py tests/test_stopcontroller_surprise_mode.py tests/test_hardrisk_frontier_status.py tests/test_rerun_independent_of_prediction.py tests/test_backward_compatibility_all_selected.py tests/test_stopcontroller_replay.py
python scripts/replay_stopcontroller_surprise_modes.py
python scripts/package_stopcontroller_trusted_surprise_hpc.py
```

Replay evidence:

```text
all_selected reconstruction rows = 31
all_selected discrepancy_check = all pass
all_selected final surprise = 47/256 = 0.18359375
trusted final surprise = 0/137 = 0.0
trusted earliest counterfactual stop iteration = 17
trusted remaining final blocker = boundary_coverage_p95
```

HPC packages:

```text
hpc_packages/stopcontroller_all_selected_baseline_snapshot.tar.gz
sha256 = c90e1188f2c20d2a6d9ed40b3143c04ce49d3855a3877aa290874a693c3c5baa
size = 7333109 bytes

hpc_packages/stopcontroller_trusted_surprise_v1.tar.gz
sha256 = b9098d92e1ad3e7d9ab83cb4f27b20de9aeeccf45680d53ef9dbb4ef22f3e090
size = 7332868 bytes
```

Current project state:

```text
The StopController implementation is ready for trusted-surprise short
validation on HPC.  The package excludes gpuh01 by default and carries LF bash
scripts.  It does not include historical ml_phase active_runs, datasets,
figures, or reports.  The next empirical blocker is the short validation run;
it has not yet been executed on the cluster in this local session.
```

## 2026-06-19 Rankcap K3 Tail Surprise Continuation Package

Files changed:

```text
scripts/package_tail_surprise_continuation_hpc.py
docs/DECISIONS.md
docs/PROJECT_SUMMARY.md
docs/report_qa/20260619_tail_surprise_continuation_package.md
hpc_packages/rankcap_k3_tail_surprise_continuation_v1/
hpc_packages/rankcap_k3_tail_surprise_continuation_v1.tar.gz
hpc_packages/rankcap_k3_tail_surprise_continuation_v1.tar.gz.sha256
hpc_packages/rankcap_k3_tail_surprise_continuation_v1.tar.gz.metadata.json
```

Summary:

```text
Built an independent HPC continuation package that starts from the downloaded
rankcap_k3 full-loop endpoint `dataset_iter031.npz` and runs additional
trusted-surprise continuation batches under a new package-local output root.
The package includes the restart dataset, tail datasets `dataset_iter026-031`,
tail iteration monitor/stop/exact/selected artifacts for `iter026-030`,
metrics and stop history, run config, and a runnable code snapshot.
```

Important decisions:

```text
The package defaults to START_ITER=31, N_ITERS=5, STOP_SURPRISE_MODE=trusted,
trusted_surprise_min_denominator=64, trusted_surprise_min_fraction=0.25,
rankcap_k3 local refinement, and EXCLUDE_NODES=gpuh01.  Large tail
candidate-score full tables are excluded because they are regenerated by new
iterations and are not needed for restart or StopController continuity.
```

Validation:

```text
python -m py_compile scripts/package_tail_surprise_continuation_hpc.py
python scripts/package_tail_surprise_continuation_hpc.py --n-iters 5 --force
custom package check: restart dataset, previous monitor, metrics/stop history,
trusted mode, START_ITER=31, rankcap flags, gpuh01 exclusion, LF/no-BOM shell
scripts, archive contents, and archive sha256 all passed.
```

HPC package:

```text
hpc_packages/rankcap_k3_tail_surprise_continuation_v1.tar.gz
sha256 = 4b1b063684983a81e3e5316ee0c607f2418508c39e056122f91ac9ab71e854b2
size = 71699040 bytes
```

Current project state:

```text
The next empirical test should upload
`rankcap_k3_tail_surprise_continuation_v1.tar.gz` and run
`scripts/submit_rankcap_k3_tail_surprise_continuation.sh`.  The returned
outputs should determine whether late-stage trusted surprise stays below the
formal tolerance, whether all-selected surprise decreases, and whether
boundary coverage remains the main blocker.
```

## 2026-06-19 Rankcap K3 Tail Surprise Continuation Return

Files changed:

```text
scripts/build_tail_surprise_continuation_return_report.py
reports/rankcap_k3_tail_surprise_continuation_return/
docs/PROJECT_SUMMARY.md
docs/report_qa/20260619_tail_surprise_continuation_return.md
```

Summary:

```text
Checked the downloaded
`rankcap_k3_tail_surprise_continuation_results/` return artifacts and generated
a synchronized Markdown/PDF/CSV/PNG report under
`reports/rankcap_k3_tail_surprise_continuation_return/`.
```

Key evidence:

```text
final evaluated iteration = iter034
final dataset = dataset_iter035
exact_call_count = 7434
stop = true
stop_reason = converged_main_phase_boundaries
convergence_pass = true
passed_condition_count = 5/5
patience_counter = 4/4
phase_map_change = 0.0016287 < 0.002
boundary_shift_normal_sc = 0.0041667 <= 0.0041667
boundary_shift_uniform_fflo = 0
boundary_coverage_p95 = 0.0046875 < 0.00625
trusted surprise = 0/127 = 0
all-selected surprise = 75/256 = 0.29296875
hard-risk surprise = 75/129 = 0.581395
rerun_required_count = 110
publication_ready = false
publication_ready_reason = hard_risk_boundary_impact_not_audited
```

Current project state:

```text
The main rankcap_k3 phase-map/boundary active-learning loop should not be
rerun simply to satisfy the historical all-selected surprise metric.  The
supported result is main-boundary convergence under the trusted-surprise gate,
with a still-active hard-risk numerical frontier.  The next calculation should
be a targeted hard-risk boundary-impact audit/cleanup before publication-grade
claims, while the broader project can move into the next major physics/report
stage using the converged trusted-gate phase map.
```

## 2026-06-20 Hard-Risk Boundary-Impact Audit

Files changed:

```text
scripts/build_hard_risk_boundary_impact_audit.py
reports/hard_risk_boundary_impact_audit/
docs/PROJECT_SUMMARY.md
docs/report_qa/20260620_hard_risk_boundary_impact_audit.md
```

Summary:

```text
Generated a report-only hard-risk boundary-impact audit from the downloaded
rankcap_k3 tail-continuation artifacts.  The audit uses final dataset
dataset_iter035, iter034 exact/rerun/selected/monitor/stop artifacts, and
offline copied dense-grid label flips only.  It does not modify acquisition,
the exact oracle, StopController, phase criteria, tolerances, datasets, or
Slurm state.
```

Key evidence:

```text
hard-risk total = 129
boundary-near hard-risk points = 88
deep/far interior hard-risk points = 17
potentially boundary-moving points = 0
single-point exceeds tolerance = 0/88
local cluster exceeds tolerance = 0/37
local single/cluster worst normal/SC p95 shift = 0
local single/cluster worst uniform/FFLO p95 shift = 0
strict global-stress worst normal/SC p95 shift = 0.002604 < 0.004167
strict global-stress worst uniform/FFLO p95 shift = 0.663439 > 0.004167
Decision = A
need_new_exact_calculation = no
targeted_rerun_point_count = 0
```

Important interpretation:

```text
The synchronized all-boundary-near SC/FFLO stress test can create distant
uniform/FFLO islands and exceed the uniform/FFLO boundary-shift tolerance, but
no single hard-risk point and no continuous local hard-risk cluster moves the
main boundaries beyond tolerance.  The main phase map can therefore be used as
publication-ready with explicit hard-risk uncertainty markers/bands; provisional
hard-risk labels should still not be promoted to definitive labels.
```

Current project state:

```text
No full-loop rerun or targeted cleanup exact calculation is required by this
audit.  The next useful work is to integrate the hard-risk uncertainty marker
layer into final figures/report text and proceed to the next major physics or
report stage, while preserving the hard-risk numerical frontier as a disclosed
caveat.
```

## 2026-06-20 Phase-II Final Audit and Report

Files changed:

```text
scripts/build_phase2_final_audit_and_report.py
reports/hard_risk_boundary_impact_audit_v2/
report_phase2_robust_al_final_202606/
docs/PHASE2_FINAL_AUDIT_PLAN.md
docs/PHASE2_FINAL_AUDIT_DECISION_LOG.md
docs/PHASE2_FINAL_REPORT_STATUS.md
docs/PROJECT_SUMMARY.md
docs/report_qa/20260620_phase2_final_audit_and_report.md
```

Summary:

```text
Completed the publication-grade hard-risk boundary-impact audit v2 and, after
the audit passed, generated the self-contained Phase-II robust active-learning
convergence and optimization report.  The report synchronizes Markdown, PDF,
CSV tables, PNG figures, a decision log, and a reproduction manifest.
```

Key evidence:

```text
publication_boundary_audit = pass
audit_decision = Decision A
need_new_exact_calculation = False
targeted_rerun_count = 0
hard_risk_total = 129
rerun_required_count = 110
non_rerun_hard_risk = 19
boundary_near_hard_risk_count = 88
local single/cluster p95 shift = 0
strict local Hausdorff diagnostic = 0.8125
significant local Hausdorff gate value = 0
meaningful topology change = False
final_report_status = complete
frozen dataset = dataset_iter035
dataset total = 7434
normal/uniform_SC/FFLO = 1867/715/4852
```

Important interpretation:

```text
The strict Hausdorff diagnostic captures isolated hard-risk uncertainty-marker
fragments on the dense monitor grid.  Under the publication gate, those
outliers are not treated as main-boundary motion unless they affect a
significant boundary arc or change significant main-boundary topology.  The
Phase-II main thermodynamic phase map is therefore publication-ready with a
documented hard-risk uncertainty layer, but the numerical frontier remains
active and should not be claimed as fully resolved.
```

Current project state:

```text
Do not launch another full active-learning loop for the Phase-II main phase
map.  Freeze dataset_iter035, rankcap_k3 production oracle settings, and the
trusted-surprise StopController configuration for this result.  Recommended
next physics stages are branch-resolved topology classification,
hidden-ground-truth evaluation, multi-seed benchmarking, and final publication
figure polishing.
```

## 2026-06-20 Phase-II LaTeX Report Build

Files changed:

```text
scripts/build_phase2_latex_reports.py
reports/hard_risk_boundary_impact_audit_v2/hard_risk_boundary_impact_audit_v2.tex
reports/hard_risk_boundary_impact_audit_v2/hard_risk_boundary_impact_audit_v2.pdf
report_phase2_robust_al_final_202606/phase2_robust_al_final_report.tex
report_phase2_robust_al_final_202606/phase2_robust_al_final_report.pdf
docs/PHASE2_FINAL_REPORT_STATUS.md
docs/report_qa/20260620_phase2_latex_report_build.md
```

Summary:

```text
Added a report-only LaTeX builder for the Phase-II final report and the
hard-risk boundary-impact audit.  The builder consumes existing CSV, JSON,
Markdown, and PNG artifacts, writes .tex files, and compiles the PDFs with
pdflatex.
```

Validation:

```text
python -m py_compile scripts/build_phase2_latex_reports.py
python scripts/build_phase2_latex_reports.py
pdflatex hard_risk_boundary_impact_audit_v2.tex
pdflatex phase2_robust_al_final_report.tex
pdftoppm spot-render checks
audit PDF pages = 6
Phase-II final report PDF pages = 11
```

Current project state:

```text
The human-facing Phase-II reports are now available as LaTeX-compiled PDFs
with retained Markdown, CSV, PNG, decision-log, and reproduction-manifest
companions.  This update did not change numerical data, production exact
oracle code, acquisition logic, StopController logic, physical criteria, or
tolerances.
```

## 2026-06-20 Phase-II Report Consolidation and Rewrite

Files changed:

```text
project_history/reports/report_phase2_robust_al_final_202606/
project_history/reports/_supporting_reports/
reports/_phase2_supporting_reports/
scripts/build_phase2_latex_reports.py
docs/PHASE2_FINAL_REPORT_STATUS.md
docs/report_qa/20260620_phase2_report_consolidation.md
```

Summary:

```text
Consolidated the Phase-II report sprawl by moving small validation and audit
reports into secondary supporting-report folders.  The first-level
project_history/reports view now contains the first-stage active-learning note,
the rewritten Phase-II final report, and the supporting-report container.
The first-level reports/ view now contains only _phase2_supporting_reports.
```

Rewritten Phase-II report:

```text
The Phase-II final report was rebuilt as a discussion-ready narrative from
random exact seed through full active learning and tail convergence.  It now
includes a random-seed active-learning loop diagram, acquisition/oracle
responsibility split, optimization timeline, validation ladder, final phase
maps, learning curves, runtime/local-box results, hard-risk uncertainty layer,
and supporting-report index.
```

Validation:

```text
python -m py_compile scripts/build_phase2_latex_reports.py
python scripts/build_phase2_latex_reports.py
pdflatex phase2_robust_al_final_report.tex
pdflatex hard_risk_boundary_impact_audit_v2.tex
pdftoppm spot-render checks
Phase-II final report PDF pages = 14
LaTeX fatal/error/overfull scan = clean
```

Current project state:

```text
The report tree is organized for discussion: first-stage and second-stage main
reports are explicit entry points, while smaller Phase-II reports are retained
as supporting evidence.  No numerical data, production exact oracle,
acquisition logic, StopController logic, physical criterion, or tolerance was
changed.
```

## 2026-06-20 Stage III Offline Topology Classification Pilot and Full Pass

Files changed:

```text
ml_phase/topology_oracle.py
scripts/run_topology_pass_dataset_iter035.py
reports/topology_pass_dataset_iter035_v1/
docs/report_qa/20260620_stageiii_topology_pass_dataset_iter035.md
```

Summary:

```text
Started Stage III on frozen dataset_iter035 with run_id
topology_pass_dataset_iter035_v1.  Implemented a report-only topology oracle
that reuses the project BdG Hamiltonian builder for bulk-gap calculations and
cross-checks the analytic Pfaffian convention against the numerical
Majorana-basis Pfaffian.  The thermodynamic labels and original dataset were
not modified.  The validation report is now maintained as both Markdown and
LaTeX source, with the PDF compiled from
reports/topology_pass_dataset_iter035_v1/topology_validation_report.tex.
```

Important implementation decisions:

```text
The current project convention gives P0 as the (mu - t cos(q/2)) branch and
Ppi as the (mu + t cos(q/2)) branch; this was verified against the existing
BdG Hamiltonian builder rather than inferred from the formula text.  Bulk
gaps are computed over the full Brillouin zone using float64/complex128
batched eigvalsh.  The 4090 GPU backend was selected by measured pilot
runtime after CPU/GPU agreement at Nk=2048.
```

Validation:

```text
python -m py_compile ml_phase/topology_oracle.py scripts/run_topology_pass_dataset_iter035.py
python scripts/run_topology_pass_dataset_iter035.py --mode pilot
python scripts/run_topology_pass_dataset_iter035.py --mode full --backend auto
pdflatex topology_validation_report.tex
pdflatex topology_validation_report.tex

Pfaffian analytic/numeric product-sign agreement: 104/104 non-boundary
validation cases.  CPU/GPU bulk-gap agreement at Nk=2048: max absolute gap
difference 7.77e-16.  Full pass processed all 5567 SC points with no failed
points.  The LaTeX PDF compiles successfully to an 8-page report with no fatal
LaTeX errors or overfull boxes in the log.
```

Current Stage III result:

```text
uniform-SC: 715 trivial, 0 topological, 0 gapless, 0 unresolved.
FFLO: 3127 trivial, 1515 topological, 195 gapless_SC, 15 topology_unresolved.
Trusted topology points: 5357/5567 SC points.  Delaunay diagnostics on trusted
FFLO gapped points found 182 candidate Z2-change edges, all through P0 sign
change, and 447 large-circumradius coverage-hole triangles.  The recommended
next decision case is Case B: topology-aware acquisition/refinement should
target Pfaffian-margin, bulk-gap, Z2-change edges, and coverage holes; the
current sparse scatter must not be treated as a final topological boundary.
```

Known unresolved issues:

```text
Parquet output was not written because neither pyarrow nor fastparquet is
installed locally; CSV and NPZ outputs were written instead.  The 15
topology_unresolved points are numerical-reliability states, not physical
gapless labels.  The Delaunay candidate edges are topology-boundary seeds only
and need topology-aware follow-up before any final topo/trivial contour claim.
```

Next recommended steps:

```text
Build a topology-aware acquisition/refinement pilot from the Stage III
diagnostic tables, prioritizing small Pfaffian margin, small bulk gap,
candidate Z2-change edges, and coverage-hole regions.  Do not restart the
thermodynamic active-learning loop for this step.
```

## 2026-06-20 Stage III Topology/Trivial Full-Loop HPC Package

Files changed:

```text
ml_phase/active_refine.py
ml_phase/config.py
ml_phase/acquisition.py
ml_phase/dataset_builder.py
ml_phase/exact_oracle.py
hpc_active_loop.sh
scripts/slurm_exact_oracle_array.sh
scripts/package_topo_trivial_full_loop_hpc.py
hpc_packages/active_phase_topology_from_scratch_full_loop_v1_hpc/
hpc_packages/active_phase_topology_from_scratch_full_loop_v1_hpc.tar.gz
reports/topo_trivial_full_loop_package/
docs/report_qa/20260620_topo_trivial_full_loop_hpc_package.md
```

Summary:

```text
Built a self-contained Stage III topology-aware topology/trivial full-loop HPC
package following docs/Topo_Trivial_FullLoop_Build.md.  The package uses run_id
active_phase_topology_from_scratch_full_loop_v1 and output root
ML_Phase_512_TopoTrivial_FullLoop.  It includes the source snapshot, topology
oracle, topology_pass_dataset_iter035_v1 reference outputs, selected docs,
submit/status/collect/preflight scripts, manifest, checksum, and validation
tables.  The package archive is
hpc_packages/active_phase_topology_from_scratch_full_loop_v1_hpc.tar.gz.
```

Important implementation decisions:

```text
Discovery initialization now supports initialization=sobol_scrambled using a
scrambled 2D SobolEngine over the full (kBT, JA) domain.  The older
random_grid initialization remains unchanged.  Exact oracle outputs can now
carry active-loop topology diagnostics for superconducting points:
Pfaffian P0/Ppi/product/margin, bulk gap, gap momentum, topology label code,
Z2 code, spectral status, and topology trust flag.  Dataset append/load
preserves these fields.  The new acquisition_profile=topo_trivial combines
thermodynamic A_phase with A_spectral from Pfaffian-margin and trusted
trivial/topological/gapless-neighbor geometry plus A_coverage from topology
fill distance.  The Stage III full-loop submit script remains guarded by
CONFIRM_TOPO_FULL_LOOP=1, uses ACQUISITION_PROFILE=topo_trivial, enables
--enable-topology-classification in exact shards, and defaults to excluding
gpuh01.
```

Validation:

```text
python -m py_compile ml_phase/topology_oracle.py ml_phase/exact_oracle.py ml_phase/acquisition.py ml_phase/active_refine.py ml_phase/config.py scripts/package_topo_trivial_full_loop_hpc.py
topo_trivial acquisition smoke test with synthetic topology context
python -m ml_phase.active_refine --run-mode discovery --candidate-domain-mode full --initialization sobol_scrambled --initial-seed-size 8 --run-id tmp_sobol_preflight --iterations 1 --dry-run --mode local --output-root .tmp_topo_sobol_check
python scripts/package_topo_trivial_full_loop_hpc.py
python scripts/preflight_topo_trivial_full_loop.py  # run inside package root

Package validation status is pass.  The package-local preflight reproduces the
Stage III v1 reference topology counts:
uniform_SC trivial=715; FFLO trivial=3127; FFLO topological=1515;
FFLO gapless_SC=195; FFLO topology_unresolved=15.  Shell scripts are ASCII/LF
and default to excluding gpuh01.  Local bash syntax checking is recorded as a
warning because the Windows bash executable is only the WSL launcher and no WSL
distribution is installed.
```

Known unresolved issues:

```text
Active-loop topology labels are acquisition diagnostics, not the final
publication-grade topology pass.  After the full loop, rerun the offline
topology pass/audit on the final frozen dataset before claiming final
topo/trivial/nodal boundaries.  The working tree also contains many unrelated
historical changes/deletions that were not part of this package task.
```

Next recommended steps:

```text
Upload hpc_packages/active_phase_topology_from_scratch_full_loop_v1_hpc.tar.gz
to the cluster, extract it, run package-local preflight, then submit with
CONFIRM_TOPO_FULL_LOOP=1.  Monitor topology_context counts in iter metrics and
topology classification counts in rank JSON snapshots.  After the run returns,
perform a publication-grade offline topology pass on the final dataset.
```

## 2026-06-23 Stage III Topology-Aware Full-Loop Return Report

Files changed/generated:

```text
scripts/build_topo_trivial_full_loop_return_report.py
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/active_learning_phase_boundary_report.tex
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/active_learning_phase_boundary_report.pdf
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topo_trivial_full_loop_summary.md
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/decision_log.md
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/figures/
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/figures/eta_topology_phase_map.png
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/figures/selected_boundary_concentration.png
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/runtime_summary.csv
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/response_region_diagnostics.csv
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/selected_boundary_concentration_summary.csv
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/tables/selected_boundary_concentration_by_iteration.csv
docs/report_qa/20260623_topo_trivial_full_loop_return.md
```

Summary:

```text
Checked the returned Stage III topology-aware active-learning full-loop output
under ML_Phase_512_TopoTrivial_FullLoop.  The original run_id v1 failed in its
first exact array because one gpuh14 shard hit CUDA devices unavailable; the
completed returned run is active_phase_topology_from_scratch_full_loop_v2.
The existing LaTeX report active_learning_phase_boundary_report.tex was
rewritten as a compact report-only summary with final phase/topology maps,
dataset growth, topology-count curves, stop metrics, runtime/local-box curves,
gap/Pfaffian diagnostics, selected-point maps, and learning curves.  A
red-blue eta map was added to match the original phase-map style: color is
eta sign/magnitude, solid black is normal/SC, dashed gray is uniform-SC/FFLO,
and purple short edges mark local online Z2 changes between cFFLO and tFFLO
samples.  A response-side rough-region diagnostic section was added for the
low-temperature edge and the diagonal high-J band, together with normalized
selected-point distance metrics showing that the topology-aware acquisition is
strongly concentrated near the final diagnostic normal/SC and cFFLO/tFFLO
boundary segments.
```

Key returned result:

```text
completed active-learning iterations: 18
final dataset: dataset_iter018.npz
final samples: 4345
phase counts: normal=746, uniform_SC=386, FFLO=3213
online topology counts: trivial=2676, topological=923, gapless_SC=0, unresolved=0, not_applicable=746
topology trusted count: 3599
stop reason: converged_main_phase_boundaries
phase_map_change: 0.001745 < 0.002
normal/SC boundary shift: 0.004167 <= 0.004167
uniform/FFLO boundary shift: 0
trusted surprise: 0/200
all-selected surprise: 0.066406
hard-risk surprise: 0.303571
boundary_coverage_p95: 0.006988 > 0.00625
estimated exact-array walltime: 20.10 h total
mean exact-array walltime: 67.0 min/iteration
max exact-array walltime: 115.9 min
rank-summed point runtime: 153.96 h
rank-summed local-refinement runtime: 121.51 h
online topology diagnostic runtime: 16.32 s
low-T edge diagnostic: 42 FFLO/topological points; eta has 33 negative and 9 positive signs; p95(|eta|)=0.7377
diagonal high-J band diagnostic: 74 points within raw distance 0.03; FFLO=39, normal=35; q_expanded=74/74; p95(|eta|)=0.01714
selection concentration all acquisition: normal/SC within 0.03 = 0.4065; cFFLO/tFFLO within 0.03 = 0.3408; either boundary within 0.03 = 0.5949
selection concentration last five acquisition batches: normal/SC within 0.03 = 0.4031; cFFLO/tFFLO within 0.03 = 0.3578; either boundary within 0.03 = 0.6016
selection concentration reference: initial seed either-boundary within 0.03 = 0.1289
```

Validation:

```text
python -m py_compile scripts/build_topo_trivial_full_loop_return_report.py
python scripts/build_topo_trivial_full_loop_return_report.py
pdflatex active_learning_phase_boundary_report.tex
pdflatex active_learning_phase_boundary_report.tex

The report PDF compiled to eight pages with no LaTeX error, missing figure,
overfull box, or stale-reference warning in the final log.  Rendered PDF pages
were visually checked for readable summary tables, phase/topology maps, dataset
growth plots, selected-boundary concentration plots, rough-region text, and
final caveats.
```

Current interpretation:

```text
The returned v2 run is a successful main-boundary convergence run with online
topology diagnostics.  It is not a continuation of dataset_iter035 and should
not be merged with the Phase-II frozen dataset without an explicit provenance
table.  The online topology labels are useful acquisition/runtime diagnostics,
but they are not a publication-grade topology boundary.  The next required
Stage III step is an offline topology pass/audit on frozen dataset_iter018.
```

### 2026-06-23: Offline cFFLO/tFFLO topology-boundary convergence audit for cold-start v2

Files changed/generated:

```text
scripts/build_topology_convergence_audit.py
active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop/reports/topology_convergence_audit_full_loop_v2/
docs/report_qa/20260623_topology_convergence_audit_full_loop_v2.md
```

Summary:

```text
Built and ran a report-only convergence audit for source run
active_phase_topology_from_scratch_full_loop_v2, final dataset_iter018.  The
audit uses the cumulative dataset_iter000..dataset_iter018 history and applies
one unified deterministic KNN inverse-distance interpolation pipeline to P0
and Ppi from each iteration's trusted gapped FFLO points.  It does not start
new exact calculations, does not run Delta-q search, does not continue active
learning, and does not modify historical datasets.
```

Key result:

```text
audit run_id: topology_convergence_audit_full_loop_v2
final iteration: 18
formal decision: Decision A
topology_main_converged: true
need_new_exact_calculation: false
last3 topology_map_change: 0.000471, 0.000908, 0.000488  (< 0.002)
last3 topology_boundary_shift_p95: 0.000815, 0.002625, 0.000871  (<= 0.004167)
final topology_boundary_coverage_p95: 0.006185  (< 0.00625)
last3 trusted_topology_surprise: 0.0119, 0.0171, 0.0126  (<= 0.02)
significant cFFLO/tFFLO boundary components last3: 1, 1, 1
topological-region components last3: 1, 1, 1
final direct/bracket support fraction: 1.0
```

Sensitivity caveat:

```text
The main audit uses k=8 and passes the coverage gate.  The lightweight kNN/IDW
sensitivity check keeps the same map/component topology, but k=6 gives final
coverage p95 = 0.006269, slightly above the 0.00625 threshold; k=8 gives
0.006185 and k=12 gives 0.005913.  This is recorded as a tight coverage-margin
caveat, not a map-change or boundary-shift failure.
```

Validation:

```text
python -m py_compile scripts/build_topology_convergence_audit.py
python scripts/build_topology_convergence_audit.py
pdflatex topology_convergence_audit_report.tex
pdftoppm PDF render check for pages 1, 2, and 7

Re-run on 2026-06-23:
python -m py_compile scripts/build_topology_convergence_audit.py
python scripts/build_topology_convergence_audit.py
pdflatex -interaction=nonstopmode topology_convergence_audit_report.tex
Result: Decision A JSON reproduced; PDF compiled successfully to 8 pages.
```

Current interpretation:

```text
The cold-start topology-aware full loop supports a formally converged main
cFFLO/tFFLO topology boundary under the configured offline audit.  No new exact
calculation is required for the main topology-boundary conclusion.  The next
work should freeze this topology-boundary result for offline reporting and, if
desired, later run a targeted spectral/coverage tail only for a stricter
coverage margin, not restart the full loop.
```

### 2026-06-23: Stage IV 3D topology-aware cold-start HPC package

Files changed/generated:

```text
ml_phase/stageiv_3d.py
ml_phase/topology_oracle.py
ml_phase/exact_oracle.py
ml_phase/hpc.py
ml_phase/dataset_builder.py
ml_phase/active_refine.py
scripts/stageiv_3d_select.py
scripts/stageiv_3d_preflight.py
scripts/package_stageiv_topology_3d_hpc.py
reports/stageiv_3d_preflight_local/
reports/stageiv_3d_selector_smoke_output/
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz.sha256
docs/report_qa/20260623_stageiv_3d_hpc_package.md
```

Summary:

```text
Prepared the Stage IV-A self-contained HPC package for a strict cold-start
3D active-learning run over (kBT/t, J_A/t, mu/t).  The run id is
active_phase_topology_3d_t_ja_mu_from_scratch_v1 and the output root is
ML_Phase_StageIV_Topology3D.  The package does not include Stage III datasets
or checkpoints and uses a 1024-point scrambled Sobol seed followed by 24
acquisition batches of 256 points.
```

Implementation decisions:

```text
The exact thermodynamic oracle is reused rather than rewritten.  Per-point mu
is propagated through selected_points.csv, shard partitioning, exact-oracle
evaluation, exact-output npz files, dataset append, and topology Pfaffian /
bulk-gap diagnostics.  Existing 2D datasets remain readable because the new
mu field defaults to the Stage III reference value 0.55 when absent.

Stage IV selection is implemented as a separate 3D selector script instead of
modifying the existing 2D active_refine loop.  This keeps the 3D cold-start
path explicit while preserving the validated exact oracle, rank-and-cap K3
local refinement, and topology diagnostic code.
```

Validation:

```text
python -m py_compile ml_phase/stageiv_3d.py scripts/stageiv_3d_select.py scripts/stageiv_3d_preflight.py scripts/package_stageiv_topology_3d_hpc.py
python scripts/stageiv_3d_preflight.py --output-dir reports/stageiv_3d_preflight_local --n-ja 8 --n-mu 8 --n-k 128
python scripts/package_stageiv_topology_3d_hpc.py
python scripts/stageiv_3d_select.py --config hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/configs/stageiv_3d_smoke.json --mode seed --iteration 0 --output-root reports/stageiv_3d_selector_smoke_output --run-id stageiv_seed_smoke --world-size 2 --partition-strategy cost_aware
python scripts/stageiv_3d_select.py --config reports/stageiv_3d_acq_smoke_config.json --mode acquisition --iteration 1 --output-root reports/stageiv_3d_acq_smoke_output --run-id stageiv_acq_smoke --dataset reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke/dataset_iter001.npz --world-size 2 --partition-strategy cost_aware --device cpu
```

Package status:

```text
package_validation_status: pass
archive sha256: 1996232501a0d7f9d37d935b0f723a19bfcd622035218162a379e46da5372f0e
archive size: 328,601 bytes
package preflight: pass with production config and lightweight guard scan
Slurm node policy: #SBATCH --exclude=gpuh01 plus runtime hostname guard
shell encoding policy: LF, no BOM, ASCII-safe
local bash -n status: skipped on Windows because WSL has no installed distro;
scripts should be syntax-checked again on the Linux login node after upload
```

Open issues:

```text
The package has not yet been submitted on the cluster.  The Stage IV online
loop currently runs a fixed 1024 seed plus 24 acquisition batches; the
publication-grade 3D convergence audit remains a post-run report-only task.
The selector now mixes global Sobol candidates, topology opposite-Z2 bracket
jitter candidates, thermodynamic opposite-phase bracket jitter candidates, and
mu-edge guard candidates.  It selects with explicit phase/spectral/coverage
channel quotas and records acquisition_channel plus candidate_source metadata.
The stability-triggered acquisition weight switch remains a future improvement;
the current package still uses the configured iteration switch.
If the cluster reports CUDA device busy/unavailable on a rank, the same
diagnostic and rerun workflow used in Stage III should be applied.
```

Next steps:

```text
Upload active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz to the
cluster, extract it as a fresh directory, and submit with
CONFIRM_STAGEIV_FULL_LOOP=1.  After results return, run a report-only 3D
topology/thermodynamic convergence audit before interpreting the final 3D
phase diagram.
```

Stage IV package finalization update:

```text
Added scripts/stageiv_3d_convergence_audit.py and packaged it with
scripts/build_stageiv_3d_convergence_audit.sh.  This is a report-only
post-run audit for returned Stage IV cumulative datasets.  It does not launch
exact calculations, Delta-q searches, or active-learning iterations.

The audit reconstructs cumulative dataset history, computes fixed-cloud proxy
phase/topology map changes, local opposite-label surface shift/coverage,
component proxies, and no-leakage trusted surprise.  It explicitly marks
insufficient history as Decision D instead of treating missing boundaries or
missing transitions as zero shift.

Smoke validation used the local one-iteration Stage IV smoke output.  The
expected result was produced:

stageiv_convergence_status = insufficient_history
decision_class = Decision D
need_new_exact_calculation = false

PDF generation was also checked on the smoke audit and produced
reports/stageiv_3d_convergence_audit_smoke_pdf/stageiv_3d_convergence_audit.pdf.
```

Additional validation:

```text
python -m py_compile scripts/stageiv_3d_convergence_audit.py scripts/package_stageiv_topology_3d_hpc.py
python scripts/stageiv_3d_convergence_audit.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_convergence_audit_smoke --config reports/stageiv_3d_acq_smoke_config.json --audit-cloud-size 512 --no-pdf
python scripts/stageiv_3d_convergence_audit.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_convergence_audit_smoke_pdf --config reports/stageiv_3d_acq_smoke_config.json --audit-cloud-size 512
python scripts/package_stageiv_topology_3d_hpc.py --print-json
```

Updated package status:

```text
package_validation_status: pass
archive sha256: 1996232501a0d7f9d37d935b0f723a19bfcd622035218162a379e46da5372f0e
archive size: 328,601 bytes
archive path: hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
post-run convergence audit script: scripts/build_stageiv_3d_convergence_audit.sh
hidden fixed-mu validation script: scripts/build_stageiv_3d_hidden_slice_audit.sh
all post-run reports script: scripts/build_stageiv_3d_all_postrun_reports.sh
```

Remaining Stage IV blockers:

```text
The production Stage IV HPC run has not yet returned in this workspace.
Therefore final 3D thermodynamic/topological convergence, mu-domain
completeness, hidden fixed-mu slice recovery, and publication-grade 3D phase
surfaces remain pending returned data.
```

### 2026-06-23: Stage IV readiness audit checkpoint

Files generated:

```text
reports/stageiv_3d_readiness_audit/stageiv_3d_readiness_audit.md
reports/stageiv_3d_readiness_audit/stageiv_readiness_decision.json
reports/stageiv_3d_readiness_audit/tables/stageiv_requirement_readiness.csv
```

Summary:

```text
Performed a requirement-by-requirement readiness audit against
docs/StageIV_MultiDim_Phase_Diagram.md and the current HPC package manifest.
The result is:

stageiv_readiness_status = package_ready_hpc_pending
package_validation_status = pass
production_hpc_run_returned = false
stageiv_scientific_convergence_verified = false
```

Why it matters:

```text
This separates package readiness from scientific completion.  The Stage IV
package is ready to upload and submit, but the full Stage IV objective remains
incomplete until returned cumulative datasets prove 3D thermodynamic/topology
convergence, mu-domain completeness, and hidden fixed-mu slice recovery.
```

Next required action:

```text
Upload hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
to the cluster and run scripts/submit_stageiv_3d_full_loop.sh with
CONFIRM_STAGEIV_FULL_LOOP=1.  After returned data are collected, run
scripts/build_stageiv_3d_convergence_audit.sh before interpreting Stage IV
phase surfaces.
```
### 2026-06-23: Stage IV hidden fixed-mu slice audit tooling

Files changed/generated:

```text
scripts/stageiv_3d_hidden_slice_audit.py
scripts/package_stageiv_topology_3d_hpc.py
reports/stageiv_3d_hidden_slice_audit_smoke_missing_ref/
reports/stageiv_3d_hidden_slice_audit_smoke_self_ref/
docs/report_qa/20260623_stageiv_3d_hpc_package.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
```

Summary:

```text
Added a report-only hidden fixed-mu validation tool for Stage IV.  It reads a
returned Stage IV run and an external Stage III frozen reference dataset,
extracts the mu_reference slice using a fixed KNN audit proxy, and writes
machine-readable map, boundary, coverage, and topology-overlap diagnostics.
The Stage III reference is validation-only and is not merged into Stage IV
training data.
```

Validation:

```text
python -m py_compile scripts/stageiv_3d_hidden_slice_audit.py scripts/package_stageiv_topology_3d_hpc.py
python scripts/stageiv_3d_hidden_slice_audit.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_hidden_slice_audit_smoke_missing_ref --config reports/stageiv_3d_acq_smoke_config.json --grid-n 41
python scripts/stageiv_3d_hidden_slice_audit.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_hidden_slice_audit_smoke_self_ref --config reports/stageiv_3d_acq_smoke_config.json --reference-dataset reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke/dataset_iter001.npz --grid-n 41
python scripts/package_stageiv_topology_3d_hpc.py --print-json
```

Results:

```text
Missing-reference smoke:
hidden_slice_status = inconclusive
decision_class = Decision D
reason = reference_dataset_missing
need_new_exact_calculation = false

Self-reference smoke:
the full output chain ran, strict JSON wrote null for unavailable quantities,
and pdflatex produced stageiv_3d_hidden_slice_audit.pdf.

Updated package:
archive sha256 = 1996232501a0d7f9d37d935b0f723a19bfcd622035218162a379e46da5372f0e
package_validation_status = pass
```

Open issue:

```text
The actual Stage III frozen reference dataset path must be supplied after the
production Stage IV data return, for example through
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz when running
scripts/build_stageiv_3d_hidden_slice_audit.sh on the cluster or local return.
```

### 2026-06-23: Stage IV post-run bundle entry point

Files changed/generated:

```text
scripts/stageiv_3d_postrun_bundle.py
scripts/stageiv_3d_postrun_report.py
scripts/package_stageiv_topology_3d_hpc.py
reports/stageiv_3d_postrun_bundle_smoke/
reports/stageiv_3d_readiness_audit/
docs/report_qa/20260623_stageiv_3d_hpc_package.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
```

Summary:

```text
Added a one-command report-only Stage IV post-run bundle.  The wrapper
scripts/build_stageiv_3d_all_postrun_reports.sh runs the lightweight post-run
summary, the 3D convergence audit, and the hidden fixed-mu slice validation,
then writes an aggregate decision JSON and Markdown summary under
ML_Phase_StageIV_Topology3D/reports/stageiv_3d_postrun_bundle/.
```

Validation:

```text
python -m py_compile scripts/stageiv_3d_postrun_bundle.py scripts/stageiv_3d_postrun_report.py scripts/package_stageiv_topology_3d_hpc.py
python scripts/stageiv_3d_postrun_bundle.py --run-dir reports/stageiv_3d_acq_smoke_output/active_runs/stageiv_acq_smoke --output-dir reports/stageiv_3d_postrun_bundle_smoke --config reports/stageiv_3d_acq_smoke_config.json --audit-cloud-size 512 --hidden-grid-n 41 --no-pdf
python scripts/package_stageiv_topology_3d_hpc.py --print-json
```

Result:

```text
Smoke bundle status = incomplete_convergence_history
decision_class = Decision D
convergence_status = insufficient_history
hidden_slice_status = inconclusive
need_new_exact_calculation = false

Updated package:
archive sha256 = 1996232501a0d7f9d37d935b0f723a19bfcd622035218162a379e46da5372f0e
archive size = 328,601 bytes
package_validation_status = pass
```

Open issue:

```text
The production Stage IV full loop still has not returned in this workspace.
The bundle command is now ready for returned data, but final Stage IV
thermodynamic/topology convergence, mu-domain completeness, hidden fixed-mu
slice recovery, and publication-grade surfaces remain pending.
```

### 2026-06-23: Stage IV standard README and environment handoff entries

Files changed/generated:

```text
scripts/package_stageiv_topology_3d_hpc.py
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/README.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/README_STAGEIV_3D_HPC.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/ENVIRONMENT_STAGEIV_3D_HPC.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
reports/stageiv_3d_readiness_audit/
docs/report_qa/20260623_stageiv_3d_hpc_package.md
```

Summary:

```text
Updated the Stage IV package builder so the extracted HPC package contains both
README.md and README_STAGEIV_3D_HPC.md.  Both files contain the same submit,
monitor, collect, resume, and post-run audit instructions.  The package also
contains ENVIRONMENT_STAGEIV_3D_HPC.md, which records the intended NV_H100
runtime, PYTHON_BIN, gpuh01 exclusion, CUDA checks, REFERENCE_DATASET role, and
encoding policy.  The readiness report now states the correct order: collect
the Stage IV result archive after Slurm completion, then run the post-run
bundle from the returned/extracted output.  This is a handoff and
archive-inspection improvement only; it does not change exact-oracle physics,
acquisition, StopController logic, Stage IV configuration, or any numerical
tolerance.
```

Validation:

```text
python -m py_compile scripts/package_stageiv_topology_3d_hpc.py
python -m py_compile scripts/stageiv_3d_postrun_bundle.py scripts/stageiv_3d_postrun_report.py scripts/stageiv_3d_convergence_audit.py scripts/stageiv_3d_hidden_slice_audit.py
python scripts/package_stageiv_topology_3d_hpc.py --print-json
Get-FileHash -Algorithm SHA256 hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
```

Updated package:

```text
archive sha256 = 1996232501a0d7f9d37d935b0f723a19bfcd622035218162a379e46da5372f0e
archive size = 328,601 bytes
package_validation_status = pass
tar listing includes README.md, README_STAGEIV_3D_HPC.md, and ENVIRONMENT_STAGEIV_3D_HPC.md
```

### 2026-06-23: Stage IV read-only HPC status checker

Files changed/generated:

```text
scripts/stageiv_3d_hpc_status.py
scripts/package_stageiv_topology_3d_hpc.py
reports/stageiv_3d_status_checker_smoke/
reports/stageiv_3d_readiness_audit/
docs/report_qa/20260623_stageiv_3d_hpc_package.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
```

Summary:

```text
Added a read-only Stage IV HPC status checker and packaged it as
scripts/check_stageiv_3d_hpc_status.sh.  The checker inspects returned or
in-progress run files: dataset_iter files, iter directories, selected points,
exact shard counts, merged/trusted exact outputs, merge summaries, stop
metrics, rerun point files, optional squeue/sacct output, result archive
presence, and post-run bundle decision presence.
```

Why it matters:

```text
This gives the cluster handoff a deterministic way to distinguish
run_dir_missing, no_datasets_yet, partial_or_running, complete_file_set_detected,
and postrun_bundle_detected without submitting jobs, merging shards, appending
datasets, continuing active learning, or running exact calculations.
```

Validation:

```text
python -m py_compile scripts\stageiv_3d_hpc_status.py scripts\package_stageiv_topology_3d_hpc.py
python scripts\stageiv_3d_hpc_status.py --output-root reports\stageiv_3d_status_checker_smoke_output --output-dir reports\stageiv_3d_status_checker_smoke --config reports\stageiv_3d_acq_smoke_config.json
python scripts\package_stageiv_topology_3d_hpc.py --print-json
```

Result:

```text
status-checker smoke: hpc_status = run_dir_missing
package_validation_status = pass
archive sha256 = 1996232501a0d7f9d37d935b0f723a19bfcd622035218162a379e46da5372f0e
archive size = 328,601 bytes
tar member count = 74
```

Remaining blocker:

```text
The production Stage IV HPC full loop still has not returned in this workspace.
The Stage IV objective remains incomplete until returned cumulative datasets are
audited with the post-run bundle and hidden fixed-mu slice validation.
```

### 2026-06-23: Stage IV goal-status audit checkpoint

Files changed/generated:

```text
scripts/audit_stageiv_3d_goal_status.py
reports/stageiv_3d_goal_status/stageiv_3d_goal_status.md
reports/stageiv_3d_goal_status/stageiv_3d_goal_status.pdf
reports/stageiv_3d_goal_status/stageiv_3d_goal_status.json
reports/stageiv_3d_goal_status/tables/stageiv_goal_requirements.csv
reports/stageiv_3d_goal_status/figures/stageiv_goal_status_counts.png
docs/PROJECT_SUMMARY.md
```

Summary:

```text
Added a current-state completion audit for the persistent Stage IV objective.
The audit reads the Stage IV plan, package manifest, readiness decision, package
archive, and local output namespace, then writes requirement-level evidence for
what is already package-ready versus what still requires returned production HPC
data.
```

Result:

```text
goal_status = package_ready_hpc_pending
package_validation_status = pass
package_sha256 = 1996232501a0d7f9d37d935b0f723a19bfcd622035218162a379e46da5372f0e
production_run_returned = false
final_dataset_exists = false
postrun_bundle_exists = false
status_counts = {pass: 13, pending: 6}
```

Validation:

```text
python -m py_compile scripts\audit_stageiv_3d_goal_status.py
python scripts\audit_stageiv_3d_goal_status.py
pdflatex -interaction=nonstopmode -halt-on-error stageiv_3d_goal_status.tex
pdflatex -interaction=nonstopmode -halt-on-error stageiv_3d_goal_status.tex
```

Interpretation:

```text
The audit confirms the goal cannot be marked complete from local evidence.
Package readiness is proven, but scientific completion remains pending
production Stage IV cumulative datasets, post-run convergence audit, hidden
fixed-mu slice validation, and final Stage IV report generation.
```

### 2026-06-23: Stage IV submit-ready checker refresh

Files changed/generated:

```text
scripts/stageiv_3d_submit_check.py
scripts/package_stageiv_topology_3d_hpc.py
scripts/audit_stageiv_3d_goal_status.py
reports/stageiv_3d_readiness_audit/
reports/stageiv_3d_goal_status/
docs/report_qa/20260623_stageiv_3d_hpc_package.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
```

Summary:

```text
Added a read-only Stage IV submit-ready checker and packaged it as
scripts/check_stageiv_3d_submit_ready.sh.  The checker validates required
package files, production config values, Python imports, py_compile, shell
encoding, Slurm gpuh01 exclusion, nvidia-smi/torch CUDA visibility as a
login-node warning, and existing run-directory collision before sbatch.
```

Result:

```text
package_validation_status = pass
package_submit_check_returncode = 0
archive sha256 = e27d222fc232b00ab21656529607eb741fc69f0db2fb3363d1cb89a78c6e0e76
archive size = 335,104 bytes
tar member count = 81
goal_status = package_ready_hpc_pending
status_counts = {pass: 14, pending: 6}
```

Validation:

```text
python -m py_compile scripts\stageiv_3d_submit_check.py scripts\package_stageiv_topology_3d_hpc.py
python scripts\package_stageiv_topology_3d_hpc.py --print-json
python scripts\stageiv_3d_submit_check.py --root hpc_packages\active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc --config configs\stageiv_3d_production.json --output-dir hpc_packages\active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc\reports\stageiv_3d_submit_check_final --python-bin python
python -m py_compile scripts\audit_stageiv_3d_goal_status.py
python scripts\audit_stageiv_3d_goal_status.py
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
```

Current state:

```text
The Stage IV package is ready for upload and HPC submission, with a submit-ready
check to run after extraction and before sbatch.  The Stage IV scientific goal
is still incomplete because no production cumulative datasets or post-run
bundle are present locally.
```

### 2026-06-23: Stage IV returned-result checker refresh

Files changed/generated:

```text
scripts/stageiv_3d_return_check.py
scripts/package_stageiv_topology_3d_hpc.py
scripts/audit_stageiv_3d_goal_status.py
reports/stageiv_3d_return_check_missing_smoke/
reports/stageiv_3d_return_check_package_tar_smoke/
reports/stageiv_3d_readiness_audit/
reports/stageiv_3d_goal_status/
docs/report_qa/20260623_stageiv_3d_hpc_package.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
```

Summary:

```text
Added a read-only Stage IV returned-result checker and packaged it as
scripts/check_stageiv_3d_return_bundle.sh.  The checker accepts either a
returned/extracted directory or a tar archive, validates whether the expected
final dataset and post-run bundle decision are present, writes machine-readable
JSON/CSV/Markdown diagnostics, and reports the next action before any post-run
interpretation.
```

Validation:

```text
python -m py_compile scripts\stageiv_3d_return_check.py scripts\package_stageiv_topology_3d_hpc.py
python scripts\stageiv_3d_return_check.py --return-path ML_Phase_StageIV_Topology3D --config hpc_packages\active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc\configs\stageiv_3d_production.json --output-dir reports\stageiv_3d_return_check_missing_smoke
python scripts\stageiv_3d_return_check.py --return-path hpc_packages\active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz --config hpc_packages\active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc\configs\stageiv_3d_production.json --output-dir reports\stageiv_3d_return_check_package_tar_smoke
python scripts\package_stageiv_topology_3d_hpc.py --print-json
python -m py_compile scripts\audit_stageiv_3d_goal_status.py
python scripts\audit_stageiv_3d_goal_status.py
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
```

Result:

```text
package_validation_status = pass
package_return_check_smoke_returncode = 0
archive sha256 = a1def9b9a32d03d70e948907472fb0acd55e9a9421787f7662dd6dabfc906796
archive size = 339,069 bytes
tar member count = 88
goal_status = package_ready_hpc_pending
status_counts = {pass: 16, pending: 6}
```

Current state:

```text
The package now has explicit checks for pre-submit readiness, in-progress HPC
status, and returned-result integrity.  The Stage IV production data still have
not returned in this workspace, so final 3D thermodynamic/topology convergence,
hidden fixed-mu slice recovery, and the final Stage IV report remain pending.
```

### 2026-06-23: Stage IV failed-rank recovery handoff

Files changed/generated:

```text
scripts/package_stageiv_topology_3d_hpc.py
scripts/audit_stageiv_3d_goal_status.py
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/scripts/recover_stageiv_failed_exact_iter.sh
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
reports/stageiv_3d_readiness_audit/
reports/stageiv_3d_goal_status/
docs/report_qa/20260623_stageiv_3d_hpc_package.md
```

Summary:

```text
Added a Stage IV HPC recovery entry point for the common case where one exact
Slurm array rank fails while the other shards complete, for example a transient
CUDA device busy/unavailable error.  The generated package command is:

ITER=<failed_iteration> FAILED_RANKS=<rank_list> bash scripts/recover_stageiv_failed_exact_iter.sh

The recovery command resubmits only the listed ranks for the failed exact
iteration, waits for the recovery job, checks that all expected shard files are
present, merges shards, appends trusted exact outputs, and prints the
START_ITER command for continuing the full loop.
```

Validation:

```text
python -m py_compile scripts\package_stageiv_topology_3d_hpc.py scripts\audit_stageiv_3d_goal_status.py
python scripts\package_stageiv_topology_3d_hpc.py --print-json
python scripts\audit_stageiv_3d_goal_status.py
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
```

Result:

```text
package_validation_status = pass
archive sha256 = 3c8cdf4c6448e09da2ec6341f47cb66e229c809fbf88553422bece4610cbcc78
archive size = 340,430 bytes
tar member count = 89
goal_status = package_ready_hpc_pending
status_counts = {pass: 17, pending: 6}
```

Current state:

```text
The Stage IV package now covers submit readiness, in-progress status checks,
single-iteration failed-rank recovery, returned-result integrity checks, and
post-run report-only audits.  The production Stage IV data still have not
returned in this workspace, so the persistent Stage IV objective remains
scientifically incomplete until the HPC run is completed and audited.
```

### 2026-06-23: Stage IV HPC command checklist

Files changed/generated:

```text
scripts/package_stageiv_topology_3d_hpc.py
scripts/audit_stageiv_3d_goal_status.py
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc/HPC_COMMANDS_STAGEIV_3D.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
reports/stageiv_3d_readiness_audit/
reports/stageiv_3d_goal_status/
docs/report_qa/20260623_stageiv_3d_hpc_package.md
```

Summary:

```text
Added an explicit Stage IV HPC operation checklist to the self-contained
package.  The checklist records commands for submit-ready checking, production
submission, monitoring, single-rank recovery, completed-prefix resume,
collection, returned-result checking, and post-run report-only audits.  It also
records do-not-do constraints: do not merge Stage III or Phase-II datasets into
Stage IV training, do not treat preflight success as convergence, do not mark
missing surfaces as zero shift, do not change production physics/tolerance
settings from this package, and do not use gpuh01.
```

Validation:

```text
python -m py_compile scripts\package_stageiv_topology_3d_hpc.py scripts\audit_stageiv_3d_goal_status.py
python scripts\package_stageiv_topology_3d_hpc.py --print-json
Get-FileHash -Algorithm SHA256 hpc_packages\active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
python scripts\audit_stageiv_3d_goal_status.py
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
```

Result:

```text
package_validation_status = pass
archive sha256 = ce18cda0d2347c02f7de76390555aafa53bea018a9091b47518cdb0223ad1244
archive size = 342,472 bytes
tar member count = 90
goal_status = package_ready_hpc_pending
status_counts = {pass: 18, pending: 6}
```

Current state:

```text
The package is ready to upload and run on HPC.  The Stage IV scientific goal
remains incomplete because no production Stage IV run output, final
dataset_iter025, or post-run bundle exists locally.
```

### 2026-06-23: Stage IV requirement traceability audit

Files changed/generated:

```text
scripts/audit_stageiv_3d_traceability.py
reports/stageiv_3d_traceability_audit/
```

Summary:

```text
Generated a report-only traceability audit mapping the Stage IV
Multidimensional Phase Diagram plan to current repository evidence.  The audit
separates package/readiness items from scientific gates that require returned
HPC data, preventing the package-ready state from being mistaken for Stage IV
convergence.
```

Validation:

```text
python -m py_compile scripts\audit_stageiv_3d_traceability.py
python scripts\audit_stageiv_3d_traceability.py
pdflatex -interaction=nonstopmode stageiv_3d_traceability_audit.tex
pdflatex -interaction=nonstopmode stageiv_3d_traceability_audit.tex
```

Result:

```text
traceability_status = package_ready_hpc_pending
status_counts = {pass: 17, pending_external: 7}
package_sha256 = ce18cda0d2347c02f7de76390555aafa53bea018a9091b47518cdb0223ad1244
production_run_returned = false
final_dataset_exists = false
postrun_bundle_exists = false
```

Current state:

```text
The Stage IV package/readiness side is complete enough for HPC submission.
Seven science-facing requirements remain externally pending: returned
production cumulative datasets, final dataset_iter025, last-five cumulative
history, joint 3D convergence, mu-domain completeness, hidden fixed-mu slice
validation, and final Stage IV report/figures.
```

### 2026-06-23: Stage IV HPC handoff report

Files changed/generated:

```text
scripts/build_stageiv_3d_hpc_handoff.py
reports/stageiv_3d_hpc_handoff/
```

Summary:

```text
Generated a concise operational handoff report for the Stage IV HPC package.
The handoff records upload, extraction, SHA256 verification, submit-ready
checking, production submission, monitoring, single-rank recovery, collection,
return checking, and post-run bundle commands in both Markdown and CSV form.
It is intentionally operational and does not alter science definitions or run
production exact calculations.
```

Validation:

```text
python -m py_compile scripts\build_stageiv_3d_hpc_handoff.py
python scripts\build_stageiv_3d_hpc_handoff.py
pdflatex -interaction=nonstopmode stageiv_3d_hpc_handoff.tex
pdflatex -interaction=nonstopmode stageiv_3d_hpc_handoff.tex
```

Result:

```text
handoff_status = ready_for_hpc_submit
package_sha256 = ce18cda0d2347c02f7de76390555aafa53bea018a9091b47518cdb0223ad1244
package_size_bytes = 342472
package_validation_status = pass
production_run_returned = false
final_dataset_exists = false
postrun_bundle_exists = false
```

Current state:

```text
The next required action is external: upload the Stage IV package, verify the
hash on the login node, run the submit-ready checker, and submit the production
HPC full loop.  Local completion remains impossible until returned cumulative
datasets and post-run audit decisions are available.
```

### 2026-06-23: Stage IV goal-status handoff evidence refresh

Files changed/generated:

```text
scripts/audit_stageiv_3d_goal_status.py
reports/stageiv_3d_goal_status/
```

Summary:

```text
Refreshed the persistent Stage IV goal-status audit so that it records the
operational HPC handoff evidence in addition to package/readiness evidence.
The audit now explicitly captures that the handoff report and command table
exist and that handoff_status = ready_for_hpc_submit.
```

Validation:

```text
python -m py_compile scripts\audit_stageiv_3d_goal_status.py
python scripts\audit_stageiv_3d_goal_status.py
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
pdflatex -interaction=nonstopmode stageiv_3d_goal_status.tex
```

Result:

```text
goal_status = package_ready_hpc_pending
handoff_status = ready_for_hpc_submit
package_validation_status = pass
status_counts = {pass: 21, pending: 6}
production_run_returned = false
final_dataset_exists = false
postrun_bundle_exists = false
```

Current state:

```text
All local pre-HPC Stage IV package and handoff checks are now represented in
the goal-status audit.  The scientific Stage IV objective is still incomplete
until the external HPC production run returns dataset_iter025 and the post-run
convergence, hidden-slice, and report audits are executed.
```

### 2026-06-23: Stage IV upload-readiness consistency check

Files changed/generated:

```text
reports/stageiv_3d_hpc_handoff/decision_log.md
```

Summary:

```text
Recorded a final pre-upload consistency check across the Stage IV package
manifest, readiness decision, handoff JSON, goal-status JSON, and package
command checklist.  All checked artifacts point to the same validated package
hash and the same operational state.
```

Validation:

```text
Get-Content hpc_packages\active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc\RUN_MANIFEST.json
Get-Content reports\stageiv_3d_readiness_audit\stageiv_readiness_decision.json
Get-Content reports\stageiv_3d_hpc_handoff\stageiv_3d_hpc_handoff.json
Get-Content reports\stageiv_3d_goal_status\stageiv_3d_goal_status.json
Get-FileHash -Algorithm SHA256 hpc_packages\active_phase_topology_3d_t_ja_mu_from_scratch_v1_hpc.tar.gz
```

Result:

```text
upload_readiness_consistency = pass
package_sha256 = ce18cda0d2347c02f7de76390555aafa53bea018a9091b47518cdb0223ad1244
package_validation_status = pass
handoff_status = ready_for_hpc_submit
goal_status = package_ready_hpc_pending
```

Current state:

```text
No local package/readiness/handoff inconsistency remains.  The next Stage IV
progress requires external HPC submission or returned production data.
```

### 2026-06-23: Stage IV external-dependency audit

Files changed/generated:

```text
scripts/audit_stageiv_3d_external_dependency.py
reports/stageiv_3d_external_dependency_audit/
```

Summary:

```text
Generated a report-only audit that separates local Stage IV package/readiness
evidence from requirements that can only be proven by returned external HPC
production outputs.  The audit does not submit jobs, run exact calculations,
merge shards, append datasets, or modify historical results.
```

Validation:

```text
python -m py_compile scripts\audit_stageiv_3d_external_dependency.py
python scripts\audit_stageiv_3d_external_dependency.py
pdflatex -interaction=nonstopmode stageiv_3d_external_dependency_audit.tex
pdflatex -interaction=nonstopmode stageiv_3d_external_dependency_audit.tex
```

Result:

```text
status_counts = {pass: 4, pending_external: 7}
remaining_external_dependency_count = 7
blocking_condition = production_hpc_outputs_not_returned
can_complete_stageiv_without_external_hpc_return = false
production_run_returned = false
final_dataset_exists = false
postrun_bundle_exists = false
```

Current state:

```text
Stage IV is locally ready for HPC execution, but the full objective cannot be
completed from the current workspace.  The remaining requirements are external:
returned production run directory, dataset_iter025, last-five cumulative
datasets, convergence audit, hidden fixed-mu validation, post-run bundle, and
the final Stage IV report built from returned production data.
```

### 2026-06-23: GBU HPC usage summary

Files changed/generated:

```text
docs/gbu_hpc_usage_summary.md
```

Summary:

```text
Added a reusable operational runbook summarizing how this project uses the GBU
Slurm cluster.  The note covers self-contained package workflow, SHA256
verification, Linux shell encoding, Python environment checks, Slurm submission,
monitoring, gpuh01/gpuh14 exclusion, common failure modes, failed-rank recovery,
merge/append commands, collection, returned-result checks, and the distinction
between package readiness and scientific completion.
```

Why it matters:

```text
The project repeatedly runs long exact-oracle and active-learning jobs on the
GBU cluster.  Capturing the operational conventions reduces repeated mistakes
such as partial tar extraction, broken PYTHON_BIN paths, CRLF shell scripts,
split nohup redirections, unexcluded bad GPU nodes, and unnecessary full-loop
restarts after a single transient Slurm array rank failure.
```

Current state:

```text
The summary is operational documentation only.  It does not change physics
definitions, numerical tolerances, acquisition logic, exact-oracle behavior, or
StopController criteria.
```

### 2026-06-24: Stage IV 3D HPC package rebuilt with frozen run identity

Files changed/generated:

```text
scripts/package_stageiv_topology_3d_hpc.py
docs/gbu_hpc_usage_summary.md
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624.tar.gz
hpc_packages/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624.tar.gz.sha256
```

Summary:

```text
Rebuilt the Stage IV 3D cold-start HPC package after an HPC submission showed
that stale shell identity variables could redirect a Stage IV job into the older
topology full-loop namespace.  The package generator now injects identity guards
into generated shell scripts: conflicting OUTPUT_ROOT, RUN_ID, or CONFIG_JSON
values are hard errors rather than silent overrides.  The package-local
PROJECT_SUMMARY.md is now Stage IV specific, avoiding accidental inclusion of
long historical 2D run records in the upload bundle.
```

Validation:

```text
python -m py_compile scripts\package_stageiv_topology_3d_hpc.py
python scripts\package_stageiv_topology_3d_hpc.py --print-json
Select-String generated package scripts for stale OUTPUT_ROOT/RUN_ID defaults
Select-String generated package files for stale 2D markers
```

Result:

```text
package_validation_status = pass
package_name = active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624
package_sha256 = 4799ecfca7ab16c8a340d731dfb74cf0ea44502b2c4582b59523e65df00fcbf4
stale_2d_marker_scan = []
expected_output_root = ML_Phase_StageIV_Topology3D
expected_run_id = active_phase_topology_3d_t_ja_mu_from_scratch_v1
gpuh01/gpuh14 exclusion = Slurm exclude plus runtime hostname guard
```

Current state:

```text
The mistaken older-namespace HPC result is preserved, but the rebuilt Stage IV
3D package is now independent and protected against stale 2D environment
variables.  Scientific completion still requires uploading/running the rebuilt
package on GBU and returning the production Stage IV outputs.
```

### 2026-06-25: Stage IV GBU node exclusion extended to gpuh14

Files changed/generated:

```text
scripts/package_stageiv_topology_3d_hpc.py
scripts/recover_stageiv_iter000_rank0_and_resume.sh
docs/gbu_hpc_usage_summary.md
docs/report_qa/20260625_stageiv_gpuh14_exclusion_and_rank0_recovery.md
```

Summary:

```text
Recorded gpuh14 as a project-excluded GBU GPU node in addition to gpuh01.
Stage IV job 82381 showed rank 0 failing after 15 seconds on gpuh14 with
CUDA-capable device busy/unavailable behavior, while ranks 1-7 completed
normally on other H100 nodes.  Future Stage IV 3D packages generated by
scripts/package_stageiv_topology_3d_hpc.py now default to
EXCLUDE_NODES=gpuh01,gpuh14, emit #SBATCH --exclude=gpuh01,gpuh14, and refuse
runtime hostnames gpuh01 or gpuh14.
Added a standalone uploadable recovery wrapper for the current iter000 rank-0
case.  The wrapper calls the package-local failed-rank recovery command,
verifies all eight shards plus merged/trusted outputs and dataset_iter001, then
starts the controller from START_ITER=1.
```

Current state:

```text
The active HPC run does not need a full restart solely because of this rank-0
node failure.  The recommended action is failed-rank recovery for iter000
rank 0 with EXCLUDE_NODES=gpuh01,gpuh14, then resume the Stage IV controller
from START_ITER=1 after dataset_iter001.npz and dataset_iter001.csv exist.
This is an operational recovery change only; it does not change physics
definitions, tolerances, acquisition logic, exact-oracle behavior, or
StopController criteria.
```

### 2026-06-27: Stage IV 3D returned result check and transparent 3D report

Files changed/generated:

```text
scripts/build_stageiv_3d_return_report.py
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/scripts/stageiv_3d_convergence_audit.py
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_postrun_bundle_local/
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/
```

Summary:

```text
Checked the returned Stage IV 3D cold-start topology-aware full-loop result.
The returned file set is complete through dataset_iter025, with no missing
dataset, merge, trusted, or shard iterations detected.  A small report-only
bug in the returned convergence audit script was fixed so missing selected
point score columns are treated as NaN rather than aborting the acquisition
channel summary.  This does not alter datasets, exact-oracle outputs, physical
labels, acquisition behavior, or convergence thresholds.
```

Key result:

```text
final dataset = dataset_iter025.npz
final dataset sha256 = 7acab805115bcfcae56e94e2541bfdaa727de7767f2d7dd34330f163e2695842
final samples = 7081
thermodynamic counts = normal 2350, uniform_SC 141, FFLO 4590
topology counts = not_applicable 2350, trivial 3265, topological 1466,
                  gapless_SC 0, unresolved 0
trusted_exact = 6767
topology_trusted = 4731
needs_rerun_exact = 0
q_unresolved = 0
delta_unresolved = 0
```

Post-run decision:

```text
file_set_status = complete
stageiv_convergence_status = not_converged
postrun_bundle_status = hidden_slice_reference_missing
hidden_slice_status = inconclusive
mu_domain_complete = false
mu_range_limited = true

Topology audit limiting values:
topology_volume_map_change_last3 = [0.001412, 0.001923, 0.002561]
topology_surface_shift_p95_last3 = [0.005584, 0.005907, 0.005996]
topology_surface_coverage_p95_final = 0.015318
trusted_topology_surprise_last3 = [0.065, 0.1123, 0.140625]
topology_surface_component_count_last3 = [3, 4, 3]
```

Report outputs:

```text
stageiv_3d_return_report.md
stageiv_3d_return_report.pdf
decision_log.md
tables/*.csv
figures/*.png and figures/*.pdf
reproduction_manifest.json
```

Current state:

```text
The Stage IV 3D returned run is data-complete and includes useful thermodynamic
and topology-aware exact samples, but it should not be presented as a formally
converged 3D phase/topology map.  The next recommended step is to inspect the
failed topology/thermodynamic surface regions and supply the frozen Stage III
fixed-mu reference dataset for hidden-slice validation before launching any
additional exact work.
```

### 2026-06-27: Stage IV 3D report boundary-surface visualization update

Files changed/generated:

```text
scripts/build_stageiv_3d_return_report.py
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/
docs/report_qa/20260627_stageiv_3d_boundary_visualization_update.md
```

Summary:

```text
Enhanced the Stage IV 3D returned-result report figures by overlaying
smoothed diagnostic boundary geometry on the existing transparent 3D and
fixed-mu 2D visualizations.  The 3D thermodynamic figure now includes
semi-transparent smooth normal/SC and uniform_SC/FFLO boundary surfaces.  The
3D topology figure now also includes a smooth cFFLO/tFFLO diagnostic
topology-boundary surface.  The fixed-mu slice atlases now include smoothed
normal/SC, uniform_SC/FFLO, and cFFLO/tFFLO boundary curves where locally
supported by the final exact labels.
```

Implementation note:

```text
The added surfaces and curves are report-only diagnostic interpolants.  The
code first extracts locally supported boundary-crossing points using Delaunay
neighborhoods with long-edge filtering.  3D surfaces are then fit as smooth
RBF thin-plate-spline surfaces \(J_A/t=f(k_B T/t,\mu/t)\), while fixed-mu
2D boundary curves use binned-median smoothing splines
\(J_A/t=f(k_B T/t)\).  They improve readability but do not replace the
official Stage IV convergence metrics and do not change any dataset,
exact-oracle result, phase label, topology label, acquisition rule, or
tolerance.
```

Validation:

```text
python -m py_compile scripts\build_stageiv_3d_return_report.py
python ..\scripts\build_stageiv_3d_return_report.py --package-root .
pdftoppm -f 1 -l 4 -png -r 120 stageiv_3d_return_report.pdf tmp/pdf_render_stageiv_return_boundaries/page
```

Result:

```text
PDF rebuilt successfully with RGB white-background PNG embeds.
Rendered pages 1-4 were visually checked.  Smooth boundary overlays are visible
in the 3D phase/topology figures and in the fixed-mu slice atlases.  The
scientific decision remains unchanged: Stage IV 3D is data-complete but not
formally converged.
```

### 2026-06-28: Stage IV-A 3D audit from `docs/Audit.md`

Files changed/generated:

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/scripts/build_stageiv_3d_audit_report.py
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_audit_local/
docs/report_qa/20260628_stageiv_3d_audit_summary.md
```

Summary:

```text
Implemented and ran the report-only Stage IV-A audit requested by
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/docs/Audit.md.
The audit reads only existing returned artifacts: dataset_iter025, cumulative
convergence tables, selected-point metadata, hidden-slice audit provenance,
and the normal-state band-count preflight scan.  It writes a Markdown/PDF
audit report, decision JSON, config YAML, CSV tables, and PNG/PDF diagnostic
figures.  It does not run exact calculations, does not continue active
learning, and does not change any thermodynamic or topology labels.
```

Key audit outputs:

```text
stageiv_3d_audit_report.md
stageiv_3d_audit_report.pdf
stageiv_3d_next_action_decision.json
tables/stageiv_3d_hidden_slice_validation.csv
tables/stageiv_3d_mu_edge_contact.csv
tables/stageiv_3d_single_band_overlap.csv
tables/stageiv_3d_convergence_failure_decomposition.csv
tables/stageiv_3d_acquisition_forensics.csv
tables/stageiv_3d_surface_support_audit.csv
tables/stageiv_3d_hard_risk_distribution.csv
figures/*.png and figures/*.pdf
```

Decision:

```text
decision_class = Decision D
hidden_slice_status = inconclusive
lower_mu_range_limited = true
upper_mu_range_limited = false
single_band_hypothesis_status = supported
tFFLO single-pair-corridor fraction = 0.9654
cFFLO single-pair-corridor fraction = 0.3568
main_limiting_factor = trusted_topology_surprise
secondary_limiting_factors = coverage, surface_shift,
                              component_instability, mu_edge_limitation
from_scratch_restart_needed = false
```

Interpretation:

```text
The current returned run is useful but not a closed 3D topology result.
The Stage III hidden fixed-mu reference artifact is absent, so hidden-slice
recovery is inconclusive.  The lower mu edge is range-limited: tFFLO and
opposite-Z2 boundary-proxy points touch the lower edge, while the upper edge
does not show tFFLO contact in the main edge bands.  The normal-state
single-pair corridor hypothesis is positively supported as a diagnostic:
96.5% of trusted tFFLO points lie inside the preflight single-pair corridor.
The next step should collect the missing Stage III reference and then choose
between a same-window topology/spectral tail and a lower-mu extension.  A
from-scratch restart is not recommended.
```

Validation:

```text
python active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_3d_audit_report.py
PyMuPDF rendered page 1 of stageiv_3d_audit_report.pdf successfully.
Visual spot-checks were performed on the rendered PDF page and on
figures/stageiv_convergence_failure_summary.png.
```

### 2026-06-28: Stage IV-A mu-slice thermodynamic-boundary audit

Files changed/generated:

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/scripts/build_stageiv_mu_slice_boundary_audit.py
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_mu_slice_boundary_audit_local/
docs/report_qa/20260628_stageiv_mu_slice_boundary_audit.md
```

Summary:

```text
Implemented and ran the report-only mu-slice thermodynamic-boundary audit from
Slice_Audit_prompt.md.  The audit uses only the frozen Stage IV-A returned
dataset_iter025 and existing metadata.  It checks whether suspicious wide-mu
normal/SC boundary panels are broad-bin projection artifacts, smooth-fit
artifacts, sampling limitations, hard-risk/numerical reliability effects, or
real thermodynamic features.  It does not run exact calculations, does not
continue active learning, and does not modify labels, acquisition logic, or
tolerances.
```

Key audit outputs:

```text
stageiv_mu_slice_boundary_audit_report.md
stageiv_mu_slice_boundary_audit_report.tex
stageiv_mu_slice_boundary_audit_report.pdf
stageiv_mu_slice_boundary_audit_summary.json
stageiv_mu_slice_boundary_audit_decision.json
stageiv_mu_slice_boundary_audit_config.yaml
stageiv_wide_bin_projection_metrics.csv
stageiv_narrow_mu_slice_metrics.csv
stageiv_normal_sc_bracket_support.csv
stageiv_boundary_curve_support_audit.csv
stageiv_thermodynamic_margin_audit.csv
stageiv_surface_fit_sensitivity.csv
stageiv_acquisition_support_by_mu_bin.csv
stageiv_single_band_boundary_relation.csv
stageiv_hard_risk_near_anomaly.csv
stageiv_phase_tail_candidates_for_anomaly_bins.csv
figures/*.png and figures/*.pdf
```

Decision:

```text
final_classification = Class A - broad-bin projection artifact
recommended_next_action = curve_extraction_fix_only
need_new_exact_calculation = false
recommended_tail_candidate_count = 0
candidate_count_generated_for_traceability = 24
baseline_reproduced = true
anomaly_bins = 0.50_0.83, 0.83_1.17, 1.17_1.50
```

Interpretation:

```text
The suspicious high-mu normal/SC panels are best treated as wide-bin
projection artifacts, with a medium contribution from smooth diagnostic curve
fitting and partial numerical/hard-risk correlations.  Narrow fixed-mu slices
and mutual-KNN bracket support do not require new exact calculations for this
specific visualization issue.  The next practical step is to fix the curve
extraction/plotting logic for wide mu bins before considering any targeted
tail calculation.  This does not change the broader Stage IV-A status:
tFFLO is discovered and the single-band diagnostic remains useful, but formal
3D topology convergence has not passed.
```

Validation:

```text
python active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_mu_slice_boundary_audit.py
python -m py_compile active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_mu_slice_boundary_audit.py
pdflatex compiled stageiv_mu_slice_boundary_audit_report.tex successfully.
PyMuPDF rendered selected PDF pages for visual QA; pages 1, 6, and 11 were
spot-checked for readable text, figures, and caveats.
```

### 2026-06-28: Stage IV-A curve-extraction fix report

Files changed/generated:

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/scripts/build_stageiv_curve_extraction_fix_report.py
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_curve_extraction_fix_local/
docs/report_qa/20260628_stageiv_curve_extraction_fix.md
```

Summary:

```text
Implemented and ran the report-only curve-extraction repair requested by
Report_Re_prompt.md.  This is the concrete follow-up to the mu-slice boundary
audit's `curve_extraction_fix_only` recommendation.  The builder reads only
the frozen Stage IV-A `dataset_iter025` and existing audit outputs.  It
rebuilds wide-bin and narrow fixed-mu phase/topology atlases, computes local
normal/SC and cFFLO/tFFLO bracket support, removes unsupported smooth-curve
segments from the revised report display, and compiles a LaTeX PDF report.
It does not run exact calculations, does not continue active learning, and
does not modify thermodynamic labels, topology labels, acquisition logic, or
tolerances.
```

Key outputs:

```text
stageiv_curve_extraction_fix_summary.json
stageiv_curve_extraction_fix_config.yaml
stageiv_curve_extraction_fix_report.md
stageiv_curve_extraction_fix_report.tex
stageiv_curve_extraction_fix_report.pdf
decision_log.md
tables/stageiv_support_restricted_normal_sc_curves.csv
tables/stageiv_filtered_curve_segments.csv
tables/stageiv_curve_segment_support_summary.csv
tables/stageiv_topology_bracket_support.csv
figures/stageiv_revised_wide_bin_phase_atlas.png/pdf
figures/stageiv_revised_wide_bin_mu_colored_atlas.png/pdf
figures/stageiv_narrow_fixed_mu_phase_atlas.png/pdf
figures/stageiv_narrow_fixed_mu_phase_bracket_atlas.png/pdf
figures/stageiv_revised_topology_slice_atlas.png/pdf
figures/stageiv_narrow_fixed_mu_topology_bracket_atlas.png/pdf
```

Decision:

```text
baseline_reproduced = true
broad_bin_projection_artifact = partial
smooth_fit_artifact = medium
sampling_support_limited = false
need_new_exact_calculation = false
recommended_next_action = curve_extraction_fix_only
optional_traceability_candidate_count = 24
old_curve_segments_removed = 144 / 1080
unsupported_arc_fraction = 0.133333
stageiv_convergence_status_changed = false
```

Interpretation:

```text
The high-mu wide-bin normal/SC irregularity should be presented as a
projection diagnostic rather than a fixed-mu physical boundary.  The revised
figures remove unsupported smooth-curve artifacts, including the high-mu
vertical spike, and show local bracket support explicitly.  The optional 24
candidate points remain traceability-only and are not recommended exact tasks
under this decision.  This report fixes the atlas/curve presentation issue
but does not close the broader Stage IV-A 3D topology convergence problem.
The next scientific decision remains: collect the missing Stage III reference,
then choose between a same-window topology/spectral tail and a lower-mu
extension.
```

Validation:

```text
python -m py_compile active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_curve_extraction_fix_report.py
python active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624\scripts\build_stageiv_curve_extraction_fix_report.py
pdflatex compiled stageiv_curve_extraction_fix_report.tex successfully.
PyMuPDF rendered selected PDF pages.  Pages 1, 6, and 10 were visually
checked for readable text, figure alignment, and caveats.
```

### 2026-06-28: Stage V acquisition-v2 implementation and HPC package

Files changed/generated:

```text
ml_phase/stagev_acqv2.py
scripts/stagev_acqv2_select.py
scripts/stagev_acqv2_update_reward.py
scripts/stagev_acqv2_smoke.py
scripts/package_stagev_acqv2_hpc.py
tests/test_stagev_acqv2.py
reports/stagev_acqv2_smoke/
reports/stagev_acqv2_hpc_package/
hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/
hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc.tar.gz
docs/report_qa/20260628_stagev_acqv2_hpc_package.md
```

Summary:

```text
Implemented Stage V cold-start acquisition-v2 for the 3D
(kBT, JA, mu) active-learning problem.  The new selector builds a 1024-point
scrambled Sobol seed, then selects 64-point micro-batches using explicit
boundary-support channels for normal/SC, uniform/FFLO, P0 topology, Ppi
topology, and gap/nodal diagnostics.  Candidate logs include A0 components,
support distances, sparse-support factors, propensities, selected metadata,
and top-K candidate controls.  A linear learned-residual reward model is
updated after merge/append, but `lambda_t` stays zero until the logged reward
denominator and validation improvement are sufficient.
```

Why it matters:

```text
This starts the acquisition-function optimization stage without changing the
BdG exact oracle, thermodynamic phase criterion, topology definitions, or
StopController thresholds.  Stage V is explicitly cold-start and does not use
Stage III or Stage IV data/checkpoints for training initialization.  Stage IV
artifacts remain comparison/reporting inputs only.
```

HPC package:

```text
run_id = stagev_acqv2_boundary_support_learned_residual_3d_v1
output_root = ML_Phase_StageV_AcqV2
archive = hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc.tar.gz
archive_sha256 = see reports/stagev_acqv2_hpc_package/stagev_acqv2_package_summary.json
excluded_nodes = gpuh01,gpuh14
```

Validation:

```text
python -m py_compile ml_phase/stagev_acqv2.py scripts/stagev_acqv2_select.py scripts/stagev_acqv2_smoke.py
python scripts/stagev_acqv2_smoke.py --output-dir reports/stagev_acqv2_smoke
python -m pytest tests/test_stagev_acqv2.py -q
python scripts/package_stagev_acqv2_hpc.py
```

Results:

```text
stagev_acqv2_smoke = pass
tests/test_stagev_acqv2.py = 6 passed
package py_compile = pass
shell encoding/LF = pass
gpuh exclusion check = pass
local bash -n = skipped because Windows WSL bash has no installed distribution
```

Next recommended steps:

```text
Upload the Stage V HPC tarball, run the package smoke test on the cluster, then
submit the production loop only after setting CONFIRM_STAGEV_PRODUCTION=1.
Monitor early reward-model state carefully: `lambda_t` should remain zero until
reward history is large enough and the learned residual improves ranking over
A0.
```

### 2026-06-29: Stage V iter002 reward-model prediction hotfix

Files changed/generated:

```text
ml_phase/stagev_acqv2.py
tests/test_stagev_acqv2.py
hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc.tar.gz
```

Summary:

```text
The first Stage V HPC run completed iter000 and iter001, wrote dataset_iter002,
then failed while preparing iter002 acquisition.  The learned reward model had
been trained with `selection_probability`, but this field is not available for
pre-selection candidate scoring.  `predict_linear_value_model` now fills missing
candidate feature columns with 0.0 instead of raising KeyError.  This is an
acquisition-layer robustness fix and does not affect completed exact outputs.
```

Validation:

```text
python -m py_compile ml_phase/stagev_acqv2.py scripts/stagev_acqv2_select.py scripts/stagev_acqv2_update_reward.py scripts/package_stagev_acqv2_hpc.py
python -m pytest tests/test_stagev_acqv2.py -q
```

Result:

```text
tests/test_stagev_acqv2.py = 7 passed
updated archive_sha256 = 7e005c1f9494a03a9c822cd5a033cc6a6f888557520db74963918a03775e2c3c
resume recommendation = keep dataset_iter002 and restart with START_ITER=2
```

### 2026-06-29: Stage V HPC hotfix upload helper

Files changed/generated:

```text
hpc_hotfix_scripts/stagev_acqv2_iter002_reward_model_patch/
hpc_hotfix_scripts/stagev_acqv2_iter002_reward_model_patch.zip
```

Summary:

```text
Created a small uploadable hotfix folder for applying the Stage V iter002
reward-model prediction patch on the cluster without pasting heredoc Python into
the shell.  The Python script auto-detects the Stage V package root, patches
`ml_phase/stagev_acqv2.py`, verifies py_compile, prints the patched function
context, and prints the START_ITER=2 resume command.
```

Validation:

```text
python -m py_compile hpc_hotfix_scripts/stagev_acqv2_iter002_reward_model_patch/apply_stagev_acqv2_iter002_hotfix.py
BOM check passed for py/sh/readme files.
```

### 2026-06-30: Stage V Slurm-selection hotfix

Files changed/generated:

```text
scripts/package_stagev_acqv2_hpc.py
hpc_hotfix_scripts/stagev_acqv2_slurm_selection_hotfix/
hpc_hotfix_scripts/stagev_acqv2_slurm_selection_hotfix.zip
hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/
hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc.tar.gz
reports/stagev_acqv2_hpc_package/stagev_acqv2_package_summary.json
docs/report_qa/20260630_stagev_slurm_selection_hotfix.md
```

Summary:

```text
The Stage V HPC run reached dataset_iter014 after iter013 exact jobs completed
normally, then the outer control process printed `Terminated` before creating
iter014.  Since Slurm exact job 84000 completed successfully on all ranks, the
likely failure point is the CPU-heavy Stage V acquisition selection/training
step running on the login node.  A hotfix now moves `stagev_acqv2_select.py`
into its own Slurm job before each exact array.  The login node only submits and
waits for the selection job and the exact array.
```

Why it matters:

```text
This is a scheduling/package fix, not a scientific-definition change.  It does
not modify the exact oracle, thermodynamic labels, topology definitions,
StopController thresholds, acquisition score formula, or tolerances.  It
preserves the completed dataset_iter014 state and allows the run to resume from
START_ITER=14 without repeating earlier exact calculations.
```

Validation:

```text
python -m py_compile scripts/package_stagev_acqv2_hpc.py hpc_hotfix_scripts/stagev_acqv2_slurm_selection_hotfix/apply_stagev_acqv2_slurm_selection_hotfix.py
python scripts/package_stagev_acqv2_hpc.py
package py_compile = pass
shell encoding/LF = pass
gpuh exclusion check = pass
hotfix encoding check = UTF-8 without BOM, LF line endings
archive_sha256 = 4e98e14ca398afb93e8842bd51d2020e435ed16ad1100011939ed178a64d62c3
hotfix v2 zip_sha256 = 0387e2f24b8bd2a7548a5a4bd9a728d3464c25dce64b6a321076d947c718dd3b
```

Next recommended steps:

```text
Upload `hpc_hotfix_scripts/stagev_acqv2_slurm_selection_hotfix.zip` into the
existing Stage V HPC package root, apply the patch, then resume from
START_ITER=14.  The first expected new log marker is
`[submit] Stage V selection iter 014: job ...`; after that selection job
completes, the exact array for iter014 should be submitted.
```

### 2026-06-30: Stage IV return report JA-kBT comparison view

Files changed/generated:

```text
scripts/build_stageiv_3d_return_report.py
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/stageiv_3d_return_report.md
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/stageiv_3d_return_report.pdf
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/figures/phase_3d_jakt_view.png
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624/ML_Phase_StageIV_Topology3D/reports/stageiv_3d_return_report_local/figures/phase_3d_jakt_view.pdf
docs/report_qa/20260630_stageiv_jakt_view_update.md
```

Summary:

```text
The Stage IV local return report now includes a thermodynamic phase-map figure
viewed primarily in the kBT/t-J_A/t plane.  The figure keeps mu/t as 3D depth in
one panel and as opacity in a collapsed comparison projection in the second
panel, so it can be compared more directly with the earlier 2D kBT-J_A phase
maps.
```

Why it matters:

```text
This is a report-only visualization update.  It does not change the Stage IV
dataset, thermodynamic phase labels, topology labels, boundary extraction
diagnostics, exact oracle, acquisition logic, or numerical tolerances.  The
report caveat explicitly states that the kBT/t-J_A/t view is a projection of the
3D data cloud and is not a single fixed-mu phase diagram.
```

Validation:

```text
python -m py_compile scripts/build_stageiv_3d_return_report.py
python scripts/build_stageiv_3d_return_report.py --package-root active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624
pdflatex returncode = 0
stageiv_3d_return_report.pdf = 5 pages, 7792630 bytes
Rendered PDF pages 2 and 5 for visual checks; the new JA-kBT comparison figure
appears on page 2 and the caveat page renders without LaTeX underscore errors.
```

### 2026-07-01: Stage V acquisition-v2 local return report

Files changed/generated:

```text
scripts/build_stagev_acqv2_return_report.py
stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/ML_Phase_StageV_AcqV2/reports/stagev_acqv2_return_report_local/
docs/report_qa/20260701_stagev_acqv2_return_report.md
```

Summary:

```text
Generated a report-only local audit for the Stage V
stagev_acqv2_boundary_support_learned_residual_3d_v1 run.  The local copy
contains cumulative datasets through dataset_iter093, with 6892 samples:
normal=4855, uniform_SC=130, FFLO=1907.  FFLO topology counts are
cFFLO/trivial=1637 and tFFLO/topological=270.  The report includes 3D phase and
topology views, mu/t slice atlases, eta slices, phase/topology growth curves, a
fixed-probe kNN phase-map-change proxy, selected feature-margin diagnostics,
and learned-residual acquisition curves.
```

Why it matters:

```text
The learned residual appears active and useful in the available local data:
lambda_t=0.7 at the latest completed reward update, learned-model rank
correlation is about 0.789 versus about 0.209 for A0.  The report-only
fixed-probe phase-map-change proxy is about 0.00025 at the latest available
cumulative state.  However, the local return is not complete to dataset_iter100:
iter093 exact shards are present but no local exact_merged_iter093,
exact_trusted_iter093, or dataset_iter094+ was found.  Do not claim formal
100-iteration convergence from this local copy.
```

Validation:

```text
python -m py_compile scripts/build_stagev_acqv2_return_report.py
python scripts/build_stagev_acqv2_return_report.py --package-root stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc
pdflatex returncode = 0
stagev_acqv2_return_report.pdf = 12 pages
Rendered PDF pages 1, 2, 3, and 12 for visual checks.
```

### 2026-07-01: Stage V-v2 multi-head acquisition refactor and HPC package

Files changed/generated:

```text
ml_phase/stagev_v2.py
scripts/stagev_v2_select.py
scripts/stagev_v2_update_reward.py
scripts/stagev_v2_smoke.py
scripts/package_stagev_v2_hpc.py
tests/test_stagev_v2.py
reports/stagev_v2_hpc_package/
reports/stagev_v2_multihead_smoke/
hpc_packages/stagev_v2_multihead_boundary_learning_3d_v1_hpc/
hpc_packages/stagev_v2_multihead_boundary_learning_3d_v1_hpc.tar.gz
docs/report_qa/20260701_stagev_v2_multihead_acquisition_package.md
```

Summary:

```text
Implemented Stage V-v2 as a new cold-start acquisition package with run_id
stagev_v2_multihead_boundary_learning_3d_v1 and output_root
ML_Phase_StageV_V2_Multihead.  The v2 acquisition keeps the Stage V-v1
transparent boundary-support A0 channels but replaces the scalar learned
residual with independent heads for ns, uf, p0, ppi, and gap.  Each head has
its own reward normalization, validation metrics, lambda_s, and fallback to
A0_s.  The combined score uses per-boundary rank normalization, automatic
alpha_s priority updates, proposal-source density correction, and stochastic
micro-batch selection with propensities.
```

Why it matters:

```text
The Stage V-v1 return report showed useful learned residual rank correlation,
but normal/SC and global_sobol selection remained dominant while P0/Ppi
topology support stayed weak.  Stage V-v2 directly targets that failure mode
without changing the exact oracle, thermodynamic phase rule, Hamiltonian,
Pfaffian convention, q/Delta search, rankcap_k3 local refinement, topology
labels, or numerical tolerances.  The generated HPC package runs selection as
a Slurm job to avoid CPU-heavy login-node selection failures and excludes
gpuh01,gpuh14 in Slurm scripts and submit commands.
```

Validation:

```text
python -m py_compile ml_phase/stagev_v2.py scripts/stagev_v2_select.py scripts/stagev_v2_update_reward.py scripts/stagev_v2_smoke.py
python -m pytest tests/test_stagev_v2.py -q
python -m pytest tests/test_stagev_v2.py tests/test_stagev_acqv2.py -q
python scripts/stagev_v2_smoke.py --output-dir reports/stagev_v2_multihead_smoke
python scripts/package_stagev_v2_hpc.py
python hpc_packages/stagev_v2_multihead_boundary_learning_3d_v1_hpc/scripts/stagev_v2_preflight.py --config hpc_packages/stagev_v2_multihead_boundary_learning_3d_v1_hpc/configs/stagev_v2_smoke.json --output-dir hpc_packages/stagev_v2_multihead_boundary_learning_3d_v1_hpc/reports/preflight_smoke
python hpc_packages/stagev_v2_multihead_boundary_learning_3d_v1_hpc/scripts/stagev_v2_smoke.py --output-dir hpc_packages/stagev_v2_multihead_boundary_learning_3d_v1_hpc/reports/stagev_v2_multihead_smoke

Stage V-v2 tests: 9 passed.
Stage V-v1 + v2 acquisition tests: 16 passed.
Package py_compile_status: pass.
Local bash -n was skipped because the Windows host has no installed WSL distribution;
run bash scripts/run_stagev_v2_smoke.sh again on the Linux HPC package after upload.
Archive SHA256 is recorded in
reports/stagev_v2_hpc_package/stagev_v2_package_summary.json.
```

Known unresolved issues:

```text
Stage V-v2 has not yet run on the cluster, so no convergence or superiority
over Stage V-v1 is claimed.  The exact oracle schema was not modified; the
package records that free_energy_gap_to_normal is the stored alias for
F_SC_minus_F_normal.  Formal matched-budget comparison must be performed after
the Stage V-v2 production run returns.
```

### 2026-07-02: Stage I-V comprehensive report compilation

Files changed/generated:

```text
reports/stagei_to_stagev_comprehensive_report/
reports/stagei_to_stagev_comprehensive_report/scripts/generate_report.py
reports/stagei_to_stagev_comprehensive_report/main_report.tex
reports/stagei_to_stagev_comprehensive_report/main_report.md
reports/stagei_to_stagev_comprehensive_report/main_report.pdf
reports/stagei_to_stagev_comprehensive_report/tables/
reports/stagei_to_stagev_comprehensive_report/figures/
reports/stagei_to_stagev_comprehensive_report/decision_log.md
```

Summary:

```text
Generated a report-only Stage I-V synthesis following docs/StageI_V_Report.md.
The report compiles the project trajectory from 2D thermodynamic active
learning, robust exact-oracle reliability, 2D topology-aware convergence, 3D
topology discovery, and Stage V acquisition-function learning.  It outputs a
LaTeX PDF, Markdown handoff report, 15 PDF/PNG/SVG figures, artifact and figure
manifests, missing-artifact table, consistency checks, and build log.
```

Scientific status recorded:

```text
Stage III 2D topology-aware convergence is treated as the strongest formal
cFFLO/tFFLO result.  Stage IV is reported as 3D tFFLO discovery but not formal
3D convergence.  Stage V-v1 is reported as a learned-acquisition prototype.
Stage V-v2 is marked as a validated package and production run in progress,
not as a returned convergence result.
```

Validation:

```text
python reports\stagei_to_stagev_comprehensive_report\scripts\generate_report.py
pdflatex -interaction=nonstopmode -halt-on-error main_report.tex
PyMuPDF rendered pages 1, 4, 8, 9, and 14 for visual QA.
main_report.pdf SHA256:
2ae62ec89e55cba4ae28f3d545153330e28ceb9dbbf2e8ff68d46b04692058ae
```

Known unresolved issues:

```text
The Stage I/II artifact evidence remains spread across several rankcap and
Phase-II reports rather than a single original Stage I-only report.  The Stage
V-v2 HPC run is still in progress, so the report must be revised after the
matched-budget return.  The current PDF contains minor LaTeX overfull warnings
from long artifact identifiers in the audit table, but the report compiles and
the rendered pages are readable.
```

### 2026-07-03: Stage V-v2 Slurm accounting race documented

Files changed:

```text
docs/report_qa/20260703_stagev_v2_slurm_accounting_race.md
```

Summary:

```text
Recorded the operational diagnosis for apparent Stage V-v2 stops at iter024 and
iter084.  In both cases the loop wrapper reported a failure because `sacct`
temporarily returned `RUNNING:0:0` after `squeue` had cleared the job.  Later
`sacct` and output-file checks showed the jobs completed successfully and the
run could be resumed from the correct iteration.
```

Why it matters:

```text
The return report should classify these events as Slurm accounting race
conditions in the wrapper, not as exact-oracle, GPU, acquisition, or numerical
failures.  A future wrapper robustness patch should retry transient `sacct`
states and verify expected output files before declaring failure.
```
