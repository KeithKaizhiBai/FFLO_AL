# Project Summary

Last updated: 2026-05-20

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
