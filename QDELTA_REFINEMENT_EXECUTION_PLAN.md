# Q/Delta Refinement Execution Plan

This plan is written for Codex to execute directly in this repository.

Scope: implement the unfinished q-window expansion, Delta local refinement,
trusted-dataset filtering, and later the active-learning loop. The current
request is to prepare an executable plan and HPC submission path; the actual
multi-iteration active-learning loop should be implemented in the next step.

## 1. Current State

Already implemented:

```text
ml_phase/exact_oracle.py records q-window metadata:
    q_min, q_max, n_q, q_index, q_edge_distance, q_edge_hit

ml_phase/exact_oracle.py records Delta ambiguity metadata:
    delta_min, delta_max, n_delta, delta_boundary_ambiguous,
    free_energy_gap_to_normal

ml_phase/acquisition.py scores:
    q_edge_risk_score
    delta_refine_risk_score
    extrapolation_risk_score

ml_phase/hpc.py writes:
    rerun_points.csv after shard merge
```

Still missing:

```text
1. actual q-window expansion reruns;
2. actual local Delta refinement rescans;
3. trusted-vs-ambiguous exact output filtering before dataset append;
4. multi-iteration active-learning loop using rerun_points.csv;
5. production H100 packaging and one-click execution flow.
```

## 2. Non-Negotiable Constraints

Do not change:

```text
Hamiltonian convention
Nambu basis convention
phase-label definitions
axis convention: x = k_B T/t, y = J_A/t
warm-start dataset convention: row = JA, col = kT
existing full-grid CUDA workflow unless a small reusable helper is required
```

The ML model remains a scheduler. Exact BdG results remain the source of truth.

## 3. Implementation Milestone A: q-Window Expansion

Target files:

```text
ml_phase/exact_oracle.py
eta_phase_diagram_cuda.py only if a small helper is unavoidable
docs/NUMERICS_SPEC.md
docs/PROJECT_SUMMARY.md
```

Required behavior:

```text
1. Evaluate a point with the default q window.
2. If q_edge_hit is false, keep the result.
3. If q_edge_hit is true, rebuild q_vec with an expanded window.
4. Rerun the exact point.
5. Repeat until q_opt is away from q_min/q_max or max expansion is reached.
6. Record q_refinement_level.
7. If max expansion is reached and q_edge_hit remains true, mark exact_status_code.
```

Recommended CLI flags:

```text
--enable-q-expansion
--q-expand-factor 1.5
--q-expand-pad-steps 50
--q-max-abs 3.141592653589793
--max-q-refinements 3
```

Acceptance checks:

```text
python -m py_compile ml_phase/exact_oracle.py

Run 1-3 cheap CPU or GPU test points with q expansion disabled and enabled.
Confirm output npz contains:
    q_refinement_level
    q_edge_hit
    exact_status_code
```

Do not append q-edge-hit points to trusted data without either correcting or
flagging them.

## 4. Implementation Milestone B: Local Delta Refinement

Target files:

```text
ml_phase/exact_oracle.py
eta_phase_diagram_cuda.py only if a helper is unavoidable
docs/NUMERICS_SPEC.md
docs/PROJECT_SUMMARY.md
```

Required behavior:

```text
1. Evaluate a point with the default Delta grid.
2. If Delta is not near the boundary and free-energy ambiguity is low, keep it.
3. If Delta_opt ~= DELTA_EPS or the free-energy minimum is shallow, rescan a
   narrower Delta interval around Delta_opt.
4. Record n_delta_refined and delta_refinement_level.
5. Use the refined Delta_opt for phase labels and observables if the exact
   helper returns a fully consistent refined q/Delta result.
```

Recommended CLI flags:

```text
--enable-delta-refinement
--delta-refine-half-width 0.03
--n-delta-refined 300
--max-delta-refinements 2
```

Acceptance checks:

```text
point near normal/SC boundary:
    delta_boundary_ambiguous should trigger refinement
    delta_refinement_level should be > 0

point deep in SC region:
    delta_refinement_level should remain 0
```

If implementing full refined current response is too invasive, first implement
the refined Delta metadata and rerun-list behavior, then only trust the original
observable values after explicit validation.

## 5. Implementation Milestone C: Trusted Dataset Filtering

Target files:

```text
ml_phase/hpc.py
ml_phase/active_refine.py
ml_phase/dataset_builder.py
```

Required behavior:

```text
1. After merge, split exact results into trusted and rerun-required subsets.
2. Trusted means exact_status_code == 0.
3. Ambiguous q/Delta points go to rerun_points.csv.
4. The future append step must append trusted points only by default.
5. Provide a CLI override such as --allow-ambiguous-append only for debugging.
```

New helper to add:

```python
def split_trusted_exact_results(merged_npz: Path) -> tuple[Path, Path]:
    """
    Write exact_trusted_iterXXX.npz and rerun_points.csv.
    """
```

Acceptance checks:

```text
synthetic merged npz with exact_status_code=[0,1,2,3]
    trusted file should include only status 0
    rerun_points.csv should include status 1/2/3
```

## 6. Implementation Milestone D: Multi-Iteration Loop

Implement only after A-C are validated.

Target files:

```text
ml_phase/active_loop.py or extend ml_phase/active_refine.py carefully
hpc_one_click_submit.sh
docs/PROJECT_SUMMARY.md
```

Required behavior:

```text
for iter in 0..N-1:
    prepare selected shards
    submit exact H100 array
    wait or resume after completion
    merge exact shards
    split trusted/rerun
    append trusted exact results
    retrain surrogate
    stop if convergence criteria are met
```

Do not block the whole design on one monolithic SLURM job. Keep CPU-side
selection, H100 exact evaluation, merge, and report as separable steps.

## 7. HPC Packaging Strategy

Use a source-focused upload bundle. Do not upload local generated outputs unless
explicitly needed.

Include:

```text
ml_phase/
scripts/
report/
docs/
eta_phase_diagram_cuda.py
tfflo_1d_cuda.py
plot_eta_phase_diagram.py
finite_T_phase_diagram.m
MODEL_SPEC.md
PROJECT_SUMMARY.md
QDELTA_REFINEMENT_EXECUTION_PLAN.md
RUN_ORDER_GBU_HPC.md
Active_Learning_Phase_Boundary_Refinement_Plan.md
hpc_one_click_submit.sh
eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/
```

Exclude:

```text
id_rsa
.env
__pycache__/
tmp/
PD_ML/
.history/
large generated ML_Phase outputs
old slurm logs
```

Create the bundle locally with:

```powershell
powershell -ExecutionPolicy Bypass -File package_hpc_upload.ps1
```

Upload the resulting archive to the cluster, then unpack:

```bash
tar -xzf hpc_upload_qdelta_YYYYMMDD_HHMMSS.tar.gz
cd hpc_upload_qdelta_YYYYMMDD_HHMMSS
```

## 8. HPC One-Click Smoke Run

On the cluster login node, after unpacking:

```bash
export PROJECT_DIR=$PWD
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
export CONDA_ENV=
export RUN_ID=active_boundary_qdelta_smoke
export POINTS_PER_ITER=32
export WORLD_SIZE=4
export N_ENSEMBLE=2
export REG_EPOCHS=20
export CLS_EPOCHS=20
export BATCH_SIZE=1024
bash hpc_one_click_submit.sh
```

Expected outputs:

```text
ML_Phase/active_runs/<RUN_ID>/iter000/selected_points.csv
ML_Phase/active_runs/<RUN_ID>/iter000/exact_merged_iter000.npz
ML_Phase/active_runs/<RUN_ID>/iter000/rerun_points.csv
ML_Phase/reports/active_learning_phase_boundary_report.tex
ML_Phase/reports/active_learning_phase_boundary_report.pdf if pdflatex exists
```

## 9. HPC One-Click Larger Run

After the smoke run succeeds:

```bash
export RUN_ID=active_boundary_qdelta_v1
export POINTS_PER_ITER=128
export WORLD_SIZE=8
export N_ENSEMBLE=5
export REG_EPOCHS=240
export CLS_EPOCHS=240
export BATCH_SIZE=512
bash hpc_one_click_submit.sh
```

Do not jump directly to 512-2048 points until q/Delta rerun rates and seconds
per point are measured.

## 10. Verification Before Declaring Success

Required checks:

```text
1. candidate_scores.csv includes q_edge_risk_score and delta_refine_risk_score;
2. exact shard npz files include q_edge_hit and delta_boundary_ambiguous;
3. merge writes rerun_points.csv even when empty;
4. report builder completes without missing figure errors;
5. H100 logs contain cuda available inside the job or exact oracle succeeds;
6. no selected point has kT < 0;
7. high-JA selected points are tracked as q-risk candidates.
```

## 11. Final Coding Order for Codex

Execute in this order:

```text
1. Add q-expansion CLI flags and helper functions.
2. Add q-expanded rerun path for one point.
3. Test exact_oracle on 1 point.
4. Add Delta-refinement CLI flags and helper functions.
5. Test Delta-refinement metadata on synthetic or small exact points.
6. Add trusted/rerun split helper to hpc.py.
7. Update active_refine append logic to trusted-only.
8. Update report_builder with q/Delta ambiguity summaries.
9. Run local dry-run.
10. Package and run H100 smoke via hpc_one_click_submit.sh.
11. Only then implement the multi-iteration loop.
```

