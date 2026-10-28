# Stage V-v2 Multi-head Acquisition Package

Date: 2026-07-01

## Context

Stage V-v1 showed that the scalar learned residual could learn useful ranking
structure, but the selected batches remained dominated by normal/SC and
`global_sobol` proposals.  P0/Ppi topology support stayed comparatively weak.

## What Changed

Stage V-v2 introduces a new cold-start package:

```text
run_id = stagev_v2_multihead_boundary_learning_3d_v1
output_root = ML_Phase_StageV_V2_Multihead
```

The acquisition keeps the transparent Stage V-v1 boundary-support base score
but replaces the single scalar learned residual with independent heads:

```text
ns, uf, p0, ppi, gap
```

Each head has its own reward normalization, validation metric, `lambda_s`, and
fallback to `A0_s`.  Scores are rank-normalized per boundary before
log-sum-exp combination.  Automatic `alpha_s` priority updates are driven by
boundary-support deficits rather than manual selected-point quotas.

## What Did Not Change

No physical definitions changed:

```text
thermodynamic phase rule unchanged
exact oracle unchanged
Hamiltonian unchanged
Pfaffian convention unchanged
q/Delta search unchanged
rankcap_k3 local refinement unchanged
topology labels unchanged
numerical tolerances unchanged
```

The exact output schema was not modified.  Existing `free_energy_gap_to_normal`
is treated as the stored \(F_{\rm SC}-F_{\rm normal}\) margin for acquisition
diagnostics.

## Validation

```text
python -m pytest tests/test_stagev_v2.py -q
9 passed

python -m pytest tests/test_stagev_v2.py tests/test_stagev_acqv2.py -q
16 passed

python scripts/stagev_v2_smoke.py --output-dir reports/stagev_v2_multihead_smoke
status = pass

python scripts/package_stagev_v2_hpc.py
py_compile_status = pass
excluded_nodes_present = true
selection_runs_under_slurm = true
```

Archive:

```text
hpc_packages/stagev_v2_multihead_boundary_learning_3d_v1_hpc.tar.gz
sha256 = 3f5224ce38e6251582a231c72f978c2342ae235bc11aaf6971f890a3a0529ea1
```

## Caveat

The local machine has no installed WSL distribution, so local `bash -n` checks
were marked as skipped.  The shell scripts are UTF-8/LF normalized and should be
checked again on the Linux HPC package with:

```bash
bash scripts/run_stagev_v2_smoke.sh
```

Do not claim Stage V-v2 convergence or superiority over Stage V-v1 before the
production run and matched-budget comparison return.
