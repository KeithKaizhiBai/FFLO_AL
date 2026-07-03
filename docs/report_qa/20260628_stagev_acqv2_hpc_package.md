# Stage V Acquisition-v2 HPC Package

Date: 2026-06-28

## Context

Stage V starts a new acquisition-function development stage after the Stage IV-A
3D topology-aware run.  The goal is not to change the BdG exact oracle or the
thermodynamic/topology definitions.  The goal is to improve point selection by
using explicit boundary-support geometry and an online learned residual that is
kept in shadow mode until validated.

## Implemented Run Identity

```text
run_id = stagev_acqv2_boundary_support_learned_residual_3d_v1
output_root = ML_Phase_StageV_AcqV2
initial_seed_size = 1024
micro_batch_size = 64
max_micro_batches = 96
default final exact iteration = 96
```

## Acquisition Design

The base score \(A_0\) combines:

```text
normal_SC
uniform_FFLO
P0_topology
Ppi_topology
gap_nodal
```

Each boundary channel uses margin likelihood, uncertainty, and sparse support.
Boundary support is constructed from local mutual-kNN opposite-label brackets,
not raw global Delaunay long edges.  The final score is:

```text
A(x) = A0(x) * exp(lambda_t * g_theta(phi(x)))
```

`lambda_t` starts at zero and remains zero until enough logged reward rows exist
and the learned residual improves rank correlation relative to \(A_0\).

## HPC Package

```text
hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/
hpc_packages/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc.tar.gz
```

The package is cold-start and self-contained for code/config execution.  It does
not include Stage III or Stage IV datasets for training initialization.

Included configs:

```text
configs/stagev_acqv2_production.json
configs/stagev_acqv2_same_window.json
configs/stagev_acqv2_lower_mu_extension.json
configs/stagev_acqv2_smoke.json
```

Slurm scripts exclude:

```text
gpuh01,gpuh14
```

## Validation

Local validation completed:

```text
python -m py_compile ml_phase/stagev_acqv2.py scripts/stagev_acqv2_select.py scripts/stagev_acqv2_smoke.py
python scripts/stagev_acqv2_smoke.py --output-dir reports/stagev_acqv2_smoke
python -m pytest tests/test_stagev_acqv2.py -q
python scripts/package_stagev_acqv2_hpc.py
```

Results:

```text
smoke status = pass
pytest = 6 passed
package py_compile = pass
shell encoding = pass
local bash -n = skipped because Windows WSL bash has no installed distribution
excluded nodes present = true
```

No exact calculation was launched locally.
