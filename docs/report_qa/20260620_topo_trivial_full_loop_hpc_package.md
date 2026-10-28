# 2026-06-20 Topology/Trivial Full-Loop HPC Package

## Question

Build the Stage III topology/trivial full-loop HPC package according to
`docs/Topo_Trivial_FullLoop_Build.md`, with topology-aware acquisition enabled
for a cold-start full loop.

## Package Built

```text
package root:
    hpc_packages/active_phase_topology_from_scratch_full_loop_v1_hpc/

archive:
    hpc_packages/active_phase_topology_from_scratch_full_loop_v1_hpc.tar.gz

sha256:
    9e024e12e95d2674b87d43154990749f41f7753012814951b5f35d429ba513b8

run_id:
    active_phase_topology_from_scratch_full_loop_v1

output_root:
    ML_Phase_512_TopoTrivial_FullLoop
```

The package includes the source snapshot, topology oracle, Stage III scripts,
selected docs, `topology_pass_dataset_iter035_v1` reference outputs, package
manifest, validation JSON/CSV, README, and submit/status/collect/preflight
scripts.

## Important Decision

The package is now production-submission ready.  The submit script keeps one
explicit full-loop guard:

```bash
export CONFIRM_TOPO_FULL_LOOP=1
```

It no longer requires `TOPOLOGY_AWARE_ACQUISITION_READY`, because the packaged
source now includes the `topo_trivial` acquisition profile and exact-oracle
topology diagnostics.

The submitted run is a cold-start topology-aware active-learning loop, not a
continuation from `dataset_iter035`.  `dataset_iter035` remains the frozen Stage
II thermodynamic dataset and reference input for offline topology work.

## Implemented Support

The active-learning initialization supports:

```text
initialization = sobol_scrambled
```

for cold-start discovery over the full \((k_B T, J_A)\) domain.

The exact oracle now writes topology diagnostic fields when enabled:

```text
topology_label_code
topology_z2
topology_p0
topology_ppi
topology_pf_product
topology_pfaffian_margin
topology_bulk_gap
topology_k_at_bulk_gap
topology_trusted
topology_runtime_sec
```

The acquisition layer adds a `topo_trivial` profile using:

```text
A_phase        thermodynamic discovery score
A_spectral     Pfaffian-margin / Z2-edge / gapless-edge score
A_coverage     distance from trusted topology-labeled samples
```

The package submit script sets:

```text
ACQUISITION_PROFILE=topo_trivial
TOPOLOGY_CLASSIFICATION_FLAG=--enable-topology-classification
TOPOLOGY_GAP_BACKEND=gpu
TOPOLOGY_GAP_NK=2048
EXCLUDE_NODES=gpuh01
```

## Validation

Commands run:

```bash
python -m py_compile ml_phase/topology_oracle.py ml_phase/exact_oracle.py ml_phase/acquisition.py ml_phase/active_refine.py ml_phase/config.py scripts/package_topo_trivial_full_loop_hpc.py
python scripts/package_topo_trivial_full_loop_hpc.py
python scripts/preflight_topo_trivial_full_loop.py
python -m py_compile ml_phase/topology_oracle.py ml_phase/exact_oracle.py ml_phase/acquisition.py ml_phase/active_refine.py ml_phase/config.py scripts/run_topology_pass_dataset_iter035.py scripts/preflight_topo_trivial_full_loop.py
```

Additional local smoke validation confirmed that `topo_trivial` acquisition
returns finite scores and topology component arrays on synthetic monitor data.

The rebuilt package validation status is:

```text
validation_status = pass
production_submission_status = ready
package_tree_hash = 6138d2b3a94bfcdae8c36b9c81a0f3633498eb87ff828f91758103c942e86d7e
archive_size_bytes = 21728217
```

Package-local preflight reproduced the Stage III v1 reference topology counts:

```text
uniform_SC trivial = 715
FFLO trivial = 3127
FFLO topological = 1515
FFLO gapless_SC = 195
FFLO topology_unresolved = 15
trusted gapped FFLO = 4642
raw Z2-change Delaunay edges = 182
raw P0 sign-change edges = 182
raw Ppi sign-change edges = 0
raw large-circumradius triangles = 447
```

The package scripts are ASCII/LF/no-BOM and default to excluding `gpuh01`.

## Caveat

The active-loop topology fields are online diagnostics used by acquisition.
Publication-grade topology claims should still be made from a post-run offline
topology pass over the final thermodynamic dataset, with the same Pfaffian and
bulk-gap audit standards used for `dataset_iter035`.

## Next Step

Upload:

```text
hpc_packages/active_phase_topology_from_scratch_full_loop_v1_hpc.tar.gz
```

Then run package preflight on the cluster and submit with
`CONFIRM_TOPO_FULL_LOOP=1`.
