# Stage V-v4 Phase-Structure Graph Acquisition Launch Note

Date: 2026-07-08

## Purpose

Stage V-v4 upgrades the acquisition logic from known-boundary score refinement
to an evolving phase-structure graph acquisition policy.  The production run is
strict cold start and does not initialize from Stage III, Stage IV, Stage V-v1,
Stage V-v2, Stage V-v2-tail, or Stage V-v3 datasets.

## Core Method

The run maintains a graph \(G_t=(V_t,E_t)\).  Nodes represent known or
provisional candidate phase regions.  Edges represent local adjacency,
boundary brackets, topology transitions, or oracle-anomaly interfaces.

The acquisition target is an approximate expected reduction of graph
uncertainty:

\[
E[\Delta U(G_t)\mid x,\mathrm{action}].
\]

The implementation includes:

- observation descriptors from exact-output thermodynamic, topology, branch,
  and reliability fields;
- graph node/edge mining with spatial connectedness and descriptor
  separation;
- graph uncertainty scores for node, edge, support, component, reliability,
  and oracle-anomaly uncertainty;
- action families for global exploration, known-boundary refinement,
  candidate-phase validation, and oracle-anomaly audit;
- graph-value reward and an online graph-value model;
- hard validation that every selected point has nonblank `proposal_source` and
  `action_family`.

## What Was Not Changed

The implementation does not change:

- exact free-energy oracle;
- BdG Hamiltonian;
- thermodynamic phase criterion;
- Pfaffian topology criterion;
- topology gap oracle;
- q/Delta refinement rules;
- production parameter domain.

Candidate phase nodes are provisional diagnostics and do not override the
canonical normal / uniform-SC / FFLO / cFFLO / tFFLO labels.

## Validation

Local validation:

```text
python -m py_compile ml_phase/stagev_v4.py scripts/stagev_v4_select.py scripts/stagev_v4_update_reward.py scripts/stagev_v4_smoke.py
python scripts/stagev_v4_smoke.py --output-dir reports/stagev_v4_graph_acq_smoke
python -m pytest tests/test_stagev_v4.py
python scripts/package_stagev_v4_hpc.py
```

Result:

```text
pytest: 7 passed
package_sha256: 0206e1abfd1d30f627aca37813c18d83d0a43f2512bab68927988e751e8963ca
remote_preflight: pass
remote_smoke: pass
```

## HPC Launch

Remote package:

```text
/public_hw/home/sci_bfu/bkz/Fu_FFLO/stagev_v4_phase_structure_graph_acquisition_3d_v1_hpc
```

Official initial jobs:

```text
selection iter000: 91088 COMPLETED
exact iter000: 91089_[0-7] RUNNING
```

The official exact array is running on non-gpuh01 nodes.

## Recovery Note

The first SSH submission timed out locally but did start the controller and
exact job 91089.  A second recovery attempt briefly produced a duplicate
iter000 exact array and an invalid iter001 branch based on 16 exact points.
The duplicate exact array 91097 and invalid iter001 exact array 91106 were
cancelled.  Invalid generated files were moved to:

```text
recovery_quarantine_invalid_iter001_20260708_1308/
```

The remaining valid controller is waiting on exact job 91089.  After 91089
finishes, verify that the regenerated `dataset_iter001` contains the full
1024 seed exact outcomes and that no duplicate Stage V-v4 exact arrays remain.
