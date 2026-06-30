# 2026-06-09 Free-Energy Flow and FFLO-Altermagnetic System Diagrams

## Context

The second major-stage report now needs explanatory material before the
stage-by-stage numerical figures. The goal is to make two things visible:

1. what physical FFLO + altermagnetic BdG system is being studied;
2. how the exact oracle turns one \((k_B T/t,J_A/t)\) point into
   \(\Delta_{\rm opt}\), \(q_{\rm opt}\), \(\Delta F_{\min}\), phase labels,
   and local-refinement metadata.

## Added Report Figures

The report directory
`project_history/reports/report_local_refinement_refactor_note/` now includes:

```text
figures/fig17_free_energy_minimization_flow.png
figures/fig18_fflo_altermagnetic_system_schematic.png
tables/diagram_source_notes.csv
scripts/plot_free_energy_flow_and_system.py
```

`fig18_fflo_altermagnetic_system_schematic.png` is a conceptual system diagram.
It shows the one-dimensional altermagnetic superconducting chain, finite-\(q\)
pairing in momentum space, schematic phase-control axes, and exact-oracle
outputs.

`fig17_free_energy_minimization_flow.png` is a code-path-aware flowchart. It
summarizes the sequence:

```text
thermodynamic point and EtaPhaseConfig
-> BdG tensor construction
-> coarse CUDA scan over q and Delta
-> DeltaF(q) branch curve and local-minimum candidates
-> basin construction and risk annotation
-> rank-and-cap target selection
-> local-box refinement
-> exact label and training metadata
```

## Important Caveat

These figures are explanatory report material. They do not change the
Hamiltonian, phase thresholds, exact oracle, acquisition function, tolerance
policy, or local-refinement production behavior.
