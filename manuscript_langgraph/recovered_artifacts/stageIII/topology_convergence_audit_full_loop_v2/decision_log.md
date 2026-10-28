# Decision Log

Decision: Decision A

topology_main_converged: True

limiting_factor: none

need_new_exact_calculation: False

recommended_next_action: freeze_topology_boundary_result_for_offline_report

Evidence:

- last3 topology map change: [0.0004709998654286099, 0.0009083110460715547, 0.0004877720590708783]
- last3 boundary p95 shift: [0.0008148330596732811, 0.002624510467144085, 0.000871245354543369]
- final coverage p95: 0.0061854216098562
- last3 trusted surprise: [0.011904761904761904, 0.017142857142857144, 0.012578616352201259]
- final direct/bracket support fraction: 1.0

Caveats:

- online topology labels are audited offline here but still derive from the cold-start full-loop dataset
- absence of finite-area gapless_SC is not treated as a failure
- strict Hausdorff distance is diagnostic only; p95 shift is the main gate
- Delaunay or kNN opposite-label edges are used only as support/bracket diagnostics, not as the final contour
- sensitivity coverage margin is tight: k=6 is slightly above the nominal coverage threshold, while main k=8 and k=12 are below
