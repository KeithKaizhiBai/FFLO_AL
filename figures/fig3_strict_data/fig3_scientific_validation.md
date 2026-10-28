# Figure 3 scientific validation

- Frozen endpoint: `dataset_iter018.npz` (4345 rows).
- Trusted gapped FFLO mask: `topology_trusted == 1 and topology_applicable == 1 and y_phase == 2`.
- Trusted topology counts: **2290 cFFLO** and **923 tFFLO**.
- Boundary layer: archived deterministic k=8 IDW contour at iteration 18; no new extraction.
- Configured result: **Decision A**; final coverage p95 **0.006185** below **0.00625**.
- Sensitivity caveat: k=6 coverage p95 **0.006269** is slightly above threshold.
- Gap closing is a transition diagnostic, not a stable third topology phase.
- Brackets are diagnostics only.

`GATE_FIG3_HUMAN_SCIENCE_REVIEW = HUMAN_WAIT`
