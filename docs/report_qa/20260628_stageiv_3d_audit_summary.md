# Stage IV-A 3D Audit Summary

This note records the report-level interpretation of the Stage IV-A audit
generated from `active_phase_topology_3d_t_ja_mu_from_scratch_v1`.

The returned run is complete through `dataset_iter025` with 7081 samples:
2350 normal, 141 uniform-SC, and 4590 FFLO.  Topology labels are present for
the superconducting data with 3265 trivial and 1466 topological points, and
no gapless-SC or unresolved topology labels in the final dataset.

The audit decision is `Decision D` because the Stage III frozen fixed-mu
reference artifact is missing, so the hidden-slice recovery test is
inconclusive.  This is an artifact/provenance blocker, not a proof that the
Stage IV run failed to reproduce the Stage III slice.

The main numerical blocker in the available 3D convergence metrics is trusted
topology surprise.  The last three trusted topology surprise values are
0.065, 0.112299, and 0.140625, all above the 0.02 gate.  Secondary blockers
are topology surface coverage, topology surface shift, component instability,
and lower-mu edge contact.

The lower edge of the production window, \(\mu/t=-0.5\), is range-limited.
At width 0.08, there are 130 trusted tFFLO points and 263 cFFLO/tFFLO
boundary-proxy points near the lower edge.  The upper edge does not show
tFFLO contact in the corresponding edge bands.

The normal-state single-pair corridor diagnostic is positively supported:
96.54% of trusted tFFLO points lie inside the preflight single-pair corridor,
compared with 35.68% of trusted cFFLO points.  This corridor remains a
diagnostic only and must not be used to relabel exact topology results.

The recommended next action is to collect the missing Stage III reference
artifact first, then choose between a same-window topology/spectral tail and
a lower-mu extension based on the edge audit.  A from-scratch restart is not
recommended.
