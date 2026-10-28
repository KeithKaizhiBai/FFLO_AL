# Figure 3 specific validation

- Status: **PASS**
- Checks: **6/6**
- Physics recomputed: **no**

| Result | Check | Detail |
|---|---|---|
| PASS | `endpoint_hash` | dataset_iter018.npz |
| PASS | `trusted_FFLO_count` | expected 3213 |
| PASS | `topology_counts` | expected 2290/923 |
| PASS | `final_boundary_rows` | iteration 18 archived geometry |
| PASS | `k8_pass_k6_caveat` | k=8 below and k=6 above threshold |
| PASS | `no_physics_recompute` | direct frozen fields and archived audit tables only |
