# Method completeness matrix

| Area | Status | Authority |
|---|---|---|
| Physical model/free energy | complete | code + model specification |
| Exact oracle | complete | exact_oracle.py + production script |
| Surrogate/training | complete | models.py + run config |
| Acquisition/microbatch | complete | acquisition.py + run config |
| Exact-call accounting | complete | frozen endpoint + selections |
| Boundary enrichment | unavailable | missing candidate-wide archived labels |
| Wall-time speedup | not authorized | no matched exhaustive baseline |
| Stopping diagnostics | complete, operational only | stop_controller.py |
| Topology | complete with sensitivity caveat | topology code + frozen tables |
