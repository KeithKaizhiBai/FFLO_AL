# Decision Log: Phase q-window and Delta-refinement audit

Status: setup complete; exact q-window and Delta-refinement reruns are pending.

## Prepared Inputs

```json
{
  "qwindow_sensitive_points": 342,
  "delta_sensitive_points": 96,
  "clean_control_points": 20,
  "combined_unique_points": 345
}
```

## Decisions Already Fixed

- Do not modify active-learning acquisition, NN training, StopController, or
  the existing active-learning dataset.
- Do not redefine the free-energy phase criterion.
- Use q-window expansion to test branch identity, q_opt stability, boundary
  robustness, and topology readiness.
- Use near-zero Delta refinement to resolve tolerance-sensitive normal/SC
  boundary points.
- Do not treat eta response as robust positive physics unless
  `eta_response_valid=True`.

## Pending Numerical Checks

- Run `scripts/submit_phase_qwindow_array.sh`.
- Run `scripts/submit_delta_refinement_array.sh`.
- After both jobs complete, run `scripts/collect_phase_audit_results.sh`.

## Expected Final Outputs

- `tables/phase_qwindow_comparison.csv`
- `tables/delta_refinement_comparison.csv`
- `tables/low_energy_local_minima.csv`
- `tables/combined_phase_robustness_summary.csv`
- final `phase_qwindow_delta_refinement.md`
- final `phase_qwindow_delta_refinement.pdf`
