# Model Specification

The current canonical model specification is maintained at the repository root:

```text
MODEL_SPEC.md
```

This pointer file exists because `AGENTS.md` requires a project-memory document
at `docs/MODEL_SPEC.md`. Keep the root `MODEL_SPEC.md` as the single source of
truth for the Hamiltonian, observables, labels, active-learning objective,
adaptive q-window policy, and Delta-refinement policy.

If the physical model or parameter convention changes, update the root
`MODEL_SPEC.md` first, then update `docs/PROJECT_SUMMARY.md` and
`docs/DECISIONS.md` as needed.

