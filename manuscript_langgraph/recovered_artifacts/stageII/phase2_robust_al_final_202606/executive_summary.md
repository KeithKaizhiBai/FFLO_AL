# Phase-II Robust Active-Learning Convergence, Numerical Reliability, and Optimization Report

## Executive Summary

Phase-II converts a sequence of small validation reports into one reproducible
story: starting from exact seed labels, the active-learning loop repeatedly
trained a surrogate, selected a full-acquisition batch, called the robust exact
oracle, appended trusted labels, and stopped only after the main phase map and
main thermodynamic boundaries stabilized.

Final frozen state:

```text
main_phase_converged = True
publication_boundary_audit = pass
publication_ready_for_main_phase_map = True
frozen dataset = dataset_iter035
total samples = 7434
normal / uniform_SC / FFLO = 1867 / 715 / 4852
trusted surprise = 0/127
hard-risk surprise = 75/129
targeted rerun count = 0
```

The hard-risk frontier remains active and must be shown as an uncertainty
layer.  It does not invalidate the main thermodynamic phase map under the
publication-grade boundary-impact audit.

