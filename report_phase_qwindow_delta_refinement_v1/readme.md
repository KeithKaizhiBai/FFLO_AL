# Phase q-window and Delta-refinement audit package

This is an audit-only production-oriented numerical robustness package for the
discovery active-learning phase diagram.

It does not modify:

- active-learning acquisition;
- neural-network training;
- StopController;
- existing active-learning datasets.

## Purpose

The package checks high-\(J_A\), low-\(T\) normal/SC boundary-sensitive points
with two independent numerical safeguards:

1. expanded free-energy q-window scans;
2. near-zero Delta refinement.

The free-energy phase criterion is not changed. Finding one
positive-\(\Delta\) superconducting state with \(F_{\rm SC}<F_N\) remains enough
for basic superconducting classification within the scanned window.

Expanded q-window scans are used to check branch identity, \(q_{\rm opt}\)
stability, boundary robustness, and topology readiness.

## Prepared inputs

```text
input_points/qwindow_sensitive_points.csv
input_points/delta_sensitive_points.csv
input_points/clean_control_points.csv
input_points/combined_phase_audit_points.csv
```

Current setup counts:

```text
qwindow_sensitive_points = 342
delta_sensitive_points = 96
clean_control_points = 20
combined_unique_points = 345
```

## HPC commands

From the repository/package root on the cluster:

```bash
cd report_phase_qwindow_delta_refinement_v1
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python

sbatch scripts/submit_phase_qwindow_array.sh
sbatch scripts/submit_delta_refinement_array.sh
```

After both jobs finish:

```bash
bash scripts/collect_phase_audit_results.sh
```

The SLURM helper scripts use Unix LF line endings and exclude `gpuh01`.

## Expected final outputs

```text
tables/phase_qwindow_comparison.csv
tables/delta_refinement_comparison.csv
tables/low_energy_local_minima.csv
tables/combined_phase_robustness_summary.csv
figures/*.png
phase_qwindow_delta_refinement.md
phase_qwindow_delta_refinement.pdf
decision_log.md
```

## Interpretation caveat

The eta response has already been downgraded to response-extraction pathology
unless `eta_response_valid=True`. This package focuses on thermodynamic
phase-side q-window and Delta robustness, not on re-promoting high-\(J_A\)
positive eta.
