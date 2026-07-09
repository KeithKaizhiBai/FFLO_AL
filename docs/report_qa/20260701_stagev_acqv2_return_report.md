# Stage V Acquisition-v2 Local Return Report

Date: 2026-07-01

## Context

The Stage V run
`stagev_acqv2_boundary_support_learned_residual_3d_v1` is a cold-start 3D
active-learning run in \((k_B T/t, J_A/t, \mu/t)\).  Its acquisition score uses
explicit boundary-support channels plus a learned residual value model:

\[
A(x) = A_0(x)\exp(\lambda_t g_\theta(\phi(x))).
\]

This note records the local return-report result generated from the downloaded
folder:

```text
stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/ML_Phase_StageV_AcqV2
```

## Local Data Completeness

The local copy currently contains cumulative datasets through:

```text
dataset_iter093
```

The corresponding latest cumulative sample count is:

```text
total = 6892
normal = 4855
uniform_SC = 130
FFLO = 1907
```

The local folder also contains `iter093` exact shards, but no local
`exact_merged_iter093`, `exact_trusted_iter093`, or `dataset_iter094+` was
found.  Therefore the local report should not be interpreted as a completed
100-iteration convergence audit until the missing final cumulative outputs are
downloaded or reconstructed.

## Main Numerical Observations

For the latest available cumulative dataset:

```text
FFLO trivial / cFFLO = 1637
FFLO topological / tFFLO = 270
SC gapless/unresolved = 0
```

The report-only fixed-probe kNN phase-map-change proxy is:

```text
latest map-change proxy = 0.0002508
supported probe fraction = 0.996875
```

This is a useful stability diagnostic, but it is not a formal StopController
convergence result.

## Acquisition Learning

At the latest completed reward update:

```text
lambda_t = 0.7
rank_correlation_A0 = 0.2089
rank_correlation_model = 0.7894
reward_history_rows = 6976
```

This supports the interpretation that the learned residual became active and
improved ranking over the transparent base acquisition score \(A_0\) on the
logged reward diagnostic.  It does not by itself prove final phase-boundary
convergence.

## Generated Report

The generated report is:

```text
stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/ML_Phase_StageV_AcqV2/reports/stagev_acqv2_return_report_local/stagev_acqv2_return_report.pdf
```

Companion Markdown, CSV tables, PNG figures, decision log, and reproduction
manifest are in the same report directory.

## Caveat for Later Writing

Use this report as a local-return diagnostic and visualization summary.  Do not
claim that the 100-iteration run has been fully audited from this local copy
unless `dataset_iter100` or the equivalent final cumulative state is available.
