# Manuscript scope, claim approval, and transferred-data presence

Date: 2026-08-12

## Question

Given that the Stage II `dataset_iter035.npz` and Stage III endpoint/
convergence bundle are missing, what does the transferred package actually
contain, and are the data needed for the paper present? How should the
manuscript claims be positioned?

## Answer

The original transfer was explicitly the `Fu_FFLO Minimal Agent Package`: a
code/HPC handoff that omitted full history and large returned datasets. It
contains canonical physics/numerics documents, solver and audit source, tests,
orchestration scripts, and report/Q&A provenance. Later work added extensive
Stage V raw endpoints/audits, Stage VI offline reports, and the manuscript
LangGraph. It did not add the missing frozen Stage II--IV production runs.

The package is sufficient to approve bounded, report-traceable claim wording.
It is not sufficient to reproduce final Figures 2--4. Specifically absent are:

- Stage II final `dataset_iter035.npz` and final table/figure bundle;
- Stage III final `dataset_iter018.npz` and convergence/table/figure bundle;
- Stage IV final `dataset_iter025.npz`, return-report/table bundle, and 3D
  source figures.

Stage V data cannot replace these artifacts because they come from different
run identities and scientific stages.

## Approved manuscript position

The main line is:

> Reliability-aware active learning enables exploration of thermodynamic,
> topology-resolved, and higher-dimensional FFLO phase spaces.

Stages I--IV are main text. Stage IV is `supported_with_caveat`: it may be
called a higher-dimensional exploration or 3D extension that reveals
thermodynamic and topology-aware structures, but the strict 3D topology closure
used for the 2D result was not passed. “Complete 3D phase diagram” and “fully
converged 3D topology atlas” are forbidden. Stage V is Discussion-only and
exploratory, with no learned-acquisition superiority claim. Stage VI is
Outlook-only, with no validated EIG claim.

Current workflow state: claim approval is complete; figure approval remains a
separate human gate. No external Introduction retrieval is authorized.
